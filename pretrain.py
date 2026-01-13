

from typing import Dict, Tuple, Union

import torch, os, math
import torch.nn as nn
import torch.nn.functional as F
import freerec
from transformers import LlamaModel, LlamaConfig
from einops import rearrange, repeat

from sampler import TrainRandomWalkSource, ValidRandomWalkSource, CUTS


DTYPE = torch.bfloat16


freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--alpha", type=float, default=0.05)
cfg.add_argument("--num-states", type=int, default=30)
cfg.add_argument("--maxlen", type=int, default=1024)

cfg.add_argument("--hidden-size", type=int, default=256)
cfg.add_argument("--num-hidden-layers", type=int, default=4)
cfg.add_argument("--num-attention-heads", type=int, default=2)

cfg.set_defaults(
    description="MPT",
    root="./data",
    dataset='Amazon2014Beauty_550_LOU',
    epochs=1001,
    batch_size=512,
    optimizer='AdamW',
    lr=3e-4,
    weight_decay=0.1,
    which4best="MODEL",
    seed=1,
)
cfg.compile()


cfg.llama_config = LlamaConfig(
    vocab_size=0,
    hidden_size=cfg.hidden_size,
    intermediate_size=cfg.hidden_size,
    num_hidden_layers=cfg.num_hidden_layers,
    num_attention_heads=cfg.num_attention_heads,
    max_position_embeddings=cfg.maxlen,
    tie_word_embeddings=True,
    attention_dropout=0.
)


class MPT(freerec.models.SeqRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
    ) -> None:
        super().__init__(dataset)

        self.model = LlamaModel(cfg.llama_config)

        self.criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')
        self.reset_parameters()

    def reset_parameters(self): ...

    def sure_trainpipe(self):
        return TrainRandomWalkSource(
            self.dataset.train(),
            datasize=cfg.batch_size * 20,
            alpha=cfg.alpha,
            min_num_states=cfg.num_states,
            max_num_states=cfg.num_states,
            minlen=cfg.maxlen + 1,
            maxlen=cfg.maxlen + 1,
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq, self.IPos)
        ).lpad_(
            cfg.maxlen, modified_fields=(self.ISeq, self.IPos),
            padding_value=self.PADDING_VALUE
        ).batch_(cfg.batch_size).tensor_()

    def sure_validpipe(self):
        return ValidRandomWalkSource(
            self.dataset.valid(),
            datasize=cfg.batch_size * 2,
            alpha=cfg.alpha,
            min_num_states=cfg.num_states,
            max_num_states=cfg.num_states,
            minlen=cfg.maxlen+1,
            maxlen=cfg.maxlen+1,
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq, self.IPos)
        ).lpad_(
            cfg.maxlen, modified_fields=(self.ISeq, self.IPos),
            padding_value=self.PADDING_VALUE
        ).batch_(cfg.batch_size).tensor_()

    def generate_ortho_vocab(self, B, V, D):
        # Batched random orthogonal embeddings
        emb_dict = torch.randn(B, max(V, D), D, dtype=torch.float32, device=self.device)
        emb_dict, _ = torch.linalg.qr(emb_dict)
        emb_dict = emb_dict[:, :V, :D]
        return F.normalize(emb_dict, dim=-1)
    
    def encode(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chains = data[self.ISeq]
        B, S = chains.shape
        voc = self.generate_ortho_vocab(
            B=B, V=chains.max() + 1,
            D=cfg.hidden_size,
        )

        row_index = torch.arange(chains.shape[0], device=self.device)
        row_index = row_index.view(-1, 1)
        emb = voc[row_index, chains]

        out = self.model(
            inputs_embeds=emb, output_attentions=False
        )

        return out.last_hidden_state, voc

    def fit(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        hiddens, voc = self.encode(data)
        indices = data[self.ISeq] != self.PADDING_VALUE
        logits = torch.einsum("BMD,BND->BMN", hiddens, voc) # (B, M, N)
        logits = logits[indices]
        labels = data[self.IPos][indices] # (*,)
        loss = self.criterion(logits, labels)

        return loss

    def recommend_from_full(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        hiddens, voc = self.encode(data)
        hiddens = hiddens[:, -CUTS:, :]
        target = data[self.IPos][:, -CUTS:]
        logits = torch.einsum("BMD,BND->BMN", hiddens, voc) # (B, M, N)
        logits = rearrange(logits, "B M N -> B N M")

        return self.criterion(logits, target), data['empirical'].mean().item(), data['oracle'].mean().item()


class CoachForMarkov(freerec.launcher.Coach):

    def save_checkpoint(self, epoch):
        super().save_checkpoint(epoch)

        if (epoch in [5, 10, 40, 100, 500, 2500, 4000, 5000, 10000, 50000]) and freerec.ddp.is_main_process():
            filename = f"model_{epoch}.pt"
            torch.save(self.model.state_dict(), os.path.join(self.cfg.LOG_PATH, filename))
        freerec.ddp.synchronize()

    def set_other(self):
        self.register_metric(
            f"MODEL", lambda x: x, best_caster=min
        )
        self.register_metric(
            f"EMPIRICAL", lambda x: x, best_caster=min
        )
        self.register_metric(
            f"ORACLE", lambda x: x, best_caster=min
        )

    def train_per_epoch(self, epoch: int):
        step = 0
        for data in self.dataloader:
            step += 1
            data = self.dict_to_device(data)

            with torch.amp.autocast('cuda', dtype=DTYPE):
                loss = self.model(data)

            loss.backward()
            if step % cfg.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
           
            self.monitor(
                loss.item(), 
                n=data[self.Size], reduction="mean", 
                mode='train', pool=['LOSS']
            )

        if step % cfg.gradient_accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def evaluate(self, epoch: int, step: int = -1, mode: str = 'valid'):
        for data in self.dataloader:
            bsz = data[self.Size]

            data = self.dict_to_device(data)
            model_loss, empirical_loss, oracle_loss  = self.model(data, ranking='full')

            self.monitor(
                model_loss,
                n=bsz, reduction="mean", mode=mode,
                pool=[f"MODEL"]
            )
            self.monitor(
                empirical_loss,
                n=bsz, reduction="mean", mode=mode,
                pool=[f"EMPIRICAL"]
            )
            self.monitor(
                oracle_loss,
                n=bsz, reduction="mean", mode=mode,
                pool=[f"ORACLE"]
            )


def main():

    dataset: freerec.data.datasets.RecDataSet
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = MPT(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe()
    validpipe = model.sure_validpipe()

    coach = CoachForMarkov(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=validpipe,
        model=model,
        cfg=cfg
    )
    coach.fit()


if __name__ == "__main__":
    main()