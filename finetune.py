

from typing import Dict, Tuple, Union, Optional

import torch, os, math
import torch.nn as nn
import torch.nn.functional as F
import freerec
from transformers import LlamaModel, LlamaConfig
from peft import LoraConfig, get_peft_model


freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()

cfg.add_argument("--path", type=str, default="")
cfg.add_argument("--ckpt-epoch", type=str, default="4000")
cfg.add_argument("--maxlen", type=int, default=50)

cfg.add_argument("--sem-feat-file", type=str, default="sentence-t5-xl_title_categories_brand.pkl")

# LLaMA Config
cfg.add_argument("--hidden-size", type=int, default=256)
cfg.add_argument("--num-hidden-layers", type=int, default=4)
cfg.add_argument("--num-attention-heads", type=int, default=2)
cfg.add_argument("--dropout-rate", type=float, default=0.2, help="attention dropout rate")

# Finetune
cfg.add_argument("--T", type=float, default=0.07, help="temperature")
cfg.add_argument("--adaptor-only", type=eval, default=True)
cfg.add_argument("--lora-rank", type=int, default=16)
cfg.add_argument("--lora-alpha", type=int, default=16)
cfg.add_argument("--lora-dropout", type=float, default=0.1)

cfg.set_defaults(
    description="MPT",
    root="./data",
    dataset='Amazon2014Beauty_550_LOU',
    epochs=200,
    batch_size=512,
    optimizer='AdamW',
    lr=1.e-3,
    weight_decay=0.1,
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
    attention_dropout=cfg.dropout_rate
)

cfg.lora_config = LoraConfig(
    r=cfg.lora_rank,
    lora_alpha=cfg.lora_alpha,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=cfg.lora_dropout,
    task_type="CAUSAL_LM",
)


class MPT(freerec.models.SeqRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
    ) -> None:
        super().__init__(dataset)

        feats = freerec.utils.import_pickle(
            os.path.join(
                self.dataset.path,
                cfg.sem_feat_file
            )
        )
        feats = self.padding(feats)

        self.Item.add_module(
            "embeddings",
            nn.Embedding.from_pretrained(
                feats,
                freeze=True,
                padding_idx=self.PADDING_VALUE
            )
        )

        self.adaptor = nn.Sequential(
            nn.RMSNorm(feats.size(1)),
            nn.Linear(feats.size(1), feats.size(1)),
            nn.LeakyReLU(),
            nn.Linear(feats.size(1), cfg.hidden_size),
        )

        self.model = LlamaModel(cfg.llama_config)
        self.model.requires_grad_(False)
        if not hasattr(self.model, 'prepare_inputs_for_generation'):
            self.model.prepare_inputs_for_generation = self._prepare_inputs_for_generation

        if not cfg.adaptor_only:
            self.model = get_peft_model(
                self.model, cfg.lora_config
            )

        self.criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')
        self.reset_parameters()

    def _prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        model_inputs = {"input_ids": input_ids}
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
        return model_inputs

    def reset_parameters(self):
        for name, m in self.adaptor.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def padding(self, feats: torch.Tensor):
        return F.pad(
            F.normalize(feats, dim=-1),
            (0, 0, self.NUM_PADS, 0),
            value=0.
        )

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return self.dataset.train().shuffled_seqs_source(
           maxlen=maxlen 
        ).seq_train_yielding_pos_(
            start_idx_for_target=1, end_idx_for_input=-1
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq, self.IPos,),
            padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def sure_validpipe(self, maxlen, ranking = 'full', batch_size = 512):
        return self.dataset.valid().ordered_user_ids_source(
        ).valid_sampling_(
            ranking
        ).lprune_(
            maxlen, modified_fields=(self.ISeq,)
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq,), 
            padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()
    
    def sure_testpipe(self, maxlen, ranking = 'full', batch_size = 512):
        return self.dataset.test().ordered_user_ids_source(
        ).test_sampling_(ranking).lprune_(
            maxlen, modified_fields=(self.ISeq,)
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq,), 
            padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def encode(
        self, 
        data: Dict[freerec.data.fields.Field, torch.Tensor],
        itemEmbds: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chains = data[self.ISeq]
        if itemEmbds is None:
            itemEmbds = self.adaptor(self.Item.embeddings.weight)

        emb = F.embedding(
            chains,
            itemEmbds, padding_idx=self.PADDING_VALUE
        )

        out = self.model(
            inputs_embeds=emb,
            attention_mask=chains.not_equal(self.PADDING_VALUE),
            output_attentions=False
        )

        return F.normalize(out.last_hidden_state, dim=-1), F.normalize(itemEmbds[self.NUM_PADS:], dim=-1)

    def fit(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        userEmbds, itemEmbds = self.encode(data)
        indices = data[self.ISeq] != self.PADDING_VALUE
        userEmbds = userEmbds[indices] # (M, D)
        logits = torch.einsum("MD,ND->MN", userEmbds, itemEmbds) / cfg.T
        labels = data[self.IPos][indices] # (M,)
        rec_loss = self.criterion(logits, labels)

        return rec_loss

    def reset_ranking_buffers(self):
        self._itemEmbds = self.adaptor(self.Item.embeddings.weight)

    def recommend_from_full(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        userEmbds, itemEmbds = self.encode(data, self._itemEmbds)
        userEmbds = userEmbds[:, -1, :]
        return torch.einsum("BD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        userEmbds, itemEmbds = self.encode(data, self._itemEmbds)
        userEmbds = userEmbds[:, -1, :]
        itemEmbds = itemEmbds[data[self.IUnseen]] # (B, K, D)
        return torch.einsum("BD,BKD->BK", userEmbds, itemEmbds)


class CoachForMarkov(freerec.launcher.Coach):

    def set_optimizer(self):
        params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=self.cfg.weight_decay
        )

    def load(self, path: str, filename: Optional[str] = None) -> None:
        filename = self.cfg.SAVED_FILENAME if filename is None else filename
        self.model.load_state_dict(
            torch.load(
                os.path.join(path, filename), 
                map_location=self.device,
                weights_only=True
            ),
            strict=False
        )
        freerec.ddp.synchronize()
        return

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
            self.optimizer.step()
           
            self.monitor(
                loss.item(), 
                n=len(data[self.User]), reduction="mean", 
                mode='train', pool=['LOSS']
            )


def main():

    dataset: freerec.data.datasets.RecDataSet
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.NextItemRecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = MPT(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(maxlen=cfg.maxlen, batch_size=cfg.batch_size)
    validpipe = model.sure_validpipe(maxlen=cfg.maxlen, batch_size=cfg.batch_size)
    testpipe = model.sure_testpipe(maxlen=cfg.maxlen, batch_size=cfg.batch_size)

    coach = CoachForMarkov(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg
    )
    coach.load(cfg.path, filename=f"model_{cfg.ckpt_epoch}.pt")
    coach.fit()


if __name__ == "__main__":
    main()