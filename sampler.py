

from typing import Iterable, Dict, Any, List

import torchdata.datapipes as dp
import torch
import numpy as np
import random
import freerec
from itertools import chain
from freerec.data.tags import ITEM, ID, SEQUENCE, POSITIVE
from freerec.data.datasets.base import RecDataSet
from freerec.data.postprocessing.base import Source
from freerec.data.postprocessing.source import OrderedSource
from freerec.data.postprocessing.sampler import ValidSampler


CUTS = 20

class TrainRandomWalkSource(Source):

    def __init__(
        self, dataset: RecDataSet, datasize: int,
        alpha: float = 0.1,
        min_num_states: int = 30, max_num_states: int = 30, 
        minlen: int = 1024, maxlen: int = 1024
    ) -> None:
        super().__init__(dataset, tuple(), datasize, shuffle=False)

        self._rng = random.Random()
        self._np_rng = np.random.default_rng()
        self.set_seed(0)

        self.alpha = alpha
        self.min_num_states = max(min_num_states, CUTS)
        self.max_num_states = max_num_states
        self.minlen = minlen
        self.maxlen = maxlen

        self.Item = self.fields[ITEM, ID]
        self.ISeq = self.Item.fork(SEQUENCE)
        self.IPos = self.Item.fork(POSITIVE)

    def set_seed(self, seed: int):
        self._rng.seed(seed)
        self._np_rng = np.random.default_rng(seed)

    def sample_transition_matrix(self, num_states: int) -> np.ndarray:
        return self._np_rng.dirichlet([self.alpha] * num_states, size=num_states)

    def sample_num_states(self):
        return self._rng.randint(self.min_num_states, self.max_num_states)

    def sample_chain_length(self):
        return self._rng.randint(self.minlen, self.maxlen)

    def sample_chain(self) -> List[int]:
        k = self.sample_num_states()
        P = self.sample_transition_matrix(num_states=k)
        n = self.sample_chain_length()
        cprobs = P.cumsum(axis=1)
        rands = self._np_rng.random(n)

        chain = np.zeros(n, dtype=int)
        for i in range(1, n):
            chain[i] = np.searchsorted(cprobs[chain[i - 1]], rands[i])
        seq, target = chain[:-1], chain[1:]
        return seq.tolist(), target.tolist()

    def __iter__(self):
        for _ in self.launcher:
            seq, target = self.sample_chain()
            yield {self.ISeq: seq, self.IPos: target}


class MarkedRandomWalkSource(TrainRandomWalkSource):

    def estimate_transition_probability(self, chain: List[int], num_states: int):
        counts = np.zeros((num_states,))
        chain, x = chain[:-1], chain[-1]

        positions = np.where(chain == x)[0][:-1]
        positions += 1

        vals = chain[positions]
        np.add.at(counts, vals, 1)

        return (counts + self.alpha) / (counts.sum() + self.alpha * num_states)

    def cross_entropy_from_probs(self, probs: np.array, target: np.ndarray):
        # probs: (CUTS, NUM_STATES)
        probs[probs == 0] = 1.e-8

        target = target.copy()[:, None]
        probs = np.take_along_axis(probs, target, axis=1)
        return np.mean(-np.log(probs)).item()

    def sample_chain(self) -> List[int]:
        k = self.sample_num_states()
        P = self.sample_transition_matrix(num_states=k)
        n = self.sample_chain_length()
        cprobs = P.cumsum(axis=1)
        rands = self._np_rng.random(n)

        chain = np.zeros(n, dtype=int)
        for i in range(1, n):
            chain[i] = np.searchsorted(cprobs[chain[i - 1]], rands[i])
        seq, target = chain[:-1], chain[1:]
        s = len(seq) - CUTS
        estimation = np.stack([
            self.estimate_transition_probability(
                seq[:s+i], k
            )
            for i in range(1, CUTS + 1)
        ], axis=0)
        oracle = np.stack([
            P[seq[s+i]]
            for i in range(CUTS)
        ], axis=0)
        empirical_loss = self.cross_entropy_from_probs(estimation, target[-CUTS:])
        oracle_loss = self.cross_entropy_from_probs(oracle, target[-CUTS:])
        return seq.tolist(), target.tolist(), empirical_loss, oracle_loss

    def __iter__(self):
        for _ in self.launcher:
            seq, target, empirical, oracle = self.sample_chain()
            yield {
                self.ISeq: seq, self.IPos: target,
                'empirical': empirical,
                'oracle': oracle
            }


class ValidRandomWalkSource(OrderedSource):

    def __init__(
        self, dataset: RecDataSet, datasize: int,
        alpha: float = 0.1,
        min_num_states: int = 30, max_num_states: int = 30, 
        minlen: int = 1024, maxlen: int = 1024
    ) -> None:

        source = MarkedRandomWalkSource(
            dataset, datasize, alpha, min_num_states, max_num_states, minlen, maxlen
        )

        super().__init__(dataset, list(source))


@dp.functional_datapipe("valid_neighbor_sampling_")
class ValidNeighborSampler(ValidSampler):

    def __init__(
        self, source, ranking: str = 'full', k: int = 1
    ):
        self.topk = k
        super().__init__(source, ranking)

    @freerec.utils.timemeter
    def prepare(self):
        seenItems = [[] for _ in range(self.User.count)]
        unseenItems = [[] for _ in range(self.User.count)]

        self.listmap(
            lambda row: seenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.train().to_seqs()
        )

        self.listmap(
            lambda row: unseenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.valid().to_seqs()
        )

        self.seenItems = seenItems
        self.unseenItems = unseenItems
        self.negItems = dict()

        edge_index = self.dataset.train().to_bigraph(edge_type='U2I')['U2I'].edge_index
        # edge_weight = torch.ones_like(edge_index[0], dtype=torch.float32)
        edge_index, edge_weight = freerec.graph.to_normalized(edge_index, normalization="left")
        R = torch.sparse_coo_tensor(
            edge_index, edge_weight.sqrt(), size=(self.User.count, self.Item.count)
        ).to_dense()
        sim_mat = R @ R.T # (#User, #User)
        sim_mat.fill_diagonal_(-10.)
        _, cols = torch.topk(sim_mat, self.topk, dim=1, largest=True)
        self.neighbors = [
            tuple(neighbors[::-1])
            for neighbors in cols.cpu().tolist()
        ]
        self.neighbors = tuple(self.neighbors)

    def _nextitem_from_full(self):
        for row in self.source:
            user = row[self.User]
            context = tuple(chain(*[self.seenItems[u] for u in self.neighbors[user]]))
            seen = self.seenItems[user]
            for k, positive in enumerate(self.unseenItems[user]):
                seq = self.seenItems[user] + self.unseenItems[user][:k]
                unseen = (positive,)
                yield {self.User: user, self.ISeq: context + seq, self.IUnseen: unseen, self.ISeen: seen}


@dp.functional_datapipe("test_neighbor_sampling_")
class TestNeighborSampler(ValidNeighborSampler):

    @freerec.utils.timemeter
    def prepare(self):
        seenItems = [[] for _ in range(self.User.count)]
        unseenItems = [[] for _ in range(self.User.count)]

        self.listmap(
            lambda row: seenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.train().to_seqs()
        )

        self.listmap(
            lambda row: seenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.valid().to_seqs()
        )

        self.listmap(
            lambda row: unseenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.test().to_seqs()
        )

        self.seenItems = seenItems
        self.unseenItems = unseenItems
        self.negItems = dict()

        edge_index = self.dataset.train().to_bigraph(edge_type='U2I')['U2I'].edge_index
        # edge_weight = torch.ones_like(edge_index[0], dtype=torch.float32)
        edge_index, edge_weight = freerec.graph.to_normalized(edge_index, normalization="left")
        R = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(self.User.count, self.Item.count)
        ).to_dense()
        sim_mat = R @ R.T # (#User, #User)
        sim_mat.fill_diagonal_(-10.)
        _, cols = torch.topk(sim_mat, self.topk, dim=1, largest=True)
        self.neighbors = [
            tuple(neighbors[::-1])
            for neighbors in cols.cpu().tolist()
        ]
        self.neighbors = tuple(self.neighbors)