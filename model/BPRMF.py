import torch
import torch.nn as nn
from typing import Optional
from util.utils import recommendation_loss, recommend

class BPRMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int
    ):
        super(BPRMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.1)

    def get_embedding(self, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding.weight
        if prompt is not None:
            emb = prompt.add(emb)
        return emb

    def forward(self, edge_index: torch.Tensor, edge_label_index: torch.Tensor, prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.get_embedding(edge_index, prompt=prompt)
        user_emb = emb[edge_label_index[0]]
        item_emb = emb[edge_label_index[1]]
        return torch.sum(user_emb * item_emb, dim=1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_users}, '
                f'{self.num_items}, embedding_dim={self.embedding_dim})')