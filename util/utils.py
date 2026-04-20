import torch
import numpy as np
import math
from torch import Tensor
from typing import Optional
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch_geometric.typing import Adj, OptTensor

def predict_link(
        model,
        prompt,
        edge_index: Adj,
        edge_label_index: OptTensor = None,
        edge_weight: OptTensor = None,
        prob: bool = False,
) -> Tensor:
    r"""Predict links between nodes specified in :obj:`edge_label_index`.

    Args:
        edge_index (torch.Tensor or SparseTensor): Edge tensor specifying
            the connectivity of the graph.
        edge_label_index (torch.Tensor, optional): Edge tensor specifying
            the node pairs for which to compute probabilities.
            If :obj:`edge_label_index` is set to :obj:`None`, all edges in
            :param model: for example LightGCN model
            :obj:`edge_index` will be used instead. (default: :obj:`None`)
        edge_weight (torch.Tensor, optional): The weight of each edge in
            :obj:`edge_index`. (default: :obj:`None`)
        prob (bool, optional): Whether probabilities should be returned.
            (default: :obj:`False`)
    """
    pred = model(edge_index, edge_label_index, edge_weight, prompt=prompt).sigmoid()
    return pred if prob else pred.round()

def recommend(
        model,
        prompt,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        src_index: OptTensor = None,
        dst_index: OptTensor = None,
        k: int = 1,
        sorted: bool = True,
) -> Tensor:
    r"""Get top-:math:`k` recommendations for nodes in :obj:`src_index`.

    Args:
        edge_index (torch.Tensor or SparseTensor): Edge tensor specifying
            the connectivity of the graph.
        edge_weight (torch.Tensor, optional): The weight of each edge in
            :obj:`edge_index`. (default: :obj:`None`)
        src_index (torch.Tensor, optional): Node indices for which
            recommendations should be generated.
            If set to :obj:`None`, all nodes will be used.
            (default: :obj:`None`)
        dst_index (torch.Tensor, optional): Node indices which represent
            the possible recommendation choices.
            If set to :obj:`None`, all nodes will be used.
            (default: :obj:`None`)
        k (int, optional): Number of recommendations. (default: :obj:`1`)
        sorted (bool, optional): Whether to sort the recommendations
            by score. (default: :obj:`True`)
    """
    out_src = out_dst = model.get_embedding(edge_index, edge_weight, prompt=prompt)

    if src_index is not None:
        out_src = out_src[src_index]

    if dst_index is not None:
        out_dst = out_dst[dst_index]

    pred = out_src @ out_dst.t()
    top_index = pred.topk(k, dim=-1, sorted=sorted).indices

    if dst_index is not None:  # Map local top-indices to original indices.
        top_index = dst_index[top_index.view(-1)].view(*top_index.size())

    return top_index

def link_pred_loss(model, pred: Tensor, edge_label: Tensor,
                   **kwargs) -> Tensor:
    r"""Computes the model loss for a link prediction objective via the
    :class:`torch.nn.BCEWithLogitsLoss`.

    Args:
        pred (torch.Tensor): The predictions.
        edge_label (torch.Tensor): The ground-truth edge labels.
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch.nn.BCEWithLogitsLoss` loss function.
    """
    loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
    return loss_fn(pred, edge_label.to(pred.dtype))

def recommendation_loss(
        model,
        prompt,
        pos_edge_rank: Tensor,
        neg_edge_rank: Tensor,
        node_id: Optional[Tensor] = None,
        lambda_reg: float = 1e-4,
        **kwargs,
) -> Tensor:
    r"""Computes the model loss for a ranking objective via the Bayesian
    Personalized Ranking (BPR) loss.

    .. note::

        The i-th entry in the :obj:`pos_edge_rank` vector and i-th entry
        in the :obj:`neg_edge_rank` entry must correspond to ranks of
        positive and negative edges of the same entity (*e.g.*, user).

    Args:
        pos_edge_rank (torch.Tensor): Positive edge rankings.
        neg_edge_rank (torch.Tensor): Negative edge rankings.
        node_id (torch.Tensor): The indices of the nodes involved for
            deriving a prediction for both positive and negative edges.
            If set to :obj:`None`, all nodes will be used.
        lambda_reg (int, optional): The :math:`L_2` regularization strength
            of the Bayesian Personalized Ranking (BPR) loss.
            (default: :obj:`1e-4`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.models.lightgcn.BPRLoss` loss
            function.
    """
    loss_fn = BPRLoss(lambda_reg, **kwargs)
    emb = model.embedding.weight if prompt is None else prompt.add(model.embedding.weight)
    emb = emb if node_id is None else emb[node_id]
    return loss_fn(pos_edge_rank, neg_edge_rank, emb)

def unlearning_loss(
        model,
        prompt,
        pos_edge_rank: Tensor,
        neg_edge_rank: Tensor,
        node_id: Optional[Tensor] = None,
        lambda_reg: float = 1e-4,
        **kwargs,
) -> Tensor:
    r"""Computes the model loss for a ranking objective via the Reversed Bayesian
    Personalized Ranking (RPR) loss.

    .. note::

        The i-th entry in the :obj:`pos_edge_rank` vector and i-th entry
        in the :obj:`neg_edge_rank` entry must correspond to ranks of
        positive and negative edges of the same entity (*e.g.*, user).

    Args:
        pos_edge_rank (torch.Tensor): Positive edge rankings.
        neg_edge_rank (torch.Tensor): Negative edge rankings.
        node_id (torch.Tensor): The indices of the nodes involved for
            deriving a prediction for both positive and negative edges.
            If set to :obj:`None`, all nodes will be used.
        lambda_reg (int, optional): The :math:`L_2` regularization strength
            of the Bayesian Personalized Ranking (BPR) loss.
            (default: :obj:`1e-4`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.models.lightgcn.BPRLoss` loss
        lambda_reg (int, optional): The :math:`L_2` regularization strength
            of the Bayesian Personalized Ranking (BPR) loss.
            (default: :obj:`1e-4`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.models.lightgcn.BPRLoss` loss
            function.
    """
    loss_fn = BPRLoss(lambda_reg, **kwargs)
    emb = model.embedding.weight if prompt is None else prompt.add(model.embedding.weight)
    emb = emb if node_id is None else emb[node_id]
    return -loss_fn(pos_edge_rank, neg_edge_rank, emb)


class BPRLoss(_Loss):
    r"""The Bayesian Personalized Ranking (BPR) loss.

    The BPR loss is a pairwise loss that encourages the prediction of an
    observed entry to be higher than its unobserved counterparts
    (see `here <https://arxiv.org/abs/2002.02126>`__).

    .. math::
        L_{\text{BPR}} = - \sum_{u=1}^{M} \sum_{i \in \mathcal{N}_u}
        \sum_{j \not\in \mathcal{N}_u} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})
        + \lambda \vert\vert \textbf{x}^{(0)} \vert\vert^2

    where :math:`\lambda` controls the :math:`L_2` regularization strength.
    We compute the mean BPR loss for simplicity.

    Args:
        lambda_reg (float, optional): The :math:`L_2` regularization strength
            (default: 0).
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch.nn.modules.loss._Loss` class.
    """
    __constants__ = ['lambda_reg']
    lambda_reg: float

    def __init__(self, lambda_reg: float = 0, **kwargs):
        super().__init__(None, None, "sum", **kwargs)
        self.lambda_reg = lambda_reg

    def forward(self, positives: Tensor, negatives: Tensor,
                parameters: Tensor = None) -> Tensor:
        r"""Compute the mean Bayesian Personalized Ranking (BPR) loss.

        .. note::

            The i-th entry in the :obj:`positives` vector and i-th entry
            in the :obj:`negatives` entry should correspond to the same
            entity (*.e.g*, user), as the BPR is a personalized ranking loss.

        Args:
            positives (Tensor): The vector of positive-pair rankings.
            negatives (Tensor): The vector of negative-pair rankings.
            parameters (Tensor, optional): The tensor of parameters which
                should be used for :math:`L_2` regularization
                (default: :obj:`None`).
        """
        log_prob = F.logsigmoid(positives - negatives).mean()

        regularization = 0
        if self.lambda_reg != 0:
            regularization = self.lambda_reg * parameters.norm(p=2).pow(2)
            regularization = regularization / positives.size(0)

        return -log_prob + regularization


def info_nce_loss(retain_embedding, forget_embedding, temperature=0.1):
    retain_embedding = F.normalize(retain_embedding, dim=1)
    forget_embedding = F.normalize(forget_embedding, dim=1)

    pos_score = torch.sum(retain_embedding * retain_embedding, dim=1)
    neg_score = torch.sum(retain_embedding * forget_embedding, dim=1)

    logits = torch.cat([pos_score.unsqueeze(1), neg_score.unsqueeze(1)], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits / temperature, labels)

def entropy(p):
    return -(p * p.log()).sum(-1)

def compute_forget_loss(student_output, teacher_output):
    """Compute forget loss using cosine similarity to maximize directional difference
    Args:
        student_output: Point-wise predictions from student model
        teacher_output: Point-wise predictions from teacher model
    Returns:
        Cosine similarity loss to maximize prediction difference
    """
    
    student_output = student_output.view(1, -1)
    teacher_output = teacher_output.view(1, -1)
    cos_sim = F.cosine_similarity(student_output, teacher_output)
    return cos_sim.mean()

def compute_retain_loss(student_output, teacher_output):  # Here we can add temperature and alpha
    student_prob = F.log_softmax(student_output, dim=-1)
    teacher_prob = F.softmax(teacher_output, dim=-1)
    return F.kl_div(student_prob, teacher_prob, reduction='batchmean')


def mse_loss(student_outputs, teacher_outputs):
    return F.mse_loss(student_outputs, teacher_outputs)


def ndcg_at_k(predictions, ground_truth, k):
    k = min(k, len(predictions))
    top_k_idxs = predictions.argsort(descending=True)[:k]
    if ground_truth.dtype == torch.bool:
        ground_truth = ground_truth.float()

    top_k_true_relevance = ground_truth[top_k_idxs]
    discounts = 1 / torch.log2(torch.arange(2, k + 2, dtype=torch.float32)).to(device=ground_truth.device)
    dcg_k = (top_k_true_relevance * discounts).sum()
    ideal_relevance = torch.sort(ground_truth, descending=True)[0][:k]
    idcg_k = (ideal_relevance * discounts).sum()

    return (dcg_k / idcg_k).item() if idcg_k > 0 else 0.0


def focal_loss(pred, truth, gamma=2.0):
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    bce_loss = criterion(pred, truth)
    pt = torch.exp(-bce_loss)
    focal_loss = ((1 - pt) ** gamma * bce_loss).mean()
    return focal_loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1.0 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss
