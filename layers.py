import torch
import torch.nn as nn
import torch.nn.functional as F

# from dhg.structure.hypergraphs import Hypergraph

from typing import Optional
from dhg.structure.hypergraphs import Hypergraph
from dhg.structure.graphs import Graph
from dhg.nn import MultiHeadWrapper

class HGNNConv(nn.Module):
    r"""The HGNN convolution layer proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).
    Matrix Format:

    .. math::
        \mathbf{X}^{\prime} = \sigma \left( \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} 
        \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X} \mathbf{\Theta} \right).

    where :math:`\mathbf{X}` is the input vertex feature matrix, :math:`\mathbf{H}` is the hypergraph incidence matrix, 
    :math:`\mathbf{W}_e` is a diagonal hyperedge weight matrix, :math:`\mathbf{D}_v` is a diagonal vertex degree matrix, 
    :math:`\mathbf{D}_e` is a diagonal hyperedge degree matrix, :math:`\mathbf{\Theta}` is the learnable parameters.

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        X = self.theta(X)
        X = hg.smoothing_with_HGNN(X)
        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X


# =======================HyperGCNConv Layer========================

class HyperGCNConv(nn.Module):
    r"""The HyperGCN convolution layer proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs <https://papers.nips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf>`_ paper (NeurIPS 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``use_mediator`` (``str``): Whether to use mediator to transform the hyperedges to edges in the graph. Defaults to ``False``.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_mediator: bool = False,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.use_mediator = use_mediator
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(
        self, X: torch.Tensor, hg: Hypergraph, cached_g: Optional[Graph] = None
    ) -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
            ``cached_g`` (``dhg.Graph``): The pre-transformed graph structure from the hypergraph structure that contains :math:`N` vertices. If not provided, the graph structure will be transformed for each forward time. Defaults to ``None``.
        """
        X = self.theta(X)
        if cached_g is None:
            g = Graph.from_hypergraph_hypergcn(
                hg, X, self.use_mediator, device=X.device
            )
            X = g.smoothing_with_GCN(X)
        else:
            X = cached_g.smoothing_with_GCN(X)
        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X


# =======================UniGCNConv Layer========================

import torch
import torch.nn as nn

from dhg.structure.hypergraphs import Hypergraph


class UniGCNConv(nn.Module):
    r"""The UniGCN convolution layer proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Sparse Format:

    .. math::
        \left\{
            \begin{aligned}
            h_{e} &= \frac{1}{|e|} \sum_{j \in e} x_{j} \\
            \tilde{x}_{i} &= \frac{1}{\sqrt{d_{i}}} \sum_{e \in \tilde{E}_{i}} \frac{1}{\sqrt{\tilde{d}_{e}}} W h_{e} 
            \end{aligned}
        \right. .
    
    where :math:`\tilde{d}_{e} = \frac{1}{|e|} \sum_{i \in e} d_{i}`.

    Matrix Format:

    .. math::
        \mathbf{X}^{\prime} = \sigma \left(  \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \tilde{\mathbf{D}}_e^{-\frac{1}{2}} \cdot \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{X} \mathbf{\Theta} \right) .

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        X = self.theta(X)
        Y = hg.v2e(X, aggr="mean")
        # ===============================================
        # compute the special degree of hyperedges
        _De = torch.zeros(hg.num_e, device=hg.device)
        # scatter_reduce() is relay on the torch 1.12.1, which may be updated in the future
        _De = _De.scatter_reduce(0, index=hg.v2e_dst, src=hg.D_v.clone()._values()[hg.v2e_src], reduce="mean")
        _De = _De.pow(-0.5)
        _De[_De.isinf()] = 1
        Y = _De.view(-1, 1) * Y
        # ===============================================
        X = hg.e2v(Y, aggr="sum")
        X = torch.sparse.mm(hg.D_v_neg_1_2, X)

        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X


# =======================UniGATConv Layer========================

class UniGATConv(nn.Module):
    r"""The UniGAT convolution layer proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Sparse Format:

    .. math::
        \left\{
            \begin{aligned}
                \alpha_{i e} &=\sigma\left(a^{T}\left[W h_{\{i\}} ; W h_{e}\right]\right) \\
                \tilde{\alpha}_{i e} &=\frac{\exp \left(\alpha_{i e}\right)}{\sum_{e^{\prime} \in \tilde{E}_{i}} \exp \left(\alpha_{i e^{\prime}}\right)} \\
                \tilde{x}_{i} &=\sum_{e \in \tilde{E}_{i}} \tilde{\alpha}_{i e} W h_{e}
            \end{aligned}
        \right. .
    
    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): The dropout probability. If ``dropout <= 0``, the layer will not drop values. Defaults to ``0.5``.
        ``atten_neg_slope`` (``float``): Hyper-parameter of the ``LeakyReLU`` activation of edge attention. Defaults to ``0.2``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        atten_neg_slope: float = 0.2,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.atten_dropout = nn.Dropout(drop_rate)
        self.atten_act = nn.LeakyReLU(atten_neg_slope)
        self.act = nn.ELU(inplace=True)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        self.atten_e = nn.Linear(out_channels, 1, bias=False)
        self.atten_dst = nn.Linear(out_channels, 1, bias=False)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        X = self.theta(X)
        Y = hg.v2e(X, aggr="mean")
        # ===============================================
        alpha_e = self.atten_e(Y)
        e_atten_score = alpha_e[hg.e2v_src]
        e_atten_score = self.atten_dropout(self.atten_act(e_atten_score).squeeze())
        # ================================================================================
        # We suggest to add a clamp on attention weight to avoid Nan error in training.
        e_atten_score = torch.clamp(e_atten_score, min=0.001, max=5)
        # ================================================================================
        X = hg.e2v(Y, aggr="softmax_then_sum", e2v_weight=e_atten_score)

        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
        return X


# =======================UniSAGEConv Layer========================

class UniSAGEConv(nn.Module):
    r"""The UniSAGE convolution layer proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Sparse Format:

    .. math::
        \left\{
            \begin{aligned}
            h_{e} &= \frac{1}{|e|} \sum_{j \in e} x_{j} \\
            \tilde{x}_{i} &= W\left(x_{i}+\text { AGGREGATE }\left(\left\{x_{j}\right\}_{j \in \mathcal{N}_{i}}\right)\right) 
            \end{aligned}
        \right. .

    Matrix Format:

    .. math::
        \mathbf{X}^{\prime} = \sigma \left( \left( \mathbf{I} + \mathbf{H} \mathbf{D}_e^{-1} \mathbf{H}^\top \right) \mathbf{X} \mathbf{\Theta} \right) .

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        X = self.theta(X)
        Y = hg.v2e(X, aggr="mean")
        X = hg.e2v(Y, aggr="sum") + X
        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X

# =======================UniGINConv Layer========================

class UniGINConv(nn.Module):
    r"""The UniGIN convolution layer proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Sparse Format:

    .. math::
        
        \left\{
            \begin{aligned}
            h_{e} &= \frac{1}{|e|} \sum_{j \in e} x_{j} \\
            \tilde{x}_{i} &= W\left((1+\varepsilon) x_{i}+\sum_{e \in E_{i}} h_{e}\right) 
            \end{aligned}
        \right. .

    Matrix Format:

    .. math::
        \mathbf{X}^{\prime} = \sigma \left( \left( \left( \mathbf{I} + \varepsilon \right) + \mathbf{H} \mathbf{D}_e^{-1} \mathbf{H}^\top \right) \mathbf{X} \mathbf{\Theta} \right) .

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``eps`` (``float``): :math:`\varepsilon` is the learnable parameter. Defaults to ``0.0``.
        ``train_eps`` (``bool``): If set to ``True``, the layer will learn the :math:`\varepsilon` parameter. Defaults to ``False``.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        eps: float = 0.0,
        train_eps: bool = False,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps]))
        else:
            self.eps = eps
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        X = self.theta(X)
        Y = hg.v2e(X, aggr="mean")
        X = (1 + self.eps) * hg.e2v(Y, aggr="sum") + X
        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X