import torch
import torch.nn as nn
import torch.nn.functional as F
import dhg
# from .layers import HGNNConv
# from dhg.nn import HGNNConv
from .layers import HGNNConv, HyperGCNConv, UniGATConv, UniGCNConv, UniGINConv, UniSAGEConv
# from dhg.nn import HGNNConv
import torch
import torch.nn as nn

import dhg
# from dhg.nn import HyperGCNConv
from dhg.structure.graphs import Graph
from dhg.nn import MultiHeadWrapper


'''
============================
HGNN and LAHGNN (also named LAHGCN)
============================
'''

class HGNN(nn.Module):
    r"""The HGNN model proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LAHGCN(nn.Module):
    def __init__(self, concat, in_channels, hid_channels, num_classes, use_bn: bool = False,
        drop_rate: float = 0.5):
        super(LAHGCN, self).__init__()
        self.hgcn1 = HGNNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        self.hgcn2 = HGNNConv(concat*hid_channels, num_classes, use_bn=use_bn, is_last=True)
        self.dropout = drop_rate

    def forward(self, x_list, hg):
        hidden_list = []
        for k in range(len(x_list)):
            x = x_list[k]
            x = x.to(device)
            hg = hg.to(device)
            x = self.hgcn1(x,hg)
            hidden_list.append(x)
        x = torch.cat((hidden_list), dim=-1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgcn2(x, hg)
        return x

# class LAHGCN(nn.Module):
#     def __init__(self, concat, in_channels, hid_channels, num_classes, dropout):
#         super(LAHGCN, self).__init__()

#         self.hgcn1_list = nn.ModuleList()
#         for _ in range(concat):
#             self.hgcn1_list.append(HGNNConv(in_channels, hid_channels))
#         self.hgc2 = HGNNConv(concat*hid_channels, num_classes)
#         self.dropout = dropout

#     def forward(self, x_list, hg):
#         hidden_list = []
#         for k, con in enumerate(self.hgcn1_list):
#             x = F.dropout(x_list[k], self.dropout, training=self.training)
#             hidden_list.append(F.relu(con(x, hg)))
#         x = torch.cat((hidden_list), dim=-1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.hgc2(x, hg)
#         return x


'''
============================
HyperGCN and LAHyperGCN
============================
'''


class HyperGCN(nn.Module):
    r"""The HyperGCN model proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs <https://papers.nips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf>`_ paper (NeurIPS 2019).
    
    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_mediator`` (``str``): Whether to use mediator to transform the hyperedges to edges in the graph. Defaults to ``False``.
        ``fast`` (``bool``): If set to ``True``, the transformed graph structure will be computed once from the input hypergraph and vertex features, and cached for future use. Defaults to ``True``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_mediator: bool = False,
        use_bn: bool = False,
        fast: bool = True,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.fast = fast
        self.cached_g = None
        self.with_mediator = use_mediator
        self.layers = nn.ModuleList()
        self.layers.append(
            HyperGCNConv(
                in_channels, hid_channels, use_mediator, use_bn=use_bn, drop_rate=drop_rate,
            )
        )
        self.layers.append(
            HyperGCNConv(
                hid_channels, num_classes, use_mediator, use_bn=use_bn, is_last=True
            )
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        if self.fast:
            if self.cached_g is None:
                self.cached_g = Graph.from_hypergraph_hypergcn(
                    hg, X, self.with_mediator
                )
            for layer in self.layers:
                X = layer(X, hg, self.cached_g)
        else:
            for layer in self.layers:
                X = layer(X, hg)
        return X


class LAHyperGCN(nn.Module):
    def __init__(self, concat, in_channels, hid_channels, num_classes, use_mediator: bool = False, use_bn: bool = False,
        drop_rate: float = 0.5):
        super(LAHyperGCN, self).__init__()
        self.hygcn1 = HyperGCNConv(in_channels, hid_channels, use_mediator=use_mediator, use_bn=use_bn, drop_rate=drop_rate)
        self.hygcn2 = HyperGCNConv(concat*hid_channels, num_classes, use_mediator=use_mediator, use_bn=use_bn, is_last=True)
        self.dropout = drop_rate

    def forward(self, x_list, hg):
        hidden_list = []
        for k in range(len(x_list)):
            x = x_list[k]
            x = x.to(device)
            hg = hg.to(device)
            x = self.hygcn1(x,hg)
            hidden_list.append(x)
        x = torch.cat((hidden_list), dim=-1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.hygcn2(x, hg)
        return x



'''
============================
UniGCN and LAUniGCN
============================
'''

class UniGCN(nn.Module):
    r"""The UniGCN model proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self, in_channels: int, hid_channels: int, num_classes: int, use_bn: bool = False, drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(UniGCNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        self.layers.append(UniGCNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True))

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X


'''
============================
UniSAGE and LAUniSAGE
============================
'''

class UniSAGE(nn.Module):
    r"""The UniSAGE model proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self, in_channels: int, hid_channels: int, num_classes: int, use_bn: bool = False, drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(UniSAGEConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        self.layers.append(UniSAGEConv(hid_channels, num_classes, use_bn=use_bn, is_last=True))

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X


class LAUniSAGE(nn.Module):
    def __init__(self, concat, in_channels, hid_channels, num_classes, use_bn: bool = False,
        drop_rate: float = 0.5):
        super(LAUniSAGE, self).__init__()
        self.unisage1 = UniSAGEConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        self.unisage2 = UniSAGEConv(concat*hid_channels, num_classes, use_bn=use_bn, is_last=True)
        self.dropout = drop_rate

    def forward(self, x_list, hg):
        hidden_list = []
        for k in range(len(x_list)):
            x = x_list[k]
            x = x.to(device)
            hg = hg.to(device)
            x = self.unisage1(x,hg)
            hidden_list.append(x)
        x = torch.cat((hidden_list), dim=-1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.unisage2(x, hg)
        return x


'''
============================
UniGAT and LAGAT
============================
'''

class UniGAT(nn.Module):
    r"""The UniGAT model proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``num_heads`` (``int``): The Number of attention head in each layer.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): The dropout probability. Defaults to ``0.5``.
        ``atten_neg_slope`` (``float``): Hyper-parameter of the ``LeakyReLU`` activation of edge attention. Defaults to 0.2.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        num_heads: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        atten_neg_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.drop_layer = nn.Dropout(drop_rate)
        self.multi_head_layer = MultiHeadWrapper(
            num_heads,
            "concat",
            UniGATConv,
            in_channels=in_channels,
            out_channels=hid_channels,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
        )
        # The original implementation has applied activation layer after the final layer.
        # Thus, we donot set ``is_last`` to ``True``.
        self.out_layer = UniGATConv(
            hid_channels * num_heads,
            num_classes,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
            is_last=False,
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        X = self.drop_layer(X)
        X = self.multi_head_layer(X=X, hg=hg)
        X = self.drop_layer(X)
        X = self.out_layer(X, hg)
        return X



class LAUniGAT(nn.Module):
    def __init__(self, concat, in_channels, hid_channels, num_classes, num_heads, use_bn: bool = False,
        drop_rate: float = 0.5, atten_neg_slope: float = 0.2):
        super(LAUniGAT, self).__init__()
        self.drop_layer = nn.Dropout(drop_rate)
        self.multi_head_layer = MultiHeadWrapper(
            num_heads,
            "concat",
            UniGATConv,
            in_channels=in_channels,
            out_channels=hid_channels,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
        )
        # The original implementation has applied activation layer after the final layer.
        # Thus, we donot set ``is_last`` to ``True``.
        self.out_layer = UniGATConv(
            concat * hid_channels * num_heads,
            num_classes,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
            is_last=False,
        )

    def forward(self, x_list, hg):
        hidden_list = []
        for k in range(len(x_list)):
            x = x_list[k]
            x = self.drop_layer(x)
            x = self.multi_head_layer(x, hg)
            x = self.drop_layer(x)
            hidden_list.append(x)
        x = torch.cat((hidden_list), dim=-1)
        x = self.out_layer(x, hg)
        return x

'''
============================
UniGIN and LAUniGIN
============================
'''

class UniGIN(nn.Module):
    r"""The UniGIN model proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``eps`` (``float``): The epsilon value. Defaults to ``0.0``.
        ``train_eps`` (``bool``): If set to ``True``, the epsilon value will be trainable. Defaults to ``False``.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        eps: float = 0.0,
        train_eps: bool = False,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            UniGINConv(in_channels, hid_channels, eps=eps, train_eps=train_eps, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            UniGINConv(hid_channels, num_classes, eps=eps, train_eps=train_eps, use_bn=use_bn, is_last=True)
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LAUniGIN(nn.Module):
    def __init__(self, concat, in_channels, hid_channels, num_classes, eps: float = 0.0,
        train_eps: bool = False, use_bn: bool = False,
        drop_rate: float = 0.5):
        super(LAUniGIN, self).__init__()
        self.unigin1 = UniGINConv(in_channels, hid_channels, eps=eps, train_eps=train_eps, use_bn=use_bn, drop_rate=drop_rate)
        self.unigin2 = UniGINConv(concat*hid_channels, num_classes, eps=eps, train_eps=train_eps, use_bn=use_bn, is_last=True)
        self.dropout = drop_rate

    def forward(self, x_list, hg):
        hidden_list = []
        for k in range(len(x_list)):
            x = x_list[k]
            x = x.to(device)
            hg = hg.to(device)
            x = self.unigin1(x,hg)
            hidden_list.append(x)
        x = torch.cat((hidden_list), dim=-1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.unigin2(x, hg)
        return x