import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from utils import generate_adj_permutation

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Net(nn.Module):
    def __init__(self, args, dim_in, dim_out, adjs, num_clusters, device, v=1):
        super(Net,self).__init__()
        self.args = args

        self.v = v

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.device = device
        
        self.setup_layers(args)

        self.relu = nn.ReLU(inplace=True)

        self.dropout_rate = args.dropout

        self.graph_num = len(adjs)

        self.mp_adjs = adjs 

        
        for i in range(len(self.mp_adjs)):
            self.mp_adjs[i] = self.mp_adjs[i].to_dense()
        self.num_samples = self.mp_adjs[0].size(0)

        self.indices = generate_adj_permutation(self.graph_num)


        self.beta = nn.Parameter(torch.ones(len(self.indices)) / len(self.indices))

        self.alpha = nn.ParameterList()
        for i in range(0, len(self.indices)):
            if len(self.indices[i]) > 1:
                weight_num = len(self.indices[i])
                self.alpha.append(nn.Parameter(torch.randn(weight_num, weight_num) / weight_num)) 



    
    def setup_layers(self, args):
        """
        Creating the layes based on the args.
        """
        self.sgc_layer = GraphConvolution(self.dim_in, self.dim_out)

    def get_weighted_adj(self, beta):
        mo_adj = self.mp_adjs[0] * beta[0]
        for i in range(1, len(beta)): 
            if len(self.indices[i]) > 1:
                alpha_idx = i - self.graph_num  
                # Get weighted adj with idx 
                alpha_mx = F.softmax(self.alpha[alpha_idx], dim=1)
                order = len(self.indices[i])
                for j in range(order):
                    sum_adj = self.mp_adjs[self.indices[i][0]] * alpha_mx[j, 0]
                    for k in range(1, order): 
                        sum_adj = sum_adj + self.mp_adjs[self.indices[i][k]] * alpha_mx[j, k]
                    if (j == 0):
                        ho_adj = sum_adj
                    else:
                        ho_adj = ho_adj.matmul(sum_adj)
                ho_adj = ho_adj + ho_adj.t() - torch.diag(ho_adj.diag())
            else:
                ho_adj = self.mp_adjs[self.indices[i][0]]  
            ## Aggregation of multi-order metapath
            mo_adj = mo_adj + ho_adj * beta[i]

        return mo_adj

    def forward(self, features):
        beta = F.softmax(self.beta, dim = 0)
        A = self.get_weighted_adj(beta)

        Z_tp = features
        Z_tp = self.sgc_layer(Z_tp, A)
        Z_tp = F.dropout(Z_tp, p=self.dropout_rate, training=self.training)

        return Z_tp, A

