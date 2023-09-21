import torch
import numpy as np
from texttable import Texttable
from torch_sparse import spspmm, SparseTensor, spmm
from sklearn.metrics import f1_score 
from itertools import combinations
from sklearn.metrics.pairwise import rbf_kernel
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def myspspmm(spmx1, spmx2):
    row1, col1, value1 = spmx1.coo()
    index1 = torch.stack([row1, col1], dim=0)
    row2, col2, value2 = spmx2.coo()
    index2 = torch.stack([row2, col2], dim=0)
    index_new, value_new = spspmm(index1, value1, index2, value2, spmx1.size(0), spmx1.size(1), spmx2.size(1))
    results = SparseTensor(row=index_new[0], col=index_new[1], value=value_new, sparse_sizes=(spmx1.size(0), spmx2.size(1)))

    return results

def spdensemm(sp_mx, mx):
    row, col, value = sp_mx.coo()

    index = torch.stack([row, col], dim=0)
    out = spmm(index, value, sp_mx.size(0), sp_mx.size(1), mx)

    return out


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def F1_value(output, labels):
    preds = output.max(1)[1].type_as(labels)
    F1 = f1_score(preds, labels, average='macro')
    return F1


def generate_adj_permutation(max_length):
    idx = np.arange(0, max_length, 1)
    return_permutation = []
    for i in range(max_length):
        length = i + 1
        for permutation_idx in combinations(idx, length):
            return_permutation.append(permutation_idx)

    return return_permutation

def spconstantmm(sp_mx, constant):
    row, col, value = sp_mx.coo()
    results = SparseTensor(row=row, col=col, value=value * constant, sparse_sizes=(sp_mx.size(0), sp_mx.size(1)))

    return results




def similarity_function(points):
    """

    :param points:
    :return:
    """
    res = rbf_kernel(points)
    for i in range(len(res)):
        res[i, i] = 0
    return res

