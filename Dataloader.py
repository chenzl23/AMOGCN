import torch
import numpy as np
from scipy.io import loadmat
import os
import scipy.sparse as sp
from torch_sparse import SparseTensor
from KNNgraph import generate_knn, load_graph


def load_mat_data(dataset_name):
    data = loadmat(dataset_name)
    features = data['X'] 
    adj = data['adj']
    gnd = data['Y']
    gnd = gnd.flatten()
    gnd = gnd - 1

    return features, adj, gnd

def count_each_class_num(gnd):

    count_dict = {}
    for label in gnd:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict


def generate_permutation(gnd, args):

    train_ratio =  args.train_ratio
    valid_ratio = args.valid_ratio 
    test_ratio = args.test_ratio
    N = gnd.shape[0]

    valid_num = max(round(N * valid_ratio), 1) 
    test_num= max(round(N * test_ratio), 1)
    train_num = max(round(N * train_ratio), 1)


    # index of labeled and unlabeled samples
    train_mask = torch.from_numpy(np.full((N), False))
    valid_mask = torch.from_numpy(np.full((N), False))
    test_mask = torch.from_numpy(np.full((N), False))

    # shuffle the data
    data_idx = np.random.permutation(range(N))


    for idx in data_idx:
        if (train_num > 0):
            train_num -= 1
            train_mask[idx] = True
        elif (valid_num > 0):
            valid_num  -= 1
            valid_mask[idx] = True
        elif (test_num > 0):
            test_num -= 1
            test_mask[idx] = True

    return train_mask, valid_mask, test_mask



def to_normalized_sparsetensor(edge_index, N, mode='DAD'):
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5) 
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    if mode == 'DA':
        return deg_inv_sqrt.view(-1,1) * deg_inv_sqrt.view(-1,1) * adj
    if mode == 'DAD':
        return deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

def load_data(args):

    # load mat data
    features, adjs, gnd = load_mat_data(os.path.join("./data", args.dataset_name))
    num_samples = features.shape[0]
    gnd.astype(np.uint8)

    train_mask, valid_mask, test_mask = generate_permutation(gnd, args)

    features = torch.from_numpy(features).float()
    gnd = torch.from_numpy(gnd)
    view_num = len(adjs[0])
    
    adj_list = []
    for i in range(view_num):
        adj = adjs[0][i].toarray()
        adj = torch.from_numpy(adj)

        row, col = np.argwhere(adj > 0)
        value = torch.ones(len(row))
        ori_adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(num_samples, num_samples))
        ori_adj = ori_adj

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + torch.eye(adj.shape[0], adj.shape[0])
        edge_index = np.argwhere(adj > 0)

        adj = to_normalized_sparsetensor(edge_index, adj.shape[0])
        adj_list.append(adj)

    '''
    Load KNN Graph
    '''
    if not os.path.exists("./data/KNN/" + args.dataset_name + "/c" + str(args.k) + ".txt"):
        generate_knn(args.dataset_name, features, args.k)
    adj_knn = load_graph(args.dataset_name, args.k, num_samples)
    row, col = np.argwhere(adj_knn > 0)
    value = torch.ones(len(row))
    ori_adj_knn = SparseTensor(row=row, col=col, value=value, sparse_sizes=(num_samples, num_samples))

    adj_knn = adj_knn + adj_knn.T.multiply(adj_knn.T > adj_knn) - adj_knn.multiply(adj_knn.T > adj_knn)
    adj_knn = adj_knn + torch.eye(adj_knn.shape[0], adj_knn.shape[0])
    edge_index = np.argwhere(adj_knn > 0)

    adj_knn = to_normalized_sparsetensor(edge_index, adj_knn.shape[0])

    if args.feature_normalize == 1:
        print("Feature Normalized.")
        features = normalize(features)

    return features, gnd, train_mask, valid_mask, test_mask, adj_list, adj_knn, ori_adj_knn

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    if isinstance(mx, np.ndarray):
        return torch.from_numpy(mx)
    else:
        return mx