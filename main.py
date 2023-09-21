import torch
import random
import numpy as np
from paraparser import parameter_parser
from utils import tab_printer
from Dataloader import load_data
from model import Net
from train import train
import os
os.environ["OMP_NUM_THREADS"] = '2'

if __name__ == "__main__":
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    args = parameter_parser()
    args.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tab_printer(args)

    features, gnd, train_mask, valid_mask, test_mask, adjs, adj_knn, ori_adj_knn = load_data(args)
    print("Data loaded.")

    features = features.to(device)
    gnd = gnd.to(device)
    train_mask = train_mask.to(device)
    valid_mask = valid_mask.to(device)
    test_mask = test_mask.to(device)
    for i in range(len(adjs)):
        adjs[i] = adjs[i].to(device)

    adj_knn = adj_knn.to(device)
    ori_adj_knn = ori_adj_knn.to(device)

    cluster_num = len(torch.unique(gnd))

    input_channels = features.size(1)
    output_channels = len(torch.unique(gnd))

    model = Net(args, input_channels, output_channels, adjs, cluster_num, device).to(device)
    print("Network initialized.")



    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=5e-4)

    train(model, optimizer, features, gnd, ori_adj_knn, train_mask, valid_mask, test_mask, args)