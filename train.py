import torch
import copy
import torch.nn.functional as F
import time 
from utils import accuracy
from evaluate import classification_evaluate


def rec_loss_3(mo_A, mask_1):
    rec_loss = - torch.log(mo_A[mask_1] + 1e-4).mean()
    return rec_loss


def train(model, optimizer, features, gnd, ori_adj_knn, train_mask, valid_mask, test_mask, args):
    best_valid_acc = 0.0
    patience = args.patience


    best_model = copy.deepcopy(model)
    best_epoch = 0


    ori_adj_knn = ori_adj_knn.to_dense()

    mask_1 = ori_adj_knn > 0


    # Training
    for epoch in range(args.epoch_num):
        tic = time.time()
        loss, valid_acc = train_fullbatch(model, optimizer, features, gnd, mask_1, train_mask, valid_mask, args)
        
        if (valid_acc >= best_valid_acc):
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model)
            best_epoch = epoch

        # if early stop
        if args.early_stop:
            if (valid_acc >= best_valid_acc):
                patience = args.patience
            else:
                patience -= 1
                if (patience < 0):
                    print("Early Stopped!")
                    break

        toc = time.time()
        train_time = toc-tic
        if args.verbose == 1:
            print("Epoch: {0:d}".format(epoch), 
                "Training loss: {0:1.5f}".format(loss.cpu().detach().numpy()), 
                "Valid accuracy: {0:1.5f}".format(valid_acc),
                "Time used: {0:1.5f}".format(train_time)
                )
    test_model = best_model

    with torch.no_grad():
        test_model.eval()
        Z_tp, mo_A = test_model(features)
        predictions = F.log_softmax(Z_tp, dim=1)
        test_f1_macro, test_f1_micro = classification_evaluate(predictions, test_mask, gnd)
        print("Best epoch:", str(best_epoch))
        print("Macro_F1_value:", test_f1_macro, "Micro_F1_value", test_f1_micro)




def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def train_fullbatch(model, optimizer, features, gnd, mask_1, train_mask, valid_mask, args):
    model.train()
    optimizer.zero_grad()

    Z_tp, mo_A = model(features)


    predictions = F.log_softmax(Z_tp, dim=1)
    loss = 0.0
    loss_semantic = rec_loss_3(mo_A,  mask_1)

    loss += args.gamma * loss_semantic
    if args.isSemi == True:
        loss_semi = F.nll_loss(predictions[train_mask], gnd[train_mask]) 
        loss += loss_semi
    loss.backward()
    optimizer.step()

    # Evaluation Valid Set
    with torch.no_grad():
        model.eval()
        Z_tp, mo_A = model(features)
        predictions = F.log_softmax(Z_tp, dim=1)
        valid_acc = accuracy(predictions[valid_mask], gnd[valid_mask]).cpu().detach().numpy()
    return loss, valid_acc


