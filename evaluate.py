import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(0)
from sklearn.metrics import f1_score

def classification_evaluate(embeds, idx_test, gnd):

    test_embs = embeds[idx_test]

    preds = torch.argmax(test_embs, dim=1)

    test_lbls = gnd[idx_test]

    f1_macro_list = []
    f1_micro_list = []
    for i in range(5):
        test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
        test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')
        f1_macro_list.append(test_f1_macro)
        f1_micro_list.append(test_f1_micro)

    avg_f1_macro = np.sum(f1_macro_list) / len(f1_macro_list)
    avg_f1_micro = np.sum(f1_micro_list) / len(f1_micro_list)

    return avg_f1_macro, avg_f1_micro

