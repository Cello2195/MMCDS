import random
import os
from model import AttentionDTI
from dataset import CustomDataSet, collate_fn
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from hyperparameter import hyperparameter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score,precision_recall_curve, auc
import datetime
from utils import *
from sklearn.metrics import confusion_matrix, f1_score

def show_result(DATASET,lable,Accuracy_List,Precision_List,Recall_List,AUC_List,AUPR_List,save_path):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)
    print("The {} model's results:".format(lable))
    with open("{}/results.txt".format(save_path), 'w') as f:
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')

    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))

def load_tensor(file_name, dtype):
    # return [dtype(d).to(hp.device) for d in np.load(file_name + '.npy', allow_pickle=True)]
    return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]

def test_precess(model,pbar,LOSS):
    model.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            compounds, proteins, labels = data
            compounds = compounds.cuda()
            proteins = proteins.cuda()
            labels = labels.cuda()

            predicted_scores = model(compounds, proteins)
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())
    Precision = precision_score(Y, P)
    Recall = recall_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)  
    return Y, P, test_loss, Accuracy, Precision, Recall, AUC, PRC

def test_model(dataset_load,save_path,DATASET, LOSS, dataset = "Train",lable = "best",save = True):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_load)),
        total=len(dataset_load))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
        test_precess(model,test_pbar, LOSS)
    if save:
        with open(save_path + "/{}_{}_{}_prediction.txt".format(DATASET,dataset,lable), 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}_set--Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(lable, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test)
    print(results)
    return results,Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test


def get_kfold_data(i, datasets, k=5):
    
    fold_size = len(datasets) // k  

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:] 
        trainset = datasets[0:val_start]

    return trainset, validset

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def dateStr():
    return (
        str(datetime.datetime.now())
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .split(".")[0]
        .replace("_", "")
    )

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == "__main__":
    """select seed"""
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    """init hyperparameters"""
    hp = hyperparameter()

    """Load preprocessed data."""
    DATASET = "BIO_true"
    K_top = 3
    print("Test in " + DATASET)
    if DATASET == "DrugBank":
        weight_CE = None
        dir_input = ('./data/{}.txt'.format(DATASET))
        print("load data")
        with open(dir_input, "r") as f:
            train_data_list = f.read().strip().split('\n')
        print("load finished")
    elif DATASET == "Davis":
        weight_CE = torch.FloatTensor([0.3, 0.7]).cuda()
        dir_input = ('./data/{}.txt'.format(DATASET))
        print("load data")
        with open(dir_input, "r") as f:
            train_data_list = f.read().strip().split('\n')
        print("load finished")
    elif DATASET == "KIBA":
        weight_CE = torch.FloatTensor([0.2, 0.8]).cuda()
        dir_input = ('./data/{}.txt'.format(DATASET))
        print("load data")
        with open(dir_input, "r") as f:
            train_data_list = f.read().strip().split('\n')
        print("load finished")
    elif DATASET == "BIO":
        weight_CE = torch.FloatTensor([0.2, 0.8]).cuda()
        train_dir_input = ('./data/{}_train.txt'.format(DATASET))
        test_dir_input = ('./data/{}_test.txt'.format(DATASET))
        print("load data")
        with open(train_dir_input, "r") as f:
            train_data_list = f.read().strip().split('\n')
        with open(test_dir_input, "r") as f:
            test_data_list = f.read().strip().split('\n')
        print("load finished")
    elif DATASET.startswith("BIO_KNN_"):
        weight_CE = torch.FloatTensor([0.2, 0.8]).cuda()
        dir_input = ('./data/{}.txt'.format(DATASET))
        print("load data")
        with open(dir_input, "r") as f:
            data_list = f.read().strip().split('\n')
        print("load finished")
    elif DATASET == "BIO_true":
        weight_CE = torch.FloatTensor([0.2, 0.8]).cuda()
        train_val_dir_input = ('./data/{}_train_val_KNN_{}.txt'.format(DATASET, K_top))
        test_dir_input = ('./data/{}_test_KNN_{}.txt'.format(DATASET, K_top))
        print("load data")
        with open(train_val_dir_input, "r") as f:
            train_val_data_list = f.read().strip().split('\n')
        with open(test_dir_input, "r") as f:
            test_data_list = f.read().strip().split('\n')
        print("load finished")

    # random shuffle
    print("data shuffle")
    train_val_dataset = shuffle_dataset(train_val_data_list, SEED)

    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

    train_dataset = CustomDataSet(train_val_dataset)
    test_dataset = CustomDataSet(test_data_list)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_val_dataset, [int(0.8 * len(train_val_dataset)), len(train_val_dataset) - int(0.8 * len(train_val_dataset))])

    train_size = len(train_dataset)
    train_dataset_load = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,drop_last=False,
                                    collate_fn=collate_fn)
    valid_dataset_load = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,drop_last=False,
                                    collate_fn=collate_fn)
    test_dataset_load = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,drop_last=False,
                                    collate_fn=collate_fn)

    """ create model"""
    model = AttentionDTI(hp).cuda()
    """weight initialize"""
    weight_p, bias_p = [], []
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    """load trained model"""
    model.load_state_dict(torch.load("checkpoints/BIO_true/checkpoint.pth"))

    criterion = SigmoidFocalLoss(gamma = 7.5, alpha = 0.95, reduction = 'mean')

    """Start testing."""
    print('Testing...')

    model.eval()
    Y, P, S = [], [], []
    with torch.no_grad():
        """test"""
        test_pbar = tqdm(
            enumerate(
                BackgroundGenerator(test_dataset_load)),
            total=len(test_dataset_load))
        model.eval()
        Y, P, S = [], [], []
        with torch.no_grad():
            for test_i, test_data in test_pbar:
                '''data preparation '''
                test_compounds, test_proteins, test_labels = test_data

                test_compounds = test_compounds.cuda()
                test_proteins = test_proteins.cuda()
                test_labels = test_labels.cuda()

                test_scores = model(test_compounds, test_proteins)
                test_labels = test_labels.to('cpu').data.numpy()
                test_scores = F.softmax(test_scores, 1).to('cpu').data.numpy()
                test_predictions = np.argmax(test_scores, axis=1)
                test_scores = test_scores[:, 1]

                Y.extend(test_labels)
                P.extend(test_predictions)
                S.extend(test_scores)
        Recall_dev = recall_score(Y, P)
        Accuracy_dev = accuracy_score(Y, P)
        tpr, fpr, _ = precision_recall_curve(Y, S)
        PRC_dev = auc(fpr, tpr)
        f1_dev = f1_score(Y, P, average='binary')

        Y = np.array(Y)
        P = np.array(P)
        conf_matrix = confusion_matrix(Y, P)
        S_ = 0.5 + 0.5 * ((np.array(S) - np.min(S)) / (np.max(S) - np.min(S)))
        S_ = sorted(S_, reverse=True)
        print("Confusion Matrix in Fig. 2b:")
        print(conf_matrix)
        print("\n")

        print("Prediction Confidence Level in Fig. 2c")
        for idx, item in enumerate(S_):
            print(f"Molecule {idx+1}: {item}")
        print("\n")