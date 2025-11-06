import os
import random
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import torch.nn.functional as F
import argparse
from copy import deepcopy
from tqdm import tqdm
import datetime
import clip
import time

from metrics import get_cindex, get_rm2
from model_clip_full import ColdDTA
# from model_clip_transformer import ColdDTA
# from model_clip_mamba_ori import MambaDTA
# from model_clip_mamba import MambaDTA
# from model_clip_mamba_ht import MambaDTA
from utils import *
from log.train_logger import TrainLogger
from torch_geometric.data import InMemoryDataset

from transE import TransE
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

sot_token = _tokenizer.encoder["<|startoftext|>"]
eot_token = _tokenizer.encoder["<|endoftext|>"]
    
log_interval = 50
PRECISION = 2

class DRPDataset(InMemoryDataset):
    def __init__(self, dataset_path=None, root = 'data/',  transform=None,
                 pre_transform=None,smile_graph=None):
        #root is required for save preprocessed data, default is '/tmp'
        super(DRPDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.data, self.slices = torch.load(dataset_path)


    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, xd, xt, y,smile_graph):
        pass

def main(SEED):
    batch_size = 256     # 参数
    lr = 0.0005
    log_interval = 20

    DATASET = "kiba+chem"
    # DATASET = "kiba_wo_chem"

    work_dir = f"{DATASET}_transe"       # 路径

    date_info = ("_" + dateStr()) if DATASET != "test" else ""

    work_dir = "./exp/" + 'filter' + "/" + work_dir + "_"  +  date_info

    if not os.path.exists("./exp/" + "filter"):
        os.mkdir("./exp/" + "filter")

    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    # copyfile(yaml_path, work_dir + "/")
    model_st = "ColdDTA"

    data_list_train = []
    pt_data = DRPDataset(dataset_path = f"./data/{DATASET}/processed/{DATASET}_train.pt")
    data_list_train.append(pt_data)
    train_data = ConcatDataset(data_list_train)
    # pdb.set_trace()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)

    device = torch.device('cuda:1')
    model = ColdDTA(device, PRECISION).to(device)
    # pdb.set_trace()
    # transe = TransE(int('1'+'0'*PRECISION),1,device, dim=128)
    # start, end = 0, 17.2
    transe = TransE(int((17.2-0)*10**PRECISION),1,device, dim=128)
    transe.to(device)

    epochs = 100
    break_flag = False

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': transe.parameters()}
    ], lr=lr)
    criterion = nn.MSELoss()

    best_mse = 9999
    # best_S2_mse = 9999
    # best_S3_mse = 9999
    # best_S4_mse = 9999
    # best_pearson = -1
    # best_epoch = -1

    model_file_name = work_dir + "/" + model_st + ".model"
    result_file_name = work_dir + "/" + model_st + ".csv"
    # loss_fig_name = work_dir + "/" + model_st + "_loss"
    # pearson_fig_name = work_dir + "/" + model_st + "_pearson"

    global_epoch = 0
    early_stop_epoch = 300
    running_best_mse = BestMeter("min")

    rankingLossFunc = torch.nn.MarginRankingLoss(
        margin=0.0, reduction='mean')       # 对比损失或三重损失
    
    with open(result_file_name, "a") as f:
        f.write("lr: "+str(lr)+"\n")
        f.write("dataset: "+str(DATASET)+"\n")
        f.write("PRECISION: " + str(PRECISION) + "\n")
        f.write("seed: "+str(SEED)+"\n")
    
    for epoch in tqdm(range(epochs)):

        loss_TransE, loss_CLIP, loss_CLIP_Num = train(model, device, train_loader, optimizer, epoch, log_interval, criterion, transe)
        print(f"loss_TransE: {loss_TransE}\tloss_CLIP: {loss_CLIP}\tloss_CLIP_num: {loss_CLIP_Num}")

        with open(result_file_name, "a") as f:
            f.write("\n " + str(epoch))
            f.write("\n loss_TransE:"+str(loss_TransE))
            f.write("\n loss_CLIP:"+str(loss_CLIP)+"\n")
        if epoch%1==0 and loss_CLIP < 0.005:
            model_file_name = work_dir + "/" + model_st + "_"+ str(epoch) +"epoch.model"
            torch.save(model.state_dict(), model_file_name)

def train(model, device, train_loader, optimizer, epoch, log_interval, criterion, transe):
    print("Training on {} samples...".format(len(train_loader.dataset)))
    model.train()
    # DC_cross_entropy_loss = torch.nn.CrossEntropyLoss()
    # T_cross_entropy_loss = torch.nn.CrossEntropyLoss()
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    transe.to(device)
    for batch_idx, data in enumerate(train_loader):
        # start = time.time()
        data = data.to(device)
        # pred = model(data)
        logits_per_dc, logits_per_text, num_logits_per_dc, num_logits_per_text = model.infer(data)
        number_features = generate_samples(model, data, 0, int('1'+'0'*PRECISION), device)
        # pdb.set_trace()
        loss_TransEs = transe(number_features,number_features)
        loss_TransE = torch.sum(loss_TransEs[0])+torch.sum(loss_TransEs[1])+torch.sum(loss_TransEs[2])
        
        labels = torch.arange(data.y.shape[0]).long().to(device)

        loss_dc = cross_entropy_loss(logits_per_dc, labels)
        loss_t = cross_entropy_loss(logits_per_text, labels)

        loss_dc_num = cross_entropy_loss(num_logits_per_dc, labels)
        loss_t_num = cross_entropy_loss(num_logits_per_text, labels)

        loss_CLIP = (loss_dc + loss_t)/2
        loss_CLIP_Num = (loss_dc_num + loss_t_num)/2
        # print(loss_TransE)
        # loss = loss_TransE*0.05 + loss_CLIP*0.95
        loss = loss_TransE*0.01 + loss_CLIP * 0.19 + loss_CLIP_Num * 0.8

        # loss = criterion(pred.view(-1), data.y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # end = time.time()
        # print(f"train batch 256: {end-start}")

        if batch_idx % log_interval == 0:
            print(
                "Train epoch: {} ({:.0f}%)  loss_TransE: {:.4f}  loss_CLIP: {:.4f}  loss_num_CLIP: {:.4f}".format(
                    epoch, 100.0 * batch_idx / len(train_loader), loss_TransE.item(), loss_CLIP.item(), loss_CLIP_Num.item()
                )
            )
    return loss_TransE.item(), loss_CLIP.item(), loss_CLIP_Num.item()
        

def predicting(model, test_loader, device):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print("Testing on {} samples...".format(len(test_loader.dataset)))
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            data_ = deepcopy(data)

            pred = model(data)
            pred = pred.cpu().view(-1, 1)
            # pdb.set_trace()
            total_preds = torch.cat((total_preds, pred), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    # pdb.set_trace()
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def generate_samples(model, data, start, end, device):
    # data: list, consist of [drug smile, cell line, ic50]
    descriptions = []
    assert end - start == int('1'+'0'*PRECISION)

    # start, end = 0, 10.0
    # start, end = 5, 10.8
    start, end = 0, 17.2
    
    # if model.training:    
    # for ic50 in range(start,end,1):
    # for idx, ic50 in enumerate(range(0,int('1'+'0'*PRECISION),1)):
    for idx, ic50 in enumerate(np.arange(start,end,1/(10**PRECISION))):
        # 
        # pdb.set_trace()
        # print(ic50)
        # des = "zero point" + num2english(ic50/int('1'+'0'*PRECISION), PRECISION)

        des = ""
        temp = int(ic50)
        alpha = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        while temp > 0:
            des = alpha[temp%10] + " " + des
            temp //= 10
        if des == "":
            des += "zero "
        des = des + "point" + num2english(ic50.item(), PRECISION=2)
        # pdb.set_trace()
        
        descriptions.append(des)

    text = clip.tokenize(descriptions,context_length=100).to(device)
    # pdb.set_trace()
    text_features = model.encode_num(text)
    # pdb.set_trace()
    
    return text_features

def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def dateStr():
    return (
        str(datetime.datetime.now())
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .split(".")[0]
        .replace("_", "")
    )

if __name__ == "__main__":
    SEED = 171
    seed_torch(SEED)      # 随机种子，保证数据一致性

    main(SEED)
