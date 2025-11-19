import os
import random
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from copy import deepcopy
from tqdm import tqdm
import datetime

from model_clip_full import ColdDTA
from utils import *
from torch_geometric.data import InMemoryDataset

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

rankingLossFunc = torch.nn.MarginRankingLoss(
        margin=0.0, reduction='mean')       # 对比损失或三重损失

def main(SEED):
    batch_size = 1024     # 参数
    lr = 0.0005
    log_interval = 20

    # DATASET = "kiba+chem"
    DATASET = "kiba_wo_chem"

    work_dir = f"{DATASET}_transe_finetuning"       # 路径

    date_info = ("_" + dateStr()) if "binding_db" != "test" else ""

    work_dir = "./exp/" + 'filter' + "/" + work_dir + "_"  +  date_info

    if not os.path.exists("./exp/" + "filter"):
        os.mkdir("./exp/" + "filter")

    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    # copyfile(yaml_path, work_dir + "/")
    model_st = "ColdDTA"

    data_list_train = []
    data_list_test = []
    pt_data = DRPDataset(dataset_path = f"./data/{DATASET}/processed/{DATASET}_train.pt")
    data_list_train.append(pt_data)
    pt_data = DRPDataset(dataset_path = f"./data/{DATASET}/processed/{DATASET}_test.pt")
    data_list_test.append(pt_data)
    train_data = ConcatDataset(data_list_train)
    test_data = ConcatDataset(data_list_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)
    # pdb.set_trace()

    device = torch.device('cuda:1')
    model = ColdDTA(device).to(device)
    model_path = "exp/filter/kiba+chem_transe__20240607091600/ColdDTA_18epoch.model"
    # model_path = "exp/filter/kiba_wo_chem_transe__20250619044638/ColdDTA_20epoch.model"
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)), strict=True)
    epochs = 3000
    break_flag = False

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_mse = 9999
    best_S2_mse = 9999
    best_S3_mse = 9999
    best_S4_mse = 9999
    best_pearson = -1
    best_epoch = -1

    model_file_name = work_dir + "/" + model_st + ".model"
    result_file_name = work_dir + "/" + model_st + ".csv"
    loss_fig_name = work_dir + "/" + model_st + "_loss"
    pearson_fig_name = work_dir + "/" + model_st + "_pearson"

    global_epoch = 0
    early_stop_epoch = 300
    running_best_mse = BestMeter("min")
    
    with open(result_file_name, "a") as f:
        f.write("lr: "+str(lr)+"\n")
        f.write("dataset: "+str(DATASET)+"\n")
        f.write("model: " + str(model_path) + "\n")
        f.write("seed: "+str(SEED)+"\n")
    
    for epoch in tqdm(range(epochs)):
        if break_flag:
            break

        loss = train(model, device, train_loader, optimizer, epoch, log_interval, criterion)

        G, P = predicting(model, test_loader, device)     # todo
        ret_val = [
            rmse(G, P), 
            mse(G, P), 
            pearson(G, P), 
            spearman(G, P), 
            rankingLossFunc(torch.tensor(G), torch.tensor(
                P), torch.ones_like(torch.tensor(P))).item()]    # 残差？
        # print(f"ret_val: {ret_val}")
        # pdb.set_trace()
        
        draw_sort_pred_gt(
            P, G, title=work_dir + "/test_" + str(epoch))

        if ret_val[1] < best_mse:
            running_best_mse.update(ret_val[1])

            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, "a") as f:
                f.write("\nepoch: " + str(epoch))
                f.write("\nrmse: "+str(ret_val[0]))
                f.write("\nmse: "+str(ret_val[1]))
                f.write("\npearson: "+str(ret_val[2]))
                f.write("\nspearman: "+str(ret_val[3]))
                f.write("\nrankingloss: "+str(ret_val[4])+"\n")
            best_epoch = epoch + 1
            best_mse = ret_val[1]
            best_pearson = ret_val[2]

            print(
                " rmse improved at epoch ",
                best_epoch,
                "; best_mse:",
                best_mse,
                model_st,
            )
        else:
            print(
                " no improvement since epoch ",
                best_epoch,
                "; best_mse, best pearson:",
                best_mse,
                best_pearson,
                model_st,
            )
            count = running_best_mse.counter()
            if count > early_stop_epoch:
                print(f"early stop in epoch {global_epoch}")
                break_flag = True
                print('best ', epoch, "'s result is: ", best_mse)
                break

def train(model, device, train_loader, optimizer, epoch, log_interval, criterion):
    print("Training on {} samples...".format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        # start = time.time()
        data = data.to(device)
        pred = model(data)

        loss_mse = criterion(pred.view(-1), data.y.view(-1))
        loss_rank = rankingLossFunc(
            torch.tensor(data.y.view(-1, 1).float().to(device)), 
            torch.tensor(pred), 
            torch.ones_like(torch.tensor(pred))).item()
        loss = loss_mse*0.95 + loss_rank*0.05

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # end = time.time()
        # print(f"train batch 256: {end-start}")

        if batch_idx % log_interval == 0:
            print(
                "Train epoch: {} ({:.0f}%)\tloss_MSE: {:.6f}\tloss_MSE: {:.6f}".format(
                    epoch, 100.0 * batch_idx / len(train_loader), loss.item(), loss.item()
                )
            )
        

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
