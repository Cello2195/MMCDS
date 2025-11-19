import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm
import datetime
import pandas as pd
from collections import defaultdict

from model_clip_full import ColdDTA
from utils import *
from torch_geometric.data import InMemoryDataset
import matplotlib.pyplot as plt

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
    batch_size = 1024     # 参数

    DATASET = "kiba+chem"
    # DATASET = "kiba_wo_chem"

    # work_dir = f"{DATASET}_filter"       # 路径

    # date_info = ("_" + dateStr()) if "binding_db" != "test" else ""

    # work_dir = "./exp/" + 'filter' + "/" + work_dir + "_"  +  date_info

    # if not os.path.exists("./exp/" + "filter"):
    #     os.mkdir("./exp/" + "filter")

    # if not os.path.exists(work_dir):
    #     os.mkdir(work_dir)

    # copyfile(yaml_path, work_dir + "/")
    # model_st = "ColdDTA"

    data_list_test = []
    data_list_noise = []
    data_list_sub = []
    pt_data = DRPDataset(dataset_path = f"./data/{DATASET}/processed/{DATASET}_test.pt")
    # pt_data = DRPDataset(dataset_path = f"./data/{DATASET}/processed/2324.pt")
    data_list_test.append(pt_data)
    test_data = ConcatDataset(data_list_test)
    # pdb.set_trace()
    # pt_data = DRPDataset(dataset_path = "./data/kiba+chem/processed/ChemDiv_noise.pt")
    # data_list_noise.append(pt_data)
    # noise_data = ConcatDataset(data_list_noise)
    # pt_data = DRPDataset(dataset_path = "./data/kiba+chem/processed/substitute.pt")
    # data_list_sub.append(pt_data)
    # sub_data = ConcatDataset(data_list_sub)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)
    # noise_loader = DataLoader(noise_data, batch_size=batch_size, shuffle=False, drop_last=False)
    # sub_loader = DataLoader(sub_data, batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device('cuda:7')
    model = ColdDTA(device).to(device)
    # model_path = "./exp/filter/kiba+chem_transe__20240607091600/ColdDTA_18epoch.model"
    model_path = "./exp/filter/kiba+chem_transe_finetuning__20240608025419/ColdDTA_1285.model"
    # model_path = './exp/filter/kiba_wo_chem_transe_finetuning__20250620021434/ColdDTA_1776.model'
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)), strict=False)
    
    # result_file_name = work_dir + "/" + model_st + ".csv"
    # hit_fig_name = work_dir + "/" + model_st + "_hit.png"
    
    # with open(result_file_name, "a") as f:
    #     f.write("dataset: "+str(DATASET)+"\n")
    #     f.write("model: " + str(model_path) + "\n")
    #     f.write("seed: "+str(SEED)+"\n")

    G, P = predicting(model, test_loader, device)     # todo

    df = pd.read_csv(f"data/{DATASET}/{DATASET}_test.csv")
    # df = pd.read_csv(f"data/{DATASET}/2324.csv")
    P_iter = iter(P)
    preds = []
    j = 0  # pointer for test_data

    for i in range(len(df)):
        if j < len(test_data) and df.iloc[i]['compound_iso_smiles'] == test_data[j].smi:
            preds.append(next(P_iter))
            j += 1
        else:
            preds.append(0)

    df['Predict'] = preds

    # MEIOB = 'MPPGVDCPMEFWTKEENQSVVVDFLLPTGVYLNFPVSRNANLSTIKQLLWHRAQYEPLFHMLSGPEAYVFTCINQTAEQQELEDEQRRLCDVQPFLPVLRLVAREGDRVKKLINSQISLLIGKGLHEFDSLCDPEVNDFRAKMCQFCEEAAARRQQLGWEAWLQYSFPLQLEPSAQTWGPGTLRLPNRALLVNVKFEGSEESFTFQVSTKDVPLALMACALRKKATVFRQPLVEQPEDYTLQVNGRHEYLYGSYPLCQFQYICSCLHSGLTPHLTMVHSSSILAMRDEQSNPAPQVQKPRAKPPPIPAKKPSSVSLWSLEQPFRIELIQGSKVNADERMKLVVQAGLFHGNEMLCKTVSSSEVSVCSEPVWKQRLEFDINICDLPRMARLCFALYAVIEKAKKARSTKKKSKKADCPIAWANLMLFDYKDQLKTGERCLYMWPSVPDEKGELLNPTGTVRSNPNTDSAAALLICLPEVAPHPVYYPALEKILELGRHSECVHVTEEEQLQLREILERRGSGELYEHEKDLVWKLRHEVQEHFPEALARLLLVTKWNKHEDVAQMLYLLCSWPELPVLSALELLDFSFPDCHVGSFAIKSLRKLTDDELFQYLLQLVQVLKYESYLDCELTKFLLDRALANRKIGHFLFWHLRSEMHVPSVALRFGLILEAYCRGSTHHMKVLMKQGEALSKLKALNDFVKLSSQKTPKPQTKELMHLCMRQEAYLEALSHLQSPLDPSTLLAEVCVEQCTFMDSKMKPLWIMYSNEEAGSGGSVGIIFKNGDDLRQDMLTLQMIQLMDVLWKQEGLDLRMTPYGCLPTGDRTGLIEVVLRSDTIANIQLNKSNMAATAAFNKDALLNWLKSKNPGEALDRAIEEFTLSCAGYCVATYVLGIGDRHSDNIMIRESGQLFHIDFGHFLGNFKTKFGINRERVPFILTYDFVHVIQQGKTNNSEKFERFRGYCERAYTILRRHGLLFLHLFALMRAAGLPELSCSKDIQYLKDSLALGKTEEEALKHFRVKFNEALRESWKTKVNWLAHNVSKDNRQ'
    MEIOB = 'MANSFAARIFTTLSDLQTNMANLKVIGIVIGKTDVKGFPDRKNIGSERYTFSFTIRDSPAHFVNAASWGNEDYIKSLSDSFRVGDCVIIENPLIQRKEIEREEKFSPATPSNCKLLLSENHSTVKVCSSYEVDTKLLSLIHLPVKESHDYYSLGDIVANGHSLNGRIINVLAAVKSVGEPKYFTTSDRRKGQRCEVRLYDETESSFAMTCWDNESILLAQSWMPRETVIFASDVRINFDKFRNCMTATVISKTIITTNPDIPEANILLNFIRENKETNVLDDEIDSYFKESINLSTIVDVYTVEQLKGKALKNEGKADPSYGILYAYISTLNIDDETTKVVRNRCSSCGYIVNEASNMCTTCNKNSLDFKSVFLSFHVLIDLTDHTGTLHSCSLTGSVAEETLGCTVHEFLAMTDEQKTALKWQFLLERSKIYLKFVLSHRARSGLKISVLSCKLADPTEASRNLSGQKHV'
    df = df[df['target_sequence'] == MEIOB]
    pdb.set_trace()
    # df.to_csv("/home/data2/xyd/DTA/coldDTA/data/kiba+chem/test_final2.csv")

    df2 = pd.read_csv("./data/merged_data_del/true_test_docking.csv")

    docking_dict = {}
    for idx, row in df2.iterrows():
        docking_dict[row['SMILES']] = row['Average_score']
    docking_list = []
    for idx, row in df.iterrows():
        if row['compound_iso_smiles'] in docking_dict.keys():
            docking_list.append(docking_dict[row['compound_iso_smiles']])
        else:
            docking_list.append(max(df2['Average_score']))

    df['Rigid'] = docking_list
    df['Rigid'] = -df['Rigid']
    df['Rigid'] = 12 + (df['Rigid'] - min(df['Rigid'])) / (max(df['Rigid']) - min(df['Rigid'])) * (14 - 12)
    pdb.set_trace()
    draw_pic(df)

    df = df.reset_index(drop=True)
    best_affinity_index = df['affinity'].idxmax()

    model_rank = find_rank_to_best(df, 'Predict', best_affinity_index)
    schrodinger_rank = find_rank_to_best(df, 'Rigid', best_affinity_index)

    print(f"模型预测中，排名前 {model_rank} 个才能找到最佳 affinity")
    print(f"薛定谔打分中，排名前 {schrodinger_rank} 个才能找到最佳 affinity")

    # G, P = predicting(model, noise_loader, device)     # todo
    # df_noise = pd.read_csv("./data/kiba+chem/ChemDiv_noise.csv")
    # df_noise = pd.read_csv("./data/kiba+chem/noise_predict.csv")
    # df_noise['Predict'] = P
    # sort_per_target(df, hit_fig_name, df_noise)
    # sort_per_target(df, hit_fig_name)

    # G, P = predicting(model, sub_loader, device)     # todo
    # df = pd.read_csv("./data/kiba+chem/substitute.csv")
    # df['Predict'] = P
    # df['Predict'] = -((df['Predict']-12)/(14-12)*(5.13-3.01)+3.01)
    # df.to_csv("./data/kiba+chem/substitute.csv", index=False)
    pdb.set_trace()


    
        
def sort_per_target(df, hit_fig_name, df_noise=None):
    affinity_matrix = defaultdict(list)
    for idx, row in df.iterrows():
        affinity_matrix[row['target_sequence']].append((row['compound_iso_smiles'], row['affinity'], row['Predict']))
    if df_noise is not None and not df_noise.empty:
        for idx, row in df_noise.iterrows():
            affinity_matrix[row['target_sequence']].append((row['compound_iso_smiles'], row['affinity'], row['Predict']))
    # pdb.set_trace()

    target_num = len(affinity_matrix)
    x, y = np.arange(13382), [0]*13382
    
    for key, value in affinity_matrix.items():
        sorted_G = sorted(value, key=lambda item: item[1], reverse=True)
        sorted_P = sorted(value, key=lambda item: item[2], reverse=True)

        PP = [item[0] for item in sorted_P]
        for i in range(13382):
            if sorted_G[0][0] in PP[:i]:
                y[i] += 1
            if i > len(PP):
                for j in range(i+1, 13382):
                    y[j] += 1
                break
        
    xx = x[:50]
    yy = y[:50]
    plt.plot(xx, yy)
    plt.savefig(f'{hit_fig_name}', dpi=300)
    pdb.set_trace()

    return y[0]/target_num, y[1]/target_num, y[2]/target_num

def draw_pic(df):
    # pdb.set_trace()
    # 按affinity列进行排序
    df_sorted = df.sort_values(by='affinity')

    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(df_sorted)), df_sorted['affinity'], color='blue', label='Real Values')
    plt.scatter(range(len(df_sorted)), df_sorted['Predict'], color='green', label='Predicted Values')
    plt.scatter(range(len(df_sorted)), df_sorted['Rigid'], color='orange', label='Docking Values')
    plt.xlabel('Molecular Index')
    plt.ylabel('Predicted Affinity')
    plt.title('Scatter Plot of True vs Predicted Affinity')
    plt.grid(True)
    # plt.show()
    plt.savefig(f'./data/kiba+chem/predict.png')

def find_rank_to_best(df, sort_by, best_index):
    sorted_df = df.sort_values(by=sort_by, ascending=False).reset_index()
    rank = sorted_df[sorted_df['index'] == best_index].index[0] + 1  # +1 表示排名从1开始
    return rank


def predicting(model, test_loader, device):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print("Testing on {} samples...".format(len(test_loader.dataset)))
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader)):
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
