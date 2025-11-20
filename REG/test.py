import pandas as pd
import torch
import numpy as np

def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
rankingLossFunc = torch.nn.MarginRankingLoss(margin=0.0, reduction='mean')

df = pd.read_csv("data/test.csv")
print("Statistical indicators in Fig. 2d:")
print(f"PCC: {pearson(df['affinity'], df['Predict'])}")
print(f"RANK: {1-rankingLossFunc(torch.tensor(df['affinity']), torch.tensor(df['Predict']), torch.ones_like(torch.tensor(df['Predict']))).item()}")
print("\nThe files for Figs. 2e and 2f are too large to display. Please run preprocess.py file to generate preprocessed data and then perform inference.")