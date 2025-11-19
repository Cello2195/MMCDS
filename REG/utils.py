import os
import torch
import numpy as np
from math import sqrt
from scipy import stats
import matplotlib.pyplot as plt

class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def save_checkpoint(model, model_dir, epoch, val_loss, val_acc):
    model_path = os.path.join(model_dir, 'epoch:%d-val_loss:%.3f-val_acc:%.3f.model' % (epoch, val_loss, val_acc))
    torch.save(model, model_path)


def load_checkpoint(model_path):
    return torch.load(model_path)


def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))


def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))


def cycle(iterable):
    while True:
        print("end")
        for x in iterable:
            yield x

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def draw_sort_pred_gt(pred,gt,title):
    # gt = gt.y.cpu().numpy()
    # pred = pred.squeeze().cpu().detach().numpy()
    # zipped = zip(gt,pred)
    # sort_zipped = sorted(zipped,key=lambda x:(x[0]))
    # data_gt, data_pred = [list(x) for x in zip(*sort_zipped)]
    # pdb.set_trace()
    data_gt, data_pred = zip(*sorted(zip(gt,pred)))
    plt.figure()
    plt.scatter( np.arange(len(data_gt)),data_gt, s=0.1, alpha=1, label='gt')
    plt.scatter( np.arange(len(data_gt)),data_pred, s=0.1, alpha=1, label='pred')
    plt.legend()
    plt.savefig(title+".png")
    plt.close()

def num2english(num, PRECISION=2):
    num = str(round(num,PRECISION)).split('.')[1]
    
    while len(num)!=PRECISION:
        num = num + '0'

    L1 = ["zero","one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    word = ""
    for i in str(num):
        # pdb.set_trace()
        word = word+" "+L1[int(i)]
   
    return word