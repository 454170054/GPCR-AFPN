from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, f1_score
import numpy as np


class Metrics:

    def __init__(self, name):
        self.y = []
        self.prediction = []
        self.probs = []
        self.name = name

    def add(self, r):
        self.y = self.y + r[0]
        self.probs = self.probs + r[1]

    def calculate_metrics(self):
        self.prediction = [1 if i >= 0.5 else 0 for i in self.probs]
        cm = confusion_matrix(self.y, self.prediction) #
        acc = accuracy_score(self.y, self.prediction)
        sp = specific(cm)
        sn = recall_score(self.y, self.prediction, pos_label=1)
        mcc = matthews_corrcoef(self.y, self.prediction)
        pre = precision_score(self.y, self.prediction, pos_label=1)
        f1 = f1_score(self.y, self.prediction, pos_label=1)
        print("{}_acc: {:.4f} ".format(self.name, acc))
        print("{}_s p: {:.4f} ".format(self.name, sp))
        print("{}_rec: {:.4f} ".format(self.name, sn))
        print("{}_mcc: {:.4f} ".format(self.name, mcc))
        print("{}_pre: {:.4f} ".format(self.name, pre))
        print("{}_f1: {:.4f} ".format(self.name, f1))
        self.prediction = np.array(self.prediction).reshape(-1, 1)
        self.probs = np.array(self.probs).reshape(-1, 1)
        return acc, sp, sn, mcc, pre, f1


def specific(confusion_matric):
    specific = confusion_matric[0][0] / (confusion_matric[0][0] + confusion_matric[0][1])
    return specific


def calculate_mean_metric(metrics, name):
    acc, sp, sn, mcc, pre, f1 = [], [], [], [], [], []
    for metrix in metrics:
        acc.append(metrix[0])
        sp.append(metrix[1])
        sn.append(metrix[2])
        mcc.append(metrix[3])
        pre.append(metrix[4])
        f1.append(metrix[5])
    print("{}_acc: {:.4f}, std:{:.4f} ".format(name, np.mean(acc), np.std(acc)))
    print("{}_s p: {:.4f}, std:{:.4f} ".format(name, np.mean(sp), np.std(sp)))
    print("{}_s n: {:.4f}, std:{:.4f} ".format(name, np.mean(sn), np.std(sn)))
    print("{}_mcc: {:.4f}, std:{:.4f} ".format(name, np.mean(mcc), np.std(mcc)))
    print("{}_rec: {:.4f}, std:{:.4f} ".format(name, np.mean(pre), np.std(pre)))
    print("{}_f1: {:.4f}, std:{:.4f} ".format(name, np.mean(f1), np.std(f1)))


def save_result(name, metrics, dir):
    f = open(f'../../resources/{dir}/{name}_metrics.csv', 'w')
    f.write('acc,sp,rec,mcc,pre,f1\n')
    for metric in metrics:
        f.write('{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(metric[0], metric[1], metric[2], metric[3], metric[4], metric[5]))
    f.close()