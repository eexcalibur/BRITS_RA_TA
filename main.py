import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np
import sys
import time
import utils
import my_models
import argparse
import data_loader
import data_loader_imputation
import pandas as pd
import ujson as json
import matplotlib.pyplot as plt

from sklearn import metrics

from ipdb import set_trace
import warnings
warnings.filterwarnings('ignore')

epochs = 20
batch_size = 64
model_name = 'brits'

def train(model):
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    data_iter = data_loader.get_loader(batch_size = batch_size)

    aucs = []

    for epoch in range(epochs):
        model.train()

        run_loss = 0.0

        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)

            ret = model.run_on_batch(data, optimizer)
            run_loss += ret['loss'].detach().cpu().numpy()

            if idx % 20 == 0:
                print('\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)))

        if epoch % 1 == 0:
            auc = evaluate(model, data_iter)
            aucs.append(auc)

    return aucs


def evaluate(model, val_iter):
    model.eval()

    labels_v = []
    preds_v = []
    labels_t = []
    preds_t = []

    evals = []
    imputations = []

    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        pred = ret['predictions'].data.cpu().numpy()
        label = ret['labels'].data.cpu().numpy()
        is_train = ret['is_train'].data.cpu().numpy()

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()

        # collect test and validation label & prediction
        pred_v = pred[np.where(is_train == 0)]
        label_v = label[np.where(is_train == 0)]
        pred_t = pred[np.where(is_train == 2)]
        label_t = label[np.where(is_train == 2)]

        labels_v += label_v.tolist()
        preds_v += pred_v.tolist()
        labels_t += label_t.tolist()
        preds_t += pred_t.tolist()

    labels_v = np.asarray(labels_v).astype('int32')
    preds_v = np.asarray(preds_v)
    labels_t = np.asarray(labels_t).astype('int32')
    preds_t = np.asarray(preds_t)

    print('AUC of validation {}'.format(metrics.roc_auc_score(labels_v, preds_v)))
    print('AUC of test {}'.format(metrics.roc_auc_score(labels_t, preds_t)))

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    print('MAE', np.abs(evals - imputations).mean())
    print('MRE', np.abs(evals - imputations).sum() / np.abs(evals).sum())

    return metrics.roc_auc_score(labels_v, preds_v)

def run():
    torch.manual_seed(3)
    model = getattr(my_models, model_name).Model()

    if torch.cuda.is_available():
        model = model.cuda()

    aucs = train(model)
    print(max(aucs))
    
if __name__ == '__main__':
    run()
