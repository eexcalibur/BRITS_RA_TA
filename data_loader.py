import os
import time

import ujson as json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pdb

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class MySet(Dataset):
    def __init__(self):
        super(MySet, self).__init__()
        #self.content = open('./json/json').readlines()
        self.content = open('/global/homes/z/zhangtao/NeuralCDE/BRITS_data/json/json').readlines()

        #indices = np.arange(len(self.content))
        #np.random.seed(0)
        #val_indices = np.random.choice(indices, len(self.content) // 5)

        ######  cross validation #####
        #np.random.seed(0)
        #indices = np.arange(len(self.content))
        #np.random.shuffle(indices)
        #cross_len = len(self.content) // 5
        #print("ncross:",ncross)
        #print("indices:", indices)
        #val_indices = indices[ncross * cross_len:(ncross+1) * cross_len]
        ####### 
        
        ###### validate and test #####
        np.random.seed(0)
        indices = np.arange(len(self.content))
        np.random.shuffle(indices)
        cross_len = len(self.content) // 5
        #test_indices = indices[4*cross_len:]
        #test_indices = indices[:cross_len]
        #val_indices = indices[cross_len:2*cross_len]
        test_indices = indices[2*cross_len:3*cross_len]
        val_indices = indices[:cross_len]
        
        
        self.val_indices = set(val_indices.tolist())
        self.test_indices = set(test_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        if idx in self.val_indices:
            rec['is_train'] = 0
        elif idx in self.test_indices:
            rec['is_train'] = 2
        else:
            rec['is_train'] = 1
            
        return rec
    
def us_score_gp(mask, value):

    if torch.all((mask == 0)):
        uu = np.zeros((len(mask)))
        return uu

    bb = np.argwhere(mask==1)[0]
    X = bb.unsqueeze(1)
    y = value[bb]

    #print(mask)
    #print(value)
    #print(X.shape, y.shape)
    #sys.exit()

    kernel = C(1.0, (1e-3, 1e3)) * RBF(8, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(X, y)


    xx = torch.arange(49).unsqueeze(1)
    y_pred, sigma = gp.predict(xx, return_std=True)

    #return 1-(sigma - sigma.min()) / (sigma.max() - sigma.min())
    return 1 - sigma
    
def us_score(aa):
    rate = 0.15

    bb = np.zeros((len(aa)))
    if torch.all((aa==0)):
        return bb

    aa = aa.cpu().numpy()
    ind1 = torch.from_numpy(np.argwhere(aa==1)[:,0])

    for iind,ind in enumerate(ind1):


        if ind == ind1[0]:
            for jj in range(ind,-1,-1):
                bb[jj] = aa[ind] - rate*(ind-jj)

        if len(ind1) == 1 and ind == ind1[-1]:
            for jj in range(ind,len(aa),1):
                bb[jj] = aa[ind] - rate*(jj-ind)

        else:
            ind2 = ind1[iind-1]
            ind3 = ind1[iind]

            diff = ind3 - ind2 

            for jj in range(ind2,ind2+int(diff/2)+1):
                bb[jj] = aa[ind2] -rate*(jj-ind2)

            for jj in range(ind3,ind3-int(diff/2)-1,-1):
                bb[jj] = aa[ind3] - rate*(ind3-jj)

            if ind == ind1[-1]:
                for jj in range(ind,len(aa),1):
                    bb[jj] = aa[ind] - rate*(jj-ind)

    return bb 
    
    
def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))

    def to_tensor_dict_forward(recs):
        values = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['values'], r)), recs)))
        #values = torch.FloatTensor(map(lambda r: map(lambda x: x['values'], r), recs))
        masks = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['masks'], r)), recs)))
        deltas = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['deltas'], r)), recs)))
        forwards = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['forwards'], r)), recs)))

        evals = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['evals'], r)), recs)))
        eval_masks = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['eval_masks'], r)), recs)))
        uncern_score = torch.zeros_like(masks, dtype=torch.double)
        
        
        #print("before uncern score")
        for i in range(masks.size()[0]):
            for j in range(masks.size()[2]):
                #uncern_score[i,:,j] = torch.DoubleTensor(self.us_score(masks[i,:,j]))
                #uncern_score[i,:,j] = torch.DoubleTensor(us_score_gp(masks[i,:,j], values[i,:,j]))
                uncern_score[i,:,j] = torch.DoubleTensor(us_score(masks[i,:,j]))

        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks,'us':uncern_score}
    
    def to_tensor_dict(recs):
        values = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['values'], r)), recs)))
        #values = torch.FloatTensor(map(lambda r: map(lambda x: x['values'], r), recs))
        masks = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['masks'], r)), recs)))
        deltas = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['deltas'], r)), recs)))
        forwards = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['forwards'], r)), recs)))

        evals = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['evals'], r)), recs)))
        eval_masks = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['eval_masks'], r)), recs)))
      
        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks}

    
    ret_dict = {'forward': to_tensor_dict_forward(forward), 'backward': to_tensor_dict(backward)}

    ret_dict['labels'] = torch.FloatTensor(list(map(lambda x: x['label'], recs)))
    ret_dict['is_train'] = torch.FloatTensor(list(map(lambda x: x['is_train'], recs)))

    return ret_dict

def collate_fn_new(recs):
    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))

    def to_tensor_dict_forward(recs):

        values = torch.FloatTensor(list(map(lambda r: r['values'],recs)))
        masks = torch.FloatTensor(list(map(lambda r: r['masks'],recs)))
        deltas = torch.FloatTensor(list(map(lambda r: r['deltas'],recs)))
        forwards = torch.FloatTensor(list(map(lambda r: r['forwards'],recs)))

        evals = torch.FloatTensor(list(map(lambda r: r['evals'],recs)))
        eval_masks = torch.FloatTensor(list(map(lambda r: r['eval_masks'],recs)))


        #values = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['values'], r)), recs)))
        #values = torch.FloatTensor(map(lambda r: map(lambda x: x['values'], r), recs))
        #masks = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['masks'], r)), recs)))
        #deltas = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['deltas'], r)), recs)))
        #forwards = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['forwards'], r)), recs)))

        #evals = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['evals'], r)), recs)))
        #eval_masks = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['eval_masks'], r)), recs)))
        
        uncern_score = torch.zeros_like(masks, dtype=torch.double)


        #print("before uncern score")
#        for i in range(masks.size()[0]):
#            for j in range(masks.size()[2]):
                #uncern_score[i,:,j] = torch.DoubleTensor(self.us_score(masks[i,:,j]))
                #uncern_score[i,:,j] = torch.DoubleTensor(us_score_gp(masks[i,:,j], values[i,:,j]))
#                uncern_score[i,:,j] = torch.DoubleTensor(us_score(masks[i,:,j]))

        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks,'us':uncern_score}

    def to_tensor_dict(recs):
        #values = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['values'], r)), recs)))
        #values = torch.FloatTensor(map(lambda r: map(lambda x: x['values'], r), recs))
        #masks = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['masks'], r)), recs)))
        #deltas = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['deltas'], r)), recs)))
        #forwards = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['forwards'], r)), recs)))

        #evals = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['evals'], r)), recs)))
        #eval_masks = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['eval_masks'], r)), recs)))

        values = torch.FloatTensor(list(map(lambda r: r['values'],recs)))
        masks = torch.FloatTensor(list(map(lambda r: r['masks'],recs)))
        deltas = torch.FloatTensor(list(map(lambda r: r['deltas'],recs)))
        forwards = torch.FloatTensor(list(map(lambda r: r['forwards'],recs)))

        evals = torch.FloatTensor(list(map(lambda r: r['evals'],recs)))
        eval_masks = torch.FloatTensor(list(map(lambda r: r['eval_masks'],recs)))

        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks}


    ret_dict = {'forward': to_tensor_dict_forward(forward), 'backward': to_tensor_dict(backward)}

    ret_dict['labels'] = torch.FloatTensor(list(map(lambda x: x['label'], recs)))
    ret_dict['is_train'] = torch.FloatTensor(list(map(lambda x: x['is_train'], recs)))

    return ret_dict

def get_loader(batch_size = 64, shuffle = True):
    data_set = MySet()
    torch.manual_seed(70)
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 4, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              worker_init_fn=np.random.seed(70), \
                              collate_fn = collate_fn_new
    )

    return data_iter
