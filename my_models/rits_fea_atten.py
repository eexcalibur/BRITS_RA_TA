import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import utils
import argparse
import data_loader

from ipdb import set_trace
from sklearn import metrics
import sys
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

SEQ_LEN = 49
RNN_HID_SIZE = 64

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.build()
        
    def us_score(self, aa):
        rate = 0.05
        
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

    def us_score_gp(self, mask, value):
        
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
        
        return 1 - sigma
        
    

    def build(self):
        self.rnn_cell = nn.LSTMCell(35 * 2, RNN_HID_SIZE)

        self.temp_decay_h = TemporalDecay(input_size = 35, output_size = RNN_HID_SIZE, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = 35, output_size = 35, diag = True)

        self.hist_reg = nn.Linear(RNN_HID_SIZE, 35)
        self.feat_reg = FeatureRegression(35)

        self.weight_combine = nn.Linear(35 * 2, 35)

        self.dropout = nn.Dropout(p = 0.25)
        self.out = nn.Linear(RNN_HID_SIZE, 1)
        
        self.encoder_attn = nn.Linear(in_features=RNN_HID_SIZE, out_features=1)
        


    def forward(self, data, direct):
        # Original sequence with 24 time steps
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']
        
        if direct == 'forward':
            uncern_score = data['forward']['us']
        else:
            uncern_score = torch.flip(data['forward']['us'],[1])
        
        #print(direct)
        #sys.exit()
        
        #uncern_score = torch.zeros_like(masks, dtype=torch.double)
        
        #print("before uncern score")
        #for i in range(masks.size()[0]):
        #    for j in range(masks.size()[2]):
                #uncern_score[i,:,j] = torch.DoubleTensor(self.us_score(masks[i,:,j]))
        #        uncern_score[i,:,j] = torch.DoubleTensor(self.us_score_gp(masks[i,:,j], values[i,:,j]))
        
        #print("after uncern score")
        uncern_score = torch.where(uncern_score < 0, 0., uncern_score)
 
        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        labels = data['labels'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)

        h = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))
        c = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))
        
        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        #xxx = torch.cat((h.repeat(values.size()[2],1,1).permute(1, 0, 2),
        #                c.repeat(values.size()[2],1,1).permute(1, 0, 2)), dim=2)
        
        xxx = h.repeat(values.size()[2],1,1).permute(1,0,2)

        
        eee = self.encoder_attn(xxx.reshape(-1, RNN_HID_SIZE))
        aaa = F.softmax(eee.view(-1, values.size()[2]))
        
        #print("alpha size:",alpha.size())
        #print(values[:,0,0])
        #print(alpha[:,0])
        #sys.exit()
        
        #for i in range(values.size()[1]):
        #    print(values[:,i,:].size())
        #    print(alpha.size())
        #    values[:,i,:] = values[:,i,:] * alpha
        
        #print("value size:", values.size())
        #print("alpha size:",alpha.size())
        
        #sys.exit()
            
        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(SEQ_LEN):
            x = values[:, t, :] * aaa
            m = masks[:, t, :]
            d = deltas[:, t, :]
            
            us = uncern_score[:,t,:]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            #print("h",h[:,0])
            x_h = self.hist_reg(h)
            #print("x_h",x_h[:,0])
            
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)
            
            #print("x",x[:,0])
            #print("m",m[:,0])
            #print("d",d[:,0])
            #print("us",us[:,0])
            #sys.exit()
            
            

            x_c =  m * x +  (1 - m) * x_h
            
            #print("x_c",x_c[:,0])

            z_h = self.feat_reg(x_c)
            
            #print("z_h",z_h[:,0])
            
            z_h = z_h * us.type(torch.float)
            
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            
            #print("c_h",c_h[:,0])
            
            #c_h = c_h * us.type(torch.float)
            
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h
            
            #print("c_c",c_h[:,0])
            #sys.exit()
            
            #c_c = c_c * us.type(torch.float)

            inputs = torch.cat([c_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))
            
            #print("h ",h.size())
            #print("us ",us.size())
            #h = h * us.type(torch.float)

            imputations.append(c_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        y_h = self.out(h)
        
        
        y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False)
        #print("y1_loss",y_loss)
        #print("is_train", is_train)
        y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)

        #print("x_loss",x_loss)
        #print("y1_loss",y_loss)
        #sys.exit()
        
        y_h = F.sigmoid(y_h)

        return {'loss': x_loss / SEQ_LEN + y_loss * 0.3, 'predictions': y_h,\
                'imputations': imputations, 'labels': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}
    
    
    def run_on_batch(self, data, optimizer):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
