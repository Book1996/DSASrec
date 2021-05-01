import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
class ConvolutionalAttention(nn.Module):
    def __init__(self,n_head,dim,temperature,dropout):
        super(ConvolutionalAttention, self).__init__()
        self.temperature = temperature
        self.num = 2
        self.h = n_head  
        self.d = 100
        self.filters1 = nn.Conv2d(dim*self.num, self.d,(self.h,1),bias=True)
        self.filters2 = nn.Conv2d(self.d,self.d,(self.h,1),bias=True)
        self.filters3 = nn.Conv2d(self.d,1,(self.h,1),bias=True)
        self.act1 = nn.ReLU()
        self.act2 = nn.Softmax(2)
        self.bn1 = nn.BatchNorm2d(self.d)
        self.bn2 = nn.BatchNorm2d(self.d)
        self.dropout = nn.Dropout(dropout)
        self.p1d = (0, 0, self.h-1, 0)
    def forward(self, q, k, v, mask=None,s2s_mask=None):
        B,S,T = q.shape
        q = q.transpose(1,2).unsqueeze(3).expand(-1,-1,-1,S)
        k = k.transpose(1,2).unsqueeze(2).expand(-1,-1,S,-1)   
        attn = torch.cat([q,k],1)        
        attn = self.filters1(attn)
        resiaul = attn
        attn = F.pad(attn, self.p1d)
        attn = self.bn1(attn)
        attn = self.filters2(self.act1(attn))
        attn = self.bn2(attn+resiaul)
        attn = self.filters3(self.act1(attn))
        attn = (attn.squeeze())/self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -1e16)
        attn = self.act2(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)#b*s*t
        return output, attn
class ConvolutionalAttention2(nn.Module):
    def __init__(self,n_head,dim,temperature,dropout):
        super(ConvolutionalAttention2, self).__init__()        
        self.num = 2
        self.d = dim*self.num
        self.temperature = temperature
        self.deep = nn.Sequential(
            nn.Linear(dim*2,50),
            nn.ReLU(),
            nn.Linear(50,25),
            nn.ReLU(),
            nn.Linear(25,1),
        )
        self.softmax = nn.Softmax(2)
        self.dropout = nn.Dropout(dropout)
        #self.deep_vw = nn.Linear(dim, dim)
        
    def forward(self, q, k, v, mask=None,s2s_mask=None):
        B,S,T = q.shape
        # shallow part
        # shallow_v = v#self.shallow_vw(v)
        # shallow_q = self.shallow_qw(q)
        # shallow_k = self.shallow_kw(k)
        # shallow_attn = torch.bmm(shallow_q, shallow_k.transpose(1, 2))
        # shallow_attn = shallow_attn/self.temperature
        # if mask is not None:
        #     shallow_attn = shallow_attn.masked_fill(mask, -1e16)  
        # shallow_attn = self.softmax(shallow_attn)
        # shallow_attn = self.dropout(shallow_attn)
        # shallow_output = torch.bmm(shallow_attn, shallow_v)#b*s*t
        
        
        
        #deep part
        deep_v = v#self.deep_vw(v)
        
        
        
        # deep_attn = torch.rand((B,S,S),device = q.device)
        # for i in range(S):
        #     q_ = q[:,i,:].unsqueeze(1).expand(B,S,T)
        #     pair = torch.cat([q_,k],-1)
        #     score = self.deep(pair).squeeze()
        #     deep_attn[:,i,:] = score
                
                
        deep_q = q.unsqueeze(2).expand(-1,-1,S,-1)
        deep_k = k.unsqueeze(1).expand(-1,S,-1,-1)
        deep_attn = torch.cat([deep_q,deep_k],-1)         
        deep_attn = self.deep(deep_attn).squeeze()




        
        deep_attn = deep_attn/self.temperature
        if mask is not None:
            deep_attn = deep_attn.masked_fill(mask, -1e16)       
        deep_attn = self.softmax(deep_attn)
        deep_attn = self.dropout(deep_attn)       
        deep_output = torch.matmul(deep_attn,deep_v)#b*s*t      
        
        
        
        
        #output
        # attn =  deep_attn+shallow_attn
        # output = self.dense(torch.cat([deep_output,shallow_output],-1))
        attn =  deep_attn
        output = deep_output
        return output,attn
class ConvolutionalAttention3(nn.Module):
    def __init__(self,dim,temperature,dropout):
        super(ConvolutionalAttention3, self).__init__()
        self.temperature = temperature
        self.d = 100
        self.H = 1
        self.p1d = (0, 0, self.H-1, 0)
        self.deep_k = nn.Sequential(
            nn.Linea(dim,dim),
            nn.ReLU(),
            nn.Linea(dim,dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nnn.Linea(dim,dim),
        )
        self.deep_q = nn.Sequential(
            nn.linear(dim,dim),
            nn.ReLU(),
            nn.linear(dim,dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.linear(dim,dim),
        )
        # self.deep_vw = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(2)
        self.shallow_qw = nn.Linear(dim,1)
        self.shallow_kw = nn.Linear(dim,1)
        # self.shallow_vw = nn.Linear(dim, dim)
        
        self.dense = nn.Linear(dim*2, dim)
        self.layer_norm = nn.LayerNorm(dim)
        # self.dropout = nn.Dropout(0.1)
    def forward(self, q, k, v, mask=None):
        B,S,T = q.shape
        # shallow part
        # shallow_v = v#self.shallow_vw(v)
        # shallow_q = self.shallow_qw(q)
        # shallow_k = self.shallow_kw(k)
        # shallow_attn = torch.bmm(shallow_q, shallow_k.transpose(1, 2))
        # shallow_attn = shallow_attn/self.temperature 
        # if mask is not None:
        #     shallow_attn = shallow_attn.masked_fill(mask, -1e16)  
        # shallow_attn = self.softmax(shallow_attn)
        # # shallow_attn = self.dropout(shallow_attn)
        # shallow_output = torch.bmm(shallow_attn, shallow_v)#b*s*t
        
        #deep part
        deep_v = v#self.deep_vw(v)
        deep_q = self.deep_q(q)
        deep_k = self.deep_k(k)
        deep_attn = torch.bmm(deep_q, deep_k.transpose(1, 2))
        deep_attn = deep_attn/self.temperature    
        if mask is not None:
            deep_attn = deep_attn.masked_fill(mask, -1e16)
        deep_attn = -self.softmax(deep_attn)
        # deep_attn = self.dropout(deep_attn)       
        deep_output = torch.matmul(deep_attn,deep_v)#b*s*t 
        
        #dense
        attn =  deep_attn
        output = deep_output
        # output = self.dense(torch.cat([deep_output,shallow_output],-1))
        return output,attn   
    
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self,n_head,d_model, temperature,dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(2)
        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None,s2s_mask=None):
        B,S,T = q.shape
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)
        attn = torch.matmul(q, k.transpose(1, 2))#B,T,S,S
        print(attn[0,-15:,-15:])
        attn = attn/self.temperature 
        
        if mask is not None:
            attn = attn.masked_fill(mask, -1e16)  
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn,v)#B,T,S,T
        return output, attn