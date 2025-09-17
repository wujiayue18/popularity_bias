"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
from itertools import chain
import torch.nn.functional as F

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    def infoNCE_loss(self, anchor, pos, neg):
        raise NotImplementedError
    
class SteerNet(nn.Module):
    def __init__(self,config, num_users, num_items):
        super(SteerNet, self).__init__()
        assert config['rank'] > 0
        self.adapter_class = config['adapter_class']
        self.epsilon = config['epsilon']
        self.rank = config['rank']
        self.latent_dim = config['latent_dim_rec']
        self.num_steers = config['num_steers']
        self.init_var = config['init_var']
        self.num_users = num_users
        self.num_items = num_items
        self.steer_values = torch.zeros(self.num_steers)
        self.vocab_size = num_items
        if self.adapter_class == 'multiply':
            self.projector1 = nn.Parameter(torch.randn(
                self.num_steers, self.latent_dim, self.rank  
            ) * self.init_var)
            self.projector2 = nn.Parameter(torch.randn(
                self.num_steers, self.latent_dim, self.rank  
            ) * self.init_var)
        elif self.adapter_class == 'add':
            self.add_vec = nn.Parameter(torch.randn(
                self.num_steers, self.latent_dim
            ))
        else:
            raise NotImplementedError
        

    def set_value(self, steer_values):
        self.steer_values = steer_values

    def forward(self, state):
        #[batch, latent_dim]
    
        if self.adapter_class == "multiply":
            delta = state[:, None,None,:].matmul(self.projector1[None])
            delta = delta * self.steer_values[:, :, None, None]
            delta = delta.matmul(
                self.projector2.transpose(1, 2)[None]).sum(1).squeeze()
            projected_state = state + self.epsilon * delta
            return projected_state, delta
        
        elif self.adapter_class == "add":
            add_values = self.steer_values.matmul(self.add_vec)
            projected_state = state + self.epsilon * add_values
            return projected_state, add_values
    
    def regularization_term(self):
        if self.adapter_class == "multiply":
            return self.projector1.pow(2).sum() + self.projector2.pow(2).sum()
        elif self.adapter_class == "add":
            return self.add_vec.pow(2).sum()
        else:
            raise NotImplementedError
        
    def state_dict(self):
        if self.adapter_class == "multiply":
            return {"projector1": self.projector1,
                    "projector2": self.projector2}
        elif self.adapter_class == "add":
            return {"add_vec": self.add_vec}
        else:
            raise NotImplementedError
        
    def load_state_dict(self, state_dict):
        if self.adapter_class == "multiply":
            self.projector1.data = state_dict["projector1"]
            self.projector2.data = state_dict["projector2"]
        elif self.adapter_class == "add":
            self.add_vec.data = state_dict["add_vec"]
        else:
            raise NotImplementedError

class Steer_model(BasicModel):
    def __init__(self,
                 rec_model,
                 config:dict, 
                 dataset:BasicDataset,
                 steer_values):
        super(Steer_model,self).__init__()
        self.rec_model = rec_model
        self.dataset = dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.config = config
        if self.config['steer_train']:
            for _params in self.rec_model.parameters():
                _params.requires_grad = False
        self.init_user_embedding = self.get_parameter_by_name('embedding_user.weight')
        self.init_item_embedding = self.get_parameter_by_name('embedding_item.weight')
        self.steer = SteerNet(config,self.num_users,self.num_items).to(world.device)
        self.steer_values = steer_values
        self.steer.set_value(steer_values)
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.n_layers = self.config['lightGCN_n_layers']
        self.latent_dim = self.config['latent_dim_rec']

    def get_parameter_by_name(self, name):
        state_dict = self.rec_model.state_dict()
        if name in state_dict:
            param = state_dict[name]
            return param
        else:
            raise ValueError(f"Parameter '{name}' not found in the rec_model.")
        

    
    def getEmbedding(self, users, high_items, low_items, neg_items):
        if world.config['model'] =='mf':
            users_emb = self.rec_model.embedding_user(users.long())
            all_items = self.rec_model.embedding_item.weight
        else:
            all_users, all_items,_,_,_,_,_,_ = self.rec_model.computer()
            users_emb = all_users[users]
        if self.config['steer_train']:
            self.steer.set_value(self.steer_values)
            all_items_0,delta_0 = self.steer(all_items)
            steer_values = self.steer_values.clone()
            steer_values[:,0] = steer_values[:,0] * -1
            self.steer.set_value(steer_values)
            all_items_1,delta_1 = self.steer(all_items)
            new_tensor = self.steer_values.clone()
            threshold_tensor = torch.tensor(self.dataset.threshold_item_popularity_label, device=self.steer_values.device)
            new_tensor[:, 0] = torch.tensor(-self.dataset.item_popularity_labels,  device=self.steer_values.device) * threshold_tensor
            self.steer.set_value(new_tensor)
            all_items_2,delta_2 = self.steer(all_items)
            
        high_emb_0 = all_items_0[high_items] #high是+
        low_emb_0 = all_items_0[low_items] #low是-
        high_emb_1 = all_items_1[high_items] #high是-
        low_emb_1 = all_items_1[low_items] #low是+
        high_emb_2 = all_items_2[high_items] #high是-,并且乘以度
        low_emb_2 = all_items_2[low_items] #low是+,并且乘以度
        neg_emb = all_items_2[neg_items] #负样本同样high是-，low是+
        return users_emb, high_emb_0, low_emb_0, high_emb_1, low_emb_1, high_emb_2,low_emb_2, neg_emb

    def bpr_loss(self, users, high,low, neg):
        (users_emb, high_emb_0, low_emb_0, high_emb_1, low_emb_1, high_emb_2,low_emb_2, neg_emb) = self.getEmbedding(users.long(), high.long(),low.long(), neg.long())
        # reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
        #                  posEmb0.norm(2).pow(2)  +
        #                  negEmb0.norm(2).pow(2))/float(len(users))
        #用户尽可能与high popularity相似，与low popularity不相似
        com_high_scores = torch.mul(users_emb, high_emb_0)
        com_high_scores = torch.sum(com_high_scores, dim=1)
        com_low_scores = torch.mul(users_emb, low_emb_0)
        com_low_scores = torch.sum(com_low_scores, dim=1)

        #high和low的信息尽可能的摸平，高的减，低的加
        appro_high_scores = torch.mul(users_emb, high_emb_1)
        appro_high_scores = torch.sum(appro_high_scores, dim=1)
        appro_low_scores = torch.mul(users_emb, low_emb_1)
        appro_low_scores = torch.sum(appro_low_scores, dim=1)

        #high比neg的相似度高，不同的popularity利用信息不一样
        pos_high_scores = torch.mul(users_emb, high_emb_2)
        pos_high_scores = torch.sum(pos_high_scores, dim=1)
        neg_high_scores = torch.mul(users_emb, neg_emb)
        neg_high_scores = torch.sum(neg_high_scores, dim=1)

        #low比neg的相似度高
        pos_low_scores = torch.mul(users_emb, low_emb_2)
        pos_low_scores = torch.sum(pos_low_scores, dim=1)
        neg_low_scores = torch.mul(users_emb, neg_emb)
        neg_low_scores = torch.sum(neg_low_scores, dim=1)

        loss1 = torch.mean(torch.nn.functional.softplus(com_low_scores - com_high_scores))
        # loss2 = torch.mean((appro_low_scores - appro_high_scores).pow(2))
        # loss2 = torch.mean((com_low_scores - com_high_scores).pow(2))
        loss2 = torch.mean((pos_low_scores - pos_high_scores).pow(2))
        loss3 = torch.mean(torch.nn.functional.softplus(neg_high_scores - pos_high_scores))
        loss4 = torch.mean(torch.nn.functional.softplus(neg_low_scores - pos_low_scores))

        loss = loss1 + loss2 + loss3 + loss4
        reg_loss_steer = self.steer.regularization_term()
        print(f"loss: {loss.item()},loss1: {loss1.item()},loss2: {loss2.item()},loss3: {loss3.item()},loss4: {loss4.item()}, reg_loss_steer: {reg_loss_steer.item()}")

        return loss, reg_loss_steer

    def state_dict(self):
        if self.config['steer_train']:
            steer_dict = self.steer.state_dict()
            return steer_dict
        
    def load_state_dict(self, state_dict):
        if self.config['steer_train']:
            self.steer.load_state_dict(state_dict)
        
    def getUsersRating(self, users):
        all_users, all_items,_,_,_,_,_,_ = self.rec_model.computer()
        users_emb = all_users[users.long()]
        items_emb_ego = all_items
        
        self.steer.set_value(self.steer_values)
        threshold_tensor = torch.tensor(self.dataset.threshold_item_popularity_label, device=self.steer_values.device)
        new_tensor = self.steer_values.clone()
        new_tensor[:, 0] = torch.tensor(-self.dataset.item_popularity_labels, device=self.steer_values.device) * threshold_tensor
        self.steer.set_value(new_tensor)
        items_emb,_ = self.steer(all_items)
        
        rating_before = self.f(torch.matmul(users_emb, items_emb_ego.t()))
        
        rating = self.f(torch.matmul(users_emb, items_emb.t()))

        diff = users_emb.unsqueeze(1) - items_emb
        dist = torch.norm(diff, p=2, dim=2) #[user_num, item_num]

        return rating, dist, items_emb


