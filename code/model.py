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
    
class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()
        

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

        # print("save_txt")

    

    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        g_droped = self.Graph    
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1) #平均完
        users, items = torch.split(light_out, [self.num_users, self.num_items]) #再切分
        
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))

        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
class SteerNet(nn.Module):
    def __init__(self,config, num_users, num_items):
        super(SteerNet, self).__init__()
        assert config['rank'] > 0
        self.adapter_class = config['adapter_class']
        self.eps = config['eps']
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
        if self.adapter_class == "multiply":
            delta = state[:, None,None,:].matmul(self.projector1[None])
            delta = delta * self.steer_values[:, :, None, None]
            delta = delta.matmul(
                self.projector2.transpose(1, 2)[None]).sum(1).squeeze()
            projected_state = state + self.eps * delta
            return projected_state, delta
        
        elif self.adapter_class == "add":
            add_values = self.steer_values.matmul(self.add_vec)
            projected_state = state + self.eps * add_values
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
        if self.config['steer']:
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
        all_users, all_items = self.rec_model.computer()
        users_emb = all_users[users]
        if self.config['steer']:
            self.steer.set_value(self.steer_values)
            all_items_0,_ = self.steer(all_items)
            new_tensor = self.steer_values.clone()
            threshold_tensor = torch.tensor(self.dataset.threshold_item_popularity_label, device=self.steer_values.device)
            new_tensor[:, 0] = torch.tensor(-self.dataset.item_popularity_labels,  device=self.steer_values.device) * threshold_tensor
            self.steer.set_value(new_tensor)
            all_items_1,_ = self.steer(all_items)
            
        high_emb_0 = all_items_0[high_items]
        low_emb_0 = all_items_0[low_items] 
        high_emb_1 = all_items_1[high_items]
        low_emb_1 = all_items_1[low_items]
        neg_emb = all_items_1[neg_items] 
        return users_emb, high_emb_0, low_emb_0, high_emb_1, low_emb_1, neg_emb

    def bpr_loss(self, users, high,low, neg):
        (users_emb, high_emb_0, low_emb_0, high_emb_1, low_emb_1, neg_emb) = self.getEmbedding(users.long(), high.long(),low.long(), neg.long())
       
        com_high_scores = torch.mul(users_emb, high_emb_0)
        com_high_scores = torch.sum(com_high_scores, dim=1)
        com_low_scores = torch.mul(users_emb, low_emb_0)
        com_low_scores = torch.sum(com_low_scores, dim=1)

       
        pos_high_scores = torch.mul(users_emb, high_emb_1)
        pos_high_scores = torch.sum(pos_high_scores, dim=1)
        neg_high_scores = torch.mul(users_emb, neg_emb)
        neg_high_scores = torch.sum(neg_high_scores, dim=1)


        pos_low_scores = torch.mul(users_emb, low_emb_1)
        pos_low_scores = torch.sum(pos_low_scores, dim=1)
        neg_low_scores = torch.mul(users_emb, neg_emb)
        neg_low_scores = torch.sum(neg_low_scores, dim=1)

        loss1 = torch.mean(torch.nn.functional.softplus(com_low_scores - com_high_scores))
        loss2 = torch.mean((pos_low_scores - pos_high_scores).pow(2))
        loss3 = torch.mean(torch.nn.functional.softplus(neg_high_scores - pos_high_scores))
        loss4 = torch.mean(torch.nn.functional.softplus(neg_low_scores - pos_low_scores))

        loss = world.config['alpha'] * loss1 + world.config['beta'] * loss2 +  world.config['gamma'] * loss3 + world.config['gamma'] * loss4
        reg_loss_steer = self.steer.regularization_term()
        return loss, reg_loss_steer

    def state_dict(self):
        if self.config['steer']:
            steer_dict = self.steer.state_dict()
            return steer_dict
        
    def load_state_dict(self, state_dict):
        if self.config['steer']:
            self.steer.load_state_dict(state_dict)
        
    def getUsersRating(self, users):
        all_users, all_items = self.rec_model.computer()
        users_emb = all_users[users.long()]
        self.steer.set_value(self.steer_values)
        threshold_tensor = torch.tensor(self.dataset.threshold_item_popularity_label, device=self.steer_values.device)
        new_tensor = self.steer_values.clone()
        new_tensor[:, 0] = torch.tensor(-self.dataset.item_popularity_labels, device=self.steer_values.device) * threshold_tensor
        self.steer.set_value(new_tensor)
        items_emb,_ = self.steer(all_items)
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating


