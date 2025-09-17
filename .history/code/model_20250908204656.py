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
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.dataset = dataset
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        #欧氏距离
        # diff = users_emb.unsqueeze(1) - items_emb
        # dist = torch.norm(diff, p=2, dim=2) # [batch_user, all_items]
        #余弦相似度归一化
        # user_emb_norm = torch.nn.functional.normalize(users_emb, p=2, dim=1)  # [num_users, dim]
        # item_emb_norm = torch.nn.functional.normalize(items_emb, p=2, dim=1)  # [num_items, dim]
        # cos_sim_matrix = torch.matmul(user_emb_norm, item_emb_norm.t())
        # dist = 1 - cos_sim_matrix
        #不归一化呢
        dist = 1 - scores
        return self.f(scores), dist, items_emb
    
    # def bpr_loss(self, users, pos, neg):
    #     users_emb = self.embedding_user(users.long())
    #     pos_emb   = self.embedding_item(pos.long())
    #     neg_emb   = self.embedding_item(neg.long())
    #     pos_scores= torch.sum(users_emb*pos_emb, dim=1)
    #     neg_scores= torch.sum(users_emb*neg_emb, dim=1)
    #     loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
    #     reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
    #                       pos_emb.norm(2).pow(2) + 
    #                       neg_emb.norm(2).pow(2))/float(len(users))
    #     return loss, reg_loss

    def bpr_loss(self, users, high_items, low_items, neg_items):
        users_emb = self.embedding_user(users.long())
        all_items = self.embedding_item.weight
        high_items = high_items.long()
        low_items = low_items.long()
        neg_items = neg_items.long()
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
        loss2 = torch.mean((pos_low_scores - pos_high_scores).pow(2))
        loss3 = torch.mean(torch.nn.functional.softplus(neg_high_scores - pos_high_scores))
        loss4 = torch.mean(torch.nn.functional.softplus(neg_low_scores - pos_low_scores))

        loss = loss1 + loss2 + loss3 + loss4
        reg_loss_steer = self.steer.regularization_term()
        print(f"loss: {loss.item()},loss1: {loss1.item()},loss2: {loss2.item()},loss3: {loss3.item()},loss4: {loss4.item()}, reg_loss_steer: {reg_loss_steer.item()}")

        return loss, reg_loss_steer
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

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
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
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
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        #indices,values,size
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob # # Scale the remaining values to account for dropout
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        #layer1
        users_layer1,items_layer1 = torch.split(embs[1], [self.num_users, self.num_items])
        #layer2
        users_layer2,items_layer2 = torch.split(embs[2], [self.num_users, self.num_items])
        #layer3
        users_layer3,items_layer3 = torch.split(embs[3], [self.num_users, self.num_items])
        
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1) #平均完
        users, items = torch.split(light_out, [self.num_users, self.num_items]) #再切分
        
        return users, items, users_layer1, items_layer1, users_layer2, items_layer2, users_layer3, items_layer3
    
    def getUsersRating(self, users):
        all_users, all_items,_,_,_,_,_,_ = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))

        #欧氏距离
        # diff = users_emb.unsqueeze(1) - items_emb
        # dist = torch.norm(diff, p=2, dim=2) # [batch_user, all_items]
        #余弦相似度归一化
        # user_emb_norm = torch.nn.functional.normalize(users_emb, p=2, dim=1)  # [num_users, dim]
        # item_emb_norm = torch.nn.functional.normalize(items_emb, p=2, dim=1)  # [num_items, dim]
        # cos_sim_matrix = torch.matmul(user_emb_norm, item_emb_norm.t())
        # dist = 1 - cos_sim_matrix
        #不归一化呢
        dist = 1 - rating

        return rating, dist, items_emb
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items,_,_,_,_,_,_ = self.computer()
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
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items,_,_,_,_,_,_ = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma


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
        self.poplarity_predict = nn.Linear(self.latent_dim, 1)

    def get_parameter_by_name(self, name):
    # 获取模型的状态字典
        state_dict = self.rec_model.state_dict()
        
        # 从字典中提取对应参数
        if name in state_dict:
            param = state_dict[name]
            # print(param.device)
            return param
        else:
            raise ValueError(f"Parameter '{name}' not found in the rec_model.")
        

    def forward(self, users, items):
        # all_users, all_items,_,_,_,_,_,_ ,_= self.computer_steer()
        print('forward')
        # print("gamma")
        #all_users, all_items = self.computer()
        # users_emb = all_users[users]
        # items_emb = all_items[items]
        # inner_pro = torch.mul(users_emb, items_emb)
        # gamma     = torch.sum(inner_pro, dim=1)
        pass


    
    # def getEmbedding(self, users, high_items, low_items, neg_items):
    #     if world.config['model'] =='mf':
    #         users_emb = self.rec_model.embedding_user(users.long())
    #         all_items = self.rec_model.embedding_item.weight
    #     else:
    #         all_users, all_items,_,_,_,_,_,_ = self.rec_model.computer()
    #         users_emb = all_users[users]
    #     if self.config['steer_train']:
    #         self.steer.set_value(self.steer_values)
    #         all_items_0,delta_0 = self.steer(all_items)
    #         steer_values = self.steer_values.clone()
    #         steer_values[:,0] = steer_values[:,0] * -1
    #         self.steer.set_value(steer_values)
    #         all_items_1,delta_1 = self.steer(all_items)
    #         new_tensor = self.steer_values.clone()
    #         threshold_tensor = torch.tensor(self.dataset.threshold_item_popularity_label, device=self.steer_values.device)
    #         new_tensor[:, 0] = torch.tensor(-self.dataset.item_popularity_labels,  device=self.steer_values.device) * threshold_tensor
    #         self.steer.set_value(new_tensor)
    #         all_items_2,delta_2 = self.steer(all_items)
            
    #     high_emb_0 = all_items_0[high_items] #high是+
    #     low_emb_0 = all_items_0[low_items] #low是-
    #     high_emb_1 = all_items_1[high_items] #high是-
    #     low_emb_1 = all_items_1[low_items] #low是+
    #     high_emb_2 = all_items_2[high_items] #high是-,并且乘以度
    #     low_emb_2 = all_items_2[low_items] #low是+,并且乘以度
    #     neg_emb = all_items_2[neg_items] #负样本同样high是-，low是+
    #     return users_emb, high_emb_0, low_emb_0, high_emb_1, low_emb_1, high_emb_2,low_emb_2, neg_emb
    def getEmbedding(self, users, pos_items, neg_items):
        if world.config['model'] =='mf':
            users_emb = self.rec_model.embedding_user(users.long())
            all_items = self.rec_model.embedding_item.weight
        else:
            all_users, all_items,_,_,_,_,_,_ = self.rec_model.computer()
            users_emb = all_users[users]
        if self.config['steer_train']:
            self.steer.set_value(self.steer_values)
            all_items_steer,steer_delta = self.steer(all_items)
            pos_items_steer = all_items_steer[pos_items]
            neg_items_steer = all_items_steer[neg_items]
        return users_emb, pos_items_steer, neg_items_steer, steer_delta



    # def bpr_loss(self, users, high,low, neg):
    #     (users_emb, high_emb_0, low_emb_0, high_emb_1, low_emb_1, high_emb_2,low_emb_2, neg_emb) = self.getEmbedding(users.long(), high.long(),low.long(), neg.long())
    #     # reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
    #     #                  posEmb0.norm(2).pow(2)  +
    #     #                  negEmb0.norm(2).pow(2))/float(len(users))
    #     #用户尽可能与high popularity相似，与low popularity不相似
    #     com_high_scores = torch.mul(users_emb, high_emb_0)
    #     com_high_scores = torch.sum(com_high_scores, dim=1)
    #     com_low_scores = torch.mul(users_emb, low_emb_0)
    #     com_low_scores = torch.sum(com_low_scores, dim=1)

    #     #high和low的信息尽可能的摸平，高的减，低的加
    #     appro_high_scores = torch.mul(users_emb, high_emb_1)
    #     appro_high_scores = torch.sum(appro_high_scores, dim=1)
    #     appro_low_scores = torch.mul(users_emb, low_emb_1)
    #     appro_low_scores = torch.sum(appro_low_scores, dim=1)

    #     #high比neg的相似度高，不同的popularity利用信息不一样
    #     pos_high_scores = torch.mul(users_emb, high_emb_2)
    #     pos_high_scores = torch.sum(pos_high_scores, dim=1)
    #     neg_high_scores = torch.mul(users_emb, neg_emb)
    #     neg_high_scores = torch.sum(neg_high_scores, dim=1)

    #     #low比neg的相似度高
    #     pos_low_scores = torch.mul(users_emb, low_emb_2)
    #     pos_low_scores = torch.sum(pos_low_scores, dim=1)
    #     neg_low_scores = torch.mul(users_emb, neg_emb)
    #     neg_low_scores = torch.sum(neg_low_scores, dim=1)

    #     loss1 = torch.mean(torch.nn.functional.softplus(com_low_scores - com_high_scores))
    #     # loss2 = torch.mean((appro_low_scores - appro_high_scores).pow(2))
    #     # loss2 = torch.mean((com_low_scores - com_high_scores).pow(2))
    #     loss2 = torch.mean((pos_low_scores - pos_high_scores).pow(2))
    #     loss3 = torch.mean(torch.nn.functional.softplus(neg_high_scores - pos_high_scores))
    #     loss4 = torch.mean(torch.nn.functional.softplus(neg_low_scores - pos_low_scores))

    #     loss = loss1 + loss2 + loss3 + loss4
    #     reg_loss_steer = self.steer.regularization_term()
    #     print(f"loss: {loss.item()},loss1: {loss1.item()},loss2: {loss2.item()},loss3: {loss3.item()},loss4: {loss4.item()}, reg_loss_steer: {reg_loss_steer.item()}")

    #     return loss, reg_loss_steer

    def state_dict(self):
        if self.config['steer_train']:
            steer_dict = self.steer.state_dict()
            # steer_dict = steer_dict.update(self.poplarity_predict.state_dict())
            return steer_dict
        
    def load_state_dict(self, state_dict):
        if self.config['steer_train']:
            self.steer.load_state_dict(state_dict)
            # self.poplarity_predict.load_state_dict(state_dict)
        
    def bpr_loss(self, users, pos_items, neg_items):
        users_emb, pos_emb, neg_emb, steer_delta = self.getEmbedding(users.long(), pos_items.long(), neg_items.long())
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        popularity = self.poplarity_predict(steer_delta).squeeze().float()
        popularity_label = torch.tensor(self.dataset.item_popularity).to(self.steer_values.device).float()
        loss_predict = F.mse_loss(popularity, popularity_label)
        loss_reg_steer = self.steer.regularization_term()
        
        return loss, reg_loss, loss_reg_steer, loss_predict

  

    def getUsersRating(self, users):
        if self.config['model'] == 'mf':
            users = users.long()
            users_emb = self.rec_model.embedding_user(users)
            all_items = self.rec_model.embedding_item.weight
            items_emb_ego = all_items
        else:
            all_users, all_items,_,_,_,_,_,_ = self.rec_model.computer()
            users_emb = all_users[users.long()]
            items_emb_ego = all_items
        
        # self.steer.set_value(self.steer_values)
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


