"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
from collections import Counter
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config = world.config,path="../data/gowalla"):
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.high_threshold = config['high_threshold']
        self.low_threshold = config['low_threshold']

        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items)) 
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    # print(l[0])
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze() 
        self.users_D[self.users_D == 0.] = 1 
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze() 
        self.items_D[self.items_D == 0.] = 1. 
        self._allPos = self.getUserPosItems(list(range(self.n_user))) #训练集用户交互物品列表
        self.__testDict = self.__build_test() #测试集字典，dict: {user: [items]}
        self.item_popularity = self._item_cal_popularity() #每一个item对应的popularity，训练集的item popularity
        self.top_items, self.tail_items = self._popularity_tail_items() #根据百分比分
        self.testtailDict = self.__build_tail_test() #尾部测试集字典
        self.highpo_samples, self.lowpo_samples = self._popularity_samples() #物品popularity高和低的item id，根据训练集， 根据定好的阈值分
        self.user_popularity = self._user_cal_popularity() #训练集每一个user交互的高item popularity的数量
        self.item_popularity_labels = self.item_popularity_label() #每一个item对应一个popularity的label，
        self.threshold_item_popularity_label = self.item_popularity_label_threshold()
        self.highItems, self.lowItems = self.getUserhigh_low_Items(list(range(self.n_user)))#每一个用户交互物品的高的和低popularity的item id，列表
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten() 
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat) 
                norm_adj = norm_adj.dot(d_mat) 
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def _popularity_tail_items(self):
        popularity_array = np.array(self.item_popularity)
        # 获取排序后的索引，从大到小
        sorted_indices = np.argsort(-popularity_array)
        # 计算前20%的分界点
        top_20_percent_count = int(len(popularity_array) * 0.2)
        # 划分高流行度和尾部数据
        top_items = sorted_indices[:top_20_percent_count]  # 前20%
        tail_items = sorted_indices[top_20_percent_count:]  # 后80%
        last_top_item_popularity = popularity_array[top_items[-1]]
        first_tail_item_popularity = popularity_array[tail_items[0]]
        if last_top_item_popularity == first_tail_item_popularity:
            same_popularity_indices = np.where(popularity_array == last_top_item_popularity)[0]
            tail_items = np.concatenate((tail_items, same_popularity_indices))
            # 移除 top_items 中的相同流行度的物品
            # top_items = top_items[top_items != same_popularity_indices[-1]]
        return top_items, tail_items

    def __build_tail_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                if item in self.tail_items:
                    test_data[user].append(item)
            else:
                if item in self.tail_items:
                    test_data[user] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getUserhigh_low_Items(self, users):
        highItems = []
        lowItems = []
        for user in users:
            items = self._allPos[user]
            high_items_for_user = np.intersect1d(items, self.highpo_samples) 
            low_items_for_user = np.intersect1d(items, self.lowpo_samples) 
            highItems.append(high_items_for_user)
            lowItems.append(low_items_for_user)

        return highItems, lowItems


    def _item_cal_popularity(self):
        item_popularity = np.array(np.sum(self.UserItemNet, axis=0))
        return item_popularity.reshape((-1,))

    def _popularity_samples(self):
        highpo_samples = np.where(self.item_popularity > self.high_threshold)[0]
        lowpo_samples = np.where(self.item_popularity <= self.low_threshold)[0]
        return highpo_samples, lowpo_samples

    def _user_cal_popularity(self):
        rows = np.arange(self.n_users)[:,None]
        rows = np.repeat(rows,len(self.highpo_samples),axis=1).flatten()
        cols = np.tile(self.highpo_samples, self.n_users) 
        data = np.ones_like(cols)
        user_pop = csr_matrix((data, (rows, cols)), shape=(self.n_users, self.m_items)) #假设每个用户都与流行物品交互一次
        user_pop = user_pop.multiply(self.UserItemNet)
        user_popularity = np.array(np.sum(user_pop, axis=1))
        return user_popularity.reshape((-1,))

    def item_popularity_label(self):
        labels = np.where(np.isin(np.arange(self.m_item), self.highpo_samples), 1, np.where(np.isin(np.arange(self.m_item), self.lowpo_samples), -1, 0))
        return labels
    
    def item_popularity_label_inference(self):
        labels = np.where(np.isin(np.arange(self.m_item), self.lowpo_samples), -1, 0)
        return labels
    
    
    def item_popularity_label_threshold(self):
        adjustments = np.zeros_like(self.items_D, dtype=float)  # 初始化修正值
        # 高流行度物品（大于阈值）
        high_mask = self.items_D > self.high_threshold
        max_value = np.max(self.items_D)
        adjustments[high_mask] = (self.items_D[high_mask] - self.high_threshold) / (max_value - self.high_threshold)  # 归一化处理
        adjustments[high_mask] = np.sqrt(adjustments[high_mask])  # 平方根调整
        # 低流行度物品（小于等于阈值）
        low_mask = self.items_D <= self.low_threshold
        adjustments[low_mask] = (self.low_threshold - self.items_D[low_mask]) / self.low_threshold  # 归一化处理
        adjustments[low_mask] = np.sqrt(adjustments[low_mask]) 
        if world.config['global_adjust'] == 1:
            adjustments = adjustments * world.config['mu']
        else:
            adjustments[low_mask] = adjustments[low_mask] * world.config['mu']
        # adjustments[high_mask] = adjustments[high_mask] * 0
        # 进一步调整高流行度和低流行度的修正幅度
        return adjustments




  