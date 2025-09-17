'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer, PCA_analyse_emb
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
from collections import Counter


CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"

def BPR_train_steer(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPR2Loss = loss_class
    with timer(name="Sample"):
        S = utils.UniformSample_original_python_popularity(dataset)
    users = torch.Tensor(S[:, 0]).long()
    highItems = torch.Tensor(S[:, 1]).long()
    lowItems = torch.Tensor(S[:, 2]).long()
    negItems = torch.Tensor(S[:, 3]).long()

    users = users.to(world.device)
    highItems = highItems.to(world.device)
    lowItems = lowItems.to(world.device)
    negItems = negItems.to(world.device)

    users, highItems, lowItems, negItems = utils.shuffle(users, highItems, lowItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_high,
          batch_low,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   highItems,
                                                   lowItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_high, batch_low, batch_neg)
        aver_loss += cri
        # if world.tensorboard:
        #     w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
# 输入[(rating_list, ground_truth),...,()]
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items) # 预测出的top-k的每个物品是否在ground_truth当中，长度为预测序列的长度，即K
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict #dict: {user: [items]}
    testtailDict: dict = dataset.testtailDict #dict: {user: [items]}
    if world.config['steer_train']:
        Recmodel: model.Steer_model
    else:
        Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        rating_index_list = []
        rating_number_list = []
        groundTrue_list = []
        distance_index_list = []
        distance_list = []
        disting_list = []
        rate_distance_list = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
            rating, dist, item_emb = Recmodel.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10) #[batch, item_num]排除所有已经交互的物品，将它们的值设为负无穷。
            dist[exclude_index, exclude_items] =  float('inf')
            _, rating_K = torch.topk(rating, k=max_K) #rating_K: [batch, max_K]#分数最高的前k个item
            _, dist_K = torch.topk(dist, k=max_K, largest=False,dim=1) #距离最近的前k个item，dist_K: [batch, max_K]
            rating_number_K = torch.gather(rating, dim=1, index=rating_K)
            rating_number_K = rating_number_K.cpu().numpy().tolist()
            rating = rating.cpu().numpy()
            distance_K = torch.gather(dist, dim=1, index=dist_K)
            distance_K = distance_K.cpu().numpy().tolist()
            rate_distance_K = torch.gather(dist, dim=1, index=rating_K)
            rate_distance_K = rate_distance_K.cpu().numpy().tolist()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            disting_list.append(dist_K.cpu())
            rating_index_list.append(rating_K.cpu().numpy().tolist())
            groundTrue_list.append(groundTrue)
            distance_index_list.append(dist_K.cpu().numpy().tolist())
            distance_list.append(distance_K)
            rating_number_list.append(rating_number_K)
            rate_distance_list.append(rate_distance_K)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list) #[(rating_K, groundTrue), ...]
        # X = zip(disting_list, groundTrue_list) 
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        distance_index_list = [item for sublist in distance_index_list for item in sublist]
        rating_index_list = [item for sublist in rating_index_list for item in sublist]
        results['distance_index_list'] = distance_index_list # [user, 50]每个用户距离最近的k个item的索引
        results['rating_index_list'] = rating_index_list # [user, 50]每个用户评分最高的k个item的索引
        distance_list = [item for sublist in distance_list for item in sublist]
        rating_number_list = [item for sublist in rating_number_list for item in sublist]
        rate_distance_list = [item for sublist in rate_distance_list for item in sublist]
        results['distance_list'] = distance_list # [user, 50]每个用户距离最近的k个item的距离
        results['rating_number_list'] = rating_number_list # [user, 50]每个用户距离最近的k个item的评分
        results['rate_distance_list'] = rate_distance_list # [user, 50]每个用户评分最高的k个item的距离
        #popularity
        ground_truth_items_array = [item for sublist1 in groundTrue_list for sublist2 in sublist1 for item in sublist2]
        ground_truth_items = set(ground_truth_items_array) #所有测试集用户真实交互物品的集合
        # gt_popularity = np.array([dataset.item_popularity[item] for item in ground_truth_items])
        
        rating_items_array = [item for sublist1 in rating_list for sublist2 in sublist1 for item in sublist2]
        rating_items_array = [t.item() for t in rating_items_array]
        rating_items = set(rating_items_array) #所有测试集用户rating预测交互物品的集合
        # rating_popularity = np.array([dataset.item_popularity[item] for item in rating_items])
        results['rating_items'] = rating_items #模型预测物品集合
        results['ground_truth_items'] = ground_truth_items #测试集用户真实交互物品集合
        
        results['rating_items_array'] = rating_items_array 
        rating_count = Counter(rating_items_array) #模型预测物品集合中每个物品出现的次数,{item:count}
        tuple_list_rating = list(rating_count.items()) #[(item, count), ...]
        # 根据键的索引排序
        rating_count = sorted(tuple_list_rating, key=lambda x: x[0])
        # 将排序后的元组列表转换回字典
        rating_count = dict(rating_count) #模型预测每个物品出现的次数：{item:count}

        ground_truth_count = Counter(ground_truth_items_array)
        tuple_list_gt = list(ground_truth_count.items())
        gt_count = sorted(tuple_list_gt, key=lambda x: x[0])
        gt_count = dict(gt_count)
        gt_popularity = np.array([dataset.item_popularity[item] for item in gt_count.keys()]) #真实{item: popularity}
        rating_popularity = np.array([dataset.item_popularity[item] for item in rating_count.keys()]) #预测{item: popularity}
        results['rating_count'] = rating_count
        results['gt_count'] = gt_count
        results['gt_popularity'] = gt_popularity
        results['rating_popularity'] = rating_popularity
        results['rating_popularity_mean'] = sum(rating_popularity) / len(rating_popularity)
        results['item_emb'] = item_emb
        if multicore == 1:
            pool.close()
        return results

           
def Test_tail(dataset, Recmodel, epoch, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testtailDict: dict = dataset.testtailDict #dict: {user: [items]}
    if world.config['steer_train']:
        Recmodel: model.Steer_model
    else:
        Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testtailDict.keys())
        # print(len(users), len(users_tail))
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testtailDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
            
            rating, dist, item_emb = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10) #[batch, item_num]排除所有已经交互的物品，将它们的值设为负无穷。
            _, rating_K = torch.topk(rating, k=max_K) #rating_K: [batch, max_K]
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list) #[(rating_K, groundTrue), ...]
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        ground_truth_items_array = [item for sublist1 in groundTrue_list for sublist2 in sublist1 for item in sublist2]
        ground_truth_items = set(ground_truth_items_array)
        rating_items_array = [item for sublist1 in rating_list for sublist2 in sublist1 for item in sublist2]
        rating_items_array = [t.item() for t in rating_items_array]
        rating_items = set(rating_items_array)
        results['rating_items'] = rating_items #模型预测物品集合
        results['ground_truth_items'] = ground_truth_items #测试集用户真实交互物品集合
        results['rating_items_array'] = rating_items_array 
        rating_count = Counter(rating_items_array) #模型预测物品集合中每个物品出现的次数,{item:count}
        tuple_list_rating = list(rating_count.items()) #[(item, count), ...]
        # 根据键的索引排序
        rating_count = sorted(tuple_list_rating, key=lambda x: x[0])
        # 将排序后的元组列表转换回字典
        rating_count = dict(rating_count) #模型预测每个物品出现的次数：{item:count}
        ground_truth_count = Counter(ground_truth_items_array)
        tuple_list_gt = list(ground_truth_count.items())
        gt_count = sorted(tuple_list_gt, key=lambda x: x[0])
        gt_count = dict(gt_count)
        gt_popularity = np.array([dataset.item_popularity[item] for item in gt_count.keys()]) #真实{item: popularity}
        rating_popularity = np.array([dataset.item_popularity[item] for item in rating_count.keys()]) #预测{item: popularity}
        results['rating_count'] = rating_count
        results['gt_count'] = gt_count
        results['gt_popularity'] = gt_popularity
        results['rating_popularity'] = rating_popularity
        results['rating_popularity_mean'] = sum(rating_popularity) / len(rating_popularity)
        results['item_emb'] = item_emb
        if multicore == 1:
            pool.close()
        return results
