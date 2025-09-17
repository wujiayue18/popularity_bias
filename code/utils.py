'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
from model import LightGCN
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import random
import os
from datetime import datetime
import register
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from register import dataset
import pandas as pd
from model import Steer_model
import torch.backends.cudnn as cudnn
import json
from datetime import date


class BPRLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
        self.config = config

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


class BPR2Loss:
    def __init__(self,
                 Steer_rec_model : PairWiseModel,
                 config : dict):
        self.model = Steer_rec_model
        self.weight_decay = config['decay']
        self.steer_decay = config['steer_decay']
        self.lr = config['lr']
        self.opt = optim.Adam(Steer_rec_model.parameters(), lr=self.lr)

    def stageOne(self, users, high,low, neg):
        loss, reg_loss_steer = self.model.bpr_loss(users, high, low, neg)
        reg_loss_steer = reg_loss_steer*self.steer_decay
        loss = loss + reg_loss_steer
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()
    

def UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    S = UniformSample_original_python(dataset)
    return S

def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)


def UniformSample_original_python_popularity(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    
    allPos = dataset.allPos
    highItems = dataset.highItems
    lowItems = dataset.lowItems

    user_num = sum(len(sublist) for sublist in highItems)
    # print(user_num)
    users = np.random.randint(0, dataset.n_users, user_num)
    
    S = []
    
    for i, user in enumerate(users):
        posForUser = allPos[user]
        highForUser = highItems[user]
        lowForUser = lowItems[user]
        if len(highForUser) == 0 or len(lowForUser) == 0:
            continue
        
        highindex = np.random.randint(0, len(highForUser))
        highitem = highForUser[highindex]
        lowindex = np.random.randint(0, len(lowForUser))
        lowitem = lowForUser[lowindex]      
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break      
        S.append([user, highitem, lowitem, negitem])
        
        
    total = time() - total_start
    return np.array(S)


# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName():
    if world.config['steer'] == 1:
        file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-steer-epsilon={world.config['epsilon']}-alpha={world.config['alpha']}-beta={world.config['beta']}-gamma={world.config['gamma']}.pth.tar"
    else:
        file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-lgn.pth.tar"
    return os.path.join(world.FILE_PATH,world.dataset,file)

def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def load_model(weight_file):
    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    Recmodel = Recmodel.to(world.device)
    
    world.cprint(f"loaded model weights from {weight_file}")
    Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
    return Recmodel


def load_steer_model(rec_model, weight_file, dataset):
    item_popularity_labels = dataset.item_popularity_labels
    steer_values = torch.Tensor(item_popularity_labels)[:,None].to(world.device)
    Recmodel = Steer_model(Recmodel, world.config, dataset, steer_values)
    Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
    Recmodel = Recmodel.to(world.device)
    return Recmodel



def plot_count_popularity_barh(popularity, counts, text):
    data = pd.DataFrame({
        'rating_popularity': popularity,
        'rating_count': list(counts)
    })

    plt.figure(figsize=(10, 6))

    bars = plt.bar(data['rating_popularity'], data['rating_count'], 
                color=plt.cm.viridis(np.linspace(0, 1, len(data))), 
                alpha=0.7)  

    plt.xlabel('Item Popularity Range', fontsize=16)
    plt.ylabel('Top-$k$ Recommendation Count', fontsize=16)
    plt.xticks(fontsize=14)
    plt.xticks(fontsize=14)

    plt.savefig(f"../imgs/{world.dataset}/{text}_items_popularity_vs_frequency.pdf", format='pdf')
    plt.show()
    

class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1) #每个用户推荐列表（前k个）中预测正确的物品数
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))]) #每个用户真实列表中物品数
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        #预测的每个值是否在groundtruth当中
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def setup_train():
    
    export_root = create_experiment_export_folder()
    export_experiments_config_as_json(export_root)

    return export_root


def create_experiment_export_folder():
    experiment_dir, experiment_description = world.config['experiment_dir'], world.config['experiment_description']
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    data_dir = os.path.join(experiment_dir,world.config['dataset'])
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    experiment_path = get_name_of_experiment_path(data_dir, experiment_description)
    os.mkdir(experiment_path)
    print('Folder created: ' + os.path.abspath(experiment_path))
    return experiment_path


def get_name_of_experiment_path(data_dir, experiment_description):
    experiment_path = os.path.join(data_dir, (experiment_description + "_" + str(date.today())))
    idx = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + "_" + str(idx)
    return experiment_path



def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + "_" + str(idx)):
        idx += 1
    return idx


def export_experiments_config_as_json(experiment_path):
    with open(os.path.join(experiment_path, 'config.json'), 'w') as outfile:
        json.dump(world.config, outfile, indent=2)

# ====================end Metrics=============================
# =========================================================
