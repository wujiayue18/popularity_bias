import world
import utils
from utils import plot_count_popularity_barh
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
from datetime import datetime
from model import Steer_model
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
import logging
log_file = f'log-{datetime.now().strftime("%Y%m%d%H%M%S")}.txt'
logging.basicConfig(filename=join(world.LOG_PATH,world.dataset, log_file),
                    filemode='w', 
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - [%(pathname)s:%(lineno)d] - %(levelname)s - %(message)s')
logging.info(f"{world.config}")


Recmodel = register.MODELS[world.model_name](world.config, dataset).to(world.device)
rec_path =  utils.getFileName()
weight_file_base = f"./checkpoints/{world.dataset}/lgn-{world.dataset}-3-64-lgn.pth.tar"
Neg_k = 1

if world.config['steer']:
    item_popularity_labels = dataset.item_popularity_labels
    steer_values = torch.Tensor(item_popularity_labels)[:,None]
    if world.config['dummy_steer']:
        steer_values = torch.cat([steer_values, torch.ones_like(steer_values[:,0])[:,None]],1)
    steer_values = steer_values.to(world.device)
    Recmodel = Steer_model(Recmodel, world.config, dataset, steer_values)
    bpr = utils.BPR2Loss(Recmodel, world.config)
else:
    bpr = utils.BPRLoss(Recmodel, world.config)

if world.LOAD:
    try:
        if world.config['steer']:
            world.cprint(f"loaded base model weights from {weight_file_base}")
            Recmodel.rec_model.load_state_dict(torch.load(weight_file_base,map_location=torch.device('cpu')))
            world.cprint(f"load from {rec_path}")
            Recmodel.load_state_dict(torch.load(rec_path,map_location=torch.device('cpu')))
        else:
            world.cprint(f"loaded model weights from {weight_file_base}")
            Recmodel.load_state_dict(torch.load(weight_file_base,map_location=torch.device('cpu'))) 
    except FileNotFoundError:
        print("weight_file not exists, start from beginning")
else:
    if world.config['steer']:
        world.cprint(f"save to {rec_path}")
        Recmodel.rec_model.load_state_dict(torch.load(weight_file_base,map_location=torch.device('cpu')))


Recmodel = Recmodel.to(world.device)

if world.LOAD:
    Recmodel.eval()
    if world.config['steer'] :
        text = "STEER"
    else:
        text = "LGN"
    eval_results = Procedure.Test(dataset, Recmodel, 0, world.config['multicore'])
    print(f"{text} Test results:{eval_results['recall']}{eval_results['ndcg']}{eval_results['precision']}")
    tail_results = Procedure.Test(dataset, Recmodel, 1, world.config['multicore'])
    print(f"{text}Tail test results: {tail_results['recall']}{tail_results['ndcg']}{tail_results['precision']}") 
    rating_items = eval_results['rating_items'] #每个用户预测top-k物品序列
    rating_popularity = eval_results['rating_popularity'] #每个用户预测top-k物品的流行度
    rating_count = eval_results['rating_count'].values()       
    plot_count_popularity_barh(rating_popularity, rating_count, text)
else:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if world.config['steer']:
            output_information = Procedure.BPR_train_steer(dataset, Recmodel, bpr, epoch, neg_k=Neg_k)
        else:
            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k)
        cprint("[TEST]")
        logging.info(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), rec_path)
