import world
import utils
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
#设置steer values

Recmodel = register.MODELS[world.model_name](world.config, dataset).to(world.device)

weight_file_to =  utils.getFileName()

weight_file_base = f"./checkpoints/{world.dataset}/lgn-{world.dataset}-3-64-lgn.pth.tar"
print(f"save to {weight_file_to}")

Neg_k = 1

if world.config['steer_train']:
    # item_popularity_labels = dataset.item_popularity_labels
    item_popularity_labels = [-1] * len(dataset.item_popularity_labels)
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
        if world.config['steer_train']:
            world.cprint(f"loaded base model weights from {weight_file_base}")
            Recmodel.rec_model.load_state_dict(torch.load(weight_file_base,map_location=torch.device('cpu')))
        else:
            world.cprint(f"loaded model weights from {weight_file_base}")
            Recmodel.load_state_dict(torch.load(weight_file_base,map_location=torch.device('cpu'))) 
    except FileNotFoundError:
        print("weight_file not exists, start from beginning")
#不加载，训练steer model
else:
    Recmodel.rec_model.load_state_dict(torch.load(weight_file_base,map_location=torch.device('cpu')))


Recmodel = Recmodel.to(world.device)
print("Trainable parameters in the model:")
for name, param in Recmodel.named_parameters():
    if param.requires_grad:
        print(f"Parameter: {name}, Shape: {param.shape}")

print(Recmodel.state_dict())

for epoch in range(world.TRAIN_epochs):
    start = time.time()
    if epoch % 10 == 0:
        cprint("[TEST]")
        eval_results = Procedure.Test(dataset, Recmodel, epoch, world.config['multicore'])
        logging.info(f"Test results: EPOCH[{epoch+1}/{world.TRAIN_epochs}]{eval_results['recall']}{eval_results['ndcg']}{eval_results['precision']}")
        tail_results = Procedure.Test_tail(dataset, Recmodel, epoch, world.config['multicore'])
        logging.info(f"Tail test results: EPOCH[{epoch+1}/{world.TRAIN_epochs}]{tail_results['recall']}{tail_results['ndcg']}{tail_results['precision']}")        
    output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k)
    if world.config['steer_train']:
        output_information = Procedure.BPR_train_steer(dataset, Recmodel, bpr, epoch, neg_k=Neg_k)
    else:
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
    print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
    logging.info(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
    torch.save(Recmodel.state_dict(), weight_file_to)
