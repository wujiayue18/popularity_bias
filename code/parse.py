'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=4096,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.0001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='gowalla',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=100)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    parser.add_argument('--adapter_class', type=str, default='multiply', help='adapter class, support [multiply, add]')
    parser.add_argument('--num_steers', type=int, default=1, help='number of steers')
    parser.add_argument('--rank', type=int, default=10)
    parser.add_argument("--eps", type=float, default=10)
    parser.add_argument("--init_var", type=float, default=0.1)
    parser.add_argument("--high_threshold", type=int, default=25)
    parser.add_argument("--low_threshold", type=int, default=25)
    parser.add_argument("--steer", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument('--dummy_steer', type=int, default=0)
    parser.add_argument('--steer_decay', type=float, default=1e-2)
    parser.add_argument('--epsilon', type=float,default=1)
    parser.add_argument('--alpha', type=float,default=1)
    parser.add_argument('--beta', type=float,default=1)
    parser.add_argument('--gamma', type=float,default=1)

    return parser.parse_args()
