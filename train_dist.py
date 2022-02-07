#!/usr/bin/env python3
'''
This code has been derived from: Lornatang Liu Changyu
https://github.com/Lornatang/SRGAN-PyTorch
which  op-for-op PyTorch reimplementation of Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.
Source of original paper results: https://arxiv.org/pdf/1609.04802v5.pdf


Runs on CPU or GPU - see hpar.yaml configuration (aka design)

tested w/ Pytorch:
shifter --image=nersc/pytorch:ngc-21.08-v2 bash

Runs  1 GPU  interactively on PM:
ssh pm
export MASTER_ADDR=`hostname`
export SLURM_NTASKS=1
export SLURM_PROCID=0
export SLURM_LOCALID=0
time shifter  --image=nersc/pytorch:ngc-21.08-v2 ./train_dist.py  --design dev0 --facility perlmutter  --expName exp07
>>> real	4m22.124s


Run on 4 A100 on PM:
salloc  -C gpu -q interactive  -t4:00:00  --gpus-per-task=1 --image=nersc/pytorch:ngc-21.08-v2 -A m3363_g --ntasks-per-node=4   -N 1  

Quick test:
salloc -N1
 export MASTER_ADDR=`hostname`  
srun -n 1 shifter --image=nersc/pytorch:ngc-21.08-v2  ./train_dist.py   --design dev0  --facility perlmutter  --expName exp2

On Summit: salloc, as corigpu, use facility=summitlogin

Production job 
srun -n 16 shifter --image=nersc/pytorch:ngc-21.08-v2  ./train_dist.py   --design dev7a  --facility perlmutter  --expName exp5a

Display TB
ssh cori-tb
cd  ~/prje/tmp_NyxHydro4kF
 module load pytorch
 tensorboard --port 9800 --logdir=exp3

ssh summit-tb
cd /gpfs/alpine/world-shared/ast153/balewski/tmp_NyxHydro4kF/
module load open-ce/1.1.3-py38-0
 tensorboard  --port 9700 --logdir=1645832

PM -N8:
INFO - T:rank 0 of 32, data loaders initialized
INFO - T:train-data: 5 steps, localBS=16, globalBS=512
INFO - T:valid-data: 2 steps


Summit -N10
INFO - T:rank 0 of 60, data loaders initialized
INFO - T:train-data: 7 steps, localBS=6, globalBS=360
INFO - T:valid-data: 3 steps
 

'''

import sys,os
from toolbox.Util_IOfunc import read_yaml, write_yaml
from toolbox.Trainer import Trainer

import logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
import time
import torch
import torch.distributed as dist
from pprint import pprint
import socket  # for worker name

import argparse
#...!...!..................
def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--design", default='dev0', help='[.hpar.yaml] configuration of model and training')

  parser.add_argument("--dataName",default="dm_density_4096",help="[.h5] name data  file")
  parser.add_argument("--basePath", default=None, help=' all outputs+TB+snapshots, default in hpar.yaml')

  parser.add_argument("--facility", default='corigpu', choices=['corigpu','summit','summitlogin','perlmutter','crusher'],help='computing facility where code is executed')  
  parser.add_argument("--expName", default='exp03', help="output main dir, train_summary stored there")
  parser.add_argument("-v","--verbosity",type=int,choices=[0,1,2,3], help="increase output verbosity", default=1, dest='verb')

  parser.add_argument("--epochs",default=None, type=int, help="(optional), replaces max_epochs from hpar")
  parser.add_argument("-n", "--numSamp", type=int, default=None, help="(optional) cut off num samples per epoch")
  parser.add_argument("--LRfactor", type=float, default=None, help="(optional) multiplier for initLR for G and D")

  args = parser.parse_args()
  return args

  
#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__ == '__main__':
    args=get_parser()
    if args.verb>2: # extreme debugging
      for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    os.environ['MASTER_PORT'] = "8886"
    
    params ={}
    if args.facility=='summit':
      import subprocess
      get_master = "echo $(cat {} | sort | uniq | grep -v batch | grep -v login | head -1)".format(os.environ['LSB_DJOB_HOSTFILE'])
      os.environ['MASTER_ADDR'] = str(subprocess.check_output(get_master, shell=True))[2:-3]
      os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
      os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
      params['local_rank'] = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    else:
      #os.environ['MASTER_ADDR'] = os.environ['SLURM_LAUNCH_NODE_IPADDR']
      os.environ['RANK'] = os.environ['SLURM_PROCID']
      os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
      params['local_rank'] = int(os.environ['SLURM_LOCALID'])

    params['master_name']=os.environ['MASTER_ADDR']
    params['world_size'] = int(os.environ['WORLD_SIZE'])
    params['world_rank'] =int(os.environ['RANK'])
    params['worker_name']=socket.gethostname()
    params['facility']=args.facility
    if 'crusher' in params['master_name']:
      params['local_rank'] =0

    
    print('M:params',params)
    if params['world_rank']==0:
        print('M:python:',sys.version,'torch:',torch.__version__)
    if params['world_size'] > 1:  # multi-GPU training
      torch.cuda.set_device(params['local_rank'])
      dist.init_process_group(backend='nccl', init_method='env://')
      assert params['world_rank'] == dist.get_rank()
      print('M:locRank:',params['local_rank'],'rndSeed=',torch.seed())
    params['verb'] =args.verb * (params['world_rank'] == 0)
    
    if params['verb']:
      logging.info('M:MASTER_ADDR=%s WORLD_SIZE=%s RANK=%s  pytorch:%s'%(os.environ['MASTER_ADDR'] ,os.environ['WORLD_SIZE'], os.environ['RANK'],torch.__version__ ))
      for arg in vars(args):  logging.info('M:arg %s:%s'%(arg, str(getattr(args, arg))))

    blob=read_yaml( args.design+'.hpar.yaml',verb=params['verb'], logger=True)
    facCf=blob.pop('facility_conf')[args.facility]
    blob.pop('Defaults')
    params.update(blob)
    params['design']=args.design
    
    #print('M:params');pprint(params)#tmp
    #... propagate facility dependent config
    
    for x in ["D_LR","G_LR"]: 
        params['train_conf'][x]=facCf[x]

    #params['model_conf']['D']['fc_layers']=facCf["D_num_fc_layer"]

    # refine BS for multi-gpu configuration
    tmp_batch_size=facCf['batch_size']
    if params['const_local_batch']: # faster but LR changes w/ num GPUs
      params['local_batch_size'] =tmp_batch_size 
      params['global_batch_size'] =tmp_batch_size*params['world_size']
    else:
      params['local_batch_size'] = int(tmp_batch_size//params['world_size'])
      params['global_batch_size'] = tmp_batch_size

    
    # capture other args values
    params['h5_path']=facCf['data_path']
    params['h5_name']=args.dataName+'.h5'
    params['exp_name']=args.expName

    if args.basePath==None:
      args.basePath=facCf['base_path']

    params['exp_path']=os.path.join(args.basePath,args.expName)

    #.... update selected params based on runtime config
    if args.numSamp!=None:  # reduce num steps/epoch - code testing
        params['max_glob_samples_per_epoch']=args.numSamp
    if args.epochs!=None:
        params['train_conf']['adv_epochs']= args.epochs
    if args.LRfactor!=None:
        for  x in ["D_LR"]: #,"G_LR"]: 
          params['train_conf'][x]['init']*= args.LRfactor
       
    trainer = Trainer(params)
    
    trainer.train()
                
    if params['world_rank'] == 0:
      sumF= params['exp_path']+'/sum_train.yaml'
      write_yaml(trainer.sumRec, sumF) # to be able to predict while training continus

      print("M:done rank=",params['world_rank'])
