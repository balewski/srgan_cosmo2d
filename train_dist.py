#!/usr/bin/env python
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
 shifter  --image=nersc/pytorch:ngc-21.08-v2 ./train_dist.py  --design dev0 --facility perlmutter  --jobId exp07


Run on 1 GPUs on 1 node w/ salloc
 salloc -N1 -C gpu  -c 10 --gpus-per-task=1 --image=nersc/pytorch:ngc-21.08-v2  -t4:00:00 --ntasks-per-node=1 
 export MASTER_ADDR=`hostname`
 srun -n2 shifter ./train_dist.py  --design dev0  --facility perlmutter

Run on 4 A100 on PM:
salloc  -C gpu -q interactive  -t4:00:00  --gpus-per-task=1 --image=nersc/pytorch:ngc-21.08-v2 -A m3363_g --ntasks-per-node=4   -N 1  


Quick test:
srun -n 1 -l ./train_dist.py  -n4

Production job ??
srun -n 2 -l ./train_dist.py --dataName 2021_05-Yueying-disp_17c --design supRes2 

Display TB
ssh cori-tb
cd  ~/prje/tmp_NyxHydro4kB/manual
 module load pytorch
 tensorboard  --port 9800 --logdir=exp03


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

import argparse
#...!...!..................
def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--design", default='dev0', help='[.hpar.yaml] configuration of model and training')

  parser.add_argument("--dataName",default="dm_density_4096",help="[.cpair.h5] name data  file")
  parser.add_argument("--outPath", default='*/manual', help=' all outputs+TB+snapshots, optional "*/" uses base-dir+job_id from hpar.yaml')

  parser.add_argument("--facility", default='corigpu', choices=['corigpu','summit','perlmutter'],help='computing facility where code is executed')  
  parser.add_argument("-j","--jobId", default='exp03', help="optional, aux info to be stored w/ summary")
  parser.add_argument("-n", "--numSamp", type=int, default=None, help="(optional) cut off num samples per epoch")
  parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')

  parser.add_argument("--epochs",default=None, type=int, help="if defined, replaces max_epochs from hpar")

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
    #print('M:facility:',args.facility)
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

    params['world_size'] = int(os.environ['WORLD_SIZE'])
    params['world_rank'] = 0
    #print('M: ws=', params['world_size'])
    if params['world_size'] > 1:  # multi-GPU training
      torch.cuda.set_device(params['local_rank'])
      try:
        dist.init_process_group(backend='nccl', init_method='env://')
      except:
        print('NCCL crash, sleep for 2h',flush=True);  time.sleep(2*3600)
                  
      params['world_rank'] = dist.get_rank()
      #print('M:locRank:',params['local_rank'],'rndSeed=',torch.seed())
    params['verb'] =args.verb * (params['world_rank'] == 0)
    #print('M: verb=', params['verb'])
    if params['verb']:
      logging.info('M:MASTER_ADDR=%s WORLD_SIZE=%s RANK=%s  pytorch:%s'%(os.environ['MASTER_ADDR'] ,os.environ['WORLD_SIZE'], os.environ['RANK'],torch.__version__ ))
      for arg in vars(args):  logging.info('M:arg %s:%s'%(arg, str(getattr(args, arg))))

    blob=read_yaml( args.design+'.hpar.yaml',verb=params['verb'], logger=True)
    params.update(blob)
    params['design']=args.design
    
    #print('M:params');pprint(params)#tmp
    # refine BS for multi-gpu configuration
    tmp_batch_size=params.pop('batch_size')
    if params['const_local_batch']: # faster but LR changes w/ num GPUs
      params['local_batch_size'] =tmp_batch_size 
      params['global_batch_size'] =tmp_batch_size*params['world_size']
    else:
      params['local_batch_size'] = int(tmp_batch_size//params['world_size'])
      params['global_batch_size'] = tmp_batch_size

    
    # capture other args values
    params['h5_path']=params['data_path'][args.facility]
    params['h5_name']=args.dataName+'.h5'
    params['job_id']=args.jobId
    
    if '*/' in args.outPath:
      outPath=args.outPath.replace('*/',params['out_base'][args.facility])
      params['out_path']=os.path.join(outPath,args.jobId)       
    else:
      params['out_path']=args.outPath # use as-is

    params['facility']=args.facility
    
    if args.numSamp!=None:  # reduce num steps/epoch - code testing
        params['max_glob_samples_per_epoch']=args.numSamp
    if args.epochs!=None:
        params['train_conf']['epochs']= args.epochs

    # deleted alternatives after choice was made
    for x in ['Defaults','data_path','out_base']:  
        params.pop(x)
        
    trainer = Trainer(params)
    
    trainer.train()
                
    if params['world_rank'] == 0:
      sumF= params['out_path']+'/sum_train.yaml'
      write_yaml(trainer.sumRec, sumF) # to be able to predict while training continus

      print("M:done rank=",params['world_rank'])
