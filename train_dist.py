#!/usr/bin/env python3
'''
Input data are 2D flux slices 
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

time shifter  --image=nersc/pytorch:ngc-21.08-v2 ./train_dist.py  --design benchmk_50eaf423 --facility perlmutter  --numGlobSamp 256 --epochs 10  --expName exp07
>>> 


Run on 4 A100 on PM:
salloc  -C gpu -q interactive  -t4:00:00  --gpus-per-task=1 --image=nersc/pytorch:ngc-21.08-v2 -A m3363_g --ntasks-per-node=4   -N 1  

Quick test:
salloc -N1
 export MASTER_ADDR=`hostname`  
srun -n 1  shifter  python -u ./train_dist.py   --numGlobSamp 256  --expName exp2   --basePath /pscratch/sd/b/balewski/tmp_Nyx2022a-flux/jobs/inter  --dataName flux-1LR4HR-Nyx2022a-r2c14 --epochs 4 --design benchmk_flux1

If you run SLurm scripts:
export SLURM_ARRAY_JOB_ID=555
export SLURM_ARRAY_TASK_ID=44
./batchShifter.slr 


On Summit: salloc,  use facility=summitlogin


***** Display TB *****
ssh pm-tb
cd  $SCRATCH/tmp_NyxHydro4kG/
OR
cd /global/homes/b/balewski/prje/tmp_NyxHydro_outFluxB
 module load pytorch
 tensorboard --port 9600 --logdir=tb

/pscratch/sd/b/balewski/tmp_NyxHydro4kG

ssh summit-tb
cd /gpfs/alpine/world-shared/ast153/balewski/tmp_NyxHydro4kF/

 shifter  --image=nersc/pytorch:ngc-21.08-v2 bash
 tensorboard  --port 9600 --logdir=tb
 http://localhost:9600

 python -c 'import torch; print(torch.__version__)'
 python -c 'import tensorboard; print(tensorboard.__version__)'


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
  parser.add_argument("--design", default='benchmk_50eaf423', help='[.hpar.yaml] configuration of model and training')

  parser.add_argument("--dataName",default="flux-Nyx2022a-r2c14",help="[.h5] name data  file")
  parser.add_argument("--basePath", default=None, help=' all outputs+TB+snapshots, default in hpar.yaml')

  parser.add_argument("--facility", default='perlmutter', choices=['summit','summitlogin','perlmutter','crusher','corigpu'],help='computing facility where code is executed')  
  parser.add_argument("--expName", default='exp03', help="output main dir, train_summary stored there")
  parser.add_argument("-v","--verbosity",type=int,choices=[0,1,2,3], help="increase output verbosity", default=1, dest='verb')

  parser.add_argument("--epochs",default=None, type=int, help="(optional), replaces max_epochs from hpar")
  parser.add_argument("-n", "--numGlobSamp", type=int, default=None, help="(optional) cut off num samples per epoch")


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
      params['local_rank'] = 0 

    params['master_name']=os.environ['MASTER_ADDR']
    params['world_size'] = int(os.environ['WORLD_SIZE'])
    params['world_rank'] =int(os.environ['RANK'])
    params['worker_name']=socket.gethostname()
    params['facility']=args.facility
    if 'crusher' in params['master_name']:
      params['local_rank'] =0

    
    #print('M:params',params)
    if params['world_rank']==0:
        print('M:python:',sys.version,'torch:',torch.__version__)
    if params['world_size'] > 1:  # multi-GPU training
      torch.cuda.set_device(params['local_rank'])
      dist.init_process_group(backend='nccl', init_method='env://')
      assert params['world_rank'] == dist.get_rank()
      #print('M:locRank:',params['local_rank'],'rndSeed=',torch.seed())
    params['verb'] =args.verb * (params['world_rank'] == 0)
    #print('M:verbA:',params['verb'],args.verb,params['world_rank'] == 0,params['world_rank'] )
    
    if params['verb']:
      logging.info('M:MASTER_ADDR=%s WORLD_SIZE=%s RANK=%s  pytorch:%s'%(os.environ['MASTER_ADDR'] ,os.environ['WORLD_SIZE'], os.environ['RANK'],torch.__version__ ))
      for arg in vars(args):  logging.info('M:arg %s:%s'%(arg, str(getattr(args, arg))))

    blob=read_yaml( args.design+'.hpar.yaml',verb=params['verb'], logger=True)
    facCf=blob.pop('facility_conf')[args.facility]
    blob.pop('Defaults') # fullfilled its role when Yaml was parsed
    
    params.update(blob)
    params['design']=args.design
    
    #print('M:params');pprint(params)#tmp
    #... propagate facility dependent config
        
    for x in ["D_LR","G_LR"]: 
         params['train_conf'][x]=facCf[x]
        
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
    else:
      params['exp_path']=args.basePath # if given it is used w/o modiffication

    #.... update selected params based on runtime config
    if args.numGlobSamp!=None:  # reduce num steps/epoch - code testing
        params['max_glob_samples_per_epoch']=args.numGlobSamp
    if args.epochs!=None:
        params['train_conf']['adv_epochs']= args.epochs
    for x in ["D_LR","G_LR"]: 
        if params['train_conf'][x]['decay/epochs']=='auto':
            params['train_conf'][x]['decay/epochs']=int(params['train_conf']['adv_epochs']*0.7)
        
    trainer = Trainer(params)
    
    trainer.train()
                
    if params['world_rank'] == 0:
      sumF= params['exp_path']+'/sum_train.yaml'
      write_yaml(trainer.sumRec, sumF) # to be able to predict while training continus

      print("M:done rank=",params['world_rank'])
