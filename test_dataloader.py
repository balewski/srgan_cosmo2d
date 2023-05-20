#!/usr/bin/env python
""" 

read test data from HD5
shifter  ./test_dataloader.py  -g 1 -v2  --dataName  univL7cola_dm2d_202204_c20 --facility corigpu

shifter  ./test_dataloader.py  -g 1 -v2 --design benchmk_flux4k --dataName flux_L80_s1-1LR8HR-Nyx4k-r1c9

"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)

from pprint import pprint,pformat
import  time
import sys,os
from toolbox.Util_IOfunc import read_yaml, write_yaml
from toolbox.Dataloader_H5 import get_data_loader
import numpy as np
import argparse

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", default='benchmk_flux2', help='[.hpar.yaml] configuration of model and training')

    parser.add_argument("--dataName",default="flux-1LR4HR-Nyx2022a-r2c14",help="[.h5] name data  file")
    parser.add_argument("--basePath", default=None, help=' all outputs+TB+snapshots, default in hpar.yaml')

    parser.add_argument("--facility", default='perlmutter', choices=['summit','summitlogin','perlmutter'],help='computing facility where code is executed')

    parser.add_argument("-g", "--numRanks", type=int, default=4, help="forces data partition")
    parser.add_argument("--numSamp", type=int, default=None, help="(optional) cut off num samples per epoch")

    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')
   
    args = parser.parse_args()

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

  
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()

    params ={}
    params['world_rank']=args.numRanks-1
    params['world_size']=args.numRanks

    blob=read_yaml( args.design+'.hpar.yaml')
    facCf=blob.pop('facility_conf')[args.facility]
    blob.pop('Defaults')
    params.update(blob)
    params['design']=args.design
    params['num_cpu_workers']=1
       
    #print('M:params');pprint(params)#tmp
    #... propagate facility dependent config
    params['facility']=args.facility
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

    #pprint(params); ok120
    # capture other args values
    params['h5_path']=facCf['data_path']
    params['h5_name']=args.dataName+'.h5'
    #.... update selected params based on runtime config
    if args.numSamp!=None:  # reduce num steps/epoch - code testing
        params['max_glob_samples_per_epoch']=args.numSamp


    logging.info('T:rank %d of %d, prime data loaders'%(params['world_rank'],params['world_size']))

    
    params['shuffle']=True  
    train_loader = get_data_loader(params, 'train',verb=args.verb)

    if 0: #do-valid-loader
        params['shuffle']=True # use False for reproducibility
        valid_loader = get_data_loader(params, 'valid', verb=args.verb)
        logging.info('T:valid-data: %d steps'%(len(valid_loader)))
    
    inpMD=train_loader.dataset.conf
    logging.info('T:meta-data from h5: %s'%pformat(inpMD))

    logging.info('T:rank %d of %d, data loaders initialized'%(params['world_rank'],params['world_size']))
    logging.info('T:train-data: %d steps, localBS=%d, globalBS=%d'%(len(train_loader),train_loader.batch_size,params['global_batch_size']))


    logging.info('M:loading completed')
    
    print('M: ....... access 1st batch sample, imag=flux')
    k=0
    for  lrFinImg,hrFinImg in train_loader: 

        if 1:  # get dimensions & normalization
            print('\nlrFin:',lrFinImg.shape,lrFinImg.dtype,'avr:',np.mean(lrFinImg.numpy(),axis=(1,2,3)),'std:',np.std(lrFinImg.numpy(),axis=(1,2,3)))
        
            print('hrFin:',hrFinImg.shape,hrFinImg.dtype,'avr:',np.mean(hrFinImg.numpy(),axis=(1,2,3)),'std:',np.std(hrFinImg.numpy(),axis=(1,2,3)))


        k+=1
        if k>3: break

    print('M: done')
