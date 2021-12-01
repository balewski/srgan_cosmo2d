#!/usr/bin/env python
""" 
read trained net : model+weights
read test data from HD5
infere for  test data 

Inference works alwasy on 1 GPU or CPUs

 sol=best2067; exp=dev4_lrFact1.
./predict.py   --expName $exp --genSol $sol  


"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import torch
from pprint import pprint
import  time
import sys,os
import logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
from toolbox.Model_2d import Generator
from toolbox.Util_IOfunc import read_yaml, write_yaml
from toolbox.Dataloader_H5 import get_data_loader
from toolbox.Util_Cosmo2d import interpolate_2Dfield
from toolbox.Util_H5io3 import  write3_data_hdf5


import argparse

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--venue', dest='formatVenue', choices=['prod','poster'], default='prod',help=" output quality/arangement")

    parser.add_argument("--basePath",
                        default='/global/homes/b/balewski/prje/tmp_NyxHydro4kE/'
                        , help="trained model ")
    parser.add_argument("--expName", default='exp03', help="main dir, train_summary stored there")
    parser.add_argument("-s","--genSol",default="last",help="generator solution")
    parser.add_argument("-n", "--numSamples", type=int, default=100, help="limit samples to predict")
    parser.add_argument("--domain",default='test', help="domain is the dataset for which predictions are made, typically: test")

    parser.add_argument("-o", "--outPath", default='same',help="output path for plots and tables")
 
    parser.add_argument( "-X","--noXterm", dest='noXterm', action='store_true', default=False, help="disable X-term for batch mode")

    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')
   
    args = parser.parse_args()
    args.expPath=os.path.join(args.basePath,args.expName)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!..................
def XXload_model(trainPar,modelPath): # old version from NeuronInverter    
    # load entirel model
    modelF = os.path.join(modelPath, trainMD['train_params']['blank_model'])
    stateF= os.path.join(modelPath, trainMD['train_params']['checkpoint_name'])

    model = torch.load(modelF)
    model2 = torch.nn.DataParallel(model)
    allD=torch.load(stateF, map_location=str(device))
    print('all model ok',list(allD.keys()))
    stateD=allD["model_state"]
    keyL=list(stateD.keys())
    if 'module' not in keyL[0]:
      ccc={ 'module.%s'%k:stateD[k]  for k in stateD}
      stateD=ccc
    model2.load_state_dict(stateD)
    return model2

#...!...!..................
def model_infer(model,data_loader,trainPar):
    #device=torch.cuda.current_device()   
    model.eval()

    # prepare output container, Thorsten's idea
    num_samp=len(data_loader.dataset)
    hr_size=trainPar['hr_size']
    lr_size=trainPar['lr_size']
    inp_chan=trainPar['num_inp_chan']
    upscale=trainPar['upscale_factor']
    print('predict for num_samp=',num_samp,', hr_size=',hr_size,inp_chan)
    
    # clever list-->numpy conversion, Thorsten's idea
    HRall=np.zeros([num_samp,inp_chan,hr_size,hr_size],dtype=np.float32)
    LRall=np.zeros([num_samp,inp_chan,lr_size,lr_size],dtype=np.float32)
    ILRall=np.empty_like(HRall)
    SRall=np.empty_like(HRall)
    print('P0',HRall.shape)
    nSamp=0
    nStep=0
    with torch.no_grad():
        for lrImg,hrImg in data_loader:
            lrImg_dev, hrImg_dev = lrImg.to(device), hrImg.to(device)
            #print('P1:',lr.shape)
            srImg_dev = model(lrImg_dev)           
            srImg=srImg_dev.cpu()
            n2=nSamp+srImg.shape[0]
            print('nn',nSamp,n2)
            # convert images to densities=rho+1
            lr=np.exp(lrImg.detach()).numpy()
            hr=np.exp(hrImg.detach()).numpy()
            sr=np.exp(srImg.detach()).numpy()

            HRall[nSamp:n2,:]=hr    
            SRall[nSamp:n2,:]=sr
            LRall[nSamp:n2,:]=lr

            # compute interploated LR
            #print('lrImg',lrImg.shape) # B,C,W,H
            #print('PR one hr:',hr[0].shape,np.sum(hr[0]),np.min(hr),', lr:',sr[0].shape,np.sum(sr[0]))
            #print('lrImg.T',lrImg.T.shape) # C,W,H
            # must put channel as the last axis
            #d=lr.shape[1]
            x2=lr.T -1 # H,W,C,B  abd  undo '1+rho'
            x3,_=interpolate_2Dfield(x2, upscale)
            #print('x3',x3.shape)
            #d*=upscale
            fact=upscale*upscale
            ilr=x3.T/fact +1  # preserve the integral, restore '1+rho' for consistency
            print('ilr',ilr.shape) # B,C,W,H
            ILRall[nSamp:n2,:]=ilr
            nSamp=n2
            nStep+=1
    
    print('infere done, nSamp=%d nStep=%d'%(nSamp,nStep),flush=True)
    bigD={'lr':LRall,'ilr':ILRall,'sr':SRall,'hr':HRall}


    return bigD,nSamp

  
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()
        
    sumF=os.path.join(args.expPath,'sum_train.yaml')
    trainMD = read_yaml( sumF)
    trainPar=trainMD['train_params']
    trainPar['world_size']=1
    trainPar['world_rank']=0
    if args.numSamples!=None:
        trainPar['max_glob_samples_per_epoch' ] = args.numSamples

    pprint(trainPar)
 
    device   = torch.device("cpu")
    #device = torch.device("cuda")
    #assert torch.cuda.is_available()
    #device=torch.cuda.current_device()
    
    if 1:
        # instantiate model.
        model      = Generator(trainPar['num_inp_chan']).to(device)
        # Load generator model weights
        sol="g-%s.pth"%args.genSol
        model_path=os.path.join(args.expPath,'checkpoints',sol)
        print('M:model_path',model_path)
    
        state_dict = torch.load(model_path, map_location=device)
        model = torch.nn.DataParallel(model) # disable if 1-gpu training was done
        model.load_state_dict(state_dict)

    trainPar['h5_path']='/global/homes/b/balewski/prje/data_NyxHydro4k/B/' #tmp
    trainPar['num_cpu_workers']=1
    data_loader = get_data_loader(trainPar, args.domain, verb=1)
 
    startT=time.time()
    bigD,nSamp=model_infer(model,data_loader,trainPar)
    predTime=time.time()-startT
    print('M: infer :   dom=%s samples=%d , elaT=%.2f min\n'% ( args.domain, nSamp,predTime/60.))

    sumRec={}
    sumRec['domain']=args.domain
    sumRec['exp_name']=trainPar['exp_name']
    #sumRec['exp_name']=trainPar['job_id']
    sumRec['predTime']=predTime
    sumRec['numSamples']=nSamp
    sumRec['modelDesign']=trainMD['train_params']['myId']
    sumRec['model_path']=model_path
    sumRec['gen_sol']=sol.replace('.pth','')[2:]
    for x in  ['sim3d','field2d']:
        sumRec[x]=trainPar[x]
    
    if args.outPath=='same' : args.outPath=args.expPath

    outF=os.path.join(args.outPath,'pred-%s-%s.h5'%(args.domain,sumRec['gen_sol']))
    write3_data_hdf5(bigD,outF,metaD=sumRec)

    print('M:done')
