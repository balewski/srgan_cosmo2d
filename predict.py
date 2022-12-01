#!/usr/bin/env python3
""" 
read trained net : model+weights
read test data from HD5
infere for  test data 

Inference works alwasy on 1 GPU or CPUs

 sol=best2067; exp=dev4_lrFact1.
./predict.py   --expName $exp --genSol $sol  (assumes basePath is common for many experiments)

OR cf to sandbox
 ENGINE=" shifter  --image=nersc/pytorch:ngc-21.08-v2 "
 srun -n1 shifter  ~/srgan_cosmo2d/predict.py --basePath . --expName .
 $ENGINE   ~/srgan_cosmo2d/predict.py --basePath . --expName .
 srun -n1 shifter  ~/srgan_cosmo2d/predict.py --basePath . --expName .

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
from toolbox.Util_Cosmo2d import  density_2Dfield_numpy,powerSpect_2Dfield_numpy, srgan2d_FOM1, median_conf_V
from toolbox.Util_H5io3 import  write3_data_hdf5


import argparse

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basePath",
                        #default='/global/homes/b/balewski/prje/tmp_srganA/'
                        default='/pscratch/sd/b/balewski/tmp_NyxHydro512A/'
                        , help="trained model ")
    parser.add_argument("--expName", default='exp03', help="main dir, train_summary stored there")
    parser.add_argument("-s","--genSol",default="last",help="generator solution")
    parser.add_argument("-n", "--numSamples", type=int, default=1000, help="limit samples to predict")
    parser.add_argument("--domain",default='test', help="domain is the dataset for which predictions are made, typically: test")

    parser.add_argument("-o", "--outPath", default='same',help="output path for plots and tables")
 
    parser.add_argument( "--doFOM",  action='store_true', default=False, help="compute FOM ")
    
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')
   
    args = parser.parse_args()
    args.expPath=os.path.join(args.basePath,args.expName)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!..................
def XXinterpolate_field_to_hr(lr,upscale):
    #print('lrImg',lrImg.shape) # B,C,W,H
    #print('PR one hr:',hr[0].shape,np.sum(hr[0]),np.min(hr),', lr:',sr[0].shape,np.sum(sr[0]))
    #print('lrImg.T',lrImg.T.shape) # C,W,H
    # must put channel as the last axis
    x2=lr.T -1 # H,W,C,B  abd  undo '1+rho'
    x3,_=interpolate_2Dfield(x2, upscale)
    #print('x3',x3.shape)
    fact=upscale*upscale
    ilr=x3.T/fact +1  # preserve the integral, restore '1+rho' for consistency
    #print('ilr',ilr.shape) # B,C,W,H
    return ilr

#...!...!..................
def histo_dens(hrA,srA,Rall):  # input: images = log(rho+1) 
    nsamp=hrA.shape[0]
    for i in range(nsamp):
        # ... compute density
        rphys,Rhr=density_2Dfield_numpy(hrA[i])
        _,Rsr=density_2Dfield_numpy(srA[i])    
        #print('Rsr-',i,Rhr.shape,Rhr.dtype)
        r_rel=Rsr/Rhr
        Rall.append(r_rel)
    
#...!...!..................
def histo_power(hrA,srA,space_step,Pall):  # input: densities
    nsamp=hrA.shape[0]
    for i in range(nsamp):
         # ... compute power spectra
        hr=hrA[i,0] # skip C-index, for now it is 1 channel
        sr=srA[i,0] 
        
        kphys,kidx,Phr,fftA2=powerSpect_2Dfield_numpy(hr,d=space_step)
        _,_,Psr,_           =powerSpect_2Dfield_numpy(sr,d=space_step)
        #print('Psr-',i,Phr.shape)
        p_rel=Psr/Phr
        Pall.append(p_rel)
    
#...!...!..................
def model_infer(model,data_loader,trainPar):
    #device=torch.cuda.current_device()   
    model.eval()

    # prepare output container, Thorsten's idea
    num_samp=len(data_loader.dataset)
    cfds=trainPar['data_shape']
    hr_size=cfds['hr_size']
    lr_size=cfds['lr_size']
    inp_chan=trainPar['num_inp_chan']
    upscale=cfds['upscale_factor']
    print('predict for num_samp=',num_samp,', hr_size=',hr_size,inp_chan)
    
    # clever list-->numpy conversion, Thorsten's idea
    class Empty: pass
    F=Empty()  # fields (not images)
    F.hrFin=np.zeros([num_samp,inp_chan,hr_size,hr_size],dtype=np.float32)
    F.hrIni=np.zeros_like(F.hrFin)
    F.srFin=np.zeros_like(F.hrFin)
    #XF.ilrFin=np.zeros_like(F.hrFin)
    F.lrFin=np.zeros([num_samp,inp_chan,lr_size,lr_size],dtype=np.float32)
    print('F-container',F.hrFin.shape,list(F.__dict__))
    
    if args.doFOM: # need more transient storage
        print('M: compute FOM ')        
        densAll=[]; powerAll=[]
        space_step=trainPar['field2d']['hr']['space_step']  # the same for SR
        
    nSamp=0
    nStep=0
    
    with torch.no_grad():
        for hrIniImg,lrFinImg,hrFinImg in data_loader:
            hrIniImg_dev, lrImg_dev, hrImg_dev = hrIniImg.to(device), lrFinImg.to(device), hrFinImg.to(device)
            #print('P1:',hrIniImg.shape, np.max(hrIniImg),np.max(hrFinImg))
            srImg_dev = model([hrIniImg_dev,lrImg_dev]) # THE PREDICTION      
            srFinImg=srImg_dev.cpu()
            n2=nSamp+srFinImg.shape[0]
            #print('nn',nSamp,n2)
            
            # convert images are the same as flux, no exp-log conversion
            lrFin=lrFinImg.detach().numpy()
            hrFin=hrFinImg.detach().numpy()
            hrIni=hrIniImg.detach().numpy()
            srFin=srFinImg.detach().numpy()
            #print('P2:',hrIni.shape, np.max(hrIni),np.max(hrFin),'std:',np.std(hrIni),np.std(hrFin))

            F.hrFin[nSamp:n2,:]=hrFin    
            F.hrIni[nSamp:n2,:]=hrIni
            F.srFin[nSamp:n2,:]=srFin
            F.lrFin[nSamp:n2,:]=lrFin
            #XF.ilrFin[nSamp:n2,:]=interpolate_field_to_hr(lrFin,upscale)
                
            if args.doFOM:
                histo_dens(hrFinImg,srFinImg,densAll)
                histo_power(hrFinImg,srFinImg,space_step,powerAll)
                
            # end-of-infering
            nSamp=n2
            nStep+=1
    
    print('infere done, nSamp=%d nStep=%d'%(nSamp,nStep),flush=True)


    fomD=None
    if args.doFOM:
        Rall=np.array(densAll)
        Rmed=median_conf_V(Rall)
        Pall=np.array(powerAll)
        Pmed=median_conf_V(Pall)
        print('M:Rmed:',Rmed.shape,'Pmed:',Pmed.shape)
        fomD=srgan2d_FOM1(Rmed[0],Pmed[0])
        fomTxt='FOM1: %.2g  = space: %.2g + fft: %.2g'%(fomD['fom'],fomD['r_fom'],fomD['f_fom'])
        
        print('M design:',trainPar['design'],fomTxt)
        pprint(fomD)

        
    #bigD={'lr':LRall,'ilr':ILRall,'sr':SRall,'hr':HRall}
    bigD=vars(F)
    return bigD,nSamp,fomD

  
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

    #?trainPar['field2d'].pop('sr')
    print('trainMD:',list(trainMD))
    if args.verb>1:pprint(trainPar)
 
    device   = torch.device("cpu")
    #device = torch.device("cuda")
    #assert torch.cuda.is_available()
    #device=torch.cuda.current_device()
    
    if 1:
        # instantiate model.
        model = Generator(trainPar['num_inp_chan'],trainPar['model_conf']['G']).to(device)
        # Load generator model weights
        sol="g-%s.pth"%args.genSol
        model_path=os.path.join(args.expPath,'checkpoints',sol)
        print('M:model_path',model_path)
    
        state_dict = torch.load(model_path, map_location=device)
        model = torch.nn.DataParallel(model) # disable if 1-gpu training was done
        model.load_state_dict(state_dict)

    trainPar['num_cpu_workers']=1
    data_loader = get_data_loader(trainPar, args.domain, verb=1)
 
    startT=time.time()
    bigD,nSamp,fomD=model_infer(model,data_loader,trainPar)
    predTime=time.time()-startT
    print('M: infer :   dom=%s samples=%d , elaT=%.2f min\n'% ( args.domain, nSamp,predTime/60.))

    sumRec={}
    sumRec['domain']=args.domain
    sumRec['exp_name']=trainPar['exp_name']
    sumRec['FOM']=fomD
    #sumRec['exp_name']=trainPar['job_id']
    sumRec['predTime']=predTime
    sumRec['numSamples']=nSamp
    sumRec['modelDesign']=trainMD['train_params']['myId']
    sumRec['model_path']=model_path
    sumRec['gen_sol']=sol.replace('.pth','')[2:]
    sumRec['inpMD']=data_loader.dataset.inpMD
    #1for x in  ['sim3d','field2d']:   sumRec[x]=trainPar[x]
    
    if args.outPath=='same' : args.outPath=args.expPath

    outF=os.path.join(args.outPath,'pred-%s-%s.h5'%(args.domain,sumRec['gen_sol']))
    write3_data_hdf5(bigD,outF,metaD=sumRec)

    print('M:done')
