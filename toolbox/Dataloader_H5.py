__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
this data loader reads one shard of data from a common h5-file upon start, there is no distributed sampler!!

reads all data at once and serves them from RAM
- optimized for mult-GPU training
- only used block of data  from each H5-file
- reads data from common file for all ranks
- allows for in-fly transformation

Shuffle: only local samples after read is compleated

Dataloader expects 2D the HR-images stack one another and split into train/val/test domains.
E.g. small file:
/global/cscratch1/sd/balewski/srgan_cosmo2d_data/univL9cola_dm2d_202204_c30.h5
h5-write : val.hr (4608, 512, 512, 2) uint8
h5-write : test.hr (4608, 512, 512, 2) uint8
h5-write : train.hr (36864, 512, 512, 2) uint8
The last dimR 0: zRed=50=ini, 1:zRed=0.=fin

Typical input will be sampled along the 1st axis,
randomized by mirroring and/or 90-rot,
the lr.fin-image is derived from  hr.fin by downsampling

'''

import time,  os
import random
import h5py
import numpy as np
import json
from pprint import pprint

import copy
from torch.utils.data import Dataset, DataLoader
import torch 
import logging
from toolbox.Util_IOfunc import read_yaml
from toolbox.Util_Cosmo2d import random_flip_rot_WHC, rebin_WHC, prep_fieldMD
from toolbox.Util_Torch import transf_field2img_torch

#...!...!..................
def get_data_loader(trainMD,domain, verb=1):
  trainMD['data_shape']={'upscale_factor': (1<<trainMD['model_conf']['G']['num_upsamp'])}
  conf=copy.deepcopy(trainMD)  # the input may be reused later in the upper level code
  cfds=conf['data_shape']
  
  dataset=  Dataset_h5_srgan2D(conf,domain,verb)
  
  # return back some info
  trainMD[domain+'_steps_per_epoch']=dataset.sanity()
  for x in ['data_shape','sim3d','field2d']:
      trainMD[x]=conf[x]

  # data dimension is know after data are read in
  trainMD['data_shape']['hr_img']=[conf['num_inp_chan'],cfds['hr_size'],cfds['hr_size']]
  trainMD['data_shape']['lr_img']=[conf['num_inp_chan'],cfds['lr_size'],cfds['lr_size']]
  dataloader = DataLoader(dataset,
                          batch_size=conf['local_batch_size'],
                          num_workers=conf['num_cpu_workers'],
                          shuffle=conf['shuffle'],
                          drop_last=True,
                          pin_memory=torch.cuda.is_available())

  return dataloader

#-------------------
#-------------------
#-------------------
class Dataset_h5_srgan2D(object):
#...!...!..................    
    def __init__(self, conf,domain,verb=1):
        conf['domain']=domain
        conf['rec_name']=domain+'.hr'
        assert conf['world_rank']>=0
        
        self.conf=conf
        self.verb=verb

        self.openH5() # computes final dimensionality of data
        assert self.numLocalSamp>0

        if self.verb :
            logging.info(' DS:load-end %s locSamp=%d, X.shape: %s type: %s'%(self.conf['domain'],self.numLocalSamp,str(self.data_block.shape),self.data_block.dtype))
            #print(' DS:Xall',self.data_frames.shape,self.data_frames.dtype)
            #print(' DS:Yall',self.data_parU.shape,self.data_parU.dtype)
            

#...!...!..................
    def sanity(self):      
        stepPerEpoch=int(np.floor( self.numLocalSamp/ self.conf['local_batch_size']))
        if  stepPerEpoch <1:
            logging.error('DLI: Have you requested too few samples per rank?, numLocalSamp=%d, BS=%d  dom=%s'%(self.numLocalSamp, self.conf['local_batch_size'],self.conf['domain']))
            exit(67)
        # all looks good
        return stepPerEpoch
        
#...!...!..................
    def openH5(self):
        cf=self.conf
        inpF=os.path.join(cf['h5_path'],cf['h5_name'])
        dom=cf['domain']
        if self.verb>0 : logging.info('DS:fileH5 %s  rank %d of %d '%(inpF,cf['world_rank'],cf['world_size']))
        
        if not os.path.exists(inpF):
            print('DLI:FAILED, missing HD5',inpF)
            exit(22)

        startTm0 = time.time()
                
        # = = = READING HD5  start
        h5f = h5py.File(inpF, 'r')
        metaJ=h5f['meta.JSON'][0]
        inpMD=json.loads(metaJ)

        cfds=cf['data_shape']
        #print('DL:inpD'); pprint(inpMD)
        cfds['hr_size']=inpMD['packing']['raw_cube_shape'][0]

        assert cfds['hr_size']%cfds['upscale_factor']==0
        cfds['lr_size']=cfds['hr_size']//cfds['upscale_factor']
        
        #print('DL:recovered meta-data with %d keys dom=%s'%(len(inpMD),dom))
        cf.update(prep_fieldMD(inpMD,cf))
        
        totSamp=inpMD['packing']['big_index'][cf['rec_name']]
        
        if 'max_glob_samples_per_epoch' in cf:            
            max_samp= cf['max_glob_samples_per_epoch']
            if dom=='valid': max_samp//=4
            oldN=totSamp
            totSamp=min(totSamp,max_samp)
            if totSamp<oldN and  self.verb>0 :
              logging.warning('GDL: shorter dom=%s max_glob_samples=%d from %d'%(dom,totSamp,oldN))            
            #idxRange[1]=idxRange[0]+totSamp

        
        self.inpMD=inpMD # will be needed later too        
        locStep=int(totSamp/cf['world_size']/cf['local_batch_size'])
        locSamp=locStep*cf['local_batch_size']
        if  self.verb>0 :
          logging.info('DLI:globSamp=%d locStep=%d BS=%d dom=%s'%(totSamp,locStep,cf['local_batch_size'],dom))
        assert locStep>0
        maxShard= totSamp// locSamp
        assert maxShard>=cf['world_size']
                    
        # chosen shard is rank dependent, wraps up if not sufficient number of ranks
        myShard=self.conf['world_rank'] %maxShard
        sampIdxOff=myShard*locSamp
        
        if self.verb: logging.info('DS:file dom=%s myShard=%d, maxShard=%d, sampIdxOff=%d '%(dom,myShard,maxShard,sampIdxOff))       
        
        # data reading starts ....
        self.data_block=h5f[ cf['rec_name']][sampIdxOff:sampIdxOff+locSamp]
        h5f.close()
        # = = = READING HD5  done
        
        if self.verb>0 :
            startTm1 = time.time()
            if self.verb: logging.info('DS: hd5 read time=%.2f(sec) dom=%s dataBlock.shape=%s dtype=%s'%(startTm1 - startTm0,dom,str(self.data_block.shape),self.data_block.dtype))
            
        # .......................................................
        #.... data embeddings, transformation should go here ....
            
        #.... end of embeddings ........
        # .......................................................
        self.numLocalSamp=self.data_block.shape[0]
        
        if 0 : # check X normalizations - from neuron inverter, never used        
            X=self.data_frames
            xm=np.mean(X,axis=1)  # average over 1600 time bins
            xs=np.std(X,axis=1)
            print('DLI:X',X[0,::80,0],X.shape,xm.shape)

            print('DLI:Xm',xm[:10],'\nXs:',xs[:10],myShard,'dom=',cf['domain'],'X:',X.shape)
                    

#...!...!..................
    def __len__(self):        
        return self.numLocalSamp

#...!...!..................
    def __getitem__(self, idx):
        cf=self.conf
        cfds=cf['data_shape']
        #print('DSI:idx=',idx,cf['domain'],'rank=',cf['world_rank'],'slide:',self.data_block.shape)
        assert idx>=0
        assert idx< self.numLocalSamp

        # primary input dtype=uint8 - this is pair of 3D HR densities for initial & final state
        hrIF=self.data_block[idx]  
        #print('DSI:hrIF=',hrIF.shape,hrIF.dtype)
              
        hrIF=random_flip_rot_WHC(hrIF) # shape: WHC  
        lrIF=rebin_WHC(hrIF,cfds['upscale_factor']) # both ini+fin
        #print('DSI:lrIF=',lrIF.shape,lrIF.dtype,'cf: lr+hr sizes:',cf['lr_size'],cf['hr_size'])
        #print('DL-GI hr shape+sum+max',hrIF.shape,np.sum(hrIF,axis=(0,1)),np.max(hrIF,axis=(0,1)),', lr:',lrIF.shape,np.sum(lrIF,axis=(0,1)))

        # final 'sample' consist of 3 images obtained from 2d densities
        # X=(lr.fin,hr.ini), Y=hr.fin
        
        # use only one C, convert WHC to CWH
        lrFin=lrIF[...,1].reshape(1,cfds['lr_size'],-1)
        hrIni=hrIF[...,0].reshape(1,cfds['hr_size'],-1)
        hrFin=hrIF[...,1].reshape(1,cfds['hr_size'],-1)
        
        #print('DL shape  X',lrFin.shape,hrIni.shape,'Y:',hrFin.shape)
        # transform field to image,computed as log(1+rho)
        lrFinImg=transf_field2img_torch(torch.from_numpy(np.copy(lrFin+1. )) )
        hrIniImg=transf_field2img_torch(torch.from_numpy(np.copy(hrIni+1. )) )
        hrFinImg=transf_field2img_torch(torch.from_numpy(np.copy(hrFin+1. )) )

        # fp32-prec data
        return lrFinImg.float(),hrIniImg.float(),hrFinImg.float()
    
        # finally cast output to fp16 - Jan could not make model to work with it
        #return lrFinImg.half(),hrIniImg.half(),hrFinImg.half()
        # RuntimeError: Input type (torch.cuda.HalfTensor) and weight type (torch.cuda.FloatTensor) should be the same
