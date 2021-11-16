__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
this data loader reads one shard of data from a common h5-file upon start, there is no distributed sampler

reads all data at once and serves them from RAM
- optimized for mult-GPU training
- only used block of data  from each H5-file
- reads data from common file for all ranks
- allows for in-fly transformation

Shuffle: only  all samples after read is compleated

Typical input is a large 3D cube, which will be sliced & sampled 
E.g:  513G  dm_density_4096.h5  contains
h5-write : dm_density (4096, 4096, 4096) float32
h5-write : meta.JSON as string (1,) object

Currently, only axis=0 is used to subdivide data on train/valid/test/skip 
using formula which adds 3.3% isolation, parametrized by skipFrac & domFrac 
See compute_samples_division(numSamp)

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


#...!...!..................
def get_data_loader(trainMD,domain, verb=1):

  conf=copy.deepcopy(trainMD)  # the input may be reused later in the upper level code
  
  dataset=  Dataset_h5_srgan2D(conf,domain,verb)
  
  # return back some of info
  trainMD[domain+'_steps_per_epoch']=dataset.sanity()
  #?trainMD['model']['inputShape']=list(dataset.data_frames.shape[1:])
  #?trainMD['model']['outputSize']=dataset.data_parU.shape[1]
  #trainMD['full_h5name']=conf['h5name']

  dataloader = DataLoader(dataset,
                          batch_size=conf['local_batch_size'],
                          num_workers=conf['num_cpu_workers'],
                          shuffle=conf['shuffle'],
                          drop_last=True,
                          pin_memory=torch.cuda.is_available())

  return dataloader

#...!...!..................
def compute_samples_division(numSamp): # build division into train/valid/test/skip
    skipFrac=0.1  # amout of samples left out to assure separation between train/valid/test subset
    domFrac={'valid':0.1,'test':0.1,'train':0.7}

    numSkip=int(numSamp * skipFrac/3.)
    assert numSkip>1

    divRange={}
    i1=numSkip
    for dom in domFrac:
        i2=i1+int(numSamp * domFrac[dom])
        assert i2>i1
        divRange[dom]=[i1,i2]
        i1=i2+numSkip

    #print('divRange:',divRange)
    return divRange

#...!...!..................
def random_crop_WHC(image,tgt_size):
    org_size=image.shape[0]
    maxShift=org_size - tgt_size
    #print('RC2d:',org_size,tgt_size, maxShift)
    assert maxShift>=0
    if maxShift>0:  # crop image to smaller size
      ix=np.random.randint(maxShift)
      iy=np.random.randint(maxShift)
      image=image[ix:ix+tgt_size,iy:iy+tgt_size]
    return  np.reshape(image,(tgt_size,tgt_size,1))

#...!...!..................
def random_flip_rot_WHC(image,prob=0.5):
    if np.random.uniform() <prob:  image=np.flip(image,axis=0)
    if np.random.uniform() <prob:  image=np.flip(image,axis=1)
    if np.random.uniform() <prob:  image=np.swapaxes(image,0,1)
    return image

#...!...!....................
def rebin_WHC(cube,nReb):  # shape: WHC
    # rebin only over axis=0,1; retain the following axis unchanged
    assert cube.ndim==3
    # cube must be symmetric
    assert cube.shape[0]==cube.shape[1]
    nb=nReb # rebin factor
    a=cube.shape[0]
    assert a%nb==0
    b=a//nb
    nz=cube.shape[2]
    #print('aaa',a,nb,b)
    sh = b,nb,b,nb,nz
    C=cube.reshape(sh)
    #print('Csh',C.shape)
    D=np.sum(C,axis=-2)
    #print('Dsh',D.shape)
    E=np.sum(D,axis=-3)
    #print('Esh',E.shape)
    return E
    
#-------------------
#-------------------
#-------------------
class Dataset_h5_srgan2D(object):
#...!...!..................    
    def __init__(self, conf,domain,verb=1):
        conf['domain']=domain
        conf['cube_name']='dm_density'
        self.conf=conf
        self.verb=verb

        self.openH5()
        if self.verb and 0:
            print('\nDS-cnst name=%s  shuffle=%r BS=%d steps=%d myRank=%d numSampl/hd5=%d'%(self.conf['name'],self.conf['shuffle'],self.localBS,self.__len__(),self.conf['myRank'],self.conf['numSamplesPerH5']),'H5-path=',self.conf['dataPath'])
        assert self.numLocalSamp>0
        assert self.conf['world_rank']>=0
        assert conf['hr_size']%conf['upscale_factor']==0
        conf['lr_size']=conf['hr_size']//conf['upscale_factor']

        if self.verb :
            logging.info(' DS:load-end %s locSamp=%d, X.shape: %s type: %s'%(self.conf['domain'],self.numLocalSamp,str(self.data_block.shape),self.data_block.dtype))
            #print(' DS:Xall',self.data_frames.shape,self.data_frames.dtype)
            #print(' DS:Yall',self.data_parU.shape,self.data_parU.dtype)
            

#...!...!..................
    def sanity(self):      
        stepPerEpoch=int(np.floor( self.numLocalSamp/ self.conf['local_batch_size']))
        if  stepPerEpoch <2:
            print('\nDS:ABORT, Have you requested too few samples per rank?, numLocalSamp=%d, BS=%d  dom=%s'%(self.numLocalSamp, self.conf['local_batch_size'],self.conf['domain']))
            exit(67)
        # all looks good
        return stepPerEpoch
        
#...!...!..................
    def openH5(self):
        cf=self.conf
        inpF=cf['h5_name']
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
        #print('DL:inpD',inpMD)
        #print('DL:recovered meta-data with %d keys dom=%s'%(len(inpMD),dom))
        
        numSamp=inpMD['cube_shape'][0]        
        idxRange=compute_samples_division(numSamp)[ dom]
        totSamp=idxRange[1]-idxRange[0]
        if 'max_samples_per_epoch' in cf:
            max_samp= cf['max_samples_per_epoch']
            if dom=='valid': max_samp//=8
            oldN=totSamp
            totSamp=min(totSamp,max_samp)
            if totSamp<oldN:
              logging.warning('GDL: shorter dom=%s max_samples=%d from %d'%(dom,totSamp,oldN))
            
            idxRange[1]=idxRange[0]+totSamp

        
        self.inpMD=inpMD # will be needad later too        
        locStep=int(totSamp/cf['world_size']/cf['local_batch_size'])
        locSamp=locStep*cf['local_batch_size']
        logging.info('DLI:totSamp=%d locStep=%d BS=%d'%(totSamp,locStep,cf['local_batch_size']))
        assert locStep>0
        maxShard= totSamp// locSamp
        assert maxShard>=cf['world_size']
                    
        # chosen shard is rank dependent, wraps up if not sufficient number of ranks
        myShard=self.conf['world_rank'] %maxShard
        sampIdxOff=myShard*locSamp
        
        if self.verb: logging.info('DS:file dom=%s myShard=%d, maxShard=%d, sampIdxOff=%d '%(cf['domain'],myShard,maxShard,sampIdxOff))       
        
        # data reading starts ....
        self.data_block=h5f[ cf['cube_name']][sampIdxOff:sampIdxOff+locSamp]
        h5f.close()
        # = = = READING HD5  done
        
        if self.verb>0 :
            startTm1 = time.time()
            if self.verb: logging.info('DS: hd5 read time=%.2f(sec) dom=%s dataBlock.shape=%s'%(startTm1 - startTm0,dom,str(self.data_block.shape)))
            
        # .......................................................
        #.... data embeddings, transformation should go here ....
            
        #.... end of embeddings ........
        # .......................................................
        self.numLocalSamp=self.data_block.shape[0]
        
        if 0 : # check X normalizations            
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
        #print('DSI:idx=',idx,cf['domain'],'rank=',cf['world_rank'],'slide:',self.data_block.shape)
        assert idx>=0
        assert idx< self.numLocalSamp

        image=self.data_block[idx]
        hr=random_crop_WHC(image,cf['hr_size'])
        assert hr.shape[0]==hr.shape[1]
        hr=random_flip_rot_WHC(hr) # shape: WHC
        lr=rebin_WHC(hr,cf['upscale_factor'])
        #print('DL-GI hr:',hr.shape,', lr:',lr.shape)
        # convert to CWH
        lr=lr.reshape(1,cf['lr_size'],-1)
        hr=hr.reshape(1,cf['hr_size'],-1)

        # transform to log
        lrImg=np.log(1+lr)
        hrImg=np.log(1+hr)
        
        return torch.from_numpy(np.copy(lrImg)), torch.from_numpy(np.copy(hrImg))

