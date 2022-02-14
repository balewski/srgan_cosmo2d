#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 tool generaring and evaluating HPAR for SRGAN-Cosmo2d
 
save: out/hpar_XX.conf.yaml and hpar_XX.sum.yaml 
uses generator for input

Steps:
* gen_CNN_block
* gen_FC_block
* gen_train_block
* build_model
* eval_speed

****Run on PM (using Shifter image)
 salloc -C gpu -q interactive -t4:00:00 --gpus-per-task=1 --image=nersc/pytorch:ngc-21.08-v2 -A m3363_g --ntasks-per-node=4  -N 1

 time srun -n1 shifter ./genHPar_srganCosmo2d.py 
shifter --image=nersc/pytorch:ngc-21.08-v2 ./genHPar_srganCosmo2d.py

'''

import socket  # for hostname
import os,sys,copy
import time
import secrets
from pprint import pprint
import torch
import numpy as np
from toolbox.Util_IOfunc import read_yaml, write_yaml
from toolbox.Trainer import Trainer

import logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hpoName', default='hpoA',help="name of the HPO effort")
    parser.add_argument('--startDesign', default='hpoA_start1.hpar.yaml',help="Starting HPAR config")
    parser.add_argument("-v","--verb",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-o","--outPath",default='/pscratch/sd/b/balewski/tmp_hpoA/b/',help="output path for HPAR")
    parser.add_argument("-n","--numTrials",default=1, type=int,help="number of HPAR sets to be generated")
    parser.add_argument("--dataName",default="dm_density_4096",help="[.h5] name data  file for short training")
    parser.add_argument("--basePath", default=None, help=' all outputs+TB+snapshots, default in hpar.yaml')
    args = parser.parse_args()
    args.epochs=2
    args.steps=8
    args.facility='perlmutter'
    args.hpoConf=args.hpoName+'.conf.yaml' #defines all contrants for HPAR generation and evaluation")
    np.random.seed() #  set the seed to a random number obtained from /dev/urandom 
    for i in range(int(50*np.random.rand())):
        np.random.rand()  # now rnd is burned in

    for arg in vars(args):  
        print( 'myArg:',arg, getattr(args, arg))
    print('seed test:',np.random.rand(4))

    return args


# - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - 
class HparGenerator_srganCosmo2d():
    def __init__(self,myConf,seedDesign,verb=1):
        self.myConf=myConf
        self.proposal=seedDesign
        self.verb=verb
        self.proposal['myId']=secrets.token_hex(nbytes=4)
        self.timeBegin = time.time()
        self.qa={}
        self.proposal['qa_hpo_1gpu']=self.qa # this is a shortcut
        
#...!...!....................
    def prep_G_block(self):
        if self.verb: print('G_block start')
        C=self.myConf['G_block']
        M=self.proposal['model_conf']['G']
        if self.verb>1:   pprint(C)
        M['first_cnn_chan']=rnd_value(C['first_cnn_chan'])
        M['num_resid_conv']=rnd_value(C['num_resid_conv'])
        
        if self.verb: print('G_block done',M)
        
        
    #...!...!....................
    def prep_D_conv_block(self):
        if self.verb: print('D_conv_block start')
        C=self.myConf['D_conv_block']
        M=self.proposal['model_conf']['D']['conv_block']
        if self.verb>1:   pprint(C)

        num_hr_bins=512 #tmp2
        numLyr=rnd_value(C['num_layer'])
        #print('M1',M)
        # clear old data
        for x in ['bn','filter','kernel','stride']: M[x]=[]
        mylen=num_hr_bins
        mychan=rnd_value(C['first_dim'])
        kernel=rnd_value(C['kernel'])
        for k in range(numLyr):
            stride=1+rnd_value(C['stride2'])  # it will impact filters
            M['bn'].append( rnd_value(C['BN']) )
            M['filter'].append(mychan)
            M['kernel'].append(kernel)
            M['stride'].append( stride )
            print(k,'ggg mylen',mylen,'filter:',mychan)
            if  mylen<4 : return True # error code
            mylen//=stride            
            mychan*=rnd_value(C['dim_fact'])
                        
        if self.verb: print('D_conv_block done');pprint(M)
        return False  # error code
    
    #...!...!....................
    def prep_D_fc_block(self):
        if self.verb: print('D_fc_block start')
        C=self.myConf['D_fc_block']
        M=self.proposal['model_conf']['D']['fc_block']
        if self.verb>1:   pprint(C)

        numLyr=rnd_value(C['num_layer'])
        mydim=rnd_value(C['last_dim'])
        dimL=[]
        for k in range(numLyr):
            dimL.append(mydim)
            mydim*=rnd_value(C['dim_fact'])
        dimL.reverse() # in place
        M['dims']=dimL
        M['dropFrac']=rnd_value(C['drop_frac'])
        if self.verb: print('D_fc_block done');pprint(M)
        
    #...!...!....................
    def prep_train_block(self):
        if self.verb>0: print('train_conf start')
        C=self.myConf['train_conf']
        M=self.proposal['facility_conf'][args.facility]
        if self.verb>1:   pprint(C)
        #print('M-LF:',M)
        for x in ['D_LR','G_LR']:
            M[x]['init']=rnd_value(C[x+'_init'])
            M[x]['reduce']=rnd_value(C[x+'_reduce'])

        M['batch_size']=rnd_value(C['localBS'])
        print('ggg localBS:',M['batch_size'])
        

#...!...!....................
def rnd_value(rec):  # zoo of random generators
    # returns single value x, a<x<b , drawn using mode-type distribution
    a,b=rec[:2]
    mode=rec[-1]
    if mode=='lin-int': # uniform linear int
        assert a<=b
        return int(np.random.randint(a,b))
    
    if mode=='power2': # uniform  power of 2s
        assert a>=0
        assert a<=b
        j=np.random.randint(a,b+1)
        return 1<<j
    
    if mode=='choice-int':
        values=rec[:-1]
        return int(np.random.choice(values))
    
    if mode=='prob-int':
        assert a>=0.
        assert a<=1.
        return int(np.random.uniform()<a)
    
    if mode=='exp-float': # uniform  in exponent
        assert a>=0
        assert a<=b
        a=np.log(a)
        b=np.log(b)            
        u=np.random.uniform(a,b)
        txt='%.3g'%np.exp(u)
        return float(txt)
    
    else:
        print('rnd_val invalid mode:',rec); exit(99)
        

#...!...!....................
def prep_exec_config(params): # modiffication added here will not be saved
    facCf=params.pop('facility_conf')[args.facility]
    params.pop('Defaults')

    #... propagate facility dependent config
    for x in ["D_LR","G_LR"]:
        params['train_conf'][x]=facCf[x]

    params['train_conf']['pre_epochs']=2
    params['train_conf']['adv_epochs']=2
    for x in ["D_LR","G_LR"]: 
        if params['train_conf'][x]['decay/epochs']=='auto':
            params['train_conf'][x]['decay/epochs']=params['train_conf']['adv_epochs']//2
    params['local_batch_size'] =facCf['batch_size']
    params['verb']=args.verb
    params['world_size']=1
    params['world_rank']=0
    params['local_rank']=0
    params['exp_path']=args.outPath
    params['opt_pytorch']=gen.proposal['opt_pytorch']
    params['global_batch_size']=params['local_batch_size']*params['world_size']

    # capture other args values
    params['h5_path']=facCf['data_path']
    params['h5_name']=args.dataName+'.h5'
    params['exp_name']='hpo_'+gen.proposal['myId']
    params['design']=gen.proposal['myId']
    params['facility']=args.facility

    if args.basePath==None:
        args.basePath=facCf['base_path']

    params['exp_path']=os.path.join(args.basePath,params['exp_name'])
    #.... update selected params based on runtime config       
    params['max_glob_samples_per_epoch']=max(2*params['local_batch_size'],256)
    return params

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':

    args=get_parser()
    if args.verb>0:
        print('M:torch:',torch.__version__,',python:',sys.version)
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print ('Available devices ', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(i,torch.cuda.get_device_name(i))

    confN=args.hpoConf
    hpoConf=read_yaml(confN)
    print('M: mpoConf:');    pprint(hpoConf)

    inpF=args.startDesign
    print('M: start design=',inpF)
    startDesign=read_yaml(inpF)
    print('M: start keys:',sorted(startDesign))
    startT0 = time.time()


    nok=0
    cnt={'shot':0,'Dmodel':0,'build':0,'size':0,'run':0,'acc':0}
    for shot in range(args.numTrials):
        print('\n = = = = = = = = = = M:new shot cnt:',cnt)
        myVerb=(shot==0)*args.verb
        seedDesign=copy.deepcopy(startDesign)
        gen=HparGenerator_srganCosmo2d(hpoConf,seedDesign,verb=myVerb)
        gen.qa['facility']=args.facility
        cnt['shot']+=1
        propF=os.path.join(args.outPath,'%s.hpar.yaml'%(gen.proposal['myId']))

        if 1:
            gen.prep_G_block()
            if gen.prep_D_conv_block(): continue
            cnt['Dmodel']+=1  
            gen.prep_D_fc_block()
            gen.prep_train_block()
    
        else: # debug only
            inpF='hpoA_start1.hpar.yaml'
            print(' test external design=',inpF)
            gen.proposal=read_yaml(inpF)
            gen.proposal['qa_hpo_1gpu']=gen.qa

        params=prep_exec_config(copy.deepcopy(gen.proposal))
        if 0: # activate if Trainer code has a bug
            trainer = Trainer(params)
            trainer.train()
            ok_stop_66
        # - --- PHASE 1 ---------    
        try:
            trainer = Trainer(params)
        except:
            continue
        
        cnt['build']+=1
        for GD in ['G','D']:
            rec=trainer.sumRec[GD+'_model_summary']
            gen.qa[GD+'_param_count']=rec['modelWeightCnt']

        #print('M: sumRec:');pprint(trainer.sumRec)
        #print('zz0',propF)
        DtotParCnt=gen.qa['D_param_count']
        if  DtotParCnt <hpoConf['constraints']['minDModelParameters'] or\
            DtotParCnt >hpoConf['constraints']['maxDModelParameters'] :
            if args.verb>1: print('M: model is too small/large=%.3e param, ABANDON'%DtotParCnt,'shot=',shot)
            propF+='bad0'
            #print('zz',propF)
            write_yaml(gen.proposal,propF)
            continue
        
        cnt['size']+=1
        if args.verb and 0:
            print('M:hpo proposal:');  pprint(gen.proposal)

        # - --- PHASE 2 ---------    
        try:
            trainer.train()
        except:
            #pprint(trainer.sumRec)
            locBS=trainer.sumRec['train_params']['local_batch_size']
            if args.verb>1: print('\nB: too large BS=%d , try next'%locBS,shot)
            propF+'bad1'
            write_yaml(gen.proposal,propF)
            continue

        cnt['run']+=1
        elaT=trainer.sumRec['last_adv_train_epoch_time']
        trnCf=trainer.sumRec['train_params']
        locSamp=trnCf['train_steps_per_epoch']*trnCf['local_batch_size']
        sampSpeed=locSamp/elaT
        gen.qa['adv_samples_per_sec']=sampSpeed
        gen.qa['date']=trainer.sumRec['train_date']
        
        #print('mmm',locSamp,elaT)

        if sampSpeed <hpoConf['constraints']['minSamplesPerSec']:
            if args.verb>1: print('M: model is too slow=%.2e sampl/sec, ABANDON'%sampSpeed,'shot=',shot)
            propF+='bad2'
            write_yaml(gen.proposal,propF)
            continue

        
        cnt['acc']+=1
        gen.proposal['model_conf']['comment']='%s %s %s'%(args.hpoName,args.facility,trainer.sumRec['train_date'])
        write_yaml(gen.proposal,propF)

        tagF=os.path.join(args.outPath,'good_%s.txt'%(gen.proposal['myId']))
        fd=open(tagF,'w')
        fd.write('HparGood design: %s  localBS: %d   param_count G:%.2e D:%.2e   adv_samp/sec: %.2e  shot: %d\n'%(gen.proposal['myId'], params['local_batch_size'], gen.qa['G_param_count'],gen.qa['D_param_count'],gen.qa['adv_samples_per_sec'],shot))
        fd.close()

        # save whole summary as well
        sumF= os.path.join(args.outPath,'sum_train_%s.yaml'%(gen.proposal['myId']))
        write_yaml(trainer.sumRec, sumF)
        
        nok+=1
        #time.sleep(3)

    totMin= (time.time()-startT0)/60.
    print('M: done all %d trials, nOK=%d, totTime=%.1f min'%(args.numTrials,nok,totMin))
    print('M:cnt:',cnt)
