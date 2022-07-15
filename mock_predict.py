#!/usr/bin/env python3
""" 

read test data from HD5
No data loader
NO inference
write output as from 'predict' - to allow all final analysis

"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
from toolbox.Util_H5io3 import read3_data_hdf5,write3_data_hdf5
from pprint import pprint
import  time
import sys,os

from toolbox.Util_Cosmo2d import rebin_WHC,prep_fieldMD, interpolate_2Dfield
#,  density_2Dfield_numpy,powerSpect_2Dfield_numpy, srgan2d_FOM1, median_conf_V

from toolbox.Util_H5io3 import  write3_data_hdf5


import argparse

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataName", default='NyxHydro4k_L7_dm2d_202106', help="[.h5] formatted data")

    parser.add_argument("-n", "--numSamples", type=int, default=200, help="limit samples to predict")
    parser.add_argument("--upscale", type=int, default=4, help="LR --> HR scale change")
    parser.add_argument("--domain",default='test', help="domain is the dataset for which predictions are made, typically: test")
    parser.add_argument("--dataPath",help="input data  path",
                        default='/global/homes/b/balewski/prje/data_NyxHydro4k/B/'
                        )
    parser.add_argument("-o", "--outPath", default='out',help="output path for plots and tables")
 
    parser.add_argument( "--doFOM",  action='store_true', default=False, help="compute FOM ")
    
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')
   
    args = parser.parse_args()
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!..................
def interpolate_field_to_hr(lr,upscale):
    #print('lrImg.T',lrImg.T.shape) # C,W,H
    # must put channel as the last axis
    x2=lr -1 # H,W,C  abd  undo '1+rho'
    x3,_=interpolate_2Dfield(x2, upscale)
    #print('x3',x3.shape)
    fact=upscale*upscale
    ilr=x3/fact +1  # preserve the integral, restore '1+rho' for consistency
    #print('ilr',ilr.shape) # C,W,H
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
def M_mock_predict(hr_data,trainPar):
    num_samp=args.numSamples
    cfds=trainPar['data_shape']
    hr_size=cfds['hr_size']
    lr_size=cfds['lr_size']
    inp_chan=trainPar['num_inp_chan']
    upscale=cfds['upscale_factor']
    print('mock_predict for num_samp=',num_samp,', hr_size=',hr_size,inp_chan)
    
    # clever list-->numpy conversion, Thorsten's idea
    class Empty: pass
    F=Empty()  # fields (not images)
    F.hrFin=np.zeros([num_samp,inp_chan,hr_size,hr_size],dtype=np.float32)
    F.hrIni=np.empty_like(F.hrFin)
    F.srFin=np.empty_like(F.hrFin)
    F.ilrFin=np.empty_like(F.hrFin)
    F.lrFin=np.zeros([num_samp,inp_chan,lr_size,lr_size],dtype=np.float32)
    print('F-container',F.hrFin.shape,list(F.__dict__))
    
    if args.doFOM: # need more transient storage
        print('M: compute FOM ')        
        densAll=[]; powerAll=[]
        space_step=trainPar['field2d']['hr']['space_step']  # the same for SR
        
    #print('hr_data',hr_data.shape)
    for i in range(args.numSamples):
        
        hrFin= hr_data[i,:,:,1:2]  # WHC
        #print('MPD:hrFin=',hrFin.shape,hrFin.dtype)
        lrFin=rebin_WHC(hrFin,args.upscale)  # WHC
        ilrFin=interpolate_field_to_hr(lrFin,args.upscale)
        #print('MPD:interp lrFin=',lrFin.shape,'ilrFin=',ilrFin.shape)

        # convert WHC to CWH to match output form training
        # now it is rho+1
        lrFin=lrFin.reshape(1,lr_size,-1) +1.
        ilrFin=ilrFin.reshape(1,hr_size,-1) +1.
        hrFin=hrFin.reshape(1,hr_size,-1) +1.
        #print('MPD: CWH lrFin=',lrFin.shape,'ilrFin=',ilrFin.shape,'hrFin=',hrFin.shape)
        F.lrFin[i]=lrFin
        F.ilrFin[i]=ilrFin
        F.hrFin[i]=hrFin
        
        '''
            F.hrIni[nSamp:n2,:]=hrIni
            F.srFin[nSamp:n2,:]=srFin
            F.lrFin[nSamp:n2,:]=lrFin
            F.ilrFin[nSamp:n2,:]=
        '''
        
        if args.doFOM:
            hrFinImg=np.log(F.hrFin)
            histo_dens(hrFinImg,srImg,densAll)
            histo_power(hrFin,sr,space_step,powerAll)
                
    print('mock infere done, nSamp=%d '%(args.numSamples),flush=True)

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
    return bigD,args.numSamples,fomD

  
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()

    inpF=os.path.join(args.dataPath,args.dataName+'.h5')
    inpD,inpMD=read3_data_hdf5(inpF)
    print('inpMD:',list(inpMD))
    if args.verb>1:pprint(inpMD)

    hr_size=inpMD['data_shape']['hr_size']
    assert hr_size%args.upscale==0
    lr_size=hr_size//args.upscale
    inpMD['data_shape']['lr_size']=lr_size
    inpMD['data_shape']['upscale_factor']=args.upscale

    startT=time.time()
    bigD,nSamp,fomD=M_mock_predict(inpD['hr_'+args.domain],inpMD)

    predTime=time.time()-startT
    print('M: infer :   dom=%s samples=%d , elaT=%.2f min\n'% ( args.domain, nSamp,predTime/60.))

     
    inp2MD={'data_shape':inpMD['data_shape']}
    trainPar=prep_fieldMD(inpMD,inp2MD)

    sumRec={}
    sumRec['domain']=args.domain
    sumRec['exp_name']='mock_predict'
    sumRec['FOM']=fomD
    #1sumRec['exp_name']='0000'
    sumRec['predTime']=predTime
    sumRec['numSamples']=nSamp
    #1sumRec['modelDesign']=trainMD['train_params']['myId']
    #1sumRec['model_path']=model_path
    sumRec['gen_sol']='mock_L%d'%inpMD['packing']['cutLevel']
    for x in  ['field2d']: sumRec[x]=trainPar[x] #'sim3d',
        
    outF=os.path.join(args.outPath,'pred-%s-%s.h5'%(args.domain,sumRec['gen_sol']))
    write3_data_hdf5(bigD,outF,metaD=sumRec)

    print('M:done')
