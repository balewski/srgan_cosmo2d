#!/usr/bin/env python3
'''
 compare power spectra  for neihbour HR slices


'''
import sys,os,copy
import numpy as np
import argparse,os

from pprint import pprint
from sim_NyxHydro.inspect_NyxHydroH5 import read_one_nyx_h5
from toolbox.Util_Cosmo2d import  powerSpect_2Dfield_numpy,density_2Dfield_numpy
from toolbox.Plotter_Backbone import Plotter_Backbone


#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3], help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='ab',help="abc-string listing shown plots")
    parser.add_argument( "-i","--index", default=33,type=int,help="image index, aka 2D-slice")
    parser.add_argument( "-X","--noXterm", action='store_true', default=False, help="disable X-term for batch mode")
   
    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")
    parser.add_argument("--cubeName", default='HR00661',help="core name of hd5:  plot[name].flux.h5")
    parser.add_argument("-d","--dataPath",
                        default='/pscratch/sd/b/balewski/tmp_NyxProd/2767632_2univ/cube_885045934'                        ,help='data location w/o expName')
 
    args = parser.parse_args()
    args.prjName='abc'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
            
    if not os.path.exists(args.outPath):
        os.makedirs(args.outPath);   print('M: created',args.outPath)
    return args

#............................
#............................
#............................
class Plotter(Plotter_Backbone):
    def __init__(self,args):
        Plotter_Backbone.__init__(self,args)
        self.args=args

#...!...!..................
    def image_2d(self,cube, md,figId=4):
        figId=self.smart_append(figId)
        ncol,nrow=3,2; xyIn=(17.5,10)
        fig=self.plt.figure(figId,facecolor='white', figsize=xyIn)
        
        cmap='Blues'
        #?if 'fft' in myType:    cmap ='Paired'
        acf=md['ana']
        idxr=acf['idx_range']
        idx0=acf['idx0']
        
        tit1='cube name:'+md['short_name']
        tit2=acf['univ_type']
        
        for i in range(idxr):
            ax=self.plt.subplot(nrow,ncol,1+i)
            myidx=idx0+i
            img=cube[:,myidx,:] # correct: x-z slice
            #img=cube[:,:,myidx]  # incorrect: x-y slice
            zScale=ax.imshow(img.T, cmap=cmap,origin='lower')
            fig.colorbar(zScale, ax=ax)

            tit=' slice x:%d'%myidx
            if i==0: tit=tit1+tit
            if i==1: tit=tit2+tit
            ax.set(title=tit)
            

#...!...!..................
    def pspectra(self,kbins,PV,md,figId=4):
        figId=self.smart_append(figId)
        ncol,nrow=2,1
        self.plt.figure(figId,facecolor='white', figsize=(10,5))

        acf=md['ana']
        idxr=acf['idx_range']
        idx0=acf['idx0']
        tit1='cube name:'+md['short_name']
        tit2=acf['univ_type']
        
        # - - - -  absolute power spec - - - - 
        ax=self.plt.subplot(nrow,ncol,1)
        for i in range(idxr):
            pspec=PV[i]
            dLab='idx=%d'%(i+idx0)
            ax.step(kbins,pspec,where='post',label=dLab)#,color=hcol)
        ax.set_yscale('log')
        ax.grid()
        ax.legend(loc='best', title='img type')
        
        tit='Power spec, '+tit1
        ax.set(title=tit)


        # - - - -  relative power spec - - - - 
        ax=self.plt.subplot(nrow,ncol,2)
        pref=PV[idxr//2+1]  # pick middle
        print('middle',idxr//2+1)
        for i in range(idxr):
            pspec=PV[i]/pref
            dLab='idx=%d'%(i+idx0)
            ax.step(kbins,pspec,where='post',label=dLab)#,color=hcol)
        
        ax.grid()
        ax.legend(loc='best', title='img type')
        
        tit='relative power spec, '+tit2
        ax.set(title=tit)
        

#...!...!..................
def compute_power_spec(cube,cell_size,md):
    acf=md['ana']
    idxr=acf['idx_range']
    idx0=acf['idx0']

    nb=cube.shape[0]//2 -1
    
    PV=np.zeros( (idxr,nb))

    for i in range(idxr):
        myidx=idx0+i
        one=cube[myidx]
        kphys,kbins,P,fftA2=powerSpect_2Dfield_numpy(one,d=cell_size)
        PV[i]=P

    return kbins,PV

    mP=np.mean(PV,axis=0)
    sP=np.std(PV,axis=0)
    print(PV.shape,mP.shape)
    srP=sP/mP

    for i in range(0,nb,5):
        print(i,kbins[i],srP[i])

   

#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__ == "__main__":
    args=get_parser()
    
    #.......... input data
    cubeF='plot%s.flux.h5'%args.cubeName
    univ_type="flux_HR"
    inpF=os.path.join(args.dataPath,cubeF)
    fieldN='tau_red'
    bigD,inpMD=read_one_nyx_h5(inpF, [fieldN],groupN="derived_fields",verb=2)
   
    
    
    #bigD,inpMD=read4_data_hdf5(inpF,acceptFilter=[univ_type])

    acf={'univ_type':univ_type}
    acf['idx_range']=5 # will compare +/- 2 slices above and below
    acf['idx0']=args.index # pick a starting index
    
    inpMD['ana']=acf
    inpMD['short_name']=args.cubeName
    pprint(inpMD)
    cube1=np.exp(-bigD['tau_red'])
    if 0:
        print('cube skewer',bigD[fieldN][30,300])
        print('flux skewer',cube1[30,300])
    
    kbins,PV=compute_power_spec(cube1,inpMD['cell_size'],inpMD)
    
    
    # - - - - - Plotting - - - - -
    
    # - - - - - PLOTTER - - - - -
    plot=Plotter(args)
    if 'a' in args.showPlots:   plot.image_2d(cube1,inpMD,figId=1)
    if 'b' in args.showPlots:   plot.pspectra(kbins,PV,inpMD,figId=4)
    
   
    plot.display_all()
     
