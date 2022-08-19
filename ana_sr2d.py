#!/usr/bin/env python3
'''
 Analysis of quality of SRGAN-2D images

exp=1612659_1 sol=epoch850 
./ana_sr2d.py  --expName $exp --genSol $sol 

'''
import sys,os,copy
import numpy as np
import argparse,os
#import scipy.stats as stats
from pprint import pprint
#from matplotlib.colors import LogNorm

from toolbox.Util_H5io3 import  read3_data_hdf5
from toolbox.Util_Cosmo2d import  powerSpect_2Dfield_numpy,density_2Dfield_numpy
from toolbox.Plotter_Backbone import Plotter_Backbone


#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3], help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='ab',help="abc-string listing shown plots")

    parser.add_argument( "-i","--index", default=33,type=int,help="image index, aka 2D-slice")
    parser.add_argument( "-X","--noXterm", action='store_true', default=False, help="disable X-term for batch mode")
    parser.add_argument("-e","--expName",default=None,help="(optional), append experiment dir to data path")
    parser.add_argument("-s","--genSol",default="last",help="generator solution, e.g.: epoch123")
    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")
    parser.add_argument("-d","--dataPath",
                        #default='/global/homes/b/balewski/prje/tmp_srganA/'
                        default='/pscratch/sd/b/balewski/tmp_NyxHydro512A/'
                        ,help='data location w/o expName')
 
    args = parser.parse_args()
    if args.expName!=None:
        args.dataPath=os.path.join(args.dataPath,args.expName)
    args.formatVenue='prod'
    args.prjName=args.expName
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
    def image_2d(self,fieldD,metaD,plDD,myType,figId=4):
        figId=self.smart_append(figId)
        ncol,nrow=3,2; xyIn=(17.5,10)
        fig=self.plt.figure(figId,facecolor='white', figsize=xyIn)
        jx_hr,jy_hr=metaD['hrFin']['zmax_xyz'][:2]
        #print('PI2D:',jx_hr,jy_hr)
        cmap='Blues'
        if 'fft' in myType:    cmap ='Paired'
        
        for i,kr in  enumerate(plDD['fL']):
            ax=self.plt.subplot(nrow,ncol,1+i)
            img=fieldD[myType][kr]

            zScale=ax.imshow(img, cmap=cmap,origin='lower')
            fig.colorbar(zScale, ax=ax)
           
            tit='%s:%d^2, idx=%d,   z=%s'%(kr,img.shape[0],args.index,myType)
            ax.set(title=tit)
            
            if kr!='lrFin' and  'fft' not in myType:
                ax.axhline(jy_hr,linewidth=0.5,color='m')
#                # compute range of vertical line
                y1=jy_hr/img.shape[0]            
                ax.axvline(jx_hr, max(0,y1-0.1), min(1,y1+0.1),linewidth=0.5,color='y')
                 

#...!...!..................
    def skewer_1d(self,fieldD,metaD,plDD,figId=4):
        figId=self.smart_append(figId)
        jx_hr,jy_hr=metaD['hrFin']['zmax_xyz'][:2]
        ncol,nrow=1,1
        self.plt.figure(figId,facecolor='white', figsize=(15,4))
        ax=self.plt.subplot(nrow,ncol,1)
        for i,kr in  enumerate(fL[2:]):
            #if 'sr' in kr: continue
            #if 'Ini' in kr: continue #tmp
            data=fieldD['rho+1'][kr][jy_hr]
            binX=np.arange(data.shape[0])
            hcol=plDD['hcol'][kr]
            ax.step(binX,data,where='post',label=kr,color=hcol,lw=1)
        ax.axvline(jx_hr,linewidth=1.,color='m',linestyle='--')
        ax.grid()
        ax.set_yscale('log')
        tit='%s  idx=%d  jy_hr=%d'%(kr,args.index,jy_hr)
        ax.set(title=tit, ylabel='1+rho',xlabel='bins')
        ax.legend(loc='best')


#...!...!..................
    def spectra(self,fieldD,metaD,plDD,figId=4):
        figId=self.smart_append(figId)
        ncol,nrow=2,2
        self.plt.figure(figId,facecolor='white', figsize=(10,8))

        # - - - -  density histo - - - - 
        ax=self.plt.subplot(nrow,ncol,1)
        for i,kr in  enumerate(fL):
            #img=fieldD['ln rho+1'][kr]
            hcol=plDD['hcol'][kr]
            x,y=metaD[kr]['density']
            ax.step(x,y,where='post',label=kr,color=hcol)
        ax.set_yscale('log')
        ax.grid()
        tit='Density,  image idx=%d '%(args.index)
        ax.set(title=tit, xlabel=r'$ln(1+\rho)$',ylabel='num bins')
        ax.legend(loc='best', title='img type')

        
#...!...!..................
def max_2d_index(A):
    [jy,jx]=np.unravel_index(A.argmax(), A.shape)
    return jy,jx,A[jy,jx]

#...!...!..................
def post_process_srgan2D_fileds(fieldD,auxMD):
    print('PPF:start, auxMD:'); pprint(auxMD)
    #  per field analysis

    metaD={}
    for kr in fL:
        metaD[kr]={}
        data=fieldD['rho+1'][kr]  # density, keep '+1'  
        #print('data %s %s mx=%.1f std=%.1f'%(kr,str(data.shape),np.max(data),np.std(data)))
        
        jy,jx,zmax=max_2d_index(data)
        if 'hr' in kr: kr2='HR'
        else: kr2='LR'
        print(jy,jx,kr,kr2,'max:',zmax,np.min(data),np.sum(data))
        metaD[kr]['zmax_xyz']=[jx,jy,zmax]
        img=np.log(data)  
        fieldD['ln rho+1'][kr]=img
        
        x,y=density_2Dfield_numpy(img,10.)
        metaD[kr]['density']=[x,y]
        
        kphys,kbins,P,fftA2=powerSpect_2Dfield_numpy(data,d=auxMD[kr2])
        fieldD['ln fftA2+1'][kr]=np.log(fftA2+1)
        metaD[kr]['power']=[kphys,P]
    return metaD
        
#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__ == "__main__":
    args=get_parser()
    
    #.......... input data
    #inpF=os.path.join(args.expPath,'pred-test-%s.h5'%args.genSol)
    inpF=os.path.join(args.dataPath,'pred-test-%s.h5'%args.genSol)
    
    bigD,predMD=read3_data_hdf5(inpF)
    #print('predMD:',list(predMD))
    if args.verb>1:pprint(predMD)
    #predMD['field2d']['ilr']=copy.deepcopy(predMD['field2d']['hr'])  # add infor ILR
    
    # stage data
    fL=[ 'lrFin', 'ilrFin','hrIni', 'hrFin', 'srFin']

    fieldD={'ln rho+1':{}, 'ln fftA2+1':{}}
    fieldD['rho+1']={ xr:bigD[xr][args.index][0] for xr in fL}  # skip C-index

    #1fieldD['rho+1']['ilr']=fieldD['rho+1']['sr']

    #pprint(predMD['field2d'])
    
    #1auxD['ilr']=auxD['hr']
    #1fL=['lr','ilr','sr','hr']

    #pprint(predMD['field2d']); ok99
    metaD=post_process_srgan2D_fileds(fieldD,predMD['inpMD']['cell_size']) #predMD['field2d'])
    #print('M:fdk',fieldD.keys())
    #print('M:fdk2',fieldD['ln rho+1'].keys())

    # - - - - - Plotting - - - - - 
    plDD={}
    plDD['hcol']={'lrFin':'green','ilrFin':'C4','srFin':'C3','hrFin':'k','hrIni':'orange'}
    plDD['fL']=fL
    # - - - - - PLOTTER - - - - -
    plot=Plotter(args)
    if 'a' in args.showPlots:   plot.image_2d(fieldD,metaD,plDD,myType='ln rho+1',figId=1)
    if 'b' in args.showPlots:   plot.image_2d(fieldD,metaD,plDD,myType='ln fftA2+1',figId=2)
    if 'c' in args.showPlots:   plot.skewer_1d(fieldD,metaD,plDD,figId=3)
    if 'd' in args.showPlots:   plot.spectra(fieldD,metaD,plDD,figId=4)
   
    plot.display_all('ana_img%d'%args.index)
     
