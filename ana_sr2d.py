#!/usr/bin/env python3
'''
 Analysis of quality of SRGAN-2D images, works with flux-data

exp=1612659_1 sol=epoch850 
./ana_sr2d.py  --expName $exp --genSol $sol 

'''
import sys,os,copy
import numpy as np
import argparse,os
from pprint import pprint

from toolbox.Util_H5io4 import  read4_data_hdf5
from toolbox.Util_Cosmo2d import  powerSpect_2Dfield_numpy, density_2Dfield_numpy
from toolbox.Plotter_Backbone import Plotter_Backbone


#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3], help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='ab', nargs='+',help="abc-string listing shown plots")

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
    args.showPlots=''.join(args.showPlots)
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
        ncol,nrow=4,1; xyIn=(17.5,4.5)
        fig=self.plt.figure(figId,facecolor='white', figsize=xyIn)
        jy_hr=plDD['skewer_iy']
        
        for i,kr in  enumerate(plDD['fL']):
            ax=self.plt.subplot(nrow,ncol,1+i)
            if 'fft' in myType:
                cmap ='Paired'
                kr+='_0'
                img=fieldD[myType][kr]
            else:
                img=fieldD[myType][kr][0]
                cmap='Blues'
                if kr=='hrIni': # Peter: show HR-SR
                    imgHR=fieldD[myType]['hrFin'][0]
                    imgSR=fieldD[myType]['srFin'][0]
                    kr='(hr-sr)Fin'
                    img=imgHR-imgSR
                    cmap ='seismic'
                    
            #print('zzz2',i,img.shape)
            zScale=ax.imshow(img.T, cmap=cmap,origin='lower')
            fig.colorbar(zScale, ax=ax)
           
            tit='%s:%d^2, idx=%d, dens=%s'%(kr,img.shape[0],args.index,myType)
            ax.set(title=tit)
            if 'fft' in myType:
                ax.set( xlabel='k(x)',ylabel='k(z*)')
            else:
                ax.set( xlabel='x',ylabel='z*')
            
            if kr!='lrFin' and  'fft' not in myType:
                ax.axhline(jy_hr,linewidth=0.5,color='m')
                 

#...!...!..................
    def skewer_1d(self,fieldD,metaD,plDD,figId=4):
        figId=self.smart_append(figId)
        #jx_hr,jy_hrjy_hr=metaD['hrFin']['zmax_xyz'][:2]
        jy_hr=plDD['skewer_iy']
        ncol,nrow=1,1
        self.plt.figure(figId,facecolor='white', figsize=(15,5))
        ax=self.plt.subplot(nrow,ncol,1)
        
        #... plot truth
        dataT=fieldD['flux']['hrFin']  #
        dataEr=fieldD['flux_std']        
        binX=np.arange(dataEr.shape[0])
        num_hrFin_chan=dataT.shape[0]
        for i in range(num_hrFin_chan):  # show 4 possible HR skewers
            data=dataT[i][jy_hr]
            sumF=np.sum(data)
            #print('zzz',i,dataT.shape,data.shape)
            ax.step(binX,data,where='post',label='HR_%d %d'%(i,sumF),color='k',lw=1)          

        #.... plot LR
        upscale=predMD['inpMD']['upscale_factor']  # hack
        jy_lr=jy_hr//upscale
        data=fieldD['flux']['lrFin'][0][jy_lr]
        sumF=np.sum(data)
        #print('lll',data.shape, binX[::upscale].shape)
        ax.step(binX[::upscale],data,where='post',label='LR    %d*'%(sumF*upscale),color='dodgerblue',lw=1.3) 
       
        #.... plot SR prediction
        data=fieldD['flux']['srFin'][0][jy_hr]
        sumF=np.sum(data)
        #print('sumF:',sumF)
        ax.step(binX,data,where='post',color='r',lw=1)
        ax.errorbar(binX[::10],data[::10],yerr=dataEr[::10],color='r',fmt='+',label='SR    %.0f'%sumF)

        
        ax.grid()       
        tit='skewer idx=%d  jy_hr=%d'%(args.index,jy_hr)
        ax.set(title=tit, ylabel='flux',xlabel='z*')
        ax.legend(loc='best',title='res   sumFlux')


#...!...!..................
    def spectra(self,fieldD,metaD,plDD,figId=4):
        figId=self.smart_append(figId)
        ncol,nrow=2,2
        self.plt.figure(figId,facecolor='white', figsize=(10,8))

        # - - - -  density histo - - - - 
        ax=self.plt.subplot(nrow,ncol,1)
        for i,kr in  enumerate(fL):
            if kr=='hrIni' : continue
            print('ddd',i,kr)
            hcol=plDD['hcol'][kr] 
            x,y=metaD[kr]['density 0']
            ax.step(x,y,where='post',label=kr,color=hcol)
            '''
            if  kr=='hrFin' :  # plot another 3
                for i in range(1,4):
                    x,y=metaD[kr]['density %d'%i]
                    ax.step(x,y,where='post',color=hcol,lw=1.)
            '''
        ax.set_yscale('log')
        ax.grid()
        tit='Density,  image idx=%d '%(args.index)
        ax.set(title=tit, xlabel='flux/bin',ylabel='num bins')
        ax.legend(loc='best', title='img type')

#............................
#............................
#............................

#...!...!..................
def post_process_srgan2D_fileds(fieldD,auxMD):
    print('PPF:start, auxMD:'); pprint(auxMD)
    #  per field analysis

    metaD={}
    for kr in fL:
        metaD[kr]={}
        data=fieldD['flux'][kr]  #  no exp-log transform for flux
        #print('data %s %s mx=%.1f std=%.1f'%(kr,str(data.shape),np.max(data),np.std(data)))

        if 'hr' in kr:
            kr2='HR'            
        else:
            kr2='LR'

        #fieldD['flux'][kr]=data

        inp_chan=data.shape[0]
        for i in range(inp_chan):
            img=data[0]
            krc='%s_%d'%(kr,i)
            #print('PPS',i,krc,img.shape)
            x,y=density_2Dfield_numpy(img) #,maxY=1.2)  
            metaD[kr]['density %d'%i]=[x,y]
        
            kphys,kbins,P,fftA2=powerSpect_2Dfield_numpy(img,d=auxMD[kr2])
            fieldD['ln fftA2'][krc]=np.log(fftA2+1.e-20)
            metaD[kr]['power %d'%i]=[kphys,P]
    return metaD
        
#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__ == "__main__":
    args=get_parser()    
    
    
    #.......... input data
    inpF=os.path.join(args.dataPath,'pred-test-%s.h5'%args.genSol)
    
    bigD,predMD=read4_data_hdf5(inpF)
    #print('predMD:',list(predMD))
    if args.verb>1:   pprint(predMD); exit(0)
   
    # stage data
    fL=[ 'lrFin', 'hrIni', 'hrFin', 'srFin']

    fieldD={'flux':{}, 'ln fftA2':{}}
    fieldD['flux']={ xr:bigD[xr][args.index] for xr in fL}  #XXX skip Chan-index
    fieldD['flux_std']=bigD['flux_std']
    
    metaD=post_process_srgan2D_fileds(fieldD,predMD['inpMD']['cell_size']) #predMD['field2d'])
    print('M:fdk',fieldD.keys(), bigD['hrFin'].shape)
 

    # - - - - - Plotting - - - - - 
    plDD={}
    plDD['hcol']={'lrFin':'green','srFin':'C3','hrFin':'k','hrIni':'orange'}
    plDD['fL']=fL
    plDD['skewer_iy']=99
    # - - - - - PLOTTER - - - - -
    plot=Plotter(args)
    if 'a' in args.showPlots:   plot.image_2d(fieldD,metaD,plDD,myType='flux',figId=1)
    if 'b' in args.showPlots:   plot.image_2d(fieldD,metaD,plDD,myType='ln fftA2',figId=2)
    if 'c' in args.showPlots:   plot.skewer_1d(fieldD,metaD,plDD,figId=3)
    if 'd' in args.showPlots:   plot.spectra(fieldD,metaD,plDD,figId=4)
   
    plot.display_all('sr_img%d'%args.index)
     
