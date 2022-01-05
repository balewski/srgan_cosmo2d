#!/usr/bin/env python3
'''
 Analysis of quality of SRGAN-2D images

 ./ana_sr2d.py -e exp22 -s best595
'''
import sys,os,copy
from toolbox.Util_H5io3 import  read3_data_hdf5
from toolbox.Util_Cosmo2d import  powerSpect_2Dfield_numpy,density_2Dfield_numpy
import numpy as np
import argparse,os
import scipy.stats as stats
from pprint import pprint
#from matplotlib.colors import LogNorm

PLOT={'image':1, 'fft':1, 'skewer':0, 'rho+power':1  }


#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "-i","--index", default=33,type=int,help="image index")
    parser.add_argument( "-X","--noXterm", action='store_true', default=False, help="disable X-term for batch mode")
    parser.add_argument("-e","--expName",default=None,help="(optional), append experiment dir to data path")
    parser.add_argument("-s","--genSol",default="last",help="generator solution, e.g.: epoch123")
    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")
    parser.add_argument("-d","--dataPath",  default='/global/homes/b/balewski/prje/tmp_NyxHydro4kF/',help='data location w/o expName')
 
    args = parser.parse_args()
    if args.expName!=None:
        args.dataPath=os.path.join(args.dataPath,args.expName)
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    if not os.path.exists(args.outPath):
        os.makedirs(args.outPath);   print('M: created',args.outPath)
    return args

#...!...!..................
def mini_plotter(args):
    import matplotlib as mpl
    if args.noXterm:
        mpl.use('Agg')  # to plot w/o X-server
        print('Graphics disabled')
    else:
        mpl.use('TkAgg') 
        print('Graphics started, canvas will pop-out')
    import matplotlib.pyplot as plt
    return plt

#...!...!..................
def save_fig( fid, ext='my', png=1):
     plt.figure(fid)
     plt.tight_layout()
     
     figName=os.path.join(args.outPath,'%s_f%d'%(ext,fid))
     if png: figName+='.png'
     else: figName+='.pdf'
     print('Graphics saving to ',figName)
     plt.savefig(figName)


#...!...!..................
def max_2d_index(A):
    [jy,jx]=np.unravel_index(A.argmax(), A.shape)
    return jy,jx,A[jy,jx]

#...!...!..................

#...!...!..................
def post_process_srgan2D_fileds(fieldD,metaD):
    print('PPF:start, metaD:'); pprint(metaD)
    #  per field analysis
    
    for kr in fL:
        data=fieldD['rho+1'][kr]  # density, keep '+1'  
        print('data %s %s '%(kr,str(data.shape)))
        jy,jx,zmax=max_2d_index(data)
        print(jy,jx,kr,'max:',zmax,np.min(data),np.sum(data))
        metaD[kr]['zmax_xyz']=[jx,jy,zmax]
        img=np.log(data)  
        fieldD['ln rho+1'][kr]=img
        
        x,y=density_2Dfield_numpy(img,10.)
        metaD[kr]['density']=[x,y]
        
        kphys,kbins,P,fftA2=powerSpect_2Dfield_numpy(data,d=metaD[kr]['space_step'])
        fieldD['ln fftA2+1'][kr]=np.log(fftA2+1)
        metaD[kr]['power']=[kphys,P]

#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__ == "__main__":
    args=get_parser()
    plt=mini_plotter(args)
    
    #.......... input data
    #inpF=os.path.join(args.expPath,'pred-test-%s.h5'%args.genSol)
    inpF=os.path.join(args.dataPath,'pred-test-%s.h5'%args.genSol)
    #1inpF='/global/homes/b/balewski/prje/tmp_NyxHydro4kB/manual/exp23j/monitor/valid-adv-epoch0.h5'
    bigD,predMD=read3_data_hdf5(inpF)
    print('predMD:');pprint(predMD)
    predMD['field2d']['ilr']=copy.deepcopy(predMD['field2d']['hr'])  # add infor ILR
    
    # stage data
    fL=['lr','ilr','sr','hr']
    #1fL=['lr','sr','hr']
    fieldD={'ln rho+1':{}, 'ln fftA2+1':{}}
    fieldD['rho+1']={ xr:bigD[xr][args.index][0] for xr in fL}  # skip C-index

    #tmp

    #1fieldD['rho+1']['ilr']=fieldD['rho+1']['sr']
    
    auxD={x:predMD['field2d'][x]  for x in fL} 
    #tmp
    #1auxD['ilr']=auxD['hr']
    #1fL=['lr','ilr','sr','hr']
    
    post_process_srgan2D_fileds(fieldD,auxD)
    

    # - - - - - Plotting - - - - - 
    plDD={}
    plDD['hcol']={'lr':'blue','ilr':'orange','sr':'C3','hr':'k'}
    png=1
    ext='img%d'%args.index
    
    if PLOT['image']: # - - - -  plot images - - - - 
        ncol,nrow=3,1; xyIn=(15,5); figId=4
        #ncol,nrow=2,2; xyIn=(12,12); figId=4
        plt.figure(figId,facecolor='white', figsize=xyIn)
        jx_hr,jy_hr=auxD['hr']['zmax_xyz'][:2]
        for i,kr in  enumerate(fL[1:]):
            ax=plt.subplot(nrow,ncol,1+i)
            img=fieldD['ln rho+1'][kr]
            ax.imshow(img,origin='lower')
            ax.set_aspect(1.)           
            tit='ln(rho+1) %s idx=%d  size=%s'%(kr,args.index,str(img.shape))
            ax.set(title=tit)
                 
            if kr!='lr':
                ax.axhline(jy_hr,linewidth=1.,color='m')
                # compute range of vertical line
                y1=jy_hr/img.shape[0]            
                ax.axvline(jx_hr, max(0,y1-0.1), min(1,y1+0.1),linewidth=1.,color='y')
        
        save_fig(figId,ext=ext,png=png)

    if PLOT['fft']: # - - - -  plot fft-images - - - - 
        ncol,nrow=3,1; xyIn=(15,5); figId=8
        plt.figure(figId,facecolor='white', figsize=xyIn)
        for i,kr in  enumerate(fL[1:]):
            ax=plt.subplot(nrow,ncol,1+i)
            img=fieldD['ln fftA2+1'][kr]
            zd=np.min(img);  zu=np.max(img); zm=np.mean(img); 
            print('fftImg',kr,zd,zm,zu)
            ax.imshow(img,origin='lower', cmap ='Paired')
            ax.set_aspect(1.)           
            tit='ln(abs(FFT)+1) %s idx=%d  size=%s'%(kr,args.index,str(img.shape))
            ax.set(title=tit)
        
        save_fig(figId,ext=ext,png=png)

    if PLOT['skewer']: # - - - -  plot skewer - - - -
        jx_hr,jy_hr=auxD['hr']['zmax_xyz'][:2]
        ncol,nrow=1,1; figId=5
        plt.figure(figId,facecolor='white', figsize=(15,4))
        ax=plt.subplot(nrow,ncol,1)
        for i,kr in  enumerate(fL[1:]):
            data=fieldD['rho+1'][kr][jy_hr]
            binX=np.arange(data.shape[0])
            hcol=plDD['hcol'][kr]
            ax.step(binX,data,where='post',label=kr,color=hcol,lw=1)
        ax.axvline(jx_hr,linewidth=1.,color='m',linestyle='--')
        ax.grid()
        ax.set_yscale('log')
        tit='%s  idx=%d  jy_hr=%d'%(kr,args.index,jy_hr)
        ax.set(title=tit, ylabel='1+rho')
        ax.legend(loc='best')

        save_fig(figId,ext=ext,png=png)
        
    if PLOT['rho+power']: 
        ncol,nrow=2,2; figId=6
        plt.figure(figId,facecolor='white', figsize=(10,8))

        # - - - -  density histo - - - - 
        ax=plt.subplot(nrow,ncol,1)
        for i,kr in  enumerate(fL):
            img=fieldD['ln rho+1'][kr]
            hcol=plDD['hcol'][kr]
            x,y=auxD[kr]['density']
            ax.step(x,y,where='post',label=kr,color=hcol)
        ax.set_yscale('log')
        ax.grid()
        tit='Density,  image idx=%d '%(args.index)
        ax.set(title=tit, xlabel=r'$ln(1+\rho)$',ylabel='num bins')
        ax.legend(loc='best', title='img type')

        if 1: # relative density
            ax=plt.subplot(nrow,ncol,3)
            x,y_hr=auxD['hr']['density']
            _,y_sr=auxD['sr']['density']
            _,y_ilr=auxD['ilr']['density']
            
            s2h=y_sr/y_hr
            il2h=y_ilr/y_hr
            ax.step(x,il2h,where='post',color=plDD['hcol']['ilr'],label='ILR/HR')           
            ax.step(x,s2h,where='post',color=plDD['hcol']['sr'],label='SR/HR')
            ax.grid()
            tit='Relative density,  image idx=%d '%(args.index)
            ax.set(title=tit, xlabel=r'$ln(1+\rho)$',ylabel='SR / HR')
            ax.axhline(1.,linewidth=1., linestyle='--', color='k')
            ax.set_ylim(0.4,1.6)
            ax.legend(loc='best')
            #print('xx',x)
            #print('hr',y_hr)
            #print('sr',y_sr)
        
        # .......power spectrum
        ax=plt.subplot(nrow,ncol,2)
        for i,kr in  enumerate(fL):
            kphys,P=auxD[kr]['power']
            hcol=plDD['hcol'][kr]
            ax.step(kphys,P ,where='post',label=kr,color=hcol)
        tit=r'Power Spectrum, FFT($\rho+1$), image idx=%d '%(args.index)
        ax.set(title=tit, xlabel='wavenumber (1/Mpc)',ylabel='P(k)')
        ax.legend(loc='best',title='img type')
        ax.grid()
        #ax.set_xscale('log');
        ax.set_yscale('log')
        y1=np.min(auxD['hr']['power'][1])  # otherwise ilr-filed blows-up the scale
        ax.set_ylim( y1/10.,)
        
        if 1: # relative power spec
            ax=plt.subplot(nrow,ncol,4)
            kphys1,y_hr=auxD['hr']['power']
            _,y_sr=auxD['sr']['power']
            _,y_ilr=auxD['ilr']['power']
            kphys2,y_lr=auxD['lr']['power']
            s2h=y_sr/y_hr
            il2h=y_ilr/y_hr
            n2=kphys2.shape[0] # LR data have less wavelengths
            l2h=y_lr/y_hr[:n2]
            ax.step(kphys2,l2h,where='post',color=plDD['hcol']['lr'],label='LR/HR') 
            ax.step(kphys,il2h,where='post',color=plDD['hcol']['ilr'],label='ILR/HR')
            ax.step(kphys,s2h,where='post',color=plDD['hcol']['sr'],label='SR/HR')
            ax.grid()
            tit='Relative power P(k),  image idx=%d '%(args.index)
            ax.set(title=tit, xlabel='wavenumber (1/Mpc)',ylabel='relative P(k)')
            #ax.set_xscale('log')
            ax.legend(loc='best')
            ax.axhline(1.,linewidth=1., linestyle='--', color='k')
            ax.set_ylim(0.4,1.6)

        save_fig(figId,ext=ext,png=png)

    plt.show()
