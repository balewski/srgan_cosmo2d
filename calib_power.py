#!/usr/bin/env python3
'''
  calibrate power spectrum and density

'''
import sys,os
from toolbox.Util_H5io3 import  read3_data_hdf5
from toolbox.Util_IOfunc import write_yaml
import numpy as np
import argparse,os
import scipy.stats as stats
from pprint import pprint

from  scipy import signal

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "-X","--noXterm", action='store_true', default=False, help="disable X-term for batch mode")
    parser.add_argument("-e","--expName",default="exp036",help="experiment predictions")
    parser.add_argument("--outPath", default='out/',help="output path for plots and tables")

    args = parser.parse_args()
    args.expPath='/global/homes/b/balewski/prje/tmp_NyxHydro4k/results/test/'+args.expName
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
     
     figName=args.outPath+'%s_f%d'%(ext,fid)
     if png: figName+='.png'
     else: figName+='.pdf'
     print('Graphics saving to ',figName)
     plt.savefig(figName)

#...!...!..................
def median_conf_1D(data,p=0.68):
    # returns : m, m-std, m+std
    assert data.ndim==1
    data = np.sort(data)
    N = data.shape[0]
    delN=int(N*p)
    lowCount=(N-delN)//2
    upCount =(N+delN)//2
    #print('MED:idx:', lowCount, upCount, N)
    #print('data sorted', data)
    return  data[N // 2],data[lowCount], data[upCount]

#...!...!..................
def median_conf_V(data,p=0.68):  # vectorized version
    # computes median vs. axis=0, independent sorting of every 'other' bin
    # returns : axis=0: m, m-std, m+std; other axis 'as-is'
    sdata=np.sort(data,axis=0)
    N = data.shape[0]
    delN=int(N*p)
    lowCount=(N-delN)//2
    upCount =(N+delN)//2
    #print('MED:idx:', lowCount, upCount, N)
    out=[sdata[N // 2],sdata[lowCount], sdata[upCount]]
    return  np.array(out)

#...!...!..................
def post_process_srgan2D_fileds(fieldD,metaD):
    print('PPF:start, metaD:'); pprint(metaD)
    #  per field analysis
    
    for kr in fL:
        data=fieldD['rho+1'][kr]  # density, keep '+1'  
        #print('data %s %s '%(kr,str(data.shape)))
        jy,jx,zmax=max_2d_index(data)
        print(jy,jx,kr,'max:',zmax)
        metaD[kr]['zmax_xyz']=[jx,jy,zmax]
        img=np.log(data)  # for plotting and density histo
        
        fieldD['ln rho+1'][kr]=img
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
    inpF=args.expPath+'/pred.h5'
    fieldD,expMD=read3_data_hdf5(inpF)
    print('expMD:',expMD)

    # assembly meta data
    space_step=0.02441  # Mpc/h
    space_bins=fieldD['hr'].shape[0]
    upscale=4

    fL=['lr','ilr','sr','hr']
    metaD={ }
    for kr in fL:
        metaD[kr]={'space_bins':space_bins}
        metaD[kr]['space_step']=space_step
        if kr=='lr':
            metaD[kr]['space_bins']//=upscale
            metaD[kr]['space_step']*=upscale
        
    #.... recover  data
    kr='hr'  # HR squares
    dataA=fieldD[kr]
    nSamp=dataA.shape[0]
    powerL=[]
    for i in range(nSamp):
        kphys,P=power_2Dcloud(dataA[i],d=metaD[kr]['space_step'])
        kidx=np.arange(kphys.shape[0])+1
        powerL.append(P)

    powerA=np.array(powerL)
    imgA=np.log(dataA)
    
    powerL=[]
    for j in range(0,nSamp-1,2):
        powerL.append(powerA[j]/powerA[j+1])
    powerR=np.array(powerL)  # ratio of power spectrum

    # compute median +/- std for power spectrum for each K-index
    powMed=median_conf_V(powerA)
    powRMed=median_conf_V(powerR)
    
    print('M:pm',powMed.shape)
    # smooth it
    for i in range(3):
        powMed[i]=signal.savgol_filter(powMed[i], window_length=11, polyorder=2, deriv=0)
        powRMed[i]=signal.savgol_filter(powRMed[i], window_length=11, polyorder=2, deriv=0)
    
    # save it
    outD={'k_wave_number':kidx.tolist(), 'k_wave_length':kphys.tolist(), 'k_wave_length_unit':'1/Mpc'}
    for x in ['h5name', 'cube_name', 'hr_size']:
        outD[x]=expMD[x]
    outD['abs_P']={'num_sampl':int(powerA.shape[0]), 'median':powMed[0].tolist(),
                   'med-std':powMed[1].tolist(), 'med+std':powMed[2].tolist() }
    outD['rel_P']={'num_sampl':int(powerR.shape[0]), 'median':powRMed[0].tolist(),
                   'med-std':powRMed[1].tolist(), 'med+std':powRMed[2].tolist() }

    outF=args.outPath+'/power_calib.yaml'
    write_yaml(outD,outF)
    
    # print it
    pow1=np.copy(powMed)
    pow1[1]/=pow1[0]
    pow1[2]/=pow1[0]
    
    pow2=np.copy(powRMed)
    
    for i in range(0,powMed.shape[1],-10):
        #print('%3d P=%.1g %.3f %.3f  prod=%.3f'%(i,pow1[0,i],pow1[1,i],pow1[2,i],pow1[1,i]*pow1[2,i]))
        print('%3d rP=%.3f %.3f %.3f  prod=%.3f'%(i,pow2[0,i],pow2[1,i],pow2[2,i],pow2[1,i]*pow2[2,i]))

        
    # - - - - - Plotting - - - - - 
    plDD={}
    ncol,nrow=2,1; figId=6
        
    if 1:
        plt.figure(figId,facecolor='white', figsize=(12,6))
        # - - - -  power 
        ax=plt.subplot(nrow,ncol,1)
        
        for i in range(nSamp):
            P=powerA[i]
            ax.step(kidx,P, linewidth=1. )
        tit=r'%s  Power Spectrum, FFT($\rho+1$), nSamp=%d'%(kr,nSamp)
        ax.set(title=tit, xlabel='wavenumber index',ylabel='P(k)')

        ax.plot(kidx,powMed[2],linewidth=3,color='k',linestyle='--',label='m+std')
        ax.plot(kidx,powMed[0],linewidth=3,color='k',label='median')
        ax.plot(kidx,powMed[1],linewidth=3,color='k',linestyle='--',label='m-std')
        ax.grid()
        ax.set_yscale('log')
        ax.legend(loc='best', title='summary stats')
        ax.set_ylim(1e8,6e12)

  
        ax=plt.subplot(nrow,ncol,2)
        kwave=np.arange(kphys.shape[0])+1
        nSamp=powerR.shape[0]
        for i in range(nSamp):
            P=powerR[i]
            ax.step(kidx,P )
        ax.plot(kidx,powRMed[2],linewidth=3,color='k',linestyle='--',label='m+std')
        ax.plot(kidx,powRMed[0],linewidth=3,color='k',label='median')
        ax.plot(kidx,powRMed[1],linewidth=3,color='k',linestyle='--',label='m-std')

        tit=r'%s  Power Spectrum ratio, nSamp=%d'%(kr,nSamp)
        ax.set(title=tit, xlabel='wavenumber index',ylabel='P1(k)/P2(k)')
        
        ax.grid()
        ax.set_yscale('log')
        ax.set_ylim(2e-3,7e2)
        ax.legend(loc='best', title='summary stats')  
        save_fig(figId)
      
    if 0: 
        # - - - -  density
        ax=plt.subplot(nrow,ncol,1)
        binsX=np.linspace(-0.5,9,50)
        for i in range(nSamp):
            img=imgA[i]
            y, x, _ =ax.hist(img.flatten(),binsX,lw=1.2,  histtype='step')
        ax.set_yscale('log')
        ax.grid()
        tit='%s  Density'%(kr)
        ax.set(title=tit, xlabel=r'$ln(1+\rho)$',ylabel='num bins')
        

        
   

    plt.show()
