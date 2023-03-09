#!/usr/bin/env python3
'''

Plots x-slice at the same depth thrugh 6 cubes 

'''

import scipy.stats as stats
from toolbox.Util_H5io4 import read4_data_hdf5, write4_data_hdf5
import numpy as np
import argparse,os,time
from pprint import pprint
from numpy.random import default_rng
from toolbox.Util_Cosmo2d import  powerSpect_2Dfield_numpy

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "-i","--lrIndex", default=80,type=int,help="2D slice index in HR image")

    parser.add_argument( "-X","--noXterm", action='store_true', default=False, help="disable X-term for batch mode")
    parser.add_argument("--dataPath",default='/global/homes/b/balewski/prje/superRes-Nyx2022a-flux/tripack_cubes',help="data location")
    parser.add_argument("--outPath", default='out/',help="output path for plots and tables")
    parser.add_argument("--triName",default='cube_82337823',help="[.tripack.h5] file name")
    parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abcd-string listing shown plots")

    args = parser.parse_args()
    args.prjName='calib_'+args.triName
    args.showPlots=''.join(args.showPlots)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    if not os.path.exists(args.outPath):
        os.makedirs(args.outPath);   print('M: created',args.outPath)
    return args


from toolbox.Plotter_Backbone import Plotter_Backbone


#............................
#............................
#............................
class Plotter(Plotter_Backbone):
    def __init__(self, args,inpMD,sumRec=None):
        Plotter_Backbone.__init__(self,args)
        self.md=inpMD

#...!...!..................
    def input_images(self,imgFlux,imgFft,figId=1):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(13,5))
        ncol,nrow=2,1
        plmd=self.md['plot']
        pprint(plmd)
        xLab,yLab=plmd['2d-axis']
        k=0
        cmap='Blues'
        #
        
        ax=self.plt.subplot(nrow,ncol,1)
        img=imgFlux
        name=cubeN
        vmin,vmax,vavr=np.min(img),np.max(img),np.mean(img)
        print('2D min:max:avr',vmin,vmax,vavr,name)
            
        tit='%s  %s'%(name,str(img.shape))           
        pim=ax.imshow(img.T,origin='lower',cmap=cmap) 

        # Create colorbar
        cbar = ax.figure.colorbar(pim, ax=ax)            
        ax.set(title=tit, xlabel=xLab, ylabel=yLab)

        # tmp
        tit='log(FFT  ampl^2)  %s'%(str(imgFft.shape))           
        ax=self.plt.subplot(nrow,ncol,2)
        cmap ='Paired' # for FFT
        pim=ax.imshow(imgFft.T,origin='lower',cmap=cmap)
        # Create colorbar
        cbar = ax.figure.colorbar(pim, ax=ax)                    
        ax.set(title=tit, xlabel=xLab, ylabel='k(z*)')
        
#...!...!..................
    def skewer_ana(self,anaD,figId=1,isFft=False):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,6))
        ncol,nrow=2,2
        plmd=self.md['plot']
        pprint(plmd)
        _,xLab=plmd['2d-axis']
        yLab='flux'
        if isFft:
            yLab='log FFT'
            xLab='k(z*)'
        #...  samples
        ax=self.plt.subplot(nrow,ncol,1)
        sampleV=anaD['sampleV']
        for ske in sampleV:  ax.plot(ske,ds='steps-mid')
        tit='skewers sample'#%s  '%(name)
        ax.set(title=tit, xlabel=xLab, ylabel=yLab)


        #...  differential samples
        ax=self.plt.subplot(nrow,ncol,2)
        sampleV=anaD['sampleD']
        for ske in sampleV:  ax.plot(ske,ds='steps-mid')
        tit='2 skewers difference'
        ax.set(title=tit, xlabel=xLab, ylabel='delta '+yLab)
        
        #...  std of differential
        ax=self.plt.subplot(nrow,ncol,3)
        stdV=anaD['std']
        ax.plot(stdV,ds='steps-mid')
        ax.set( xlabel=xLab, ylabel='std dev (del1)')
        tit=' std dev of delat flux'
        ax.set(title=tit, xlabel=xLab, ylabel='std skewers diff '+yLab)
        
       
#............................
#............................
#............................
       
#...!...!..................
def ana_skewers(square,isFft=False):  # 2nd dim is preserved - it should be Z
    print('ASK:',square.shape)
    assert square.ndim==2
    nske=square.shape[0]
    
    #.... produce sample sqewers
    idxL=rng.choice(nske-1, size=10, replace=False)
    print('my idxL',idxL)
    skeV=square[idxL]
    if isFft:  # do ratios
        skeD=square[idxL]/square[idxL+1]
        squD=square[::2] /square[1::2] # difference shifted by 1
    else: #do difference
        skeD=square[idxL]-square[idxL+1]
        squD=square[::2] -square[1::2] # difference shifted by 1
    
    avrV=np.mean(squD, axis=0)  # preserve last axis
    stdV=np.std(squD, axis=0).astype(np.float32)
    #print('zzzz',squD.shape,avrV.shape)

    print('avrV : mean',avrV.mean(),'std:',avrV.std())
    print('stdV : std ',stdV.mean(),'std:',stdV.std())
    outD={'sampleV':skeV,'sampleD':skeD,'avr':avrV,'std':stdV}
    return outD

#...!...!..................
def compute_fft_plane(cube):
    nx=cube.shape[0]
    cell_size=triMD['cell_size']['HR']
    logfftSqr=np.zeros((nx,nx//2))  # output storage
    #print('C:logfftSqr',logfftSqr.shape,nx); ok0
    for i in range(nx):
        square=cube[i]
        kphys,kbins,P,fftA2=powerSpect_2Dfield_numpy(square,d=cell_size)
        #return np.log(fftA2) # single FFT splane, messes up whole code downstream, only for plotting
        logfftSqr[i]=np.log(fftA2[0]+1e-20)  # take just 1 skewer
        
    return   logfftSqr

#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__ == "__main__":
    args=get_parser()
    rng = default_rng()

    inpF=os.path.join(args.dataPath,args.triName+'.tripack.h5')
    triD,triMD=read4_data_hdf5(inpF)

    pprint(triMD)
    print(triD.keys())

    cubeN='flux_HR_z3'
    cube=triD[cubeN]
    sizeD=triMD['cube_bin']
    vmin,vmax,vavr=np.min(cube),np.max(cube),np.mean(cube)
    print('3D min:max:avr',vmin,vmax,vavr,cubeN)
    
    #.......  select& process 2d squares
    big2D={}
    bigD={}  # density
    bigP={}  # power spectrum
    plmd={}
                     
    ix=args.lrIndex
   
    sliN=cubeN+'-x%d'%ix
    plmd[sliN]=ix
    plmd['2d-axis']=['y','z*']
    square=cube[ix]        
    big2D[sliN]=square
    print('M:  I2D:',sliN,big2D[sliN].shape,'avr:',np.mean(big2D[sliN]))
    bigD=ana_skewers(square)
    logfftSqr=compute_fft_plane(cube)
    print('M:logfftSqr',logfftSqr.shape);  assert logfftSqr.shape==(512,256)
    bigP=ana_skewers(logfftSqr)


    #...... construct loss weight functions for flux(z*) and fft(flux)(k(z*)
    normD={}
    normD['std diff log fft']=bigP['std']
    normD['std diff flux']=bigD['std']
    # update MD
    triMD['comment']='x-axis: log_fft(k(z*)), flux(z*)'
    outF=os.path.join(args.outPath,args.triName+'.normFlux.h5')
    write4_data_hdf5(normD,outF,metaD=triMD)
    
    # ...... only plotting ........
    plot=Plotter(args,triMD)
    triMD['plot']=plmd
    
    if 'a' in args.showPlots:
        plot.input_images(square,logfftSqr)

    if 'b' in args.showPlots:
        plot.skewer_ana(bigD)

    if 'c' in args.showPlots:
        plot.skewer_ana(bigP,isFft=True)


    plot.display_all()
   
