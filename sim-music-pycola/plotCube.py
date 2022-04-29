#!/usr/bin/env python3
'''
plot one cube

'''
import sys,os,time
from pprint import pprint
from toolbox.Util_H5io3 import  read3_data_hdf5

from toolbox.Plotter_Backbone import Plotter_Backbone
import numpy as np
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-X","--noXterm", action='store_true', default=False,
                         help="disable X-term for batch mode")

    parser.add_argument("-d", "--dataPath",  default='out/',help="scored data location")
    parser.add_argument("-s", "--showPlots",  default='ab',help="abc-string listing shown plots")

    parser.add_argument("-z", "--zRedShift",  default='49', type=int, help="(int) z red shift")
    
    parser.add_argument("--dataName",  default='univers0.music', help="[.dm.h5] desnity cube")
    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")

    args = parser.parse_args()
    args.formatVenue='prod'
    args.prjName=args.dataName

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#............................
#............................
#............................
class Plotter(Plotter_Backbone):
    def __init__(self,args):
        Plotter_Backbone.__init__(self,args)
        self.args=args


#...!...!..................
    def dm_cube_3d(self,X,figId=1):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,8))
        ncol,nrow=1,1

        X1d=X.flatten()
        r1=X1d.min(); r2=X1d.max(); rm=np.median(X1d)
        rp=np.percentile(X1d,99.9)
        print('dm_cube, shape=',X.shape,'min/max=%.1f %.1f , median=%.1f 1pk=%.1f'%(r1,r2,rm,rp))
        wthr=max(rp,3)
        print('zRed=%d, will apply wthr=%.1f'%(self.args.zRedShift,wthr),X.shape)
        j=0
        xs=np.zeros_like(X1d)
        ys=np.zeros_like(X1d)
        zs=np.zeros_like(X1d)
        t0=time.time()
        for i0 in range(X.shape[0]):
            for i1 in range(X.shape[1]):
                for i2 in range(X.shape[2]):
                    if cube[i0,i1,i2] <wthr: continue
                    xs[j]=i0
                    ys[j]=i1
                    zs[j]=i2
                    j+=1
        t1=time.time()
        print('wthr=',wthr,' nbin=',j,'elaT=%.1f min'%((t1-t0)/60.))
        ax=self.plt.subplot(nrow,ncol,1, projection='3d')    
        ax.scatter(xs[:j], ys[:j], zs[:j],alpha=0.8, s=0.4,c='r')
        ax.view_init(30, 120)
        tit='%s, zRed=%d, thres>=%d, voxels=%d,  maxMass=%d'%(self.args.dataName,self.args.zRedShift,wthr,j,r2)
        ax.set(title=tit,xlabel='bins',ylabel='bins')

        
#...!...!..................
    def dm_density_h(self,X,figId=2):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(5,4))
        ncol,nrow=1,1

        X1d=X.flatten()
        r1=X1d.min(); r2=X1d.max(); rm=np.median(X1d)
        rp=np.percentile(X1d,99.9)
        
        print('dm_dens, shape=',X1d.shape,'min/max=%.1f %.1f , median=%.1f proc90=%.1f'%(r1,r2,rm,rp))
        binsX=min(100,int(r2*1.05))

        ax=self.plt.subplot(nrow,ncol,1)
        (binCnt,_,_)=ax.hist(X1d,binsX,color='b')
        ax.set_yscale('log')
        ax.grid()
        tit='%s, zRed=%d, thres=%d,  maxMass=%d'%(self.args.dataName,self.args.zRedShift,rp,r2)
        ax.set(title=tit,xlabel='mass',ylabel='num voxels')
        ax.axvline(rp,color='m')
        
#...!...!..................
    def dm_slices_2d(self,X,figId=3):
        figId=self.smart_append(figId)
        nrow,ncol=1,3
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,4.5))
        from matplotlib.colors import LogNorm

        numBin=X.shape[0]
        N=nrow*ncol
        iSlice=numBin//2
        
        for i in range(3):
            ax = self.plt.subplot(nrow, ncol, i+1) 
            if i==0:
                img=X[iSlice]; dLab='XY'
            if i==1:
                img=X[:,iSlice,:]; dLab='XZ'
            if i==2:
                img=X[:,:,iSlice]; dLab='YZ'
            
            pos=ax.imshow(img, cmap='Blues',norm=LogNorm(vmin=1.))
            fig.colorbar(pos, ax=ax)
            ax.grid()
            tit='%s zRed=%d slice=%s'%(self.args.dataName,self.args.zRedShift,dLab)
            ax.set(title=tit,xlabel='bins',ylabel='bins')
            #ax.set_zscale('log')
   
            
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()

    inpF=os.path.join(args.dataPath,args.dataName+'.dm.h5')
    bigD,inpMD=read3_data_hdf5(inpF, verb=1)
    #pprint(inpMD)
    zrs='z%d'%args.zRedShift
    cube=bigD[zrs]
    
    # - - - - - PLOTTER - - - - -
    plot=Plotter(args)
    if 'a' in args.showPlots:   plot.dm_cube_3d(cube)
    if 'b' in args.showPlots:   plot.dm_density_h(cube)
    if 'c' in args.showPlots:   plot.dm_slices_2d(cube)
        
    plot.display_all('colaDM_'+zrs)
