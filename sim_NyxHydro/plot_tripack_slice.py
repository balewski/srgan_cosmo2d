#!/usr/bin/env python3
'''

Plots x-slice at the same depth thrugh 6 cubes 

'''

import scipy.stats as stats
from toolbox.Util_H5io4 import read4_data_hdf5
import numpy as np
import argparse,os,time
from pprint import pprint
from numpy.random import default_rng

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "-i","--lrIndex", default=20,type=int,help="2D slice index in LR image")

    parser.add_argument( "-X","--noXterm", action='store_true', default=False, help="disable X-term for batch mode")
    parser.add_argument("--dataPath",default='/global/homes/b/balewski/prje/superRes-Nyx2022a-flux/tripack_cubes',help="data location")
    parser.add_argument("--outPath", default='out/',help="output path for plots and tables")
    parser.add_argument("--triName",default='cube_82337823',help="[.tripack.h5] file name")
    parser.add_argument("-p", "--showPlots",  default='ab', nargs='+',help="abcd-string listing shown plots")

    args = parser.parse_args()
    args.prjName='triPack'
    args.showPlots=''.join(args.showPlots)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    if not os.path.exists(args.outPath):
        os.makedirs(args.outPath);   print('M: created',args.outPath)
    return args


from toolbox.Plotter_Backbone import Plotter_Backbone


#............................
#............................
#............................
class Plotter_TriCube(Plotter_Backbone):
    def __init__(self, args,inpMD,sumRec=None):
        Plotter_Backbone.__init__(self,args)
        self.md=inpMD

#...!...!..................
    def input_images(self,big2D,figId=1):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(17,5))
        ncol,nrow=3,1
        plmd=self.md['plot']
        pprint(plmd)
        xLab,yLab=plmd['2d-axis']
        k=0
        cmap='Blues'
        #cmap ='Paired' # for FFT
        for name in big2D:                    
            ax=self.plt.subplot(nrow,ncol,1+k)
            k+=1
            
            img=big2D[name]
            vmin,vmax,vavr=np.min(img),np.max(img),np.mean(img)
            print('2D min:max:avr',vmin,vmax,vavr,name)
            
            tit='%s %s'%(name,str(img.shape))           
            pim=ax.imshow(img.T,origin='lower',cmap=cmap)  #,vmax=zmax)

            # Create colorbar
            cbar = ax.figure.colorbar(pim, ax=ax)            
            ax.set(title=tit, xlabel=xLab, ylabel=yLab)

#...!...!..................
    def skewer_ana(self,bigD,figId=1):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(15,9))
        ncol,nrow=3,3
        plmd=self.md['plot']
        pprint(plmd)
        _,xLab=plmd['2d-axis']

        k=0
        for name in bigD:                    
            ax1=self.plt.subplot(nrow,ncol,1+k)
            ax2=self.plt.subplot(nrow,ncol,1+k+ncol)
            ax3=self.plt.subplot(nrow,ncol,1+k+ncol*2)
            k+=1
            anaD=bigD[name]
            sampleV=anaD['sampleV']
            sampleD=anaD['sampleD']
            stdV=anaD['std']
            yLab=name.split('_')[0]
            
            tit='%s  '%(name)
            for ske in sampleV:  ax1.plot(ske,ds='steps-mid')
            for ske in sampleD:  ax2.plot(ske,ds='steps-mid')
            
            ax1.set(title=tit, xlabel=xLab, ylabel=yLab)
            ax2.set(title=tit, xlabel=xLab, ylabel='del1 '+yLab)

            ax3.plot(stdV,ds='steps-mid')
            ax3.set( xlabel=xLab, ylabel='std dev (del1)')
           

#...!...!..................
def ana_skewers(square):  # 2nd dim is preserved - it should be Z
    print('ASK:',square.shape)
    assert square.ndim==2
    nske=square.shape[0]
    
    #.... produce sample sqewers
    idxL=rng.choice(nske-1, size=10, replace=False)
    print('my idxL',idxL)
    skeV=square[idxL]
    #do difference
    skeD=square[idxL]-square[idxL+1]
    squD=square[::2] -square[1::2] # difference shifted by 1
    
    avrV=np.mean(squD, axis=0)  # preserve last axis
    stdV=np.std(squD, axis=0)
    #print('zzzz',squD.shape,avrV.shape)

    print('avrV : mean',avrV.mean(),'std:',avrV.std())
    print('stdV : std ',stdV.mean(),'std:',stdV.std())
    outD={'sampleV':skeV,'sampleD':skeD,'avr':avrV,'std':stdV}
    return outD
    
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

    sizeD=triMD['cube_bin']
    binFact=sizeD['HR']//sizeD['LR']

    #.......  select matching 2d squares
    big2D={}
    bigD={}  # density
    plmd={}
    for x in triD:
        cube=triD[x]
        vmin,vmax,vavr=np.min(cube),np.max(cube),np.mean(cube)
        print('3D min:max:avr',vmin,vmax,vavr,x)
                     
        ix=args.lrIndex
        if 'HR' in x: ix*=binFact
           
        sliN=x+'-x%d'%ix
        #print(x,ix,binFact,sizeD['HR'],sizeD['LR'])
        plmd[sliN]=ix
        plmd['2d-axis']=['y','z']
        square=cube[ix]        
        big2D[sliN]=square
        print('M:  I2D:',sliN,big2D[sliN].shape,'avr:',np.mean(big2D[sliN]))
        bigD[sliN+'skewerFlux']=ana_skewers(square)   
                
    plot=Plotter_TriCube(args,triMD)


    triMD['plot']=plmd
    
    if 'a' in args.showPlots:
        plot.input_images(big2D)

    if 'b' in args.showPlots:
        plot.skewer_ana(bigD)

 

    plot.display_all()
   
