#!/usr/bin/env python3
'''

Plots x-slice at the same depth thrugh 6 cubes 

'''

import scipy.stats as stats
from toolbox.Util_H5io4 import read4_data_hdf5
import numpy as np
import argparse,os,time
from pprint import pprint

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "-i","--lrIndex", default=20,type=int,help="2D slice index in LR image")

    parser.add_argument( "-X","--noXterm", action='store_true', default=False, help="disable X-term for batch mode")
    parser.add_argument("--dataPath",default='/global/homes/b/balewski/prje/superRes-Nyx2022a/sixpack_cubes',help="data location")
    parser.add_argument("--outPath", default='out/',help="output path for plots and tables")
    parser.add_argument("--sixName",default='cube_1181930045',help="[.sixpack.h5] file name")
 
    args = parser.parse_args()
    args.prjName='sixPack'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    if not os.path.exists(args.outPath):
        os.makedirs(args.outPath);   print('M: created',args.outPath)
    return args


from toolbox.Plotter_Backbone import Plotter_Backbone


#............................
#............................
#............................
class Plotter_SixCube(Plotter_Backbone):
    def __init__(self, args,inpMD,sumRec=None):
        Plotter_Backbone.__init__(self,args)

#...!...!..................
    def input_images(self,big2D,fieldN,figId=4):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(14,9))
        ncol,nrow=3,2
        k=0
        for hlr in ['LR','HR']:
            for zred in['z200','z5','z3']:                
                ax=self.plt.subplot(nrow,ncol,1+k)
                k+=1
                ax.set_aspect(1.)
                name='%s_%s_%s'%(fieldN,hlr,zred)
                data=big2D[name]
                img=np.log(1+data)
                tit='%s %d %s'%(name,args.lrIndex,str(img.shape))
                zmax=None
                if zred!='z200': zmax=1.5 
                pim=ax.imshow(img,origin='lower',vmax=zmax)

                # Create colorbar
                if zred=='z200': cbar = ax.figure.colorbar(pim, ax=ax)#, **cbar_kw)
                #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
                ax.set_title(tit)
                


#...!...!..................
    def one_image(self,big2D,name,figId=5):
        print('kkk',big2D.keys())
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(11,9))
        ncol,nrow=1,1
        ax=self.plt.subplot(nrow,ncol,1)
        ax.set_aspect(1.)
        data=big2D[name]
        img=np.log(1+data)
        tit='%s %d %s'%(name,args.lrIndex,str(img.shape))
        zmax=None
        zmax=1.5 
        pim=ax.imshow(img,origin='lower',vmax=zmax)

        # Create colorbar
        cbar = ax.figure.colorbar(pim, ax=ax)#, **cbar_kw)
        #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        ax.set_title(tit)
                

     
#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__ == "__main__":
    args=get_parser()
   
    inpF=os.path.join(args.dataPath,args.sixName+'.sixpack.h5')
    sixD,sixMD=read4_data_hdf5(inpF)

    fieldN="baryon_density"
    #fieldN="dm_density"
    
    pprint(sixMD)
    print(sixD.keys())

    sizeD=sixMD['cube_bin']
    binFact=sizeD['HR']//sizeD['LR']
    big2D={}
    for x in sixD:
        if  fieldN not in x : continue
        ix=args.lrIndex
        if 'HR' in x: ix*=binFact
        #print(x,ix,binFact,sizeD['HR'],sizeD['LR'])
        big2D[x]=sixD[x][ix]
        
    plot=Plotter_SixCube(args,sixMD)
    plot.input_images(big2D,fieldN)
    plot.one_image(big2D,fieldN+'_HR_z3')
    
    plot.display_all()
   
