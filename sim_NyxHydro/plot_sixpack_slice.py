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
    parser.add_argument("-p", "--showPlots",  default='c', nargs='+',help="abcd-string listing shown plots")

    args = parser.parse_args()
    args.prjName='sixPack'
    args.showPlots=''.join(args.showPlots)
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
        self.md=inpMD

#...!...!..................
    def input_images(self,big2D,fieldN,figId=1):
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
    def one_image(self,big2D,name,figId=2):
        #print('kkk',big2D.keys())
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
                

#...!...!..................  fixed resolution
    def mass_correlationA(self,bigD,fieldN,figId=3):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,4.5))
        ncol,nrow=2,1
        k=0
        sumZ={}
        nameL=[]
        for zred in['z200','z5','z3']:
            name='%s_%s'%(fieldN,zred)
            data=bigD[name]
            print(name,data.shape)
            sumz=np.sum(data,axis=(1,2))
            sa=np.mean(sumz); sd=np.std(sumz)
            nameL.append([name,sa,sd,zred])
            sumZ[name]=sumz

        pprint(nameL)
        area=data.shape[1]**2

        #... make plots
        for k in range(2):
            ax=self.plt.subplot(nrow,ncol,1+k)

            name1,sa1,sd1,zred1=nameL[k]
            name2,sa2,sd2,zred2=nameL[2]
            V1=sumZ[name1]
            V2=sumZ[name2]

            corr=np.corrcoef(V1,V2)
            #print('aa',corr)
            txt='%s sum=%.1e +/- %.1f%c '%(zred1,sa1,100*sd1/sa1,37)
            txt+='\n%s sum=%.1e +/- %.1f%c '%(zred2,sa2,100*sd2/sa2,37) 
            txt+='\narea=%.1e  \ncorr=%.2f'%(area,corr[0,1])
            tit='2Dsum %s sixpack:%s'%(fieldN,self.md['ic_seed'])
            
            ax.plot(V1,V2)
            ax.set(xlabel=zred1,ylabel=zred2,title=tit)
            #ax.set_ylim(area*0.5, area*1.5)
            #ax.set_xlim(area*0.5, area*1.5)
         
            ax.text(0.05,0.80,txt,transform=ax.transAxes,color='g',fontsize=12)
            ax.grid()

#...!...!..................  between resolutions
    def mass_correlationB(self,bigD,fieldN,figId=3):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,4.5))
        ncol,nrow=2,1
        sizeD=self.md['cube_bin']
        binFact=sizeD['HR']//sizeD['LR']
        binOff=binFact//2  # hardcoded 1/2 of HR/LR step
        print('binfact', binFact,binOff)

        sumZ={}
        nameL=[]
        k=0
        for hlr in ['LR','HR']:
            for zred in['z200','z3']:                
                name='%s_%s_%s'%(fieldN,hlr,zred)
                data=bigD[name]
                if hlr=='HR': data=data[binOff::binFact]
                print(name,data.shape)
                sumz=np.sum(data,axis=(1,2))
                sa=np.mean(sumz); sd=np.std(sumz)
                nameL.append([name,sa,sd,hlr,zred])
                sumZ[name]=sumz

        area=data.shape[1]**2
        #... make plots
        for k in range(2):
            ax=self.plt.subplot(nrow,ncol,1+k)

            if k==0:
                name1,sa1,sd1,hlr1,zred1=nameL[1]
                name2,sa2,sd2,hlr2,zred2=nameL[3]
            if k==1:
                name1,sa1,sd1,hlr1,zred1=nameL[2]
                name2,sa2,sd2,hlr2,zred2=nameL[3]
            V1=sumZ[name1]
            V2=sumZ[name2]

            corr=np.corrcoef(V1,V2)
            #print('aa',corr)
            txt='%s sum=%.1e +/- %.1f%c '%(zred1,sa1,100*sd1/sa1,37)
            txt+='\n%s sum=%.1e +/- %.1f%c '%(zred2,sa2,100*sd2/sa2,37) 
            txt+='\narea=%.1e  \ncorr=%.2f'%(area,corr[0,1])
            tit='2Dsum %s sixpack:%s'%(fieldN,self.md['ic_seed'])
            
            ax.plot(V1,V2)
            ax.set(xlabel='%s_%s'%(hlr1,zred1),ylabel='%s_%s'%(hlr2,zred2),title=tit)
        
            ax.text(0.05,0.80,txt,transform=ax.transAxes,color='g',fontsize=12)
            ax.grid()
            #ax.set_xscale('log');ax.set_yscale('log')
            ax.locator_params(axis='both', nbins=4)
            #break
                

#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__ == "__main__":
    args=get_parser()
    fieldN="baryon_density"
    fieldN="dm_density"
    
    inpF=os.path.join(args.dataPath,args.sixName+'.sixpack.h5')
    sixD,sixMD=read4_data_hdf5(inpF,acceptFilter=[fieldN])

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
        
        print('I:',x,big2D[x].shape,'sum:',np.sum(big2D[x]))
        
    plot=Plotter_SixCube(args,sixMD)

    if 'a' in args.showPlots:
        plot.input_images(big2D,fieldN)

    if 'b' in args.showPlots:
        plot.one_image(big2D,fieldN+'_HR_z3')
    
    if 'c' in args.showPlots:
        plot.mass_correlationA(sixD,fieldN+'_LR')
    if 'd' in args.showPlots:
        plot.mass_correlationA(sixD,fieldN+'_HR')

    if 'e' in args.showPlots:
        plot.mass_correlationB(sixD,fieldN)

    plot.display_all()
   
