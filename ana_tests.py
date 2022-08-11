#!/usr/bin/env python3
'''
?????   discard ??
 Analysis of quality of SRGAN-2D images

 ./ana_sr2dX.py -e exp22 -s best595
'''
import sys,os
from toolbox.Util_H5io3 import  read3_data_hdf5
from toolbox.Util_Cosmo2d import powerSpect_2Dfield_numpy,density_2Dfield_numpy
import numpy as np
import argparse,os

from pprint import pprint

import torch  # testing PowerSepctrum
from toolbox.Util_Torch import transf_img2field_torch,transf_field2img_torch

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "-i","--index", default=3,type=int,help="image index")
    parser.add_argument( "-X","--noXterm", action='store_true', default=False, help="disable X-term for batch mode")
    parser.add_argument("-e","--expName",default="exp036",help="experiment predictions")
    parser.add_argument("-s","--genSol",default="last",help="generator solution")
    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")
    parser.add_argument("-d","--dataPath",  default='/global/homes/b/balewski/prje/tmp_NyxHydro4kE/',help='data location w/o expName')

 
    args = parser.parse_args()
    #args.expPath=os.path.join('/global/homes/b/balewski/prje/tmp_NyxHydro4kB/manual',args.expName)
    
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


def f_get_rad(img):
    ''' Get the radial tensor for use in f_torch_get_azimuthalAverage '''

    height,width=img.shape[-2:]
    # Create a grid of points with x and y coordinates
    y, x = np.indices([height,width])

    center=[]
    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    # Get the radial coordinate for every grid point. Array has the shape of image
    r = torch.tensor(np.hypot(x - center[0], y - center[1]))

    # Get sorted radii
    ind = torch.argsort(torch.reshape(r, (-1,)))

    return r.detach(),ind.detach()

#...!...!..................
def f_torch_get_azimuthalAverage(image,r,ind):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).
    source: https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
    """

#     height, width = image.shape
#     # Create a grid of points with x and y coordinates
#     y, x = np.indices([height,width])

#     if not center:
#         center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

#     # Get the radial coordinate for every grid point. Array has the shape of image
#     r = torch.tensor(np.hypot(x - center[0], y - center[1]))

#     # Get sorted radii
#     ind = torch.argsort(torch.reshape(r, (-1,)))

    r_sorted = torch.gather(torch.reshape(r, ( -1,)),0, ind)
    i_sorted = torch.gather(torch.reshape(image, ( -1,)),0, ind)

    # Get the integer part of the radii (bin size = 1)
    r_int=r_sorted.to(torch.int32)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = torch.reshape(torch.where(deltar)[0], (-1,))    # location of changes in radius
    nr = (rind[1:] - rind[:-1]).type(torch.float)       # number of radius bin

    # Cumulative sum to figure out sums for each radius bin

    csum = torch.cumsum(i_sorted, axis=-1)
    tbin = torch.gather(csum, 0, rind[1:]) - torch.gather(csum, 0, rind[:-1])
    radial_prof = tbin / nr  # disable to make it identical w/ my nupy-code

    return radial_prof,nr

#...!...!..................
def f_torch_compute_spectrum(arr,r,ind):

    fourier_image = torch.fft.fft2(arr)
    fourier_image=torch.fft.fftshift(fourier_image)
    print('FTCS:arr',arr.shape,'fft:',fourier_image.shape)
    fourier_amplitudes2= torch.abs(fourier_image)**2

    z1,nr=f_torch_get_azimuthalAverage(fourier_amplitudes2,r,ind)     ## Compute radial profile
    print('FTCS:z1',z1.shape)
    return z1,nr

#...!...!..................
def powerSpect_2Dimage_torch(image): # use torch only computation
    #print('FGR: hrImgB',type(lrImgB),len(lrImgB),len(hrImgB),hrImgB.shape)
    # FGR: hrImgB <class 'torch.Tensor'> 16 16 torch.Size([16, 1, 128, 128])

    pwspD={}
    # Precompute radial coordinates in numpy
    r,ind=f_get_rad(image)#.numpy())
    print('FGR:r:', type(r),r.shape,'ind:', type(ind),ind.shape)

    field=transf_img2field_torch(image)
    z1,nr=f_torch_compute_spectrum(field,r,ind)
    numK=image.shape[0]//2
    print('dump z1',z1.shape,numK)
    #print('dump nr',nr)
    pwspD['spec_r']=r
    pwspD['spec_ind']=ind
    Abins=z1[:numK]
    kbins = np.arange(0.5,Abins.shape[0], 1.)
    
    return kbins,Abins


#...!...!..................
def post_process_srgan2D_fileds(fieldD,metaD):
    print('PPF:start, metaD:'); pprint(metaD)
    #  per field analysis
    
    for kr in fL:
        data=fieldD['rho+1'][kr]  # density, keep '+1'  
        print('data %s %s '%(kr,str(data.shape)))
        jy,jx,zmax=max_2d_index(data)
        print(jy,jx,kr,'max:',zmax)
        metaD[kr]['zmax_xyz']=[jx,jy,zmax]
        img=np.log(data)  # for plotting and density histo
        kphys,kbins,P,fftA2=powerSpect_2Dfield_numpy(data,d=metaD[kr]['space_step'])
        fieldD['ln rho+1'][kr]=img
        metaD[kr]['power']=[kphys,kbins,P]

        kbins2,P2=powerSpect_2Dimage_torch(torch.from_numpy(img))
        #print('tt',kbins2.shape,P2.shape)
        metaD[kr]['power2']=[kbins2,P2]
        
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
    inpF=os.path.join(args.dataPath,args.expName,'pred-test-%s.h5'%args.genSol)
   
    bigD,predMD=read3_data_hdf5(inpF)
    print('predMD:');pprint(predMD)
        
    # stage data
    fL=['sr','hr']
    fieldD={'ln rho+1':{}}
    fieldD['rho+1']={ xr:bigD[xr][args.index][0] for xr in fL}  # skip C-index
    auxD={x:predMD['field2d'][x]  for x in fL} 
    
    post_process_srgan2D_fileds(fieldD,auxD)
    

    # - - - - - Plotting - - - - - 
    plDD={}
    plDD['hcol']={'lr':'g','ilr':'orange','sr':'C3','hr':'k'}
    png=1
    ext='img%d'%args.index
    
    if 0: # - - - - -   power spectrum
        ncol,nrow=2,1; figId=6
        plt.figure(figId,facecolor='white', figsize=(10,5))
        
        # .......power spectrum numpy
        ax=plt.subplot(nrow,ncol,1)
        for i,kr in  enumerate(fL):
            kphys,kbins,P=auxD[kr]['power']
            hcol=plDD['hcol'][kr]
            #print('pp',i,kr,kbins[:5],P[:5])
            ax.step(kbins,P ,label=kr,color=hcol)
            #if i>0: break
        tit=r'Power Spectrum, Numpy, image idx=%d '%(args.index)
        #ax.set(title=tit, xlabel='wavenumber (1/Mpc)',ylabel='P(k)')
        ax.set(title=tit, xlabel='wave index',ylabel='P(k)')
        ax.legend(loc='best',title='img type')
        ax.grid()
        ax.set_yscale('log')

    if 0:       # .......power spectrum torch
        ax=plt.subplot(nrow,ncol,2)
        for i,kr in  enumerate(fL):
            kbins,P=auxD[kr]['power2']
            hcol=plDD['hcol'][kr]
            ax.step(kbins,P ,label=kr,color=hcol)
        tit=r'Power Spectrum,PyTorch, image idx=%d '%(args.index)
        ax.set(title=tit, xlabel='wave index',ylabel='P(k)')
        ax.legend(loc='best',title='img type')
        ax.grid()        
        ax.set_yscale('log')
        
        save_fig(figId,ext=ext,png=png)

    if 1: # - - - - -    density
        ncol,nrow=2,1; figId=7
        plt.figure(figId,facecolor='white', figsize=(10,5))
        binsX=np.linspace(-0.,9,50)
        #binsX=50
        
        # --- use mpl-histo
        ax=plt.subplot(nrow,ncol,1)
        for i,kr in  enumerate(fL):
            img=fieldD['ln rho+1'][kr]
            hcol=plDD['hcol'][kr]
            y, x, _ =ax.hist(img.flatten(),binsX,label=kr,edgecolor=hcol,lw=1.2,  histtype='step')
        ax.set_yscale('log')
        ax.grid()
        tit='Density, mpl,  image idx=%d '%(args.index)
        ax.set(title=tit, xlabel=r'$ln(1+\rho)$',ylabel='num bins')
        ax.legend(loc='best', title='img type')

    if 1:   # --- use numpy-histo
        ax=plt.subplot(nrow,ncol,2)
        for i,kr in  enumerate(fL):
            img=fieldD['ln rho+1'][kr]
            hcol=plDD['hcol'][kr]
            x,y=density_2Dfield_numpy(img,9.)
            ax.step(x,y,where='post',label=kr,color=hcol)
            
            #print('dens',kr,np.sum(y),np.sum(x))
        ax.set_yscale('log')
        ax.grid()
        tit='Density, numpy, image idx=%d '%(args.index)
        ax.set(title=tit, xlabel=r'$ln(1+\rho)$',ylabel='num bins')
        ax.legend(loc='best', title='img type')
 
        
    plt.show()
