import numpy as np
from scipy.interpolate import RegularGridInterpolator

#...!...!..................
def prep_fieldMD(inpMD,trainPar):
    # assembly meta data for FFT
    
    space_step=inpMD['cell_size']
    space_bins=trainPar['hr_size']
    upscale=trainPar['upscale_factor']
    fieldMD={'space_step_unit':'1/Mpc','upscale_factor':upscale}
    kr='hr'
    fieldMD[kr]={'space_bins':space_bins}
    fieldMD[kr]['space_step']=space_step
    fieldMD['sr']=fieldMD['hr']
    kr='lr'
    fieldMD[kr]={'space_bins':space_bins//upscale}
    fieldMD[kr]['space_step']=space_step*upscale
    outMD={'sim3d':inpMD,'field2d':fieldMD}
    #print('BB',outMD)
    return outMD
 

#...!...!..................
def interpolate_2Dimage(A,nZoom=2):    
    # it will work with mutli-channel array, format : W,H,C

    sizeX=A.shape[0]
    sizeY=A.shape[1]
    binX=np.linspace(0,sizeX-1,sizeX, endpoint=True)
    binY=np.linspace(0,sizeY-1,sizeY, endpoint=True)
    #print('A:',A,'nZoom=',nZoom,'\nbinX:',binX,'\nbinY:',binY)
    
    interpA=RegularGridInterpolator((binX,binY),A)

    # new bins for the target B
    binU=np.linspace(0,sizeX-1,sizeX*nZoom, endpoint=True)
    binV=np.linspace(0,sizeY-1,sizeY*nZoom, endpoint=True)
    #print('\nbinU',binU.shape, binU)
    #print('\nbinV',binV.shape, binV)
    bin1,bin2 = np.meshgrid(binV,binU)
    
    bin2D=np.stack((bin2, bin1), axis=-1)
    #print('\nbin2D',bin2D.shape, bin2D)
    B=interpA(bin2D) 
    return B,bin2D
    

#...!...!..................
def random_crop_WHC(image,tgt_size):
    org_size=image.shape[0]
    maxShift=org_size - tgt_size
    #print('RC2d:',org_size,tgt_size, maxShift)
    assert maxShift>=0
    if maxShift>0:  # crop image to smaller size
      ix=np.random.randint(maxShift)
      iy=np.random.randint(maxShift)
      image=image[ix:ix+tgt_size,iy:iy+tgt_size]
    return  np.reshape(image,(tgt_size,tgt_size,1))

#...!...!..................
def random_flip_rot_WHC(image,prob=0.5):
    if np.random.uniform() <prob:  image=np.flip(image,axis=0)
    if np.random.uniform() <prob:  image=np.flip(image,axis=1)
    if np.random.uniform() <prob:  image=np.swapaxes(image,0,1)
    return image

#...!...!....................
def rebin_WHC(cube,nReb):  # shape: WHC
    # rebin only over axis=0,1; retain the following axis unchanged
    assert cube.ndim==3
    # cube must be symmetric
    assert cube.shape[0]==cube.shape[1]
    nb=nReb # rebin factor
    a=cube.shape[0]
    assert a%nb==0
    b=a//nb
    nz=cube.shape[2]
    #print('aaa',a,nb,b)
    sh = b,nb,b,nb,nz
    C=cube.reshape(sh)
    #print('Csh',C.shape)
    D=np.sum(C,axis=-2)
    #print('Dsh',D.shape)
    E=np.sum(D,axis=-3)
    #print('Esh',E.shape)
    return E

#...!...!....................
