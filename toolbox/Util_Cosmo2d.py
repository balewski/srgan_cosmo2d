import numpy as np
from scipy.interpolate import RegularGridInterpolator
import scipy.stats as stats
from pprint import pprint



#...!...!..................
def density_2Dfield_numpy(field1,maxY=9.,nbin=150,pad1=True): # ln_field1=ln(filed+1)  
    #print('rho2D: ln_field1',ln_field1.shape)
    vmin,vmax,vavr=np.min(field1),np.max(field1),np.mean(field1)
    #print('D2DN  vmin,vmax,vavr', vmin,vmax,vavr)
    binsX=np.linspace(0.,maxY,nbin,endpoint=False)
    y, x= np.histogram(field1, bins=binsX)  # will flatten input array
    if pad1: # to avoid NaN when computing ratio
        y[y==0.]=1.11111
        
    return x[:-1],y  # x contains begin of bins
    #  plot with:    ax.step(x[:-1],y,where='post') 
   
#...!...!..................
def powerSpect_2Dfield_numpy(field,d=1):  # d: Sample spacing (inverse of the sampling rate)
    #print('Pow2D: field',field.shape)
    npix = field.shape[0]
    assert npix == field.shape[1]
    assert npix%2==0  # for computation of kvals

    fourier_image = np.fft.fftn(field)
    fourier_amplitudes2= np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    amplitudes2 = fourier_amplitudes2.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    #kvals = 0.5 * (kbins[1:] + kbins[:-1])

    Abins, _, _ = stats.binned_statistic(knrm, amplitudes2,  statistic = "mean", bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    kvals=np.fft.fftfreq(npix, d=d)
    kphys=kvals[:npix//2]
    absFftA=fourier_amplitudes2[:npix//2,:npix//2]  # only lower-left quadrant 
    #print('dd',kphys.shape,kbins.shape)
    return kphys[1:],kbins[1:-1],Abins[1:], absFftA # k,P, skip 0-freq, k  units: 1/d


#...!...!..................
def XXinterpolate_2Dfield(A,nZoom=2):    # needed for some pprediction plots
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
def XXrandom_crop_WHC(image,tgt_size):
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
def random_flip_rot_WHC(image,rndV,prob=0.5): # format HWC, retain C-axis
    if rndV[0] <prob:  image=np.flip(image,axis=0)
    if rndV[1] <prob:  image=np.flip(image,axis=1)
    if rndV[2] <prob:  image=np.swapaxes(image,0,1)
    return image

#...!...!....................
def XXrebin_WHC(cube,nReb):  # shape: WHC
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
    med=data[N // 2]
    return  med,data[lowCount]-med, data[upCount]-med

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
def srgan2d_FOM1(densV,fftV):  # vectorized, version1 uses abs
    print('SFV: densV ',densV.shape,densV.dtype,' fftV:',fftV.shape,fftV.dtype)
    # use only median, skip up/low CL
    r=densV-1.
    f=fftV-1.
    #print('xx1',r.shape,r)
    r_fom=np.mean(np.abs(r))
    f_fom=np.mean(np.abs(f))
    fom=r_fom+f_fom
    fomD={'fom':fom,'r_fom':r_fom,'f_fom':f_fom,'name':'fom1','r_bins':r.shape[0],'f_bins':f.shape[0]}
    return fomD
