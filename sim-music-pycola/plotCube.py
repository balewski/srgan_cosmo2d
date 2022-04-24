#!/usr/bin/env python3
'''
plot one cube

'''
import sys,os
from pprint import pprint
from toolbox.Util_H5io3 import  read3_data_hdf5

from toolbox.Plotter_Backbone import Plotter_Backbone
import numpy as np
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-X","--noXterm", dest='noXterm',
        action='store_true', default=False,help="disable X-term for batch mode")

    parser.add_argument("-d", "--dataPath",  default='out/',help="scored data location")

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

#............................
    def dm_cube(self,X,idx=-1,tit='',figId=7):
        assert len(X) > idx # do not ask for a not existing frame

        frameDim=X.ndim
        print('xx',X.shape,frameDim)
        if frameDim==3: 
            nrow,ncol=2,2
            figSize=(10,9)
            '''
            cube4D=X[0,msidx]
            nTime=cube4D.shape[3]  
            sh1=tuple(cube4D.shape[-4:-1])          
            cubeV=np.split(cube4D,nTime,axis=-1)

            for iT in range(nTime):
                #print('res it=',iT,cubeV[iT].shape,sh1)
                cubeV[iT]=cubeV[iT].reshape(sh1)
            '''
        else:
            error12

        #print('plot input for trace idx=',idx,' frameDim=',frameDim, 'nTime=',nTime)
    
        fig=self.plt.figure(figId,facecolor='white', figsize=figSize)        
        figId=self.smart_append(figId)
        nrow,ncol=1,1
        nTime=1
        cubeV=[X]
        angle=120
        wmax=np.max(cubeV[0])
        #wthr=0.3*wmax
        wthr=3
        print('will apply wthr=',wthr,cubeV[0].shape)
        for iT in range(nTime):
            ax = self.plt.subplot(nrow, ncol, iT+1, projection='3d')  
            cube=cubeV[iT]
            wmin=np.min(cube)
            wmax=np.max(cube)
            wsum=np.sum(cube)
            print('cube wmin=',wmin, ' wmax=',wmax, 'w sum=',wsum,cube.shape) 
 
            xs=[]; ys=[];zs=[]
            for i0 in range(cube.shape[0]):
                for i1 in range(cube.shape[1]):
                    for i2 in range(cube.shape[2]):
                        if cube[i0,i1,i2] <wthr: continue
                        xs.append(i0)
                        ys.append(i1)
                        zs.append(i2)
            print('wthr=',wthr,' nbin=',len(xs),'iT=',iT)
            ax.scatter(xs, ys, zs,alpha=0.8, s=0.4,c='r')

            ax.view_init(30, angle)
            return
            if iT==0:
                tit=tit
                ptxtL=[ '%.2f'%x for x in Y[idx]]
                tit+=' U:'+', '.join(ptxtL)
                tit+=', thr=%d, msidx=%d'%(wthr,msidx)
            else:
                tit='time-bin=%d'%iT

            xtit='idx%d'%(idx) 
            ytit='sum=%.2e, max=%.2e'%(wmax,wsum)            
            ax.set(title=tit[:65], xlabel=xtit, ylabel=ytit)

        
#...!...!..................
    def XXXwaveArray(self,bigD,plDD,figId=5):
        figId=self.smart_append(figId)
        nrow,ncol=4,2; yIn=9
        #nrow,ncol=2,2; yIn=5
        fig=self.plt.figure(figId,facecolor='white', figsize=(14,yIn))
        
        timeV=bigD['time']
        ia=plDD['iStimAmpl']

        M=bigD['sweepCnt'][ia]
        spikeC=bigD['spikeCount'][ia]
        spikeT=bigD['spikeTrait'][ia]
        sweepT=bigD['sweepTrait'][ia]
                
        stimA=bigD['stim'][ia]*5 # better visibility
        waveA=bigD['waveform'][ia]
        
        #print('wdif0',waveA.shape,stimA.shape,len(spikeA[0]))
        idxL,idxR,idxS=plDD['idxLR']
        for n in range(idxL,idxR,idxS):
            if n>=M : continue
            j=n-idxL
            if j>=nrow*ncol: continue
            
            ax = self.plt.subplot(nrow,ncol,1+j)
            ks=spikeC[n]
            [sweepId, sweepTime, serialRes]=sweepT[n]  #*nSweeps
            
            ax.plot(timeV,stimA[n], 'C1',linewidth=0.5,label='stim')
            ax.plot(timeV,waveA[n], 'b',linewidth=0.7,label='soma AP')
            txt='numSpike=%d  sweep=%d  wallTime=%.1f min'%(ks,n,sweepTime/60.)
            ax.text(0.35,0.9,txt,transform=ax.transAxes,color='g')
            if 1: # show valid spikes
                spikes=spikeT[n][:ks]
                #print('spikes=',spikes)
                tPeak=spikes[:,0]
                yPeak=spikes[:,1]
                ax.plot(tPeak,yPeak,"*",color='m')
    
            if j==0:   ax.text(0.01,0.9,'cell='+plDD['shortName'],transform=ax.transAxes,color='m')
            if j==1:  ax.text(0.01,0.9,plDD['text1'],transform=ax.transAxes,color='m')


            ax.axhline(plDD['peak_thr'],color='m', linestyle='--',linewidth=0.5)
        
            yLab='AP (mV)'
            xLab='time (ms), '+plDD['shortName']+', n=%d'%n
            ax.set(xlabel=xLab,ylabel=yLab)
            ax.grid()
            if j==0: ax.legend(loc='best')

            if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))
            if 'amplLR' in plDD: ax.set_ylim(tuple(plDD['amplLR']))

            
#...!...!..................
    def score_stimAmpl(self,bigD,plDD,figId=6):
        figId=self.smart_append(figId)
        nrow,ncol=1,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,5))
        ax = self.plt.subplot(nrow,ncol,1)

        sweepC=bigD['sweepCnt']
        N=sweepC.shape[0] # num stim ampl
        #stimA=plDD['stimAmpl']
        scoreA=bigD['spikeCount'] ; yTit='spike count'
        stimAmplA=plDD['stimAmpl']
        #print('P:spikeCount',bigD['spikeCount'] )

        data=[]
        avrScore=[]; stdScore=[]
        for n in range(N): # loop over stim-ampl          
           k=sweepC[n]
           scoreV=scoreA[n][:k]
           data.append(scoreV)
           stimV=[stimAmplA[n]]*k
           avrScore.append(np.mean(scoreV))
           stdScore.append(np.std(scoreV))
           
        ax.violinplot(data,positions=stimAmplA,widths=0.01,showmeans=True)
        ax.plot(stimAmplA,avrScore)
        ax.axhline(0,color='g',linestyle='--',linewidth=0.5)
        ax.axhline(10,color='g',linestyle='--')
            
        tit1='experiment: %s'%plDD['shortName']
        ax.set(ylabel=yTit,xlabel='stim ampl (FS)',title=tit1)
        ax.legend(loc='best')
        ax.grid()

        return [stimAmplA,np.array(avrScore),np.array(stdScore),sweepC]

#...!...!..................
    def score_wallTime(self,bigD,plDD,figId=6):
        figId=self.smart_append(figId)
        nrow,ncol=1,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(6,5))
        ax = self.plt.subplot(nrow,ncol,1)

        sweepC=bigD['sweepCnt']
        traitA=bigD['sweepTrait']
        N=sweepC.shape[0] # num stim ampl
        stimA=plDD['stimAmpl']
        scoreA=bigD['spikeCount'] 
        stimAmplA=plDD['stimAmpl']
        #print('P:spikeCount',bigD['spikeCount'] )

        avrScore=[]; jj=0
        for n in range(N):  # loop over stim-ampl
           k=sweepC[n]
           traits=traitA[n][:k]  #[sweepId, sweepTime, serialRes]*nSweeps
           #print('q3',traits.shape,traits)
           wallTA=np.array(traits[:,1])/60.
           scoreV=scoreA[n][:k]
           if np.max(scoreV)<=0: continue
           dLab='ampl=%.2f'%(stimAmplA[n])
           dmk=self.mL7[jj%7]; jj+=1
           ax.plot(wallTA,scoreV,dmk+"-",label=dLab)
        ax.legend(bbox_to_anchor=(0., 1.06, 1., .106), loc=2,
                   ncol=4, mode="expand", borderaxespad=0.,
                  title=plDD['shortName']+', stim ampl (FS)' )
        
        yTit='spike count'
        ax.set(ylabel=yTit,xlabel='wall time (min)')
        ax.grid()
        ax.set_ylim(0,10.5)
        
#...!...!..................
    def spikes_survey2D(self,bigD,plDD,figId=6):
        figId=self.smart_append(figId)
        nrow,ncol=4,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,12))
        #fig.suptitle("Title for whole figure", fontsize=16)

        spikeC=bigD['spikeCount']
        spikeA=bigD['spikeTrait']
        traitA=bigD['sweepTrait']
        N=spikeC.shape[0] # num stim ampl
        
        
        # repack data for plotting
        tposA=[]; widthA=[]; amplA=[]; stimA=[]; swTimeA=[]; resA=[]
        for ia in range(N): # loop over stmAmpl
            if 'iStimAmpl' in plDD:
                if ia!=plDD['iStimAmpl']: continue
            #if n!=6: continue  # show only 1 stim-ampl
            stimAmpl=plDD['stimAmpl'][ia]
            #print('q2',traits)
            M=bigD['sweepCnt'][ia] # num of waveform/ampl
            
            for j in range(M): #loop over waveforms
                ks=spikeC[ia,j]
                #print('qwq',n,j,ks)
                if ks==0: continue
                
                traits=traitA[ia,j]
                #print('q3',traits)
                sweepId, sweepTime, serialRes=traits
                spikes=spikeA[ia,j][:ks] # use valid spikes
                for rec in spikes:
                    tPeak,yPeak,twidth,ywidth,_=rec
                    tposA.append(tPeak)
                    widthA.append(twidth)
                    amplA.append(yPeak)
                    stimA.append(stimAmpl)
                    swTimeA.append(sweepTime)
                    resA.append(serialRes)
        wallTA=np.array(swTimeA)/60.

        ax = self.plt.subplot(nrow,ncol,1)
        ax.scatter(tposA,amplA, alpha=0.6)
        ax.set(xlabel='stim time (ms)',ylabel='spike ampl (mV)')
        ax.text(0.01,0.9,plDD['shortName'],transform=ax.transAxes,color='m')
        if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))

        ax = self.plt.subplot(nrow,ncol,2)
        ax.scatter(stimA,amplA, alpha=0.6)
        ax.set(xlabel='stim ampl (FS)',ylabel='spike ampl (mV)')
                
        ax = self.plt.subplot(nrow,ncol,3)
        ax.scatter(tposA,widthA, alpha=0.6)
        ax.set(xlabel='stim time (ms)',ylabel='spike width (ms)')
        if 'fwhmLR' in plDD: ax.set_ylim(tuple(plDD['fwhmLR']))
        if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))

        ax = self.plt.subplot(nrow,ncol,4)
        ax.scatter(stimA,widthA, alpha=0.6)
        ax.set(xlabel='stim ampl (FS)',ylabel='spike width (ms)')

        ax = self.plt.subplot(nrow,ncol,5)
        ax.scatter(wallTA,amplA, alpha=0.6,color='g')
        ax.set(xlabel='wall time (min)',ylabel='spike ampl (mV)')
        ax.grid()

        ax = self.plt.subplot(nrow,ncol,6)
        ax.scatter(wallTA,widthA, alpha=0.6,color='g')
        ax.set(xlabel='wall time (min)',ylabel='spike width (ms)')
        ax.grid()

        ax = self.plt.subplot(nrow,ncol,7)
        ax.scatter(resA,widthA, alpha=0.6,color='b')
        ax.set(xlabel='serial resistance (MOhm)',ylabel='spike width (ms)')
        ax.grid()
            
#...!...!..................
    def spikes_survey1D(self,bigD,plDD,figId=6):
        figId=self.smart_append(figId)
        nrow,ncol=1,4
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,2.5))

        ia=plDD['iStimAmpl']
        stimAmpl=plDD['stimAmpl'][ia]
        spikeC=bigD['spikeCount'][ia]
        spikeA=bigD['spikeTrait'][ia]

        M=bigD['sweepCnt'][ia] # num of waveform/ampl
        print('pl1D: ampl=%.2f, M=%d'%(stimAmpl,M))
        # repack data for plotting
        widthA=[]; amplA=[]; tbaseA=[]
        for j in range(M): #loop over waveforms
            ks=spikeC[j]
            if ks==0: continue                
            spikes=spikeA[j][:ks] # use valid spikes
            for rec in spikes:
                tPeak,yPeak,twidth,ref_amp,twidth_base=rec                  
                widthA.append(twidth)
                amplA.append(yPeak)
                tbaseA.append(twidth_base)

        #print('pl1D: widthA:',widthA)

        tit='%s stim ampl=%.2f'%(plDD['shortName'],stimAmpl)
        ax = self.plt.subplot(nrow,ncol,1)
        binsX= np.linspace(-0.5,10.5,12)  # good choice
        ax.hist(spikeC[:M],binsX,facecolor='g')        
        ax.set(xlabel='num spikes per sweep',ylabel='num sweeps',title=tit)
        ax.grid()

        ax = self.plt.subplot(nrow,ncol,2)
        binsX= np.linspace(0.5,3.5,20)
        ax.hist(widthA,bins=binsX,facecolor='b')        
        ax.set(xlabel='spike half-width (ms), aka FWHM',ylabel='num spikes')
        if 'fwhmLR' in plDD: ax.set_xlim(tuple(plDD['fwhmLR']))
        ax.grid()
                
        ax = self.plt.subplot(nrow,ncol,3)
        binsX= np.linspace(20,60,40)
        ax.hist(amplA,bins=binsX,facecolor='C1')        
        ax.set(xlabel='spike peak ampl (mV)',ylabel='num spikes')
        ax.grid()
                
        ax = self.plt.subplot(nrow,ncol,4)
        binsX= np.linspace(0,10,40)
        ax.hist(tbaseA,bins=binsX,facecolor='C3')        
        ax.set(xlabel='spike base width (ms)',ylabel='num spikes')
        ax.grid()
            
#...!...!..................
def M_save_summary(sumL):
    sum2D=np.array(sumL)
    outF=args.dataName+'.sum.h5'
    print('sum2D:',sum2D.shape)
    outD={'spikeCount':sum2D}
    write3_data_hdf5(outD,args.outPath+outF)
    
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()

    inpF=os.path.join(args.dataPath,args.dataName+'.dm.h5')
    bigD,inpMD=read3_data_hdf5(inpF, verb=1)
    pprint(inpMD)
    
    # - - - - - PLOTTER - - - - -
    plot=Plotter(args)
    plot.dm_cube(bigD['music'])
    plot.display_all('scoreExp')
