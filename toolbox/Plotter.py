__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np

from matplotlib import cm as cmap
#import matplotlib as mpl # for LogNorm()
from toolbox.Plotter_Backbone import Plotter_Backbone


#............................
#............................
#............................
class Plotter_SRGAN_2D(Plotter_Backbone):
    def __init__(self, args,inpMD,sumRec=None):
        Plotter_Backbone.__init__(self,args)
        self.inpMD=inpMD
        self.sumRec=sumRec
        self.formatVenue=args.formatVenue
        

#...!...!..................
    def input_images(self,plDD,figId=4): 
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,5))
        ncol,nrow=2,1

        rL=['lr','hr']
        for i,xr in  enumerate(rL):
            ax=self.plt.subplot(nrow,ncol,1+i)
            ax.set_aspect(1.)
            img=plDD[xr]
            #tit='%s %d %s'%(xr,args.index,str(img.shape))
            ax.imshow(img.T,origin='lower') #,vmax=plDD['img_vmax'])
            
        
