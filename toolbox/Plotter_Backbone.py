__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os
import numpy as np

import socket  # for hostname
import time

#...!...!..................
def roys_fontset(plt):
    print('load Roys fonts')
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    #plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    tick_major = 6
    tick_minor = 4
    plt.rcParams["xtick.major.size"] = tick_major
    plt.rcParams["xtick.minor.size"] = tick_minor
    plt.rcParams["ytick.major.size"] = tick_major
    plt.rcParams["ytick.minor.size"] = tick_minor

    font_small = 12
    font_medium = 13
    font_large = 14
    plt.rc('font', size=font_small)          # controls default text sizes
    plt.rc('axes', titlesize=font_medium)    # fontsize of the axes title
    plt.rc('axes', labelsize=font_medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=font_small)    # legend fontsize
    plt.rc('figure', titlesize=font_large)   # fontsize of the figure title

#............................
#............................
#............................
class Plotter_Backbone(object):
    def __init__(self, args):
        import matplotlib as mpl
        if args.noXterm:
            mpl.use('Agg')  # to plot w/o X-server
            print(self.__class__.__name__,':','Graphics disabled')
        else:
            mpl.use('TkAgg') # on baci-desktop
            print(self.__class__.__name__,':','Graphics started')
        #mpl.rcParams['text.usetex'] = True
        import matplotlib.pyplot as plt
        plt.close('all')
        self.plt=plt
        self.figL=[]
        self.outPath=args.outPath
        self.prjName=args.prjName
        if args.formatVenue=='poster':
            roys_fontset(plt)
        
        
        
    #............................
    def display_all(self, ext='', png=1):
        if len(self.figL)<=0: 
            print('display_all - nothing to plot, quit')
            return
        for fid in self.figL:
            self.plt.figure(fid)
            self.plt.tight_layout()
            #self.plt.subplots_adjust(bottom=0.06, top=0.97) # for 21-pred
            
            figName=self.outPath+'%s_%s_f%d'%(self.prjName,ext,fid)
            if png: figName+='.png'
            else: figName+='.pdf'            
            print('Graphics saving to ',figName)
            self.plt.savefig(figName)             
        self.plt.show()

# figId=self.smart_append(figId)
#...!...!....................
    def smart_append(self,id): # increment id if re-used
        while id in self.figL: id+=1
        self.figL.append(id)
        return id

