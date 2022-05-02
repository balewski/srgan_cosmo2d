#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 tool generating MUSIC config based on the templet

Uses plain python
'''

import os,sys,copy
import time
import secrets
from pprint import pprint
import numpy as np
import configparser
from numpy.random import default_rng


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outConf', default='myMusic',help=" [.ics.conf] music config name")
    parser.add_argument('--intSeed', default='123',type=int,help="aditional seed ")
    parser.add_argument('--startConf', default='univ_base7', help=" [.ics.conf] music config name")
    parser.add_argument("-v","--verb",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1)
    parser.add_argument("-o","--outPath",default='out',help="output path for produced config")
    parser.add_argument("--inpPath", default='./', help='input location')
    args = parser.parse_args()
    
    for arg in vars(args):  
        print( 'myArg:',arg, getattr(args, arg))

    return args


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()

    rng = default_rng()
    iseed=(time.time_ns()+args.intSeed)%(1<<31)
    np.random.RandomState(iseed)
    x=rng.integers(1e10)
    print('M:seed test:',x,iseed)


    cfgF=os.path.join(args.inpPath,args.startConf+'.ics.conf')
    cfg = configparser.RawConfigParser()
    cfg.optionxform = str  # make option names case sensitive
    cfg.read(cfgF)
    print('conf sections:',cfg.sections(),cfgF)

    if args.outConf==None:
        tag='%010d'%rng.integers(1e10)
        coreN='myMusic_'+tag
    else:
        coreN=args.outConf
    print('M:coreN:',coreN)

    cfg['output']['filename']=coreN+'.music.h5'

    for i in range(7,12):
        key='seed[%d]'%i
        cfg['random'][key]=str(rng.integers(1e8))

    outF=os.path.join(args.outPath,coreN+'.ics.conf')
    print('M:saving;',outF)
    with open(outF, 'w') as fd:
        cfg.write(fd)
    print('M:saved',outF)
