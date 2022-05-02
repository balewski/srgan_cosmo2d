#!/usr/bin/env python3
# DM NBody simulation

########################################################################
########################################################################
#    Copyright (c) 2013,2014       Svetlin Tassev
#                       Princeton University,Harvard University
#
#   This file is part of pyCOLA.
#
#   pyCOLA is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   pyCOLA is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with pyCOLA.  If not, see <http://www.gnu.org/licenses/>.
#
########################################################################
########################################################################

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os,sys,time
import configparser
import numpy as np
from pprint import pprint
from toolbox.Util_H5io3 import read3_data_hdf5, write3_data_hdf5

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("--dataPath",default='out')
    parser.add_argument("--simConf",  default='univ_base0', help=" [.ics.conf] music config name")
    parser.add_argument( "-E","--noEvol", action='store_true',default=False,
         help="disable evolution to z=0")

    args = parser.parse_args()

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!..................
def cfg2dict(cfg):
    metaD={}
    #print('sections:',cfg.sections())
    for sectN in cfg.sections():
        sectObj=cfg[sectN]
        sect={}
        metaD[sectN]=sect
        #print('\nsubs:',sectObj)
        for k,v in sectObj.items():
            #print(k,v)
            sect[k]=v

    return metaD

# - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - 

def runPycola(infile,  omM, zstart,zstop,boxlength,levelmax):

    import numpy as np
    import matplotlib.pyplot as plt
    from pycola3.aux import boundaries
    from pycola3.ic import ic_2lpt,import_music_snapshot
    from pycola3.evolve import evolve
    from pycola3.ic import ic_2lpt,import_music_snapshot
    from pycola3.cic import CICDeposit_3
    from pycola3.potential import initialize_density
    
    # Set up the parameters from the MUSIC ic snapshot:
    music_file=infile

    # Set up according to instructions for 
    # aux.boundaries()
    
    boxsize=boxlength # in Mpc/h
    level=levelmax
    level_zoom=levelmax
    gridscale=3
    
    # Set up according to instructions for 
    # ic.import_music_snapshot()
    level0='%02d'%level # should match level above
    level1='%02d'%level_zoom # should match level_zoom above
    
    # Set how much to cut from the sides of the full box. 
    # This makes the COLA box to be of the following size in Mpc/h:
    # (2.**level-(cut_from_sides[0]+cut_from_sides[1]))/2.**level*boxsize
    
    # This is the full box. Set FULL=True in evolve() below
    cut_from_sides=[0,0]# 100Mpc/h. 
    #
    # These are the interesting cases:
    #cut_from_sides=[64,64]# 75Mpc/h
    #cut_from_sides=[128,128] # 50Mpc/h
    #cut_from_sides=[192,192]  # 25Mpc/h

    print('jj1 music_file:',music_file, boxsize,level0,level1)
    t0=time.time()
    sx_full1, sy_full1, sz_full1, sx_full_zoom1, sy_full_zoom1, \
        sz_full_zoom1, offset_from_code \
        = import_music_snapshot(music_file, boxsize,level0=level0,level1=level1)
    
    NPART_zoom=list(sx_full_zoom1.shape)

    print("Starting 2LPT on full box...")
    #Get bounding boxes for full box with 1 refinement level for MUSIC.
    BBox_in, offset_zoom, cellsize, cellsize_zoom, \
        offset_index, BBox_out, BBox_out_zoom, \
        ngrid_x, ngrid_y, ngrid_z, gridcellsize  \
        = boundaries(boxsize, level, level_zoom, \
                     NPART_zoom, offset_from_code, [0,0], gridscale)

    #print('jj2 BBox_in',BBox_in.shape,type(BBox_in),BBox_in.dtype)
    #for xx in [  ,  , , ,, ]:
    sx_full1=sx_full1.astype(np.float32)
    sy_full1=sy_full1.astype(np.float32)
    sz_full1=sz_full1.astype(np.float32)
    sx_full_zoom1=sx_full_zoom1.astype(np.float32)
    sy_full_zoom1=sy_full_zoom1.astype(np.float32)
    sz_full_zoom1=sz_full_zoom1.astype(np.float32)
        
    #print('jj2b',type(sx_full1),sx_full1.dtype)
    #print('jj2c',type(sx_full_zoom1),sx_full_zoom1.dtype)

    sx2_full1, sy2_full1, sz2_full1,  sx2_full_zoom1, \
        sy2_full_zoom1, sz2_full_zoom1 \
        = ic_2lpt( 
            cellsize,
            sx_full1 ,
            sy_full1 ,
            sz_full1 ,
            
            cellsize_zoom=cellsize_zoom,
            sx_zoom = sx_full_zoom1,
            sy_zoom = sy_full_zoom1,
            sz_zoom = sz_full_zoom1,
            
            boxsize=boxsize,
            ngrid_x_lpt=ngrid_x,ngrid_y_lpt=ngrid_y,ngrid_z_lpt=ngrid_z,
                       
            offset_zoom=offset_zoom,BBox_in=BBox_in)

    print('jj3 sx2_full1:',sx2_full1.shape,sx2_full1.sum(),sy2_full_zoom1.shape,sy2_full_zoom1.sum())

    #Get bounding boxes for the COLA box with 1 refinement level for MUSIC.
    BBox_in, offset_zoom, cellsize, cellsize_zoom, \
        offset_index, BBox_out, BBox_out_zoom, \
        ngrid_x, ngrid_y, ngrid_z, gridcellsize \
        = boundaries(
            boxsize, level, level_zoom, \
            NPART_zoom, offset_from_code, cut_from_sides, gridscale)
    #print('jj4 BBox_in:',BBox_in.shape)
    
    # Trim full-box displacement fields down to COLA volume.
    sx_full       =       sx_full1[BBox_out[0,0]:BBox_out[0,1],  \
                                   BBox_out[1,0]:BBox_out[1,1],  \
                                   BBox_out[2,0]:BBox_out[2,1]]
    sy_full       =       sy_full1[BBox_out[0,0]:BBox_out[0,1],  
                                   BBox_out[1,0]:BBox_out[1,1],  \
                                   BBox_out[2,0]:BBox_out[2,1]]
    sz_full       =       sz_full1[BBox_out[0,0]:BBox_out[0,1],  \
                                   BBox_out[1,0]:BBox_out[1,1],  \
                                   BBox_out[2,0]:BBox_out[2,1]]
    sx_full_zoom  =  sx_full_zoom1[BBox_out_zoom[0,0]:BBox_out_zoom[0,1],  \
                                   BBox_out_zoom[1,0]:BBox_out_zoom[1,1],  \
                                   BBox_out_zoom[2,0]:BBox_out_zoom[2,1]]
    sy_full_zoom  =  sy_full_zoom1[BBox_out_zoom[0,0]:BBox_out_zoom[0,1],  \
                                   BBox_out_zoom[1,0]:BBox_out_zoom[1,1],  \
                                   BBox_out_zoom[2,0]:BBox_out_zoom[2,1]]
    sz_full_zoom  =  sz_full_zoom1[BBox_out_zoom[0,0]:BBox_out_zoom[0,1],  \
                                   BBox_out_zoom[1,0]:BBox_out_zoom[1,1],  \
                                   BBox_out_zoom[2,0]:BBox_out_zoom[2,1]]               
    del sx_full1, sy_full1, sz_full1, sx_full_zoom1, sy_full_zoom1, sz_full_zoom1

    sx2_full       =       sx2_full1[BBox_out[0,0]:BBox_out[0,1],  \
                                     BBox_out[1,0]:BBox_out[1,1],  \
                                     BBox_out[2,0]:BBox_out[2,1]]
    sy2_full       =       sy2_full1[BBox_out[0,0]:BBox_out[0,1],  \
                                     BBox_out[1,0]:BBox_out[1,1],  \
                                      BBox_out[2,0]:BBox_out[2,1]]
    sz2_full       =       sz2_full1[BBox_out[0,0]:BBox_out[0,1],  \
                                     BBox_out[1,0]:BBox_out[1,1],  \
                                     BBox_out[2,0]:BBox_out[2,1]]
    sx2_full_zoom  =  sx2_full_zoom1[BBox_out_zoom[0,0]:BBox_out_zoom[0,1],  \
                                     BBox_out_zoom[1,0]:BBox_out_zoom[1,1],  \
                                     BBox_out_zoom[2,0]:BBox_out_zoom[2,1]]
    sy2_full_zoom  =  sy2_full_zoom1[BBox_out_zoom[0,0]:BBox_out_zoom[0,1],  \
                                     BBox_out_zoom[1,0]:BBox_out_zoom[1,1],  \
                                     BBox_out_zoom[2,0]:BBox_out_zoom[2,1]]
    sz2_full_zoom  =  sz2_full_zoom1[BBox_out_zoom[0,0]:BBox_out_zoom[0,1],  \
                                     BBox_out_zoom[1,0]:BBox_out_zoom[1,1],  \
                                     BBox_out_zoom[2,0]:BBox_out_zoom[2,1]]
    del sx2_full1, sy2_full1, sz2_full1, sx2_full_zoom1, sy2_full_zoom1, sz2_full_zoom1

    t1=time.time()
    print ("2LPT on full box is done, elaT=%.1f min"%( (t1-t0)/60.))
    print ("Starting COLA! sx_full:",sx_full.shape,'Z start=%.1f --> stop=%.1f'%(zstart,zstop))

    #--------------------------  pass 1 ----- full evolution ------
    n_steps=10
    print ("cellsize:", cellsize,'n_steps:',n_steps)
    # Jan: adjust arguments to pycola3:
    # https://github.com/philbull/pycola3/blob/main/pycola3/evolve.py
    px, py, pz, vx, vy, vz, \
        px_zoom, py_zoom, pz_zoom, vx_zoom, vy_zoom, vz_zoom \
        = evolve( 
            cellsize,
            sx_full, sy_full, sz_full, 
            sx2_full, sy2_full, sz2_full,
            covers_full_box=True,  #was: FULL=True,
            
            cellsize_zoom=cellsize_zoom,
            sx_full_zoom  = sx_full_zoom , 
            sy_full_zoom  = sy_full_zoom , 
            sz_full_zoom  = sz_full_zoom ,
            sx2_full_zoom = sx2_full_zoom,
            sy2_full_zoom = sy2_full_zoom,
            sz2_full_zoom = sz2_full_zoom,
            
            offset_zoom=offset_zoom,
            bbox_zoom=BBox_in,  #was:  BBox_in=BBox_in,
            
            ngrid_x=ngrid_x,
            ngrid_y=ngrid_y,
            ngrid_z=ngrid_z,
            gridcellsize=gridcellsize,
            
            ngrid_x_lpt=ngrid_x,
            ngrid_y_lpt=ngrid_y,
            ngrid_z_lpt=ngrid_z,
            gridcellsize_lpt=gridcellsize,
            
            Om = float(omM),
            Ol = 1.0-float(omM), 
            a_final=1./(1.+zstop),  # 'a' is cosmological scale factor
            a_initial=1./(1.+zstart),
            n_steps=n_steps,

            #was: save_to_file=True,  # set this to True to output the snapshot to a file
            #was: file_npz_out=outfile,
            )
    outD={'z0.px':px, 'z0.py':py, 'z0.pz':pz, 'z0.vx':vx, 'z0.vy':vy, 'z0.vz':vz}    
    t2=time.time()

    
    #--------------------------  pass 2 ----- minimal evolution ------
    n_steps=1
    zstop2=zstart-1
    print ("cellsize:", cellsize,'n_steps:',n_steps,'zstop:',zstop)
    # Jan: adjust arguments to pycola3:
    # https://github.com/philbull/pycola3/blob/main/pycola3/evolve.py
    px, py, pz, vx, vy, vz, \
        px_zoom, py_zoom, pz_zoom, vx_zoom, vy_zoom, vz_zoom \
        = evolve( 
            cellsize,
            sx_full, sy_full, sz_full, 
            sx2_full, sy2_full, sz2_full,
            covers_full_box=True,  #was: FULL=True,
            
            cellsize_zoom=cellsize_zoom,
            sx_full_zoom  = sx_full_zoom , 
            sy_full_zoom  = sy_full_zoom , 
            sz_full_zoom  = sz_full_zoom ,
            sx2_full_zoom = sx2_full_zoom,
            sy2_full_zoom = sy2_full_zoom,
            sz2_full_zoom = sz2_full_zoom,
            
            offset_zoom=offset_zoom,
            bbox_zoom=BBox_in,  #was:  BBox_in=BBox_in,
            
            ngrid_x=ngrid_x,
            ngrid_y=ngrid_y,
            ngrid_z=ngrid_z,
            gridcellsize=gridcellsize,
            
            ngrid_x_lpt=ngrid_x,
            ngrid_y_lpt=ngrid_y,
            ngrid_z_lpt=ngrid_z,
            gridcellsize_lpt=gridcellsize,
            
            Om = float(omM),
            Ol = 1.0-float(omM), 
            a_final=1./(1.+zstop2),  # 'a' is cosmological scale factor
            a_initial=1./(1.+zstart),
            n_steps=n_steps,

            #was: save_to_file=True,  # set this to True to output the snapshot to a file
            #was: file_npz_out=outfile,
            )
    zz='z%d'%zstop2
    outD.update({zz+'.px':px, zz+'.py':py, zz+'.pz':pz, zz+'.vx':vx, zz+'.vy':vy, zz+'.vz':vz}  )
    t3=time.time()

    
    print ("COLA done, took %.2f min, elaT=%.1f min "%( (t2-t1)/60., (t3-t0)/60.))
    
    sumD={'evol_time':t2-t1,'pycola_time':t2-t0,'date':time.strftime("%Y%m%d_%H%M%S_%Z",time.localtime()),'zRedShift':[zstop2,zstop],'zRedShift_label':['z%d'%zstop2,'z%d'%zstop]}
    return outD,sumD


# - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - 

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()
    cfgF=os.path.join(args.dataPath,args.simConf+'.ics.conf')
    cfg = configparser.RawConfigParser()   
    cfg.read(cfgF)
    metaD=cfg2dict(cfg)
    print('conf sections:',cfg.sections(),cfgF)
    assert len(cfg.sections())>0
    levMin=int(cfg['setup']['levelmin'])
    levMax=int(cfg['setup']['levelmax'])
    assert levMin==levMax
    zstart=float(cfg['setup']['zstart'])
    zstop=float(cfg['pycola']['zstop'])
    
    # Peter: redshift z=0-->now, z=50 -->40 million years after big-bang
    
    musF = cfg['output']['filename']
    boxlength = int(cfg['setup']['boxlength'])
    omega_m=float(cfg['cosmology']['Omega_m'])
    inpF=os.path.join(args.dataPath,musF)
    outF=inpF.replace('.music','.cola')
    print('M:inpF',inpF)
    bigD,sumD=runPycola(inpF,  omega_m, zstart,zstop,boxlength,levMax)
    
    metaD['pycola'].update(sumD)
    write3_data_hdf5(bigD,outF,metaD=metaD)
    print('M: pycola done')
    #pprint(metaD)
    
