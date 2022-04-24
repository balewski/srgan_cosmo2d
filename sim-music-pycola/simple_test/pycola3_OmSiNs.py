#!/usr/bin/env python3
'''
Use with shifter image at NERSC

shifter --image balewski/ubu20-music-pycola3:v2 bash
 ./pycola3_OmSiNs.py ~/prjs/superRes3D-sim/music  cosmoMeta.yaml

'''

import sys, os
from pprint import pprint
from ruamel.yaml import YAML

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

def runit(infile, outfile, omM,boxlength,levelmax):

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
    level0='0%d'%level # should match level above
    level1='0%d'%level_zoom # should match level_zoom above
    
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
    sx_full1, sy_full1, sz_full1, sx_full_zoom1, sy_full_zoom1, \
        sz_full_zoom1, offset_from_code \
        = import_music_snapshot(music_file, boxsize,level0=level0,level1=level1)
    
    NPART_zoom=list(sx_full_zoom1.shape)

    print("Starting 2LPT on full box.")
    
    #Get bounding boxes for full box with 1 refinement level for MUSIC.
    BBox_in, offset_zoom, cellsize, cellsize_zoom, \
        offset_index, BBox_out, BBox_out_zoom, \
        ngrid_x, ngrid_y, ngrid_z, gridcellsize  \
        = boundaries(boxsize, level, level_zoom, \
                     NPART_zoom, offset_from_code, [0,0], gridscale)

    #print('jj2',type(BBox_in),BBox_in.dtype)
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





    #Get bounding boxes for the COLA box with 1 refinement level for MUSIC.
    BBox_in, offset_zoom, cellsize, cellsize_zoom, \
        offset_index, BBox_out, BBox_out_zoom, \
        ngrid_x, ngrid_y, ngrid_z, gridcellsize \
        = boundaries(
            boxsize, level, level_zoom, \
            NPART_zoom, offset_from_code, cut_from_sides, gridscale)

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


    print ("2LPT on full box is done.")
    print ("Starting COLA!")

    print ("cellsize:", cellsize, "outfile:",outfile)
# adjust arguments to pycola3:
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
            a_final=1.,
            a_initial=1./10.,
            n_steps=10,

            filename_npz=outfile,
            #was: save_to_file=True,  # set this to True to output the snapshot to a file
            #was: file_npz_out=outfile,
            )

    del vx_zoom,vy_zoom,vz_zoom
    del vx,vy,vz



# - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - 


def read_yaml(yaml_fn,verb=1):
        data={}
        if verb:  print('  read  yaml:',yaml_fn)
        with open(yaml_fn) as yamlfile:
            for key, val in YAML().load(yamlfile).items():
                print('hpar:',key, val)
                data[key]=val
        return data


if __name__ == '__main__':

    #ymlF='outMusic/cosmoMeta.yaml'
    ioPath=sys.argv[1]
    ymlF=sys.argv[2]
    print ("read YAML from ",ymlF,' and pprint it:')
    #inpFile=open(ymlF)
    blob=read_yaml(ymlF)

    pprint(blob)
 
    infile=os.path.join(ioPath,blob['coreStr']+'.h5')
    #infile='/global/homes/b/balewski/prjs/superRes3D-sim/music/supres_ver1.hdf5'
    outfile=infile.replace('.hdf5','.npz')
    runit(infile, outfile, blob['physOmega_m'], blob['boxlength'],blob['levelmax'])
