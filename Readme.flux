In the Fall of 2022 we changed training data from 3D mass to 3D flux cubes

density --> flux  and 1lr:1hr packing
https://docs.google.com/document/d/1a20m8_gDbatyS4EUY0BqzUTyKxc3SEzoIvkNHNcc7_4/edit?usp=sharing


1a=====  raw flux cubes are stored at:
/pscratch/sd/b/balewski/tmp_NyxProd  14 universes
/pscratch/sd/b/balewski/tmp_NyxProd2  ~200 universes

example
/pscratch/sd/b/balewski/tmp_NyxProd/2767632_2univ/cube_20054028

97M Oct 21 19:43 plotLR00407.flux.h5
6.1G Oct 21 19:53 plotHR00776.flux.h5


1b=======   pack 3  cubes in hd5  into a single h5
z=3 I'll use [LR,HR]   record:   derived_fields/tau_red  
 HR z=50 density filed:   native_fields/baryon_density  

Persistent loaction at:  ~/prje/superRes-Nyx2022a-flux/tripack_cubes
215 GB

./make_tripack_NyxHydroH5.py --basePath  /pscratch/sd/b/balewski/tmp_NyxProd2  --jobName 2968683_100univ


data location1 :  14 universes
/pscratch/sd/b/balewski/tmp_NyxProd/tripack_cubes> ls
cube_1064170349.tripack.h5  cube_173605359.tripack.h5  cube_609994690.tripack.h5
cube_1121666308.tripack.h5  cube_173840049.tripack.h5  cube_75335460.tripack.h5
cube_1166586326.tripack.h5  cube_20054028.tripack.h5   cube_82337823.tripack.h5
cube_1181930045.tripack.h5  cube_415462772.tripack.h5  cube_885045934.tripack.h5
cube_129663520.tripack.h5   cube_505143543.tripack.h5

data location2 :  ~200 universes
/pscratch/sd/b/balewski/tmp_NyxProd/tripack_cubes>

example content
h5ls cube_1121666308.tripack.h5
flux_HR_z3               Dataset {512, 512, 512}
flux_LR_z3               Dataset {128, 128, 128}
invBarDens_HR_z200       Dataset {512, 512, 512}
meta.JSON                Dataset {1}




2====  packing 2D-planes  1+1  (1lr_1hr)
Formatting tripacks for srgan2d
~/srgan_cosmo2d/sim_NyxHydro> ./format_tripack_2_srgan2D_1lr1hr.py

/global/homes/b/balewski/prje/superRes-Nyx2022a-flux/
dataName=flux-1lr1hr-Nyx2022a-r2c14
dataName=flux-1lr1hr-Nyx2022a-r2c99  :  52G

content:
meta.JSON                Dataset {1}
test_flux_HR_z3          Dataset {2304, 512, 512}
test_flux_LR_z3          Dataset {2304, 128, 128}
test_invBarDens_HR_z200  Dataset {2304, 512, 512}
train_flux_HR_z3         Dataset {20736, 512, 512}
train_flux_LR_z3         Dataset {20736, 128, 128}
train_invBarDens_HR_z200 Dataset {20736, 512, 512}
valid_flux_HR_z3         Dataset {2304, 512, 512}
valid_flux_LR_z3         Dataset {2304, 128, 128}
valid_invBarDens_HR_z200 Dataset {2304, 512, 512}


3====  packing 2D-planes  1+4  (1lr_4hr)
Formatting tripacks for srgan2d
~/srgan_cosmo2d/sim_NyxHydro> ./format_tripack_2_srgan2D.py

/global/homes/b/balewski/prje/superRes-Nyx2022a-flux/
 29G Dec  1 14:41 flux-1LR4HR-Nyx2022a-r2c14.h5

meta.JSON                Dataset {1}
test_flux_HR_z3          Dataset {256, 4, 512, 512}
test_flux_LR_z3          Dataset {256, 128, 128}
test_invBarDens_HR_z200  Dataset {256, 4, 512, 512}
train_flux_HR_z3         Dataset {3072, 4, 512, 512}
train_flux_LR_z3         Dataset {3072, 128, 128}
train_invBarDens_HR_z200 Dataset {3072, 4, 512, 512}
valid_flux_HR_z3         Dataset {256, 4, 512, 512}
valid_flux_LR_z3         Dataset {256, 128, 128}
valid_invBarDens_HR_z200 Dataset {256, 4, 512, 512}

