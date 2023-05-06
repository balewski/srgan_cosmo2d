#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value
set -o errexit ;  # exit if any statement returns a non-true return value
k=0

#inpPath=/global/cfs/cdirs/m3363/balewski/superRes-Nyx2022a-flux
inpPath=/pscratch/sd/b/balewski/tmp_NyxProd2/
hsiPath=superRes-Nyx2022a-flux

core=tripack_cubes
echo core=$core
tgt=${hsiPath}/${core}.tar
time  htar -cf $tgt ${inpPath}/$core  >& out/log-htar.$core 

