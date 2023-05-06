#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value
set -o errexit ;  # exit if any statement returns a non-true return value
k=0

#inpPath=/global/cfs/cdirs/m3363/balewski/superRes-Nyx2022a-flux
inpPath=/pscratch/sd/b/balewski/tmp_NyxProd2/2968685_100univ/
hsiPath=superRes-Nyx2022a-flux/NyxHydro/

for dirN in ${inpPath}/*/     # list directories in the form "/tmp/dirname/"
do
    dirN=${dirN%*/}      # remove the trailing "/"
    k=$[ ${k} + 1 ]
    #echo $k $dirN
    core="${dirN##*/}"    # print everything after the final "/"
    echo $k $core
    tgt=${hsiPath}/${core}.tar
    time  htar -cf $tgt ${inpPath}/$core  >& out/log-htar.${k}-$core 
    #hsi ls -B $tgt
    #exit
done

echo
echo HTAR: processed $k dirs

exit
screen: cori1

To restore a  dir from HPSS
cd abc
time htar -xf neuronBBP2-data_67pr/bbp0541.tar
