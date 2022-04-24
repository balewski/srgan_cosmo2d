#!/bin/bash 
 
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;    #  bash exits if any statement returns a non-true return value
set -o errexit ;  # exit if any statement returns a non-true return value

#( sleep 40; echo "TTTTTTTTT1";  date; hostname; free -g; top ibn1)&
#( sleep 500; echo "TTTTTTTT2";  date; hostname; free -g; top ibn1)&

procIdx=${SLURM_PROCID}
module unload darshan
module swap PrgEnv-intel PrgEnv-gnu
module load cray-hdf5 cray-fftw gsl
FFTW3_LIBRARIES=/opt/cray/pe/fftw/3.3.8.10/haswell/lib FFTW3_INCLUDE_DIR=/opt/cray/pe/fftw/3.3.8.10/haswell/include FFTW3_ROOT=/opt/cray/pe/fftw/3.3.8.10/haswell HDF5_ROOT=/opt/cray/pe/hdf5/1.12.1.1/GNU/8.2



echo U: PWD1= `pwd`
coreStr=`grep coreStr ./cosmoMeta.yaml |  awk  '{printf "%s", $3 }' `
musicConf=${coreStr}_${procIdx}.conf
echo 'U: prep MUSIC procIdx='$procIdx  'musicCon='$musicConf

outPath=out_$procIdx
mkdir $outPath
./runMusic.sh   $musicConf  $outPath

module unload python/3.6-anaconda-4.4
module load python/2.7-anaconda-4.4
source activate cola_jan1
 

echo U: start PyCola coreStr=$coreStr procIdx=$procIdx
#On Haswell:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/common/software/fftw/3.3.4/hsw/gnu/lib/
#OR on Edison:
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/common/software/fftw/3.3.4/gnu/lib/

time ./pycola-OmSiNs-jan.py  $outPath cosmoMeta.yaml

echo U: start projection 
time ./projectNBody.py $outPath ./cosmoMeta.yaml
ls -l  $outPath
echo U: start slicing 
./sliceBigCube.py $outPath ./cosmoMeta.yaml

echo U: done $outPath
touch  $outPath/done.job
ls -lh $outPath/*hdf5
rm $outPath/*hdf5
rm $outPath/*npz
rm $outPath/wnoise*bin


exit


echo U: optional produce  input for Pk
module unload python/2.7-anaconda-4.4
module load python/3.6-anaconda-4.4
time ./pack_hd5_Pk.py $outPath/$coreStr

# this part will fail because it is calling srun inside srun
srun -n 32 -c 2 --cpu_bind=cores /project/projectdirs/mpccc/balewski/cosmo-gimlet2/apps/matter_pk/matter_pk.ex $outPath/${coreStr}.nyx.hdf5  $outPath/${coreStr}

#gnuplot> set logscale
#gnuplot> plot "ics_2018-12_a12383763_rhom_ps3d.txt" u 3:4 w lines
echo U: plot Pk
gnuplot  <<-EOFMarker
    set title "Pk for ${coreStr}" font ",14" textcolor rgbcolor "royalblue"
    set pointsize 1
    set logscale
    set terminal png   
    set output  "$outPath/${coreStr}_rhom_ps3d.png"    
    plot "$outPath/${coreStr}_rhom_ps3d.txt" u 3:4 w lines
EOFMarker

