#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
#set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

numUniv=${1-1}  
jid=${SLURM_JOBID-123}

execPath=/global/homes/b/balewski/prje/simu_Nyx2022_exec
srcPath=/global/homes/b/balewski/srgan_cosmo2d/sim_NyxHydro
outPath=/pscratch/sd/b/balewski/tmp_NyxProd/${jid}_${numUniv}univ      

function gen_int {  #...............
    MIN=0
    MAX=1234567890
    while
	rnd=$(cat /dev/urandom | tr -dc 0-9 | fold -w${#MAX} | head -1 | sed 's/^0*//;')
	[ -z $rnd ] && rnd=0
	(( $rnd < $MIN || $rnd > $MAX ))
    do :
    done
    my_rnd=$rnd
}

#...............  initial setup
echo outPath=$outPath
mkdir -p $outPath
pwd

#...............  main loop

for i in `seq 1 ${numUniv}` ; do
    gen_int
    path=${outPath}/cube_$my_rnd
    echo seed=$my_rnd  path=$path
    
    mkdir -p $path  
    cd ${path}
    
    sed "s/<seed>/${my_rnd}/" ${srcPath}/CosmicIC_input.templ >CosmicIC_input
    echo  link binaries form other people ${execPath}
    for xx in convertLR2hdf5.sh convertHR2hdf5.sh Nyx3d.gnu.TPROF.MTMPI.OMP.CUDA.ex  TREECOOL_middle convert3d.gnu.x86-milan.PROF.MPI.ex ; do
	ln -s ${execPath}/${xx} .
    done
    
    cd ${execPath}
    echo  creates low-res IC with this seed '...'  #real	2m25.756s
    time  ./init_sub ${path}/CosmicIC_input ./cmb.tf ${path}/ICs_LR  >& ${path}/log.ic_lr

    #  create high-res image IC 
    time  ./init ${path}/CosmicIC_input ./cmb.tf ${path}/ICs_HR  >& ${path}/log.ic_hr

    cp  ${srcPath}/Nyx_input_LR  ${path}
    cp  ${srcPath}/Nyx_input_HR  ${path}

    cd ${path}  # this will make slurm.out to be stored there
    echo evolve both
    pwd
    sbatch ${srcPath}/batch_Nyx_LRandHR.slr $path 
    
done

exit 0

