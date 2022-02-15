#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
#set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

#prjquota m3363  # only works on cori

if [ ! -f toolbox/dummy.exe ] ; then
    echo S: fix-me1 in perlmutter;  #  exit
fi

nodes=1 ; numGlobSamp=160  ; nG=1
nodes=1 ; numGlobSamp=320  ; nG=2
#nodes=1 ; numGlobSamp=640  ; nG=4
nodes=1 ; numGlobSamp=1280  ; nG=8
#nodes=2 ; numGlobSamp=2560  ; nG=16
#nodes=4 ; numGlobSamp=2560  ; nG=32
#nodes=8 ; numGlobSamp=2560  ; nG=64

epochs=20
design=bench_50eaf423

if [ ! -f $design.hpar.yaml  ] ; then
    echo S: fixe-me2 ;    exit
fi

for tag in g h ; do
    #jobId=lrfG${lrFact}
    jobId=weekG${nG}_${numGlobSamp}_${tag}
  
    echo start jobId=$jobId
    export SRCOS2D_WRK_SUFIX="$jobId"
    export SRCOS2D_OTHER_PAR=" --epochs ${epochs}  --expName ${jobId} --design $design --numGlobSamp $numGlobSamp  " # will overwrite any other settings

    #sbatch   -N $nodes -J week${nG} batchShifter.slr      # PM
    #./batchShifter.slr      # PM  - interactive
    #bsub  batchSummit.lsf      # Summit
    # ./batchSummit.lsf      # Summit - interactive
    sbatch   -N $nodes -J week${nG} batchCrusher.slr      # Crusher
    sleep 1
    k=$[ ${k} + 1 ]
    #exit
done


echo
echo SCAN: submitted $k jobs " "`date`
#Cancel all my jobs:
#squeue -u $USER -h | awk '{print $1}' | xargs scancel
