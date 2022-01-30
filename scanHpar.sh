#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
#set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

#prjquota m3363  # only works on cori

if [ ! -f toolbox/dummy.exe ] ; then
    echo S: fixe-me1 ;    exit
fi

nodes=8
epochs=5700
design=dev7d

if [ ! -f $design.hpar.yaml  ] ; then
    echo S: fixe-me2 ;    exit
fi


#for lrFact in  0.69 0.83 1.00 1.20 1.44  0.58  ; do
for tag in a b c ; do
    #jobId=lrfG${lrFact}
    jobId=N${nodes}_${design}_${tag}
   
    echo start jobId=$jobId
    export SRCOS2D_WRK_SUFIX="$jobId"
    export SRCOS2D_OTHER_PAR=" --epochs ${epochs}  --expName ${jobId} --design $design "  # will overwrite any other settings

    sbatch   -N $nodes  batchShifter.slr      # PM
    #./batchShifter.slr      # PM  - interactive
    #bsub  batchSummit.lsf      # Summit
    # ./batchSummit.lsf      # Summit - interactive
    sleep 1
    k=$[ ${k} + 1 ]
    #exit
done


echo
echo SCAN: submitted $k jobs " "`date`
#Cancel all my jobs:
#squeue -u $USER -h | awk '{print $1}' | xargs scancel
