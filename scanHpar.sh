#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0


for lrFact in 0.5 0.7 1.0 1.5 2.0  ; do 
#for lrFact in 1.5 ; do # one-off
    jobId=lrFact${lrFact}

    echo
    echo start jobId=$jobId

    export SRCOS2D_LR_FACT=$lrFact
    export SRCOS2D_JOBID=$jobId

    sbatch  batchShifter.slr      # PM
    #./batchShifter.slr      # PM  - interactive
    #bsub  batchSummit.lsf      # Summit
    # ./batchSummit.lsf      # Summit - interactive
    sleep 1
    k=$[ ${k} + 1 ]
    #exit
done


echo
echo SCAN: submitted $k jobs " "`date`
