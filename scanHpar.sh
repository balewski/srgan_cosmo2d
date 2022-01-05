#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0


#for lrFact in 0.02 0.05 0.10 0.20 0.50  1.00 2.00  ; do
for lrFact in 0.44 0.67 1.00 1.50 2.30 3.50  ; do    
#for lrFact in 1.0 ; do 
    jobId=lrFactG${lrFact}

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
