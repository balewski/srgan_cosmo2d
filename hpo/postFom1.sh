#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
#set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

base0=/pscratch/sd/b/balewski/tmp_NyxHydro4kG/
exp0=1618095_147
sol=last
outPath=out5

for i in {1..40}; do  # takes ~40 min
    exp=${exp0}
    e=$[ 400 + $i * 50 ]
    sol=epoch$e
    echo  "*** predict" $exp  sol=$sol
    time ./predict.py --basePath $base0  --expName $exp --genSol $sol  --doFOM  >& $outPath/log.pred_$sol
    k=$[ ${k} + 1 ]
    #exit 1
done

echo
echo SCAN:  $k jobs 

grep 'FOM1' $outPath/log.pred*
