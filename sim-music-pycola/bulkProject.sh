#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
#set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0
dataPath=/global/cscratch1/sd/balewski/univers2/
#jobN=univL7_58411332 ; jid=univL7_58411332

jobN=univL9_58363933_3 ; jid=univL9_58363933 
#jobN=univL9_58363933_2 ; jid=univL9_58363939 
#jobN=univL9_58363933_1 ; jid=univL9_58363938  # done 0
#jobN=univL9_58364013_4 ; jid=univL9_58364013  # done 0,1

for i in {0..119} ; do
   out=${dataPath}/${jobN}/out_${i}
   univN=${jid}_${i}
    ./projectNBody.py --dataName $univN  --dataPath $out
   
   outF=${out}/${univN}.dm.h5

   univN2=${jobN}_${i}
   outF2=${out}/${univN2}.dm.h5
   ls -l $outF
   mv  $outF  $outF2

   rm ${dataPath}/*png
   ./plotCube.py --dataName  $univN2  --dataPath $out --outPath $out --show cd -X
done

exit 0


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
