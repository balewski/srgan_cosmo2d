#!/bin/bash 
 
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;    #  bash exits if any statement returns a non-true return value
set -o errexit ;  # exit if any statement returns a non-true return value

#( sleep 40; echo "TTTTTTTTT1";  date; hostname; free -g; top ibn1)&
#( sleep 500; echo "TTTTTTTT2";  date; hostname; free -g; top ibn1)&

procIdx=${SLURM_PROCID}
jid=${MYJID}
univN=univL${LEVEL}_${jid}_${procIdx}
sleep $[ 30 * $SLURM_LOCALID ]  # not sure if it is needed - it may releave RAM congestion

echo U: PWD1= `pwd` LEVEL=$LEVEL jid=$jid univN=$univN
out=out_$procIdx
mkdir -p $out

# prep music par , seeds
#uni=univ_base$LEVEL  #tmp
#cp $uni.ics.conf $out
#tmp-end
./genMusicConf.py  --startConf univ_base$LEVEL --intSeed $procIdx  --outConf $univN --outPath $out >& ${out}/log.gen

echo U: procIdx=$procIdx uni=$univN
cd $out
time  $MUSIC $univN.ics.conf >&log.music
cd -

echo run PyCola 
time  python3 -u ./pycola3_main.py  --dataPath $out --simConf $univN >& ${out}/log.pycola

echo U:project DM vector filed to density
./projectNBody.py --dataName $univN  --dataPath $out >& ${out}/log.project

echo U:plot DM density cube
time   ./plotCube.py --dataName  $univN  --dataPath $out --outPath $out --show bcd -X   >& ${out}/log.plot

echo U:done
exit 0

THIS WILL CRASH - but may be it is usefull?
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

