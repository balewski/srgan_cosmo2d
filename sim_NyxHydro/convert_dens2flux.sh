#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value
set -o errexit ;  # exit if any statement returns a non-true return value

# binary  from Hyunbae  converting desnity to Flux
module load gsl
EXE=/global/cfs/cdirs/m3363/balewski/simu_Nyx2022_exec/lya_fields.ex
LIB1=/global/homes/b/balewski/prje/simu_Nyx2022_exec/TREECOOL_middle

# set defaults
#basePath=/pscratch/sd/b/balewski/tmp_NyxProd/2767632_2univ

name=nameX
pidfile=pidX
logfile=logX

while [ "$#" -gt 0 ]; do
  case "$1" in
    -n) name="$2"; shift 2;;
    -p) basePath="$2"; shift 2;;
    -l) logfile="$2"; shift 2;;

    --name) name="${1#*\ }"; shift 1;;
    --basePath=*) basePath="${1#*=}"; shift 1;;
    --logfile=*) logfile="${1#*=}"; shift 1;;
    --name|--pidfile|--logfile) echo "$1 requires an argument" >&2; exit 1;;

    -*) echo "unknown option: $1" >&2; exit 1;;
    *) handle_argument "$1"; shift 1;;
  esac
done

echo name=$name
echo basePath=$basePath
echo logfile=$logfile


# ........................
function find_univers {
    echo  baseParh=$basePath
    uniL=`ls -d $basePath/cube*`    
}

# ........................
function convert_last_z {
    HLR=$1
    pwd
    echo  select_last_z:  $HLR
    nameL=`ls *${HLR}*.h5`
    ls -lh $nameL
    for name in $nameL ; do
	name1=${name/_converted./.dens.}
	name2=${name/_converted./.flux.}
	echo move $name1 $name2
	mv  $name $name1
    done
    echo start flux conversion ...
    cp $name1 $name2
    time $EXE $name2
    ls -lh $name2
    
}

# = = = = = =
#  main
# = = = = = =
[ ! -d $basePath ] && echo "Directory $basePath DOES NOT exists." && exit

find_univers

#uniL="cube_926684783 cube_930945067 cube_950098718 cube_976235394 cube_979839988 cube_985229966 cube_998390396"

echo M:uniL $uniL
for univ in $uniL ; do
    #univ=$basePath/$univ0
    echo univ $univ
    cd $univ
    #ln -s /global/homes/b/balewski/prje/simu_Nyx2022_exec/TREECOOL_middle
    convert_last_z  LR
    convert_last_z  HR
    echo DONE  universe $univ "*********************"
    date
done
    
