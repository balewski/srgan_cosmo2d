#!/bin/bash 

if [ ${SLURM_PROCID} -eq 0 ] ; then
    [[ -z "${SHIFTER_RUNTIME}" ]]  &&  echo NOT-in-shifter  || echo in-shifter
    date
    echo D: CMD=$CMD
    echo D: job=${SLURM_JOBID} `hostname`
    echo D: nodes:$SLURM_NODELIST
    cat /etc/*release |grep PRETTY_NAME
    free -g
    echo D: num-cpus:`nproc --all`
    nvidia-smi --list-gpus
    python -V
    nvcc --version  # CUDA version
    echo cudann version: `cat /usr/include/cudnn_version.h |grep "e CUDNN_MAJOR" -A 2`
    python -c 'import torch; print("D: pytorch:",torch.__version__)'

    echo D: survey-end
    #nvidia-smi -l 5 >&L.smi_${SLURM_JOBID} &
fi
if [ ${SLURM_LOCALID} -eq 0 ] ; then
    echo -n "D:`hostname` rank=${SLURM_PROCID}  check ECC  on device :"
    nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.device_memory --format=csv,noheader
else
    sleep 1
fi 
# the task command executed here:
${CMD}


# from Xander
#!/bin/bash
#while true;
#do nvidia-smi --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory --format=csv >> gpu_utillization.log; sleep 1; 
#done
