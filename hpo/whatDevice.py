#!/usr/bin/env python3
import torch
import sys
print('M:torch:',torch.__version__,',python:',sys.version)
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print ('Available devices ', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i,torch.cuda.get_device_name(i))
  
