__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import time, os
import ruamel.yaml  as yaml
import warnings
#1warnings.simplefilter('ignore', yaml.error.MantissaNoDotYAML1_1Warning)

from pprint import pprint

import csv
import h5py
import hashlib
import  logging

#...!...!..................
def read_yaml(ymlFn,verb=1,logger=False):
    if verb:
        txt='U:  read  yaml:'+ymlFn
        logging.info(txt) if logger  else print(txt)        

    ymlFd = open(ymlFn, 'r')
    bulk=yaml.load( ymlFd, Loader=yaml.CLoader)
    ymlFd.close()

    return bulk

#...!...!..................
def write_yaml(rec,ymlFn,verb=1):
    start = time.time()
    ymlFd = open(ymlFn, 'w')
    yaml.dump(rec, ymlFd, Dumper=yaml.CDumper)
    ymlFd.close()
    xx=os.path.getsize(ymlFn)/1024
    if verb:
            print('  closed  yaml:',ymlFn,' size=%.1f kB'%xx,'  elaT=%.1f sec'%(time.time() - start))



#...!...!..................
def write_data_hdf5(dataD,outF,verb=1):
    h5f = h5py.File(outF, 'w')
    start = time.time()
    if verb>0:
            print('saving data as hdf5:',outF)

    for item in dataD:
        rec=dataD[item]
        if verb>1: print('x=',item,type(rec))
        h5f.create_dataset(item, data=rec)
        if verb>0:print('h5-write :',item, rec.shape,rec.dtype)
    h5f.close()
    xx=os.path.getsize(outF)/1048576
    print('closed  hdf5:',outF,' size=%.2f MB, elaT=%.1f sec'%(xx,(time.time() - start)))

    
#...!...!..................
def read_data_hdf5(inpF,verb=1):
    if verb>0:
            print('read data from hdf5:',inpF)
            start = time.time()
    h5f = h5py.File(inpF, 'r')
    objD={}
    for x in h5f.keys():
        if verb>1: print('item=',x,type(h5f[x]),h5f[x].shape)
        obj=h5f[x][:]
        if verb>0: print('read ',x,obj.shape,obj.dtype)
        objD[x]=obj
    if verb>0:
            print(' done h5, num rec:%d  elaT=%.1f sec'%(len(objD),(time.time() - start)))

    h5f.close()
    return objD


#...!...!..................
def read_one_csv(fname,delim=','):
    print('read_one_csv:',fname)
    tabL=[]
    with open(fname) as csvfile:
        drd = csv.DictReader(csvfile, delimiter=delim)
        print('see %d columns'%len(drd.fieldnames),drd.fieldnames)
        for row in drd:
            tabL.append(row)
            
        print('got %d rows \n'%(len(tabL)))
    #print('LAST:',row)
    return tabL,drd.fieldnames
    # use case
    # mapT,_=read_one_csv(inpF)
    #for rec in mapT:
    #    if args.cellName != rec['short_name']: continue
 

#...!...!..................
def write_one_csv(fname,rowL,colNameL):
    print('write_one_csv:',fname)
    print('export %d columns'%len(colNameL), colNameL)
    with open(fname,'w') as fou:
        dw = csv.DictWriter(fou, fieldnames=colNameL)#, delimiter='\t'
        dw.writeheader()
        for row in rowL:
            dw.writerow(row)    


''' - - - - - - - - - 
Offset-aware time, usage:

*) get current date:
t1=time.localtime()   <type 'time.struct_time'>

*) convert to string: 
timeStr=dateT2Str(t1)

*) revert to struct_time
t2=dateStr2T(timeStr)

*) compute difference in sec:
t3=time.localtime()
delT=time.mktime(t3) - time.mktime(t1)
totSeconds=delT..total_seconds()
'''

#...!...!..................
def dateT2Str(xT):  # --> string
    nowStr=time.strftime("%Y%m%d_%H%M%S_%Z",xT)
    return nowStr

#...!...!..................
def dateStr2T(xS):  #  --> datetime
    yT = time.strptime(xS,"%Y%m%d_%H%M%S_%Z")
    return yT

#...!...!..................
def md5hash(text):
    hao = hashlib.md5(text.encode())
    hastr=hao.hexdigest()
    return hastr,hastr[-8:]


#...!...!..................    
def build_name_hash(prjName,nameSuffix):
    t1=time.localtime()
    dataNF='%s_%s_%s'%(prjName,nameSuffix,dateT2Str(t1))
    dataNH,dataNH8=md5hash(dataNF)
    tc={}
    tc['task_name']=prjName
    tc['hash32_name']=dataNH
    tc['hash8_name']=dataNH8
    tc['full_name']=dataNF
    dataName_short='%s_%s'%(tc['task_name'],dataNH8)
    tc['short_name']=dataName_short
    print('hash of data name=',dataName_short,' full data name:',tc['full_name'])
    return tc

#...!...!..................
def expand_dash_list(kL=['']): 
    # expand list if '-' are present
    #   if present [,] : removes them and only expands the portion inside
    #print('kL',type(kL))
    assert type(kL)==type([])
    kkL=[]
    for x in kL:
        #print('aa',x, '-' not in x)
        if '-' not in x:  # nothing to expand
            kkL.append(x) ; continue

        core=''
        if '[' in x and x[-1]==']':
            j=x.index('[')
            core=x[:j]; y=x[j+1:-1]
            #print('mm',core,y)
            x=y
        xL=x.split('-')
        #print('b',xL)
        for i in range(int(xL[0]),int(xL[1])+1):
            z='%s%d'%(core,i)
            kkL.append(z)
    #print('EL:',kL,'  to ',kkL)
    return kkL
