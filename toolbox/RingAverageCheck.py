#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

# condition on the average over ring of latest values
# condition is defined as lambda avr,std: ... in constructor 

import numpy as np
#............................
#............................
#............................
class RingAverageCheck():
#...!...!..................
    def __init__(self, func,numCell=10, initVal=0):
        self.buf=np.full(numCell, initVal,dtype='float32')
        self.func=func  # condition used by check(.)
        self.n=numCell  # size of the data buffer
        self.i=0 # current index
        assert numCell>=2
#...!...!..................
    def update(self,x):
        i=self.i%self.n
        self.buf[i]=x
        self.i+=1
#...!...!..................
    def check(self):
        self.avr=np.mean(self.buf)
        self.std=np.std(self.buf)/np.sqrt(self.n)
        self.cond=self.func(self.avr,self.std)
        #print('Ring:check() mean=%.2e  std=%.2e  i=%d'%(self.avr,self.std,self.i), '==> func:',self.cond)
        return self.cond
#...!...!..................
    def dump(self):
        import inspect
        print('Ring func=',inspect.getsource(self.func))
        print('Ring i=%d buf.shape=%s'%(self.i,self.buf.shape))
        print('Ring buf',self.buf)


#=================================
#=================================
#   U N I T   T E S T
#=================================
#=================================

if __name__=="__main__":
    numCell=10; initVal=0.1
    ring=RingAverageCheck(func= lambda avr,std: avr<1.0, numCell=numCell,initVal=initVal )
    ring.dump()
    for j in range(4): # outer loop to check condition
        for i in range(5): # inner loop to update values
            x=np.random.uniform()+j/3.
            ring.update(x)
        #ring.dump()
        print('M:j=%d decision=%r'%(j,ring.check()))
    
    print('M:done')
