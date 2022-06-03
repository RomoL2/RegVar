from __future__ import print_function
import numpy as np
import numpy.random as rnd
def digitize(data, bins, dtype=int):
    if dtype == np.uint8:
        assert len(bins) <= 257
    res = np.zeros(data.size, dtype=dtype)

    #res = np.zeros(data.size, dtype=int)
    for n,x in enumerate(data.flatten()):
        i = 0
        j = len(bins)-1
        #print "value", x
        #c = 0
        while j - i > 1:
            pivot = max(1, int((j - i)/2 )) + i
            #print pivot, bins[pivot]
            #print "indices",i,j, pivot
            #print "values",'?', bins[i], bins[j], bins[pivot]
            
            if x >= bins[pivot]:
                i = pivot
            else:
                j = pivot
            #c += 1
            
        if x >= bins[j]:
            res[n] = j
        else:
            res[n] = i

        #print x,"->", res[n], "in {0} steps".format(c)
        #steps.append(c)
        
    #print np.array(steps).mean(), "average steps"
    return res.reshape(data.shape) + 1
        
if __name__ == "__main__":        
    bins = np.array([-1,1,3,5,10,15,20], dtype=np.float32)

    #data = np.array([0,2,4,6,7,8,9,10,16,30,100, -1], dtype=np.float32)
    data = np.array(rnd.random(100000000) * 22 - 1, dtype=np.float32)[np.newaxis,:]

    from time import time
    t0 = time()
    res2 = np.digitize(data, bins) - 1 
    print((time() - t0)* 1000.)
    print(res2)
    
    t0 = time()
    from .cyska import digitize_32fp_8bit
    res3 = digitize_32fp_8bit(data, bins) - 1
    print((time() - t0)* 1000.)
    print(res3)

    #res = digitize(data, bins)
    #print res
