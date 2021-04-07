from . import values as vl
import time
import os
import numpy as np
from numba import jit, f8, i8, b1, void

#張力判定用の関数(F=kΔx -> σ=EA/l Δx = E/l Δxなので現在の質点座標さえわかればいいはず ＋ 操作距離の取得)
@jit(i8(f8[:,:]))
def calctension(vectors):
    ret = 0
    for i in range(1,len(vectors)-1):
        if(vectors[i,4] == 2):
            l_0 = vl.l_0_short
        else:
            l_0 = vl.l_0
        rM = ((vectors[i,0]-vectors[i-1,0])**2 + (vectors[i,1]-vectors[i-1,1])**2)**(1/2)
        s = vl.E/vl.l_0 * max(rM-l_0,0)
        if(abs(s) > vl.sigma_par_p):
            ret = 1
            break
        elif(abs(s) < vl.sigma_par_m):
            ret = -1
            break
    return ret        
