from . import values as vl
from . import equation
import time
import os
import numpy as np
from numba import jit, f8, i8, b1, void

#ルンゲクッタ
@jit(f8[:,:](f8[:,:],f8[:],f8[:],f8[:],f8[:],f8[:,:],f8,i8,i8,f8,i8))
def rk4(vectors,masses,spring,Areas,damps,climbers,dt,counter,counter_2,dl,case):
    N_vector = vectors.shape[0]
    temps = np.zeros((N_vector,6))
    k1 = equation.equation(vectors,spring,masses,Areas,damps,climbers,counter,counter_2,dl,case)*dt
    temps = vectors + k1/2.0
    k2 = equation.equation(temps,spring,masses,Areas,damps,climbers,counter,counter_2,dl,case)*dt
    temps = vectors + k2/2.0
    k3 = equation.equation(temps,spring,masses,Areas,damps,climbers,counter,counter_2,dl,case)*dt
    temps = vectors + k3
    k4 = equation.equation(temps,spring,masses,Areas,damps,climbers,counter,counter_2,dl,case)*dt
    vectors += (k1+2*k2+2*k3+k4)/6.0

    return vectors
