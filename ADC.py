import matplotlib.pyplot as plt
import numpy as np
from random import random
from numpy import sin
from math import pi

T_LEN=1 #length of generated signal in s
T_RES=1000 #samples per s

def wh_noise(t_series, ampl):
    for i in range(len(t_series)):
        t_series[i]+=random()
    return t_series

def freq_domain(y,t):
    y=np.real(np.fft.fft(y))
    y=y[range(len(y)/2)]
    y=abs(y)
    y=y/max(y)
    f=np.arange(len(y))*(float(T_RES)/len(y))
    #frequencies in Hz
    plt.plot(f,y)

#Time series, in ns
v=np.zeros(T_LEN*T_RES)
t=np.linspace(0,T_LEN,T_RES*T_LEN)

#v=wh_noise(v,1.0)
v=sin(2*pi*t)

v_fft=freq_domain(v,t)

plt.subplot(2,1,1)
plt.plot(t,v)
plt.subplot(2,1,2)
freq_domain(v,t)
plt.show()
