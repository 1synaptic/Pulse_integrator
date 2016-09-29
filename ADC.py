import matplotlib.pyplot as plt
import numpy as np
from numpy import sin
from math import pi

T_LEN=1 #length of generated signal in s
T_RES=10000 #samples per s

def wh_noise(t_series, ampl):
    for i in range(1,len(t_series)):
        t_series[i]+=ampl*np.random.normal()
    return t_series

def freq_domain(y,t):
    y=np.real(np.fft.fft(y))
    y=y[range(len(y)/2)]
    y=abs(y)/len(y)
    f=0.5*np.arange(len(y))*(float(T_RES)/len(y))
    #frequencies in Hz
    plt.plot(f,y)

def filt(y,freq_curve):
    y=np.real(np.fft.fft(y))
    freq_curve=freq_curve/max(freq_curve)
    y*=freq_curve
    return np.fft.ifft(y)
    
#Time series, in ns
v=np.zeros(T_LEN*T_RES)
t=np.linspace(0,T_LEN,T_RES*T_LEN)

#filter curve (freq domain)
filt_curve=np.linspace(0,100,len(v))

#add white noise
v=wh_noise(v,2.0)
#v=sin(2*pi*t) #FFT test

#filter
v=filt(v,filt_curve)

plt.subplot(2,1,1)
plt.plot(t,v)
plt.subplot(2,1,2)
freq_domain(v,t)
plt.show()
