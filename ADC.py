from matplotlib.pyplot import plot, show, subplot, semilogx, loglog, xlabel,ylabel,subplots_adjust
import numpy as np
from numpy import sin, exp, sqrt
from math import pi

T_LEN=20 #length of generated signal in s
T_RES=1000 #samples per s

def wh_noise(v_series, ampl):
    for i in range(1,len(v_series)):
        v_series[i]+=ampl*np.random.normal()
    return v_series

def pulse(v_series, ampl, start, width):
    for i in range(int(start),int(start+width+1)):
        v_series[i]+=ampl
    return v_series    

def freq_domain(y,t):
    global f
    y=np.real(np.fft.fft(y))
    y=y[range(len(y)/2)]
    y=abs(y)/max(y)
    f=0.5*np.arange(len(y))*(float(T_RES)/len(y))
    #frequencies in Hz
    semilogx(f,y)

def filt(y,freq_curve):
    y=np.fft.fft(y)
    freq_curve=freq_curve/max(freq_curve)
    y*=freq_curve
    return np.fft.ifft(y)

#Time series, in ns
v=np.zeros(T_LEN*T_RES)
t=np.linspace(0,T_LEN,T_RES*T_LEN)

#filter curve (freq domain)
filt_curve=1/sqrt(1+1000*t**2)
filt_curve=abs(filt_curve)

#add white noise
v=wh_noise(v,2.0)#+10*sin(t)+5*sin(5*t)
v=pulse(v, 100.0, 5.0*T_RES, 1.0*T_RES)
#v=sin(2*pi*t) #FFT test
vf=freq_domain(v,t)

subplot(5,1,1) #original signal
plot(t,v)
#xlabel('Time')
ylabel('Amplitude')
subplots_adjust(hspace=0.4)

subplot(5,1,2) #freq domain
freq_domain(v,t)
#xlabel('Freq')
ylabel('Amplitude')

subplot(5,1,3) #low pass filter
loglog(f,filt_curve[range(len(filt_curve)/2)]/max(filt_curve))
#xlabel('Freq')
ylabel('Gain')

#filter
v=filt(v,filt_curve)

subplot(5,1,4) #filtered signal
plot(t,v)
#xlabel('Time')
ylabel('Amplitude')

subplot(5,1,5) #filtered freq domain
freq_domain(v,t)
#xlabel('Freq')
ylabel('Amplitude')
show()
