#Steven Horvath
#EE466 Exam 2
#Q1_AsymmetricFilter

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

np.set_printoptions(precision=500, suppress=True)

sample_freq = 100000
num_taps = 151
cutoff_freq = 10000
center_freq = 30000
#high pass filter: specify 'pass_zero=False'
h = signal.firwin(num_taps, cutoff_freq, pass_zero=False, nyq= sample_freq/2)
# Shift the filter in frequency by multiplying by exp(j*2*pi*f0*t)
f0 = center_freq # amount we will shift
Ts = 1.0/sample_freq # sample period
t = np.arange(0.0, Ts*len(h), Ts) # time vector. args are (start, stop, step)
exponential = np.exp(2j*np.pi*f0*t) # this is essentially a complex sine wave

h_new = h * exponential # do the shift

print(repr(h_new))

# plot impulse response
plt.figure('impulse')
plt.plot(np.real(h_new), '.-')
plt.plot(np.imag(h_new), '.-')
plt.legend(['real', 'imag'], loc=1)

# plot the frequency response
H = np.abs(np.fft.fft(h_new, 1024)) # take the 1024-point FFT and magnitude
H = np.fft.fftshift(H) # make 0 Hz in the center
w = np.linspace(-sample_freq/2, sample_freq/2, len(H)) # x axis
plt.figure('freq')
plt.plot(w, H, '.-')
plt.xlabel('Frequency [Hz]')
plt.show()
