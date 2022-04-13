#Steven Horvath
#EE466 Exam 2
#Q1_SymmetricFilter

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

np.set_printoptions(precision=20, linewidth=10000, suppress=True)

sample_freq = 100000
num_taps = 151
cutoff_freq = 10000

h = signal.firwin(num_taps, cutoff_freq, nyq = sample_freq/2)
print(repr(h))

#plot impulse response
plt.figure('impulse')
plt.plot(h, '.-')
plt.show()

#plot frequency response

H = np.abs(np.fft.fft(h, 1024))
H = np.fft.fftshift(H)
w = np.linspace(-sample_freq/2, sample_freq/2, len(H))
plt.figure('freq')
plt.plot(w, H, '.-')
plt.show()
