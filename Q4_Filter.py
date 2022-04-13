#Steven Horvath
#EE466 Exam 2
#Q4_Filter

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

np.set_printoptions(precision=20, linewidth=10000, suppress=True)

#wide enough to show frequency response
sample_freq = 350000
#Trying to imitate an analog filter, so I used more taps than before
num_taps = 551
#calculated cutoff freq
cutoff_freq = 159155

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
