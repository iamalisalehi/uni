import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

#get sigma
sigma_file = os.path.abspath(os.path.join(sys.path[0], 'sigma2.out'))
mag, sigma = np.loadtxt(sigma_file, unpack=True)
s0 = sigma[0]
mag0 = mag[0]

def fitted(mag, a, b, c):
    global s0, mag0
    return a * np.exp(b * (mag/mag0)**c) + s0


popt, pcov = op.curve_fit(fitted, mag, sigma, method='trf')

print(popt)
print(mag0, s0)

def Sigma(mag):
    return 6.51e-09 * np.exp(9.27 * (mag/13.31)**9.35e-01) + 0.001

plt.plot(mag, sigma, '.')
plt.plot(mag, Sigma(mag))
plt.yscale('log')
plt.show()


magni_file = os.path.abspath(os.path.join(sys.path[0], os.pardir, 'ev04/files/magni.txt'))
mag = np.loadtxt(magni_file)

yerrd = Sigma(mag[:,-2])

plt.figure()
plt.subplot(211)
plt.errorbar(mag[:,0], mag[:,-2], yerr=yerrd, fmt='.')
plt.gca().invert_yaxis()
plt.ylabel('mag')

plt.subplot(212)
plt.plot(mag[:,0], yerrd)
plt.xlabel('t [days]')
plt.ylabel('error')

plt.show()

mag_and_error_file = os.path.abspath(os.path.join(sys.path[0], os.pardir, 'ev04/files/W149.out'))
f = np.stack((mag[:,0], mag[:,-2], yerrd), axis=-1)
np.savetxt(mag_and_error_file, f)
