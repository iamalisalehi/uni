import numpy as np
import matplotlib.pyplot as plt

#Given values
rho_star = 0.1  #Normalized radius of star
u0 = 0.2        #Normalized impact parameter
t0 = 0
tE = 20         #days
I0 = 1200

#Main function
def lens(t, rho):
    n = 6800
    phi = np.linspace(0, 2*np.pi, n)    #Azimuthal angle
    phi_r = phi.reshape((n, 1))
    location = np.ones((n, 2), float) * [(t-t0)/tE, u0]

    r = rho * np.hstack((np.cos(phi_r), np.sin(phi_r))) + location
    u = np.linalg.norm(r, axis=1)

    A = (u**2 + 2)/(u * np.sqrt(u**2 + 4))  #Magnification
    I = (I0/n) * np.sum(A)                  #First Stokes parameter
    U = (I0/n) * np.sum(A * np.cos(2 * phi))    #Second Stokes parameter
    Q = (I0/n) * np.sum(A * np.sin(2 * phi))    #Third Stokes parameter

    return I, Q, U

#Initialization
Nt = 200
Nr = 31
t = np.linspace(-30, 30, Nt)
rho = np.linspace(0, rho_star, Nr)
I = np.zeros(Nt)
Q = np.zeros(Nt)
U = np.zeros(Nt)
p = np.zeros(Nt)

#Calculation loop
for j in range(Nt):
    for k in range(1, Nr):
        S = np.pi * (rho[k]**2 - rho[k-1]**2)   #Area
        Ij, Qj, Uj = lens(t[j], rho[k])
        I[j] += (Ij * S)
        Q[j] += (Qj * S)
        U[j] += (Uj * S)

    p[j] = np.sqrt(Q[j]**2 + U[j]**2)/I[j]      #The degree of polarization


#Plotting
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 30,
        }

plt.subplot(211)
plt.plot(t, I/Nr, 'darkred', linewidth=4)
plt.tick_params(labelsize=25, width=3, length=10)
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['left'].set_linewidth(3)
plt.ylabel('Normalised Flux', fontdict=font)
plt.ylim([1, 7])

plt.subplot(212)
plt.plot(t, p*100/Nr, 'darkred', linewidth=4)
plt.tick_params(labelsize=25, width=3, length=10)
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['left'].set_linewidth(3)
plt.ylabel('% Polarization', fontdict=font)
plt.ylim([0, 1])
plt.xlabel('t [days]', fontdict=font)
plt.show()
