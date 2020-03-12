import numpy as np
import matplotlib.pyplot as plt

#Given values
loc_star = [0, 0.5]         #Star's center
loc_stain = [0.025, 0.535]    #Stain's center
rho_star = 0.1              #Normalized radius of star
rho_stain = 0.02            #Normalized radius of stain

#This function builds the location vectors of images
def lens(loc, rho):
    n = 1000
    alpha = np.linspace(0, 2*np.pi, n).reshape((n, 1))                #Azimuthal angle
    location = np.ones((n, 2), float) * loc
    u = rho * np.hstack((np.cos(alpha), np.sin(alpha))) + location    #Source
    U = np.linalg.norm(u, axis=1)

    theta_p = (U + np.sqrt(U**2 + 4))/2
    pi = u * (theta_p/U)[:, None]          #Positive image

    theta_n = (U - np.sqrt(U**2 + 4))/2
    ni = u * (theta_n/U)[:, None]          #Negative image

    return u, pi, ni

#Plotting
star, p_star, n_star = lens(loc_star, rho_star)
stain, p_stain, n_stain = lens(loc_stain, rho_stain)

plt.title("$\\rho_{\star}$ = %.1f\n$u_{0}$ = %.1f" % (rho_star, loc_star[1]))
plt.fill(star[:,0], star[:,1], label="Source")
plt.fill(p_star[:,0], p_star[:,1], label="Positive image")
plt.fill(n_star[:,0], n_star[:,1], label="Negative image")
plt.legend()

plt.fill(stain[:,0], stain[:,1], color='yellow')
plt.fill(p_stain[:,0], p_stain[:,1], color='yellow')
plt.fill(n_stain[:,0], n_stain[:,1], color='yellow')

plt.gca().set_aspect('equal')
plt.xlim(-1.7, 1.7)
plt.ylim(-1, 1.5)
plt.show()
