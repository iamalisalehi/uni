import os
import sys
import numpy as np
import MulensModel as mm
import scipy.optimize as op
import matplotlib.pyplot as plt

#Loading main files
event_dir = 'ev03'
Mag = np.loadtxt(os.path.join(sys.path[0], event_dir, 'files/magni.txt'))
Disk = np.loadtxt(os.path.join(sys.path[0], event_dir, 'files/disk.txt'))
Params = np.loadtxt(os.path.join(sys.path[0], event_dir, 'files/param.txt'))
mag_file = os.path.join(sys.path[0], event_dir, 'files/W149.out')
mag_data = mm.MulensData(file_name=mag_file)

#Building model and event with initial parameters
t0 = 0
u0 = 0.02
tE = 100
dt = 70/3000

pspl_model = mm.Model({'t_0': t0, 'u_0': u0, 't_E': tE})
pspl_model.set_datasets([mag_data])
event = mm.Event(datasets=mag_data, model=pspl_model)

#Fitting
parameters_to_fit = ["t_0", "u_0", "t_E"]
initial_guess = [t0, u0, tE]

def chi2_for_model(theta, event, parameters_to_fit):
    for (key, parameter) in enumerate(parameters_to_fit):
        setattr(event.model.parameters, parameter, theta[key])
    return event.get_chi2()

result = op.minimize(chi2_for_model, x0=initial_guess, args=(event, parameters_to_fit),
                     method='Nelder-Mead', options={'maxiter': 2000})

print("PSPL Model\n")
(fit_t0, fit_u0, fit_tE) = result.x

# Save the best-fit parameters
chi2 = chi2_for_model(result.x, event, parameters_to_fit)
print('Chi2 = {0:.2f}'.format(chi2))

# Plot the fitted model with the data
font = {'family': 'serif',
        'weight': 'normal',
        'size': 30,
        }

label_params_out = {'font.size': 30,
                    'font.family': 'serif',
                   }

label_params_in = {'font.size': 16,
                   'font.family': 'serif',
                  }

plt.figure()
plt.rcParams.update(label_params_out)

event.plot_data()
event.plot_model(c='red', dt=dt)

plt.title('$t_{0}$ =  %.2f,\t$u_{0}$ = %.4f,\t$t_{E}$ = %.3f\n$\\chi^{2}$ = %.2f' % (fit_t0, fit_u0, fit_tE, chi2), fontdict=font)
plt.ylim([10.9, 8.1])
plt.tick_params(labelsize=25, width=3, length=10)
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['left'].set_linewidth(3)

plt.rcParams.update(label_params_in)
plt.axes([0.643, 0.67, 0.25, 0.2])

event.plot_data()
event.plot_model(c='red', linewidth=1.5, t_range=[-5.2, 4.8], dt=dt)

plt.ylim([8.5, 8.12])
plt.xlim([-3.6, 3.4])
plt.xticks([])
plt.yticks([])

plt.axes([0.14, 0.6, 0.22, 0.22])

plt.title('Path')
plt.fill(Disk[:,8], Disk[:,9], 'greenyellow')
plt.fill(Disk[:,6], Disk[:,7], 'white')
plt.fill(Disk[:,4], Disk[:,5], 'lightcoral')
plt.fill(Disk[:,2], Disk[:,3], 'white')
plt.fill(Disk[:,0], Disk[:,1], 'gold')
plt.plot(Mag[:,1], Mag[:,2], 'k')
plt.gca().set_aspect('equal')
plt.xticks([])
plt.yticks([])

plt.show()
