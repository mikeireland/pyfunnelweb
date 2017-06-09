"""
n_neighbors 20, n_known 570
Teff had MAD: 139.1, StDev: 1777.5
logg had MAD:  0.07, StDev:  0.34
Fe_H had MAD:  0.07, StDev:  0.20
alpha_Fe had MAD:  0.06, StDev:  0.14

n_neighbors 20, n_known 570
Teff had MAD: 208.9, StDev: 1728.1
logg had MAD:  0.11, StDev:  0.40
Fe_H had MAD:  0.07, StDev:  0.19
alpha_Fe had MAD:  0.05, StDev:  0.14

n_neighbors 20, n_known 1140
Teff had MAD: 445.1, StDev: 2686.3
logg had MAD:  0.17, StDev:  0.58
Fe_H had MAD:  0.07, StDev:  0.23
alpha_Fe had MAD:  0.04, StDev:  0.12

n_neighbors 20, n_known 1140
Teff had MAD: 388.4, StDev: 2850.1
logg had MAD:  0.16, StDev:  0.53
Fe_H had MAD:  0.06, StDev:  0.20
alpha_Fe had MAD:  0.04, StDev:  0.12

n_neighbors 20, n_known 285
Teff had MAD: 123.8, StDev: 3169.5
logg had MAD:  0.09, StDev:  0.43
Fe_H had MAD:  0.10, StDev:  0.27
alpha_Fe had MAD:  0.08, StDev:  0.16

n_neighbors 10, n_known 570
Teff had MAD: 248.4, StDev: 2957.9
logg had MAD:  0.12, StDev:  0.73
Fe_H had MAD:  0.06, StDev:  0.24
alpha_Fe had MAD:  0.04, StDev:  0.16

n_neighbors 10, n_known 570
Teff had MAD: 160.6, StDev: 2542.1
logg had MAD:  0.07, StDev:  0.61
Fe_H had MAD:  0.06, StDev:  0.26
alpha_Fe had MAD:  0.05, StDev:  0.16

n_neighbors 15, n_known 570
Teff had MAD: 270.7, StDev: 2862.6
logg had MAD:  0.12, StDev:  0.55
Fe_H had MAD:  0.06, StDev:  0.19
alpha_Fe had MAD:  0.05, StDev:  0.14

n_neighbors 25, n_known 570
Teff had MAD: 205.4, StDev: 2022.6
logg had MAD:  0.13, StDev:  0.46
Fe_H had MAD:  0.08, StDev:  0.18
alpha_Fe had MAD:  0.07, StDev:  0.14

n_neighbors 30, n_known 570
Teff had MAD: 244.2, StDev: 1991.7
logg had MAD:  0.13, StDev:  0.44
Fe_H had MAD:  0.10, StDev:  0.23
alpha_Fe had MAD:  0.07, StDev:  0.14


"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import astropy.io.fits as pyfits
import pdb
plt.ion()

n_neighbors = 40 #Called "k" in Yang (2006)
n_dim = 4
n_known=570*2
beta = 1e3 #1e3 or 1

#Import the data
spect = pyfits.getdata('ccd1.fits')
spect += np.random.normal(size=spect.shape)*1e-2
params = pyfits.getdata('ccd1.fits', 1) #params.names are the names of parameters.

for s in spect:
    s /= np.mean(s)

#Pick some random reference spectra.
ref_obj = (np.random.random(500)*spect.shape[0]).astype(int)

print("Finding pairwise distances")
ssum = np.sum(spect**2, 1)
dist_squared = np.tile(ssum, spect.shape[0]).reshape((spect.shape[0], spect.shape[0])) + \
    np.repeat(ssum, spect.shape[0]).reshape((spect.shape[0], spect.shape[0])) - \
    2*np.dot(spect, spect.T)
I_neighborhood = np.eye(n_neighbors)
M_align = np.zeros( (spect.shape[0], spect.shape[0]) )

print("Creating Matrix M")

#Now find the neighborhood of each spectrum
neighbors = np.zeros( (spect.shape[0], n_neighbors), dtype=np.int)
for i in range(spect.shape[0]):
    s = np.argsort(dist_squared[i])
    neighbors[i] = s[:n_neighbors]
    #Find the principle components.
    X = spect[neighbors[i],:]
    X_mn = X.mean(axis=0)
    for j in range(n_neighbors):
        X[j] -= X_mn 
    cov = np.dot(X, X.T)
    W, V = np.linalg.eigh(cov)
    G = np.append(np.ones( (n_neighbors,1) )/np.sqrt(n_neighbors), V[:,-n_dim:], 1)
    M_align[np.meshgrid(neighbors[i], neighbors[i], indexing='ij')] +=\
        I_neighborhood - np.dot(G, G.T)

#OK... now we go through and add the constraints.
known_objects = (np.random.random( n_known )*spect.shape[0]).astype(int)


y_ltsa = np.zeros( (n_dim, spect.shape[0]) )
for i in range(np.min([len(params.names), n_dim])):
    y_ltsa[i, known_objects] = beta*params[params.names[i]][known_objects] #!!! includes beta

unknown_objects = np.where(y_ltsa[0] ==0)[0]

if beta == 1:
    print("Solving system of equations")
    #Equation (12) of Yang et al.
    M12 = M_align[np.meshgrid(unknown_objects, known_objects, indexing='ij')]
    M22 = M_align[np.meshgrid(unknown_objects, unknown_objects, indexing='ij')]
    for i in range(n_dim):
        y_ltsa[i, unknown_objects] = np.linalg.solve(M22, -np.dot(M12, params[params.names[i]][known_objects]))
else:
    M_nonsup = M_align.copy()
    M_align[known_objects, known_objects] += beta
    for i in range(n_dim):
        #Equation (21) of Yang et al.
        y_ltsa[i] = np.linalg.solve(M_align, y_ltsa[i])

#Median absolute deviations...
deviates = np.zeros_like(y_ltsa)
for i in range(n_dim):
    deviates[i] = y_ltsa[i] - params[params.names[i]]

mad_unknown = np.median(np.abs(deviates[:,unknown_objects]), axis=1)
std_unknown = np.std(deviates[:,unknown_objects], axis=1)

mad_known = np.median(np.abs(deviates[:,known_objects]), axis=1)
std_known = np.std(deviates[:,known_objects], axis=1)

print("n_neighbors {}, n_known {}".format(n_neighbors, n_known))
print(params.names[0] + " had unknown MAD: {:5.1f}, StDev: {:5.1f}".format(mad_unknown[0], std_unknown[0]))
for i in range(1,4):
    print(params.names[i] + " had unknown MAD: {:5.2f}, StDev: {:5.2f}".format(mad_unknown[i], std_unknown[i]))
print(params.names[0] + " had known MAD: {:5.1f}, StDev: {:5.1f}".format(mad_known[0], std_known[0]))
for i in range(1,4):
    print(params.names[i] + " had known MAD: {:5.2f}, StDev: {:5.2f}".format(mad_known[i], std_known[i]))

log_teff = np.log10(np.minimum(y_ltsa[0,unknown_objects], 30e3))
plt.clf()
plt.scatter(y_ltsa[2,unknown_objects], y_ltsa[3,unknown_objects],c=log_teff, cmap=cm.jet_r,s=1)
plt.colorbar()
plt.scatter(y_ltsa[2,known_objects], y_ltsa[3,known_objects], color='black', s=20, marker='x')
plt.axis([-2.5,1,-0.2,0.6])
plt.xlabel('[M/H]')
plt.ylabel(r'[$\alpha$/Fe]')




    