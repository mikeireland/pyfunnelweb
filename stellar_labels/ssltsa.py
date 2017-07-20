"""
Semi-Supervised Local Tangent Space Alignment (SS-LTSA)

Algorithm from "Yang et al. 2006"
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import astropy.io.fits as pyfits
from sklearn.neighbors import NearestNeighbors
import pdb
plt.ion()

# ----------------------------------------------------------------------------------------
# Initialisation
# ----------------------------------------------------------------------------------------

n_neighbors = 40 #Called "k" in Yang (2006)
n_dim = 4
n_known = 570*2

# Regularisation parameter: --> inf if fully confident, = 0 if unsupervised
beta = 1 #e3 #1e3 or 1

# Import the data
# 5700 spectra with 4096 pixels
spect = pyfits.getdata('ccd1.fits')
spect += np.random.normal(size=spect.shape)*1e-2
params = pyfits.getdata('ccd1.fits', 1) #params.names are the names of parameters.

# Normalise the spectra
for s in spect:
    s /= np.mean(s)

#Pick some random reference spectra.
#ref_obj = (np.random.random(500)*spect.shape[0]).astype(int)

# ----------------------------------------------------------------------------------------
# SS-LTSA
# ----------------------------------------------------------------------------------------

print("Finding pairwise distances")

# Determine the nearest neighbours
nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute").fit(spect)
neighbour_distances, neighbors = nbrs.kneighbors(spect)


# Squared sum of each spectrum - [n_spectra, 1]
#ssum = np.sum(spect**2, 1)

# Steps
# 1 - N x N square matrix of squared sum (horizontal)
# 2 - N x N square matrix of squared sum (vertical)
# 3 - 2 x dot product of spectra & spectra.T (symmetric about diagonal)
#dist_squared = np.tile(ssum, spect.shape[0]).reshape((spect.shape[0], spect.shape[0])) + \
    #np.repeat(ssum, spect.shape[0]).reshape((spect.shape[0], spect.shape[0])) - \
    #2*np.dot(spect, spect.T)
    
# Initialise I to be identity matrix    
I_neighborhood = np.eye(n_neighbors)

print("Creating Matrix M")

# Initialise M matrix to be zeroes
M_align = np.zeros( (spect.shape[0], spect.shape[0]) )

#Now find the neighborhood of each spectrum
#neighbors = np.zeros( (spect.shape[0], n_neighbors), dtype=np.int)

# Find the principal components
for i in range(spect.shape[0]):
    # Determine the n_neighbours nearest neighbours for each spectrum
    #s = np.argsort(dist_squared[i])
    #neighbors[i] = s[:n_neighbors]
    
    # Find the mean of each spectral pixel in the neighbourhood
    X = spect[neighbors[i],:]
    X_mn = X.mean(axis=0)
    
    # Subtract the mean of each spectral pixel (so the new mean is zero)
    for j in range(n_neighbors):
        X[j] -= X_mn 
    
    # Obtain eigenvalues and eigenvectors of the local neighbourhood    
    cov = np.dot(X, X.T)
    W, V = np.linalg.eigh(cov)
    
    # G_i = [e/sqrt(k), g_i1, g_i2,...,g_id]
    G = np.append(np.ones( (n_neighbors,1) )/np.sqrt(n_neighbors), V[:,-n_dim:], 1)
    
    # M(I_i,I_i) <-- M(I_i,I_i) + I _ G_i G_i^T
    M_align[np.meshgrid(neighbors[i], neighbors[i], indexing='ij')] +=\
        I_neighborhood - np.dot(G, G.T)

# Incorporate prior information

# Select those objects with "known" parameters
known_objects = (np.random.random( n_known )*spect.shape[0]).astype(int)

# Initialise results vector
y_ltsa = np.zeros( (n_dim, spect.shape[0]) )

# For each dimension, 
for i in range(np.min([len(params.names), n_dim])):
    y_ltsa[i, known_objects] = beta*params[params.names[i]][known_objects] #!!! includes beta

# Determine unknown objects
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

# ----------------------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------------------
#Median absolute deviations
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

# Plotting
#log_teff = np.log10(np.minimum(y_ltsa[0,unknown_objects], 30e3))
log_teff = np.minimum(y_ltsa[0,unknown_objects], 30e3)
plt.clf()
plt.scatter(y_ltsa[2,unknown_objects], y_ltsa[3,unknown_objects],c=log_teff, cmap=cm.jet_r,s=1)
cbar = plt.colorbar()
cbar.set_label(r"T${_{\rm Eff}}$ (K)")
plt.scatter(y_ltsa[2,known_objects], y_ltsa[3,known_objects], color='black', s=20, marker='x')
plt.axis([-2.5,1,-0.2,0.6])
plt.xlabel('[M/H]')
plt.ylabel(r'[$\alpha$/Fe]')
plt.legend(loc="best")
plt.savefig("ssltsa_prototype.png", bbox_inches="tight", dpi=500)