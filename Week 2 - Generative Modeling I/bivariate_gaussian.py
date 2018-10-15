

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import ipywidgets as widgets
import IPython.display as display
from ipywidgets import interact, interactive, fixed, interact_manual, IntSlider


def bivariate_plot(var1, var2, corr):
   #
    # Set parameters of Gaussian
    mu = [0,0]
    covariance = corr * np.sqrt(var1) * np.sqrt(var2)
    sigma = [[var1,covariance], [covariance,var2]]
    np.set_printoptions(precision=2)
    print "Covariance matrix:"
    print np.around(sigma, decimals=2)
    #
    # Draw samples from the distribution
    n = 100
    x = np.random.multivariate_normal(mu,sigma,size=n)
    #
    # Set up a plot for the samples and the density contours
    lim = 10.0
    plt.xlim(-lim, lim) # limit along x1-axis
    plt.ylim(-lim, lim) # limit along x2-axis    
    plt.axes().set_aspect('equal', 'datalim')
    #
    # Plot the sampled points as blue dots
    plt.plot(x[:,0], x[:,1], 'bo')
    #
    # To display contour lines, first define a fine grid
    res = 200
    xg = np.linspace(-lim, lim, res)
    yg = np.linspace(-lim, lim, res)
    z = np.zeros((res,res))
    # Compute the density at each grid point
    rv = multivariate_normal(mean=mu, cov=sigma)
    for i in range(0,res):
        for j in range(0,res):
            z[j,i] = rv.logpdf([xg[i], yg[j]]) 
    sign, logdet = np.linalg.slogdet(sigma)
    normalizer = -0.5 * (2 * np.log(6.28) + sign * logdet)
    # Now plot a few contour lines of the density
    for offset in range(1,4):
        plt.contour(xg,yg,z, levels=[normalizer - offset], colors='r', linewidths=2.0, linestyles='solid')

    # Finally, display
    plt.show()



if __name__ == "__main__":
    var1 = (1,9)
    var2 = (1,9) 
    corr=(-0.95,0.95,0.05)
    bivariate_plot(var1, var2, corr)