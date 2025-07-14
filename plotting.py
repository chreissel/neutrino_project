from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
my_viridis = cm.get_cmap("viridis", 1024).with_extremes(under="white")

def make_bias(idx, var, true, pred):

    label = var.split('_')[0]
    unit = var.split('_')[1]

    fig = plt.figure()
    #plt.scatter(true[:,idx-1],pred[:,idx-1])
    heatmap, xedges, yedges = np.histogram2d(true[:,idx-1], pred[:,idx-1], bins=100)
    plt.imshow(heatmap.T, origin='lower', cmap=my_viridis, aspect='auto',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=1)
    plt.colorbar(label='Density')
    plt.xlabel('true ' + label + ' '+ '['+unit+']')
    plt.ylabel('predicted ' + label + ' ' +'['+unit+']')
    return fig


def make_resolution(idx, var, true, pred):
    label = var.split('_')[0]
    unit = var.split('_')[1]

    nentries = true.shape[0]
    data = true[:,idx-1]-pred[:,idx-1]

    fig = plt.figure()
    hist, bins, _ = plt.hist(data,bins=100,weights=np.ones(nentries)*1/float(nentries))
    def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=[max(hist), np.mean(data), np.std(data)])
    amplitude_fit, mean_fit, stddev_fit = popt

    x_fit = np.linspace(min(bins), max(bins), 100)
    plt.plot(x_fit, gaussian(x_fit, *popt), 'r-', label='Fit: mu={:.4f}, std={:.4f}'.format(mean_fit, stddev_fit))
    plt.xlabel('residual '+label+ ' '+'['+unit+']')
    plt.ylabel('A.U.')
    plt.legend()
    return fig
