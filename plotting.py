from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
my_viridis = cm.get_cmap("viridis", 1024).with_extremes(under="white")

# General formatting
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIG_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title

def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)
    
def get_label_unit(var):
    # Generate useful values for plotting
    # 
    # INPUTS
    #    var: the variable name from the original file (e.g., 'start_carrier_frequency_Hz')
    #      Will probably come from model.variables
    #
    # OUTPUTS
    #    label: axis label, without units
    #    unit: plotted unit
    #    diff_unit: unit for a plot of true-pred difference, which may be different than unit
    #    factor: factor corresponding to the unit
    #    diff_factor: factor corresponding to diff_unit
    
    # Interpret the variable name
    splits = var.split('_')
    label = ' '.join(splits[:-1])
    unit = splits[-1]
    diff_unit = splits[-1]
    
    factor = 1
    diff_factor = 1
    
    # We usually display the carrier frequency in GHz and the resolution in kHz
    if(var=='start_carrier_frequency_Hz'):
        factor = 1E9
        diff_factor = 1E3
        unit = 'GHz'
        diff_unit = 'kHz'
        
    # We usually display both the axial frequency and resolution in kHz
    if(var=='avg_axial_frequency_Hz'):
        factor = 1E3
        diff_factor = 1E3
        unit = 'kHz'
        diff_unit = 'kHz'
        
    return label, unit, diff_unit, factor, diff_factor

def make_bias(variables, true, pred):
    # Plot the true parameter vs. the pred parameter
    #
    # INPUTS
    #    variables: The list of all variables from the original file (e.g., 'start_carrier_frequency_Hz')
    #    true: array of true values from the model
    #    pred: array of predicted values from the model
    #
    # OUTPUTS
    #    fig: figure with the plot
    
    # There will be one plot per variable
    fig, ax = plt.subplots(1, len(variables), figsize=(4*len(variables), 4))
    
    # Loop over the variables
    for vind, var in enumerate(variables):
        
        var_label, var_unit, _, factor, _ = get_label_unit(var)
        
        # Divide by factor to get the desired units
        true_var = true[:, vind]/factor
        pred_var = pred[:, vind]/factor
        
        ax[vind].hist2d(true_var, pred_var, bins=((np.linspace(min(true_var), max(true_var), 300), np.linspace(min(true_var), max(true_var), 300))), cmin=1)
        ax[vind].set_xlabel('true ' + var_label + ' ['+var_unit+']')
        ax[vind].set_ylabel('pred ' + var_label + ' ['+var_unit+']')
        
    plt.tight_layout()

    return fig

def make_res(variables, true, pred):
    # Extract the resolution using a Gaussian fit
    #
    # INPUTS
    #    variables: The list of all variables from the original file (e.g., 'start_carrier_frequency_Hz')
    #    true: array of true values from the model
    #    pred: array of predicted values from the model
    #
    # OUTPUTS
    #    fig: figure with the plot
    
    # There will be one plot per variable
    fig, ax = plt.subplots(1, len(variables), figsize=(4*len(variables), 4))
    
    # Loop over the variables
    for vind, var in enumerate(variables):
        
        var_label, var_unit, diff_unit, factor, diff_factor = get_label_unit(var)
        
        # Divide by factor to get the desired units
        true_var = true[:, vind]
        pred_var = pred[:, vind]
        diff = (true_var-pred_var)/diff_factor
        
        hist, bins, _ = ax[vind].hist(diff,bins=500,weights=np.ones(len(true_var))*1/float(len(true_var)))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=[max(hist), np.mean(diff), np.std(diff)])
        amplitude_fit, mean_fit, stddev_fit = popt

        x_fit = np.linspace(min(bins), max(bins), 1000)
        ax[vind].plot(x_fit, gaussian(x_fit, *popt), 'r-', label='Fit: mu={:.4f}, std={:.4f}'.format(mean_fit, stddev_fit))
        ax[vind].set_xlim(mean_fit-5*stddev_fit, mean_fit+5*stddev_fit)
        ax[vind].set_xlabel('residual '+var_label+ ' '+'['+diff_unit+']')
        ax[vind].set_ylabel('A.U.')
        ax[vind].legend()
        
    plt.tight_layout()
    
    return fig
    
def make_energy_res(variables, true, pred):
    # Plot the energy resolution as a function of all variables.  The mean and stddev of the events in each bin are plotted
    # The goal is to determine whether the energy resolution is better in certain regions of the parameter space
    #
    # INPUTS
    #    variables: The list of all variables from the original file (e.g., 'start_carrier_frequency_Hz')
    #    true: array of true values from the model
    #    pred: array of predicted values from the model
    #
    # OUTPUTS
    #    fig: figure with the plot

    # There will be one plot per variable
    fig, ax = plt.subplots(1, len(variables), figsize=(4*len(variables), 4))
    
    eind = variables.index('energy_eV')
    energy_diff = true[:, eind]-pred[:, eind]
    
    # Loop over the variables
    for vind, var in enumerate(variables):
        var_label, var_unit, _, factor, _ = get_label_unit(var)
        
        # Divide by factor to get the desired units
        true_var = true[:, vind]/factor
        pred_var = pred[:, vind]/factor
        
        # Define 20 bins across the full parameter space over which to take means and stdevs
        var_bins = np.linspace(min(true_var), max(true_var), 20)
        bincenters = (var_bins[:-1] + var_bins[1:]) / 2
        
        # Find the indices corresponding to each of the variable bins
        idxs_all = [np.where((true_var >= var_bins[i]) & (true_var <= var_bins[i+1])) for i in range(len(var_bins)-1)]
        # Find the energy differences corresponding to these bins, then take the mean and stddev
        energy_diffs = [np.squeeze(true[idxs, eind]-pred[idxs, eind]) for idxs in idxs_all] 
        means = [np.mean(energy_diff) for energy_diff in energy_diffs]
        stds = [np.std(energy_diff) for energy_diff in energy_diffs]
        
        # The error bars are the stddevs
        ax[vind].errorbar(bincenters, means, stds, color='k', marker='o', ls='')
        ax[vind].axhline(0, color='r', ls='--', lw='2')
        # This region corresponds to the desired resolution of 0.3 eV stddev
        ax[vind].fill_between(var_bins, np.ones(len(var_bins))*-0.3, np.ones(len(var_bins))*0.3, alpha=0.5, label="0.3 eV std")
        ax[vind].legend()
        ax[vind].set_xlim(min(var_bins), max(var_bins))
        ax[vind].set_ylim(-3,3)
        ax[vind].set_xlabel('true ' + var_label + ' ['+var_unit+']')
        ax[vind].set_ylabel('true-pred energy [eV]')
        
    plt.tight_layout()
    
    return fig
        
def make_all_vs_all(variables, true, pred):
    # Plot the true-pred difference of all variables as a function of all variables
    # Note that the unit for the difference may be different than the unit for the variable
    # This also includes true-pred difference for a variable as a function of itself
    #
    # INPUTS
    #    variables: The list of all variables from the original file (e.g., 'start_carrier_frequency_Hz')
    #    true: array of true values from the model
    #    pred: array of predicted values from the model
    #
    # OUTPUTS
    #    fig: figure with the plot

    # It will be a grid of size len(variables) x len(variables)
    fig, ax = plt.subplots(len(variables), len(variables), figsize=((4*len(variables), 4*len(variables))))
    
    # Loop over the variables
    for vind, var in enumerate(variables):
        var_label, var_unit, diff_unit, factor, diff_factor = get_label_unit(var)
        
        # Divide by the factor to get the desired units
        true_var = true[:, vind]/factor
        
        # Loop over the variables again to get all combinations
        for vind2, var2 in enumerate(variables):
            # We will need the diff_unit and diff_factor here
            var_label2, var_unit2, diff_unit2, factor2, diff_factor2 = get_label_unit(var2)
            
            # Don't make the same plot twice
            if(vind2 > vind):
                ax[vind, vind2].axis("off")
                continue
                
            diff = (true[:, vind2]-pred[:, vind2])/diff_factor
            ax[vind, vind2].hist2d(true_var, diff, bins=((np.linspace(min(true_var), max(true_var), 100), np.linspace(np.mean(diff)-5*np.std(diff), np.mean(diff)+5*np.std(diff), 100))), cmin=1)
            ax[vind, vind2].axhline(0, color='r', ls='--', lw='2')
            ax[vind, vind2].set_xlabel('true ' + var_label + ' ['+var_unit+']')
            ax[vind, vind2].set_ylabel('true-pred '+ var_label2 + ' ['+diff_unit2+']')
    plt.tight_layout()

    return fig

def make_all_plots(variables, true, pred):
    # Make all plots
    #
    # INPUTS
    #    variables: The list of all variables from the original file (e.g., 'start_carrier_frequency_Hz')
    #    true: array of true values from the model
    #    pred: array of predicted values from the model
    # OUTPUTS
    #    res, bias, all_vs_all, energy_res: figures to be saved later, if desired
    
    res = make_res(variables, true, pred)
    bias = make_bias(variables, true, pred)
    all_vs_all = make_all_vs_all(variables, true, pred)
    # Only make the energy resolution plot if the energy variable is present
    if 'energy_eV' in variables:
        energy_res = make_energy_res(variables, true, pred)
        return res, bias, all_vs_all, energy_res
    res, bias, all_vs_all