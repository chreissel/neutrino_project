from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
my_viridis = cm.get_cmap("viridis", 1024).with_extremes(under="white")
import os

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

def _softplus(x):
    # Numerically-stable softplus, matching torch.nn.functional.softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def split_gaussian_nll_pred(pred, n_vars):
    # If pred has last-dim 2*n_vars it's interpreted as a GaussianNLL output:
    # the first n_vars entries are the mean predictions and the remaining
    # n_vars entries are the pre-softplus variance parameters.  Returns
    # (means, stds) with softplus applied to recover a positive variance; if
    # pred is not a GaussianNLL output, returns (pred, None).
    pred = np.asarray(pred)
    if pred.shape[-1] != 2 * n_vars:
        return pred, None
    means = pred[..., :n_vars]
    var = _softplus(pred[..., n_vars:])
    return means, np.sqrt(var)

def compute_fwhm(values, bins):
    # Smooth the histogram before peak finding so that bin-level Poisson noise
    # doesn't produce a spuriously high "peak" (which shrinks the half-max
    # threshold and collapses the FWHM).
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    smooth_sigma = max(1.0, len(hist) / 100.0)
    hist_s = gaussian_filter1d(hist, smooth_sigma)

    peak = np.max(hist_s)
    half_max = peak / 2.0

    above = hist_s >= half_max
    indices = np.where(above)[0]
    if len(indices) < 2:
        return np.nan, np.nan, np.nan

    left_idx = indices[0]
    right_idx = indices[-1]

    def interp_x(i1, i2):
        x1, x2 = centers[i1], centers[i2]
        y1, y2 = hist_s[i1], hist_s[i2]
        if y2 == y1:
            return x1
        return x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)

    if left_idx > 0:
        x_left = interp_x(left_idx - 1, left_idx)
    else:
        x_left = centers[left_idx]

    if right_idx < len(hist_s) - 1:
        x_right = interp_x(right_idx, right_idx + 1)
    else:
        x_right = centers[right_idx]

    return x_right - x_left, x_left, x_right

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

def make_distribution(variables, observables, true, meta):
    # Plot truth distribution of variables - this gives a sense of whether the underlying distribution of sims is even
    #
    # INPUTS
    #    variables: The list of all variables from the original file (e.g., 'start_carrier_frequency_Hz')
    #    true: array of true values from the model
    #
    # OUTPUTS
    #    fig: figure with the plot
    
    num_plots = len(variables) + len(observables)
    
    # There will be one plot per variable
    fig, ax = plt.subplots(1, num_plots, figsize=(4*num_plots, 4), squeeze=False)
    
    # Loop over the variables
    for vind, var in enumerate(variables):
        
        var_label, var_unit, _, factor, _ = get_label_unit(var)
        
        # Divide by factor to get the desired units
        true_var = true[:, vind]/factor
        
        ax[0, vind].hist(true_var, bins=np.linspace(min(true_var), max(true_var), 100), label='%d events'%len(true_var))
        ax[0, vind].set_xlabel('true ' + var_label + ' ['+var_unit+']')
        ax[0, vind].set_ylabel('events')
        ax[0, vind].set_xlim(min(true_var), max(true_var))
        ax[0, vind].legend()
        
    for oind, obs in enumerate(observables):
        
        obs_label, obs_unit, _, factor, _ = get_label_unit(obs)
        
        # Divide by factor to get the desired units
        meta_factor = meta[:, oind]/factor
        ax[0, vind+oind+1].hist(meta_factor, bins=np.linspace(min(meta_factor), max(meta_factor), 100), label='%d events'%len(meta_factor))
        ax[0, vind+oind+1].set_xlabel('true ' + obs_label + ' ['+obs_unit+']')
        ax[0, vind+oind+1].set_ylabel('events')
        ax[0, vind+oind+1].set_xlim(min(meta_factor), max(meta_factor))
        ax[0, vind+oind+1].legend()

        
    plt.tight_layout()

    return fig


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
    fig, ax = plt.subplots(1, len(variables), figsize=(4*len(variables), 4), squeeze=False)
    
    # Loop over the variables
    for vind, var in enumerate(variables):
        
        var_label, var_unit, _, factor, _ = get_label_unit(var)
        
        # Divide by factor to get the desired units
        true_var = true[:, vind]/factor
        pred_var = pred[:, vind]/factor
        
        ax[0, vind].hist2d(true_var, pred_var, bins=((np.linspace(min(true_var), max(true_var), 300), np.linspace(min(true_var), max(true_var), 300))), cmin=1)
        ax[0, vind].set_xlabel('true ' + var_label + ' ['+var_unit+']')
        ax[0, vind].set_ylabel('pred ' + var_label + ' ['+var_unit+']')
        
    plt.tight_layout()

    return fig

def make_res(variables, true, pred, fit_gaussian=True):
    # Extract the resolution using FWHM estimation and an optional Gaussian fit
    #
    # INPUTS
    #    variables: The list of all variables from the original file (e.g., 'start_carrier_frequency_Hz')
    #    true: array of true values from the model
    #    pred: array of predicted values from the model
    #    fit_gaussian: if True, overlay a Gaussian fit on the histogram (default: True)
    #
    # OUTPUTS
    #    fig: figure with the plot

    # There will be one plot per variable
    fig, ax = plt.subplots(1, len(variables), figsize=(4*len(variables), 4), squeeze=False)
    # Loop over the variables
    for vind, var in enumerate(variables):
        var_label, var_unit, diff_unit, factor, diff_factor = get_label_unit(var)

        # Divide by factor to get the desired units
        true_var = true[:, vind]
        pred_var = pred[:, vind]
        diff = (true_var-pred_var)/diff_factor

        bins = np.linspace(np.mean(diff)-5*np.std(diff), np.mean(diff)+5*np.std(diff), 2500)
        hist, bins_out, _ = ax[0, vind].hist(diff, bins=bins, weights=np.ones(len(true_var))*1/float(len(true_var)))
        bincenters = (bins_out[:-1] + bins_out[1:]) / 2

        # Always compute and display FWHM
        fwhm, x_left, x_right = compute_fwhm(diff, bins)
        ax[0, vind].axvline(x_left, color='b', ls=':', lw=1)
        ax[0, vind].axvline(x_right, color='b', ls=':', lw=1, label='FWHM={:.4f} {}'.format(fwhm, diff_unit))

        if fit_gaussian:
            try:
                popt, pcov = curve_fit(gaussian, bincenters, hist, p0=[max(hist), np.mean(diff), np.std(diff)])
                amplitude_fit, mean_fit, stddev_fit = popt
                x_fit = np.linspace(min(bins_out), max(bins_out), 1000)
                ax[0, vind].plot(x_fit, gaussian(x_fit, *popt), 'r-', label='Fit: mu={:.4f}, std={:.4f}'.format(mean_fit, stddev_fit))
                ax[0, vind].set_xlim(-5*stddev_fit, 5*stddev_fit)
            except Exception:
                pass

        ax[0, vind].set_xlabel('residual '+var_label+ ' '+'['+diff_unit+']')
        ax[0, vind].set_ylabel('A.U.')
        ax[0, vind].legend()

    plt.tight_layout()

    return fig
    
def make_energy_res(variables, observables, true, pred, meta, fit_gaussian=True):
    # Plot the energy resolution as a function of all variables.  The mean and stddev of the events in each bin are plotted
    # The goal is to determine whether the energy resolution is better in certain regions of the parameter space
    #
    # INPUTS
    #    variables: The list of all variables from the original file (e.g., 'start_carrier_frequency_Hz')
    #    true: array of true values from the model
    #    pred: array of predicted values from the model
    #    fit_gaussian: if True, use Gaussian fits to extract mean/std per bin; if False, use mean and FWHM/2.355 (default: True)
    #
    # OUTPUTS
    #    fig: figure with the plot

    all_var_names = variables + observables
    all_true = np.zeros((len(true), len(all_var_names)))
    all_true[:, :len(variables)] = true
    all_true[:, len(variables):] = meta

    # There will be one plot per variable
    fig, ax = plt.subplots(1, len(all_var_names), figsize=(4*len(all_var_names), 4), squeeze=False)
    fig2, ax2 = plt.subplots(1, len(all_var_names), figsize=(4*len(all_var_names), 4), squeeze=False)

    eind = all_var_names.index('energy_eV')
    
    maxnum = 0
    # Loop over the variables
    for vind, var in enumerate(all_var_names):
        var_label, var_unit, _, factor, _ = get_label_unit(var)
        # Divide by factor to get the desired units
        true_var = all_true[:, vind]/factor
        # Define 20 bins across the full parameter space over which to take means and stdevs
        var_bins = np.linspace(min(true_var), max(true_var), 20)
        bincenters = (var_bins[:-1] + var_bins[1:]) / 2
        
        # Find the indices corresponding to each of the variable bins
        idxs_all = [np.where((true_var >= var_bins[i]) & (true_var <= var_bins[i+1])) for i in range(len(var_bins)-1)]
        # Find the energy differences corresponding to these bins, then take the mean and stddev
        energy_diffs = [np.squeeze(all_true[idxs, eind]-pred[idxs, eind]) for idxs in idxs_all] 
        #lens = [len(ediff) for ediff in energy_diffs]
        means = []
        stds = []
        lens = []
        for energy_diff in energy_diffs:
            if(energy_diff.size <= 1):
                means.append(np.nan)
                stds.append(np.nan)
                lens.append(np.nan)
                continue
            fwhm, _, _ = compute_fwhm(energy_diff, 100)
            if fit_gaussian:
                hist, bins, _ = ax[0, vind].hist(energy_diff, bins=100)
                ax[0, vind].clear()
                bincenters_g = (bins[:-1] + bins[1:]) / 2
                try:
                    popt, pcov = curve_fit(gaussian, bincenters_g, hist, p0=[max(hist), np.mean(energy_diff), np.std(energy_diff)])
                    amplitude_fit, mean_fit, stddev_fit = popt
                    means.append(mean_fit)
                    stds.append(np.abs(stddev_fit))   # added absolute value
                    lens.append(len(energy_diff))
                except:
                    means.append(np.nan)
                    stds.append(np.nan)
                    lens.append(np.nan)
            else:
                means.append(np.mean(energy_diff))
                stds.append(fwhm / 2.3548)  # convert FWHM to sigma-equivalent
                lens.append(len(energy_diff))

        if(maxnum==0):
            maxnum = np.nanmax(lens) + 10
        ax2[0, vind].plot(lens, stds, 'k.')
        ax2[0, vind].set_xlabel('events in '+var_label+' bin')
        ax2[0, vind].set_ylabel('std [eV]')
        ax2[0, vind].fill_between([0, np.nanmax(lens)], [0, 0], [0.3, 0.3], alpha=0.5)
        ax2[0, vind].axhline(0.3, ls='--', color='b')
        ax2[0, vind].set_ylabel('std ['+var_unit+']')
        #ax2[0, vind].set_ylim(0, 1)#np.nanmax(stds)+0.1)
        ax2[0, vind].set_xlim(0, np.nanmax(lens))


        # The error bars are the stddevs
        #print(stds)
        ax[0, vind].errorbar(bincenters, means, stds, color='k', marker='o', ls='')
        ax[0, vind].axhline(0, color='r', ls='--', lw='2')
        # This region corresponds to the desired resolution of 0.3 eV stddev
        ax[0, vind].fill_between(var_bins, np.ones(len(var_bins))*-0.3, np.ones(len(var_bins))*0.3, alpha=0.5, label="0.3 eV std")
        ax[0, vind].legend()
        ax[0, vind].set_xlim(min(var_bins), max(var_bins))
        #ax[0, vind].set_ylim(-1,1)
        ax[0, vind].set_xlabel('true ' + var_label + ' ['+var_unit+']')
        ax[0, vind].set_ylabel('true-pred energy [eV]')
        
    fig.tight_layout()
    fig2.tight_layout()
    
    return fig, fig2
        
def make_all_vs_all(variables, observables, true, pred, meta):
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

    num_plots = len(variables)# + len(observables)
        
    # It will be a grid of size len(variables) x len(variables)
    fig, ax = plt.subplots(num_plots, num_plots, figsize=((4*len(variables), 4*len(variables))), squeeze=False)
    
    # Loop over the variables
    for vind, var in enumerate(variables):
        var_label, var_unit, diff_unit, factor, diff_factor = get_label_unit(var)
        
        # Divide by the factor to get the desired units
        true_var = true[:, vind]/factor
        
        # Loop over the variables again to get all combinations
        for vind2, var2 in enumerate(variables):
            # We will need the diff_unit and diff_factor here
            var_label2, var_unit2, diff_unit2, factor2, diff_factor2 = get_label_unit(var2)
            
            diff = (true[:, vind2]-pred[:, vind2])/diff_factor
            ax[vind, vind2].hist2d(true_var, diff, bins=((np.linspace(min(true_var), max(true_var), 100), np.linspace(np.mean(diff)-5*np.std(diff), np.mean(diff)+5*np.std(diff), 100))), cmin=1)
            ax[vind, vind2].axhline(0, color='r', ls='--', lw='2')
            ax[vind, vind2].set_xlabel('true ' + var_label + ' ['+var_unit+']')
            ax[vind, vind2].set_ylabel('true-pred '+ var_label2 + ' ['+diff_unit2+']')
    plt.tight_layout()

    return fig

def make_uncertainty_vs_params(variables, observables, true, pred_std, meta):
    # Plot the per-track predicted uncertainty (standard deviation) for each
    # regression variable as a function of every parameter, in bins.
    # Intended for models trained with GaussianNLLLoss.
    #
    # INPUTS
    #    variables: The regression target variable names
    #    observables: Additional observable variable names
    #    true: (N, n_vars) array of true target values in physical units
    #    pred_std: (N, n_vars) array of predicted standard deviations
    #    meta: (N, n_observables) array of observable values
    #
    # OUTPUTS
    #    fig: figure with one row per regression variable and one column per
    #         parameter (variables + observables).  Points are the mean
    #         predicted sigma within each bin; error bars are the spread of
    #         the predicted sigma within the bin.

    all_var_names = list(variables) + list(observables)
    all_true = np.zeros((len(true), len(all_var_names)))
    all_true[:, :len(variables)] = true
    all_true[:, len(variables):] = meta

    n_vars = len(variables)
    n_params = len(all_var_names)

    fig, ax = plt.subplots(n_vars, n_params, figsize=(4*n_params, 4*n_vars), squeeze=False)

    for vind, var in enumerate(variables):
        var_label, _, diff_unit, _, diff_factor = get_label_unit(var)
        pred_sigma = pred_std[:, vind] / diff_factor

        for pind, param in enumerate(all_var_names):
            p_label, p_unit, _, p_factor, _ = get_label_unit(param)
            param_true = all_true[:, pind] / p_factor

            param_bins = np.linspace(np.min(param_true), np.max(param_true), 20)
            bincenters = 0.5 * (param_bins[:-1] + param_bins[1:])

            idxs_all = [np.where((param_true >= param_bins[i]) & (param_true <= param_bins[i+1]))[0]
                        for i in range(len(param_bins) - 1)]

            means, spreads = [], []
            for idxs in idxs_all:
                if idxs.size < 1:
                    means.append(np.nan)
                    spreads.append(np.nan)
                else:
                    means.append(np.mean(pred_sigma[idxs]))
                    spreads.append(np.std(pred_sigma[idxs]))

            ax[vind, pind].errorbar(bincenters, means, spreads, color='k', marker='o', ls='')
            ax[vind, pind].set_xlabel('true ' + p_label + ' [' + p_unit + ']')
            ax[vind, pind].set_ylabel('pred sigma ' + var_label + ' [' + diff_unit + ']')
            ax[vind, pind].set_xlim(np.min(param_bins), np.max(param_bins))

    plt.tight_layout()
    return fig


def make_all_plots(variables, observables, true, pred, meta, folder=[], savefigs=False, fit_gaussian=True):
    print(folder)
    # Make all plots
    #
    # INPUTS
    #    variables: The list of all variables from the original file (e.g., 'start_carrier_frequency_Hz')
    #    true: array of true values from the model
    #    pred: array of predicted values from the model.  Accepts either
    #          (N, n_vars) for point predictions or (N, 2*n_vars) for
    #          GaussianNLL outputs where the second half is the pre-softplus
    #          variance parameter.  When a GaussianNLL shape is detected an
    #          extra per-track uncertainty plot is produced.
    #    fit_gaussian: if True, overlay Gaussian fits on residual histograms (default: True)
    # OUTPUTS
    #    res, bias, all_vs_all, energy_res, [uncertainty]: figures to be saved later, if desired

    pred_mean, pred_std = split_gaussian_nll_pred(pred, len(variables))

    dist = make_distribution(variables, observables, true, meta)
    res = make_res(variables, true, pred_mean, fit_gaussian=fit_gaussian)
    bias = make_bias(variables, true, pred_mean)
    all_vs_all = make_all_vs_all(variables, observables, true, pred_mean, meta)
    # Only make the energy resolution plot if the energy variable is present

    if 'energy_eV' in variables:
        energy_res, nums = make_energy_res(variables, observables, true, pred_mean, meta, fit_gaussian=fit_gaussian)

    uncertainty = None
    if pred_std is not None:
        uncertainty = make_uncertainty_vs_params(variables, observables, true, pred_std, meta)

    if(savefigs):
        if(folder==[]):
            print("Please provide folder to save figures")
        else:
            if not os.path.isdir(folder):
                os.mkdir(folder)
            dist.savefig(folder+'/distributions.png')
            res.savefig(folder+'/resolutions.png')
            bias.savefig(folder+'/biases.png')
            all_vs_all.savefig(folder+'/all_vs_all.png')
            if('energy_eV' in variables):
                nums.savefig(folder+'/std_vs_num.png')
                energy_res.savefig(folder+'/energy_res.png')
            if uncertainty is not None:
                uncertainty.savefig(folder+'/uncertainty_vs_params.png')

    results = [dist, res, bias, all_vs_all]
    if 'energy_eV' in variables:
        results.append(energy_res)
    if uncertainty is not None:
        results.append(uncertainty)
    return tuple(results)
