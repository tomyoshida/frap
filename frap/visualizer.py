import pickle
import numpy as np
import matplotlib.pyplot as plt


from scipy.interpolate import interp1d

import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

def calc_kde( file, nskip = 100, r_grid_fac = 20, n_kde = 400, save = None ):

    with open(file, 'rb') as f:
        data = pickle.load(f)

    R = data.r
    samples = data.sample['posterior_f']


    keys = list(samples.keys())

    param_set = data.param_set


    ylims = []
    islog = []
    keys_true = []


    log_prefix = 'log10_'
    for _key in keys:

        if _key.startswith(log_prefix):

            keys_true.append(_key[len(log_prefix):])

            f_min = param_set[_key]['f_min']
            f_max = param_set[_key]['f_max']

            ylims.append( (10**f_min, 10**f_max) )

            islog.append( True )

        else:
            
            keys_true.append(_key)

            f_min = param_set[_key]['f_min']
            f_max = param_set[_key]['f_max']

            ylims.append( (f_min, f_max) )

            islog.append( False )


    Nchain,  Nsample, _ = np.shape( samples[keys[0]] )


    R_fine = np.linspace(R.min(), R.max(), len(R) * r_grid_fac)
    kde_results = {'R_fine': R_fine, 'y_grids': {}, 'density_matrices': {}}


    print("Starting KDE calculation...")

    for i, key in enumerate(keys):
        print(f"Processing key: {key}")
        ymin, ymax = ylims[i]
        
        if islog[i]:
            y_grid = np.logspace(np.log10(ymin), np.log10(ymax), n_kde)
        else:
            y_grid = np.linspace(ymin, ymax, n_kde)
        
        full_samples = np.reshape(samples[key], (Nsample*Nchain, len(R)))
        
        
        # 4. 指定された間隔 (nskip2) で間引き
        plot_samples = full_samples[::nskip, :]
        
        
        f_sample_interp = interp1d(R, plot_samples, axis=1, kind='cubic')
        plot_samples_fine = f_sample_interp(R_fine)

        # calc KDE
        density_matrix_fine = np.zeros((len(y_grid), len(R_fine)))
        
        # logスケールの場合は評価点もlogに
        kde_eval_points = np.log10(y_grid) if islog[i] else y_grid

        for ir in range(len(R_fine)):
            column_data = plot_samples_fine[:, ir]
            
            kernel = gaussian_kde(column_data)
            density_matrix_fine[:, ir] = kernel(kde_eval_points)
            
        kde_results['y_grids'][key] = y_grid
        kde_results['density_matrices'][key] = density_matrix_fine / density_matrix_fine.max(axis=0)
        kde_results['key_true'][key] = keys_true[i]
        kde_results['key'][key] = key

    if save:
        with open(save, 'wb') as f:
            pickle.dump(kde_results, f)
        print(f"KDE calculation completed and saved to {save}.")

    else:
        return kde_results












        





