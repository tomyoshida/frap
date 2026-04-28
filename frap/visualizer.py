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

            f_min = np.min(param_set[_key]['f_min'])
            f_max = np.max(param_set[_key]['f_max'])

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
    kde_results = {'R_fine': R_fine, 'y_grids': {}, 'density_matrices': {}, 'key_true' : {}, 'key' : {}, 'islog': {}, 'ylims' : {}}


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
        kde_results['islog'][key] = islog[i]
        kde_results['ylims'][key] = ylims[i]

    if save:
        with open(save, 'wb') as f:
            pickle.dump(kde_results, f)
        print(f"KDE calculation completed and saved to {save}.")

    else:
        return kde_results


def calc_hdi(kde_data, D):

    
    R_fine = kde_data['R_fine'] * D

    Rmax = np.max(R_fine)
    keys = kde_data['key']
    islog = kde_data['islog']
    ylims =  kde_data['ylims']

    #keys_true = kde_data['key_true']

    res = {}
   
    for i, key in enumerate(keys):
        
        y_grid = kde_data['y_grids'][key]
        density_matrix = kde_data['density_matrices'][key]
        
        X_fine, Y_fine = np.meshgrid(R_fine, y_grid)
    
        # --- HDI map ---
        col_sums = density_matrix.sum(axis=0)
        norm_density = np.zeros_like(density_matrix)
        valid_cols = col_sums > 0
        norm_density[:, valid_cols] = density_matrix[:, valid_cols] / col_sums[valid_cols]
    
            
        hdi_map = np.zeros_like(norm_density)
        mass_levels = [0.9973, 0.9545, 0.6827]
            
        for k in range(len(R_fine)):
            d_col = norm_density[:, k]
            if not valid_cols[k]: continue
                
            sort_idx = np.argsort(d_col)[::-1]
            sorted_d = d_col[sort_idx]
            cumsum_d = np.cumsum(sorted_d)
                
            for val, mass in enumerate(mass_levels, start=1):
                #
                idx_in_mass = sort_idx[cumsum_d <= mass]
                if len(idx_in_mass) > 0:
                    hdi_map[idx_in_mass, k] = val


        res[key] = { 'X_fine':X_fine, 'Y_fine':Y_fine, 'hdi_map':hdi_map, 'ylims': ylims[key], 'islog' : islog[key], 'Rmax' : Rmax}

    
    return res


def plot_kde(hdi_data):

    keys = list(hdi_data.keys())


    plt.rcParams['font.size'] = 18
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.tight_layout()
    
    
    
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 4*len(keys)), sharex=True, sharey='row')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.align_ylabels(axes)
    
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]
    
    
    j = 0
    
    for i, key in enumerate(keys):
        ax = axes[i, j]
        
    
        hdi_colors = [plt.cm.Blues(0.3), plt.cm.Blues(0.5), plt.cm.Blues(0.8)]
            
        X_fine = hdi_data[key]['X_fine']
        Y_fine = hdi_data[key]['Y_fine']
        hdi_map = hdi_data[key]['hdi_map']
            
        ax.contourf(X_fine, Y_fine, hdi_map, levels=[0.5, 1.5, 2.5, 3.5],
                        colors=hdi_colors, alpha=0.8, zorder=0)
    
            
        #ax.set_ylabel(labels[i])
        ax.set_xlim(1, hdi_data[key]['Rmax'] )
        ax.set_ylim(hdi_data[key]['ylims'])

        ax.set_ylabel( key.removeprefix("log10_") )
        
        if hdi_data[key]['islog']: ax.set_yscale('log')
            
            
        
    axes[-1, 0].set_xlabel(r'$r\ {\rm (au)}$')
    
    
    return axes








        





