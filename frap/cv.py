
import jax
import jax.numpy as jnp
import numpy as np
from ._constants import *
from astropy import units as u
import warnings
import numpy as np



def _get_indices(n_folds, bands, obs, seed = 1):

    np.random.seed(seed)

    
    # all_fold_indices[data_id][sub_key] = [fold0_idx, fold1_idx, ...]
    all_fold_indices = {}

    for band in bands:

        all_fold_indices[band] = {}

        for ch_id in obs[band]['q'].keys():

            n_vis = len(obs[band]['q'][ch_id])

            indices = np.arange(n_vis)
            
            np.random.shuffle(indices)

            all_fold_indices[band][ch_id] = np.array_split(indices, n_folds)

    return all_fold_indices



def split_data(n_folds, bands, obs, seed = 1):
    
    all_fold_indices = _get_indices(n_folds, bands, obs, seed = seed)

    test_obs = {}
    train_obs = {}

    for fold_idx in range(n_folds):

        test_obs[fold_idx] = {}
        train_obs[fold_idx] = {}

        for band in bands:
            
            train_q, train_V, train_s = {}, {}, {}
            val_q, val_V, val_s = {}, {}, {}

            for ch_id in obs[band]['q'].keys():
                
                # get index
                folds = all_fold_indices[band][ch_id]
            
                val_idx = folds[fold_idx]
                train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx])
                    
                # slice for training
                train_q[ch_id] = obs[band]['q'][ch_id][train_idx]
                train_V[ch_id] = obs[band]['V'][ch_id][train_idx]
                train_s[ch_id] = obs[band]['s'][ch_id][train_idx]
                    
                # slice for test
                val_q[ch_id] = obs[band]['q'][ch_id][val_idx]
                val_V[ch_id] = obs[band]['V'][ch_id][val_idx]
                val_s[ch_id] = obs[band]['s'][ch_id][val_idx]
                
            test_obs[fold_idx][band] = {'q': val_q, 'V': val_V, 's': val_s, 'nu' : obs[band]['nu'], 'Nch': obs[band]['Nch']}
            train_obs[fold_idx][band] = {'q': train_q, 'V': train_V, 's': train_s, 'nu' : obs[band]['nu'], 'Nch': obs[band]['Nch']}
        
    return train_obs, test_obs