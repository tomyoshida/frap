#########################################################################
### FRAP: Flexible Radial Analysis of Protoplanetary disks　          ###
#########################################################################


import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import jax.random as random
import numpy as np
import warnings
import numpy as np
from scipy.special import j0


from jax.scipy.interpolate import RegularGridInterpolator
from jax.scipy.signal import convolve as jax_convolve

import numpyro
from numpyro.distributions import Normal, Uniform, MultivariateNormal
from numpyro.infer import MCMC as numpyro_MCMC
from numpyro.infer import NUTS, init_to_median, init_to_uniform
import matplotlib.pyplot as plt
from numpyro.infer import Predictive
from numpyro.infer import SVI, Trace_ELBO, init_to_sample
from numpyro.optim import Adam
from numpyro.infer.autoguide import AutoDelta


from ._constants import *
from ._utilities import *

from scipy.special import j0 as J0
from scipy.special import j1 as J1
from scipy.special import jn_zeros



from jax.debug import print as jax_print

jax.config.update("jax_enable_x64", True) 




class model:
    '''class for FRAP model

    Attributes:
        set_parameters: method to set model parameters
        set_observations: method to set observational data
        set_opacity: method to set dust opacity model
    '''

    def __init__(self, incl, r_out, N_GP,  pa = 0.0, spacing = 'FB', r_in = None, intensity_function = 'SL20', userdef_vis_model = None, flux_uncert = True, jitter=1e-4, hyperparameters_fixed=True):
        '''initialize the model

        Args:
            incl (float): inclination angle in degrees
            pa (float) : position angle in degrees. Only used for primary beam correction.
            r_out (float): outer radius in arcseconds
            N_GP (int): number of Gaussian process points
            intensity_function (str or callable, optional): 'SL20' (default) or a user-defined function with signature (obs, incl, f_latents, output_tau). The function should return intensity in cgs units, or optical depth when output_tau=True.
            userdef_vis_model (function, optional): user-defined visibility model function. Defaults to None. The function should take (V, obs, f_latents) as arguments and return modified V.
            flux_uncert (bool, optional): whether to include flux uncertainty. Defaults to True.
            jitter (float, optional): jitter value for numerical stability. Defaults to 1e-6.
            hyperparameters_fixed (bool, optional): whether to fix hyperparameters. Defaults to True.

        '''
        
        self._free_parameters = {}
        self._Nparams_forGP = 0
        self._fixed_parameters = {}

        self._r_out = r_out
        self._r_out_rad = jnp.deg2rad( self._r_out/3600 )

        self._N_GP = N_GP


        self._spacing = spacing

        if spacing == 'FB':
            self._j0_zeros = jn_zeros(0, self._N_GP + 1)
            self._j0N_plus = self._j0_zeros[-1]
            self._j0k = self._j0_zeros[:-1]
            
            self._r_GP = self._r_out * self._j0k / self._j0N_plus
            self._r_GP_rad = np.deg2rad(self._r_GP/3600)

            self._HT_prefactor = 4.0 * np.pi * self._r_out_rad ** 2/ (self._j0N_plus ** 2 * J1(self._j0k) ** 2)

        elif spacing == 'linear':
            
            warnings.warn('Please use spacing = \'FB\' if you work with the visibilities', UserWarning)

            self._r_GP = jnp.linspace( r_in, self._r_out, self._N_GP )

            self._dr = self._r_GP[1] - self._r_GP[0]
            self._r_GP_rad = jnp.deg2rad( self._r_GP/3600 )

            self._HT_prefactor = 2.0 * np.pi * (self._r_out_rad / self._N_GP) **2
   
        self._jitter = jitter

        self._incl = np.deg2rad(incl)
        self._pa = np.deg2rad(pa)

        
        self._observations = {}
        self._s_fs = {}
        self._mean_fs = {}
        self._bands = []

        if intensity_function == 'SL20':
            self._intensity_function = f_I_SL20
        elif callable(intensity_function):
            self._intensity_function = intensity_function
        else:
            raise ValueError("intensity_function must be 'SL20' or a callable")

        self._is_userdef_intensity = callable(intensity_function)
                
        self._userdef_vis_model = userdef_vis_model

        self._flux_uncert = flux_uncert

        self._hyperparameters_fixed = hyperparameters_fixed

        self._opacity_config = None



    def summary(self, file=None, verbose=False):
        """Print and return a summary of the FRAP model."""

        def fmt(x):

            a = np.asarray(x)

            if a.size == 1:
                try:
                    return f'{float(a.reshape(-1)[0]):.4g}'
                except (TypeError, ValueError):
                    return str(a.reshape(-1)[0])

            if a.size == 0:
                return f'shape={a.shape}, empty'

            return f'shape={a.shape}, range=[{np.min(a):.4g}, {np.max(a):.4g}]'

        def fmt_bounds(lo, hi):

            a = np.asarray(lo)
            b = np.asarray(hi)

            if a.size == 1 and b.size == 1:
                return f'({fmt(lo)}, {fmt(hi)})'

            return f'profile: lower {fmt(lo)}, upper {fmt(hi)}'

        def func_name(func):

            if func is None:
                return 'None'

            if isinstance(func, str):
                return func

            return getattr(func, '__qualname__', type(func).__name__)

        def public_names(obs):

            names = {
                name for name in vars(obs)
                if not name.startswith('_')
            }

            # nu is a property, so it is not included in vars(obs).
            if hasattr(obs, 'nu'):
                names.add('nu')

            return names

        lines = [
            'FRAP model summary',
            '==================',
            '',
            '[Model]',
            f'  Inclination      : {np.rad2deg(self._incl):.3f} deg',
            f'  Position angle   : {np.rad2deg(self._pa):.3f} deg',
            f'  Radial grid      : {self._spacing}, N={self._N_GP}, r_out={self._r_out:.4g} arcsec',
            f'  Radial range     : {float(self._r_GP[0]):.4g} - {float(self._r_GP[-1]):.4g} arcsec',
            f'  Jitter           : {self._jitter:.2g}',
            f'  GP hyperparams   : {"fixed" if self._hyperparameters_fixed else "free"}',
            f'  Flux uncertainty : {self._flux_uncert}',
            '',
            '[Emission]',
            f'  Intensity  : {func_name(self._intensity_function)}',
            f'  Visibility : {func_name(self._userdef_vis_model)}',
        ]

        # Opacity
        opac = getattr(self, '_opacity_config', None)

        if opac is None:
            lines.append('  Opacity    : not configured')
        else:
            lines.append(
                f'  Opacity    : configured '
                f'(Na={opac["Na"]}, Nq={opac["Nq"]}, smooth={opac["smooth"]})'
            )

            if verbose:
                for key in ['log10_a_smooth', 'a_min', 'a_max']:
                    lines.append(f'    {key}: {opac[key]}')

        # Parameters
        n_free = len(self._free_parameters)
        n_fixed = len(self._fixed_parameters)

        n_latents = sum(
            self._N_GP if p['GP'] else 1
            for p in self._free_parameters.values()
        )

        lines.extend([
            '',
            f'[Parameters]  {n_free} free, {n_fixed} fixed, {n_latents} latent variables',
        ])

        if n_free == 0 and n_fixed == 0:
            lines.append('  (none)')

        for name, p in self._free_parameters.items():

            if p['GP']:

                info = (
                    f'GP, bounds={fmt_bounds(p["f_min"], p["f_max"])}, '
                    f'lengthscale={fmt(p["g_lengthscale"])} arcsec'
                )

                if verbose:
                    info += (
                        f', variance={fmt(p["g_variance"])}'
                        f', mean={fmt(p["g_mean"])}'
                    )

            else:

                info = (
                    f'Normal(mean={fmt(p["f_mean"])}, '
                    f'std={fmt(p["f_s"])})'
                )

            lines.append(f'  {name:<18}: {info}')

        for name, p in self._fixed_parameters.items():
            lines.append(f'  {name:<18}: fixed, {fmt(p["profile"])}')

        # Observations
        lines.extend(['', '[Observations]'])

        rows = []

        for band in self._bands:

            observations = self._observations[band]

            if not observations:
                rows.append((str(band), '-', 0, '-', 0))
                continue

            kinds = ', '.join(sorted({obs.kind for obs in observations}))

            frequencies = np.asarray([
                float(obs.nu) for obs in observations
            ]) / 1e9

            if np.isclose(frequencies.min(), frequencies.max(), rtol=0, atol=1e-10):
                freq_str = f'{frequencies[0]:.3f}'
            else:
                freq_str = f'{frequencies.min():.3f}-{frequencies.max():.3f}'

            data_attrs = {
                'visibility': '_V',
                'photometry': '_F',
                'radialprofile': '_Tb',
            }

            n_data = sum(
                int(np.size(getattr(obs, data_attrs[obs.kind])))
                for obs in observations
            )

            rows.append((
                str(band),
                kinds,
                len(observations),
                freq_str,
                n_data,
            ))

        if not rows:
            lines.append('  (none)')

        else:

            band_width = max(5, *(len(row[0]) for row in rows))
            kind_width = max(4, *(len(row[1]) for row in rows))

            lines.append(
                f'  {"Band":<{band_width}}  '
                f'{"Type":<{kind_width}}  '
                f'{"Nch":>3}  '
                f'{"Frequency [GHz]":<19}  '
                f'{"Ndata":>6}'
            )

            for band, kind, nch, freq_str, n_data in rows:
                lines.append(
                    f'  {band:<{band_width}}  '
                    f'{kind:<{kind_width}}  '
                    f'{nch:>3}  '
                    f'{freq_str:<19}  '
                    f'{n_data:>6}'
                )

            if verbose:

                lines.append('')
                lines.append('  Flux scale / opacity details:')

                for band in self._bands:

                    observations = self._observations[band]

                    n_opac = sum(
                        hasattr(obs, 'log10_k_abs_tot_itp')
                        for obs in observations
                    )

                    lines.append(
                        f'    {band}: '
                        f'f_mean={self._mean_fs[band]}, '
                        f'f_std={self._s_fs[band]}, '
                        f'opacity={n_opac}/{len(observations)}'
                    )

        # User-defined intensity function
        if self._is_userdef_intensity:

            lines.extend([
                '',
                '[User-defined intensity function]',
                '  Signature: intensity_function(obs, incl, f_latents, output_tau)',
                '',
                '  Arguments:',
                '    incl       : inclination [rad]',
                '    f_latents  : parameter dictionary (see [Parameters])',
                '    output_tau : bool (True: return optical depth)',
                '',
                '  Accessible observation attributes:',
            ])

            descriptions = {
                'nu': 'frequency [Hz]',
                'kind': 'observation type',
                'H': 'Hankel transform matrix',
                'f_pbcor': 'primary beam correction',
                'log10_a_dense': 'grain size grid [log10(cm)]',
                'q_dense': 'grain size distribution index grid',
                'log10_k_abs_tot': 'absorption opacity table',
                'log10_k_sca_eff_tot': 'effective scattering opacity table',
                'log10_k_abs_tot_itp': 'absorption opacity interpolator',
                'log10_k_sca_eff_tot_itp': 'effective scattering opacity interpolator',
                'Cov': 'covariance matrix',
                'L_Cov': 'Cholesky factor of covariance matrix',
            }

            obs_by_kind = {}

            for band in self._bands:
                for obs in self._observations[band]:
                    obs_by_kind.setdefault(obs.kind, []).append(obs)

            all_obs = [
                obs
                for observations in obs_by_kind.values()
                for obs in observations
            ]

            def append_attributes(names, observations):

                for name in sorted(names):

                    available = [
                        obs for obs in observations
                        if name in public_names(obs)
                    ]

                    if not available:
                        continue

                    value = getattr(available[0], name)

                    description = descriptions.get(name, '')

                    if not description:
                        if callable(value):
                            description = 'callable'
                        elif hasattr(value, 'shape'):
                            description = 'array'
                        else:
                            description = type(value).__name__

                    if verbose and hasattr(value, 'shape'):
                        description += f', example shape={np.shape(value)}'

                    if len(available) < len(observations):
                        description += f' (available in {len(available)}/{len(observations)})'

                    lines.append(f'      obs.{name:<28} : {description}')

            if not all_obs:
                lines.append('    (no observations registered)')

            else:

                # Attributes available in every observation
                common = set.intersection(
                    *(public_names(obs) for obs in all_obs)
                )

                lines.append('')
                lines.append('    Common:')
                append_attributes(common, all_obs)

                # Attributes grouped by observation type
                for kind, observations in obs_by_kind.items():

                    names = set.union(
                        *(public_names(obs) for obs in observations)
                    ) - common

                    if not names:
                        continue

                    lines.append('')
                    lines.append(f'    {kind}:')
                    append_attributes(names, observations)

        text = '\n'.join(lines)

        print(text, file=file)

        return text


        
    def _set_latent_params( self, calc_cond_num = False ):
        '''
        Set latent parameters using Gaussian processes.
        Returns a dictionary of latent parameters.
        '''
    
        R = self._r_GP[:, None]

        f_latents = {}

        for param_name, priors in self._free_parameters.items():

            if priors['GP']:

                _g_mean = priors['g_mean']

                if self._hyperparameters_fixed:
                    L_K = priors['L_K']

                else:
                    # not really tested this. and do not reccommend to use this option for most cases.

                    u_variance = (numpyro.sample(f"variance_{param_name}", Uniform(-1.0, 1.0)) + 1.0) / 2.0
                    u_lengthscale = (numpyro.sample(f"lengthscale_{param_name}", Uniform(-1.0, 1.0)) + 1.0) / 2.0

                    _g_variances = priors['g_variance'] + u_variance * (1.0 - priors['g_variance'])
                    _g_lengthscales = priors['g_lengthscale'] + u_lengthscale * (jnp.max(R) / 4.0 - priors['g_lengthscale'])

                    K = jnp.zeros((R.shape[0], R.shape[0]))
                    for variance, lengthscale in zip(_g_variances, _g_lengthscales):
                        K += rbf_kernel(R, R, variance, lengthscale)

                    relative_jitter = jnp.nanmean(jnp.diag(K)) * self._jitter
                    K += jnp.eye(R.shape[0]) * relative_jitter

                    if calc_cond_num:
                        eigenvalues = jnp.linalg.eigvalsh(K)
                        cond_num = jnp.max(eigenvalues) / jnp.maximum(jnp.min(eigenvalues), 1e-18)
                        jax_print("Param: {name}, Condition Number: {cond}", name=param_name, cond=cond_num)

                    L_K = jnp.linalg.cholesky(K)



                z = numpyro.sample(
                    f'g_{param_name}_z',
                    Normal(0.0, 1.0).expand([R.shape[0]])
                )

                _g_latent = _g_mean + L_K @ z

                numpyro.deterministic(f'g_{param_name}', _g_latent)
                
                penalty = check_range(_g_latent, lower=-5.0, upper=5.0, alpha=10.0)
                numpyro.factor(f"penalty_{param_name}", jnp.sum(penalty))

                f_latents[param_name] = sigmoid_transform(
                    _g_latent,
                    min_val=priors['f_min'],
                    max_val=priors['f_max']
                )
         
                    
            else:
                f_latents[param_name] = numpyro.sample(
                                f'g_{param_name}',
                                Normal( 0.0, 1.0 )
                            ) * priors['f_s'] + priors['f_mean']
                

                numpyro.deterministic(f'{param_name}', f_latents[param_name])

        
        for param_name, profile in self._fixed_parameters.items():
            f_latents[param_name] = profile['profile']
        
    
        return f_latents


    def set_parameter(self, kind, free = True, GP =True, 
                      bounds = (10, 20), mean_std = (0.0, 1.0),
                      variance = 1.0, lengthscale = 0.3, mean = 0.0, 
                      profile = None):
        '''set a parameter as free or fixed.

        This method allows you to define model parameters as either free (to be inferred) or fixed (with a specified profile).
        
        Args:
            kind (str): name of the parameter
            free (bool, optional): whether the parameter is free. Defaults to True.
            dust_prop (bool, optional): whether the parameter is related to dust properties. Defaults to False.
            GP (bool, optional): whether to use Gaussian process for the parameter. Defaults to True.
            bounds (tuple, optional): (min, max) bounds for the parameter if GP is used. Defaults to (10, 20).
            mean_std (tuple, optional): (mean, std) for the parameter if not using GP. Defaults to (0.0, 1.0).
            variance (float, optional): variance for the GP kernel. Defaults to 1.0.
            lengthscale (float, optional): lengthscale for the GP kernel. Defaults to 0.3.
            mean (float, optional): mean for the GP prior. Defaults to 0.0.
            profile (array or function, optional): fixed profile for the parameter if not free. Defaults to None.

        '''
    
        
        if free:

            _g_variances = np.atleast_1d(variance)
            _g_lengthscales = np.atleast_1d(lengthscale)

            R = self._r_GP[:, None]
            K = jnp.zeros((R.shape[0], R.shape[0]))

            for  _g_variance, _g_lengthscale in zip( _g_variances, _g_lengthscales ):
                K += rbf_kernel(R, R, _g_variance, _g_lengthscale)

            relative_jitter = jnp.nanmean(jnp.diag(K)) * self._jitter
            K += jnp.eye(R.shape[0]) * relative_jitter

       
            L_K = jnp.linalg.cholesky(K)

            self._free_parameters[kind] = { 'f_min' : bounds[0], 'f_max' : bounds[1], 'GP' : GP,
                                          'f_s' : mean_std[1], 'f_mean' : mean_std[0],
                                           'g_variance':np.atleast_1d(variance),
                                           'g_lengthscale':np.atleast_1d(lengthscale), 'L_K' : L_K,
                                           'g_mean':mean }
            
            if GP:
                self._Nparams_forGP += 1
            

        else:
            if profile is not None:
                
                if callable(profile):
                    self._fixed_parameters[kind] = { 'profile' : profile(self._r_GP), 'GP' : GP, }        
                else:
                    self._fixed_parameters[kind] = { 'profile' : profile, 'GP' : GP, }
            else:
                raise ValueError(f'Profile for {kind} is not set.')


    def _expansion_model( self, f_latents, obs, dryrun = False, output_tau = False):
        '''
        Generate the expansion model for a given observation.
        f_latents: dictionary of latent parameters
        obs: observation object
        dryrun: boolean, if True, return intermediate results for debugging
        '''

        _I = self._intensity_function(obs, self._incl, f_latents, output_tau)
            
        if output_tau:

            return _I
        
        else:

            if obs.kind == 'visibility':

                # Hankel transform
                V = jnp.dot(obs.H, obs.f_pbcor * _I) / 1e-23 # Jy

                if self._userdef_vis_model is not None:
                    V = self._userdef_vis_model( V, obs, f_latents )

                if dryrun:

                    return V, _I

                else:

                    obs.V_model = V

            elif obs.kind == 'photometry':
                # Beam-corrected observed total flux
                if self._spacing == 'FB':
                    F = jnp.sum(self._HT_prefactor * _I)
                elif self._spacing == 'linear':
                    F = jnp.trapezoid(2.0 * jnp.pi * self._r_GP_rad * _I, self._r_GP_rad)

                F = F * jnp.cos(self._incl) / 1e-23  # Jy

                if self._userdef_vis_model is not None:
                    F = self._userdef_vis_model(jnp.atleast_1d(F), obs, f_latents )[0] # this is for the total flux, so it is ok...


                if dryrun:
                    return F, _I
                else:
                    obs.F_model = F

            elif obs.kind == 'radialprofile':

                # Convolve with beam

                # interpolate the model intensity to the observed radial points
                I_itp = jnp.interp(obs._r, self._r_GP, _I)

                # padding
                kernel_size = obs._beam_kernel.shape[0]
                pad_width = kernel_size // 2

                
                if kernel_size % 2 == 0:
                    # even
                    I_itp_padded = jnp.pad(I_itp, pad_width=(pad_width, pad_width - 1), mode='symmetric')
                else:
                    # odd
                    I_itp_padded = jnp.pad(I_itp, pad_width=(pad_width, pad_width), mode='symmetric')

                # convolve
                I_itp_convolved = jax_convolve(I_itp_padded, obs._beam_kernel, mode='valid')
                
                if dryrun:

                    return I_itp_convolved, _I

                else:

                    obs.I_model = I_itp_convolved
            

    def _generate_model( self, f_latents ):
        '''
        Generate the model for all observations.
        f_latents: dictionary of latent parameters
        '''

        for band in self._bands:
            
            obs = self._observations[band]
            
            for _obs in obs:

                self._expansion_model( f_latents, _obs )

            

    def _GP_sample( self ):
        '''
        NumPyro model for Gaussian process sampling.
        '''

        f_latents = self._set_latent_params()
   
        self._generate_model( f_latents )

        self._sample_model( )

        


    def _sample_model( self ):
        '''
        Sample the model for all observations.
        '''
        
        for band in self._bands:
            
            obs = self._observations[band]

            if self._flux_uncert:

                
                if self._mean_fs[band] is None or self._s_fs[band] is None:
                    # this is only for photometry, where the flux scaling factor is fixed to 1.0 and no channel correlated uncertainty is considered.
                    f_band = 1.0
                    numpyro.deterministic(f"f_band_{band}", 1.0)

                else:
                    
                    # sampling from the normal dist
                    _g_f_band = numpyro.sample(
                                            f"g_f_scale_{band}",
                                            Normal(0, 1)
                                        )
                    
                    # change it to log normal dist
                    # f_band = exp(0 + 0.1 * standard_normal)
                    f_band = jnp.exp( jnp.log(self._mean_fs[band]) + _g_f_band * self._s_fs[band])
                    
                    numpyro.deterministic(f"f_band_{band}", f_band)

                    

                for _obs in obs:

                    if _obs.kind == 'visibility':

                        # If you want to fit the visibility dicretly....

                        V_final = _obs.V_model / f_band 

                        numpyro.deterministic(f"V_final_{_obs._name}", V_final)

                        numpyro.sample(
                                    f"Y_observed_{_obs._name}",
                                    Normal(loc= V_final, scale= _obs._s),
                                    obs = _obs._V
                                )

                    elif _obs.kind == 'photometry':

                        # For photometry, the flux is not scaled by f_band, as it should be independent.
                        F_final = _obs.F_model

                        numpyro.deterministic(f"F_final_{_obs._name}", F_final)

                        numpyro.sample(
                                    f"Y_observed_{_obs._name}",
                                    Normal(loc= F_final, scale= _obs._s),
                                    obs = _obs._F
                                )
                        
                    elif _obs.kind == 'radialprofile':

                        # Instead if you fit the radial profiles...
                        # UNDER DEVELOPMENT

                        Tb_final = I2Tb( _obs._nu, _obs.I_model ) / f_band 

                        numpyro.deterministic(f"Tb_final_{_obs._name}", Tb_final)

                        numpyro.sample(
                                    f"Y_observed_{_obs._name}",
                                    MultivariateNormal(loc= Tb_final, scale_tril=_obs.L_Cov),
                                    obs = _obs._Tb
                                )

            else:
                # if you do not want to fit the flux uncertainty...
                
                for _obs in obs:

                    if _obs.kind == 'visibility':

                        numpyro.sample(
                                    f"Y_observed_{_obs._name}",
                                    Normal(loc= _obs.V_model, scale= _obs._s),
                                    obs = _obs._V
                                )

                    elif _obs.kind == 'photometry':

                        numpyro.sample(
                                    f"Y_observed_{_obs._name}",
                                    Normal(loc= _obs.F_model, scale= _obs._s),
                                    obs = _obs._F
                                )
                        
                    elif _obs.kind == 'radialprofile':

                        numpyro.sample(
                                    f"Y_observed_{_obs._name}",
                                    MultivariateNormal(loc= I2Tb( _obs._nu, _obs.I_model), scale_tril=_obs.L_Cov),
                                    obs = _obs._Tb
                                )

    def set_visibility( self, band, q, V, s, f_s, f_mean,  nu, Nch, pbcor = 'ALMA-12m' ):
        '''set observations for a given band.

        This method allows you to input observational data for a specific band, including spatial frequencies, visibilities, uncertainties, flux scaling factors, and frequencies.
        
        Args:
            band (str): name of the observation band
            q (dict): spatial frequencies in 1/arcsec. Should be a dictionary of arrays for each channel.
            V (dict): observed visibilities in Jy. Should be a dictionary of arrays for each channel.
            s (dict): uncertainties in Jy. Should be a dictionary of arrays for each channel.
            f_s (float): flux scaling factor standard deviation
            f_mean (float): mean flux scaling factor
            nu (array): frequencies in Hz
            Nch (int): number of channels. Shound be consistent with q, V, s.
            pbcor (str): 'ALMA-12m' for primary beam correction. Set None NOT to do it.
        '''

        obs_tmp = []

        for nch in range(Nch):
            
            _obs = observation( f'{band}_ch_{nch}', nu[nch], kind='visibility' )
            _obs._visibility( q[nch], V[nch], s[nch] )

            #_obs.r_rad = jnp.arange( jnp.min(jnp.deg2rad(self._r_GP/3600)), jnp.max(jnp.deg2rad(self._r_GP/3600)), 1/jnp.max(_obs.q)/self.ndr )

            #kr_matrix = _obs.q[:, jnp.newaxis] * _obs.r_rad[jnp.newaxis, :]
        

            arg = 2.0 * jnp.pi* _obs._q[:, None] * self._r_out_rad * self._j0k[None, :] / self._j0N_plus

            H = self._HT_prefactor * J0(arg)
 
            _obs.H = H

            if pbcor:
                #_obs.f_pbcor = pbcor_fac_ALMA_12m( self._r_GP, nu[nch] ) # r_GP is in arcsec, incl is in rad, nu is in Hz.
                _obs.f_pbcor = get_deprojected_beam_arcsec( self._r_GP, nu[nch], self._incl, pa=self._pa, dx=0.0, dy=0.0, n_theta=360)
            else:
                _obs.f_pbcor = 1.0
            
            obs_tmp.append( _obs )
            
        self._bands.append(band)
            
        self._observations[band] = obs_tmp
        self._s_fs[band] = f_s
        self._mean_fs[band] = f_mean



    def set_photometry( self, band, F, s, nu):
        '''set unresolved photometry data for a given band.

        This method allows you to input unresolved photometry data, including observed fluxes, uncertainties, flux scaling factors, and frequencies.
        The data should be provided as arrays for each channel.
        Assumes that the uncertainties are independent across channels.

        Args:
            band (str): name of the observation band or telescope (just to identify the data)
            F (array): observed flux in Jy
            s (array): uncertainty in Jy
            nu (array): frequencies in Hz
        '''

        obs_tmp = []

        for nch in range(len(nu)):

            _obs = observation( f'{band}_ch_{nch}', nu[nch], kind='photometry' )
            _obs._photometry( F[nch], s[nch] )
            
            obs_tmp.append( _obs )

        self._bands.append(band)
            
        self._observations[band] = obs_tmp
        self._s_fs[band] = 0.0
        self._mean_fs[band] = 1.0
        

    def set_radialprofile(  self, band, r, Tb, f_s, f_mean, nu, Nch, FWHM, Cov): 

        obs_tmp = []

        for nch in range(Nch):
            
            _obs = observation( f'{band}_ch_{nch}', nu[nch], kind='radialprofile' )
            _obs._radialprofile( r[nch], Tb[nch], FWHM, Cov[nch] )
            
            obs_tmp.append( _obs )
            
        self._bands.append(band)
            
        self._observations[band] = obs_tmp
        self._s_fs[band] = f_s
        self._mean_fs[band] = f_mean


    def set_opacity( self, opac_dict, Na = 1000, Nq = 1000, smooth = True, log10_a_smooth = 0.05, a_min = None, a_max = None ):
        '''Set dust opacity model.
        
        This method allows you to define the dust opacity model using precomputed opacity data.

        Args:
            opac_dict (dict): dictionary containing dust opacity data with keys 'a', 'lam', 'k_abs', 'k_sca', 'g'
            Na (int, optional): number of dust size grid points. Defaults to 1000.
            Nq (int, optional): number of size distribution index grid points. Defaults to 1000.
            smooth (bool, optional): whether to apply smoothing to the opacity table. Defaults to True.
            log10_a_smooth (float, optional): smoothing scale in log10 grain size. Defaults to 0.05.
            a_min (float, optional): minimum grain size in microns. Defaults to the minimum grain size in the opacity data.
            a_max (float, optional): maximum grain size in microns. Defaults to the maximum grain size in the opacity data. 
        '''
        
        a      = opac_dict['a']
        lam    = opac_dict['lam']
        k_abs  = opac_dict['k_abs']
        k_sca  = opac_dict['k_sca']
        gsca   = opac_dict['g']

        one_min_g = (1 - gsca)
        one_min_g[ one_min_g < 0 ] = 1e-6

        k_sca_eff = one_min_g * k_sca

        if a_min is not None and a_max is not None:
            a_dense = jnp.array(jnp.logspace( np.log10(a_min), np.log10(a_max), Na ))
        else:
            a_dense = jnp.array(jnp.logspace( np.log10(jnp.min(a)), np.log10(jnp.max(a)), Na ))

        log10_a_dense = jnp.log10(a_dense)

        q_dense = jnp.linspace( 0.0, 6.0, Nq )


        # We tried to calc kappa_tot from kappa at a single dust size at each iteration
        # to allow the opacity model to be flexible, but it did not work well probably because of the large correlation between parameters.
        # We therefore pre-calculate interpolators for each observation here.


        for band in self._bands:
            
            obs = self._observations[band]
            
            for _obs in obs:

                lam0 = c/_obs._nu

                log10_k_abs_tot, log10_k_sca_eff_tot = create_opacity_table( lam, a, k_abs, k_sca_eff, lam0, log10_a_dense, q_dense, smooth=smooth, 
                                                                            log10_a_smooth=log10_a_smooth, log10_a_min = jnp.log10(a_dense[0]) )

                
                _obs.log10_a_dense = log10_a_dense
                _obs.q_dense = q_dense
                _obs.log10_k_abs_tot = log10_k_abs_tot
                _obs.log10_k_sca_eff_tot = log10_k_sca_eff_tot


                _obs.log10_k_abs_tot_itp = RegularGridInterpolator(
                                            (log10_a_dense, q_dense),
                                            log10_k_abs_tot,
                                            method='linear',
                                            bounds_error=False,
                                            fill_value=None
                                        )
                
                _obs.log10_k_sca_eff_tot_itp = RegularGridInterpolator(
                                            (log10_a_dense, q_dense),
                                            log10_k_sca_eff_tot,
                                            method='linear',
                                            bounds_error=False,
                                            fill_value=None
                                        )

        self._opacity_config = {
            'Na': Na,
            'Nq': Nq,
            'smooth': smooth,
            'log10_a_smooth': log10_a_smooth,
            'a_min': a_min,
            'a_max': a_max,
        }
                

    def calc_model( self, parameters ):
        """Compute model visibilities using the current parameters.

        This method calculates the model visibilities and intensities for all observations based on the provided parameters.

        Args:
            parameters (dict): dictionary of model parameters. Should contain keys corresponding to free and fixed parameters.

        Returns:
            V_res (dict): dictionary of model visibilities for each band and observation.
            I_res (dict): dictionary of model intensities for each band and observation.

        """

        f_latents = parameters


        V_res = {}
        I_res = {}
        
        
        for band in self._bands:
            
            V_res[band] = {}
            I_res[band] = {}
            
            obs = self._observations[band]

            for _obs in obs:


                _V_res , _I_res = self._expansion_model( f_latents, _obs, dryrun=True )

                V_res[band][_obs._name] = jnp.array(_V_res)
                I_res[band][_obs._name] = jnp.array(_I_res)


        return V_res, I_res
    

    def _calc_model_cv_duplicated( self, MAP_results ):
        '''Compute model visibilities for cross-validation.
        
        This method calculates the model visibilities for cross-validation using the MAP results.

        Args:
            MAP_results (results): results object containing MAP samples and related information.

        Returns:
            V_final (dict): dictionary of model visibilities for each band and observation, corrected for flux scaling.
        
        '''

        V_final = {}

        MAP_f = MAP_results.sample['MAP_f']
        MAP_g = MAP_results.sample['MAP_g']


        # if a parameter is not assigned in MAP_f, the default value is added:
        for param_name, profile in self._fixed_parameters.items():
            MAP_f[param_name] = profile['profile']

        Vmod, _ = self.calc_model( MAP_f )

        for band in self._bands:

            obs = self._observations[band]

            V_final[band] = {}

            for _obs in obs:

                f_band = MAP_g[f'f_band_{band}']
                V_final[band][_obs._name] = Vmod[band][_obs._name] / f_band

                
        return V_final

    def calc_chi2(self, MAP_results):

        '''Compute chi-squared for cross-validation.
                
            This method calculates the chi-squared value for cross-validation using the MAP results. Note that the MAP estimate should be done with the data splitted from the same dataset.
        
            Args:
                MAP_results (results): results object containing MAP samples and related information.
                
            Returns:
                chi2 (float): chi-squared value for the cross-validation.
                
        '''
        
        MAP_f = dict(MAP_results.sample['MAP_f'])
        MAP_g = MAP_results.sample['MAP_g']

        for param_name, profile in self._fixed_parameters.items():
            MAP_f[param_name] = profile['profile']

        model_values, _ = self.calc_model(MAP_f)
        chi2 = 0.0

        for band in self._bands:
            
            if self._flux_uncert:
                f_band = jnp.squeeze(MAP_g[f'f_band_{band}'])
            else:
                f_band = 1.0

            for obs in self._observations[band]:
                prediction = model_values[band][obs._name]

                if obs.kind == 'visibility':
                    prediction = jnp.reshape(prediction, obs._V.shape) / f_band
                    chi2 += jnp.sum(((obs._V - prediction) / obs._s) ** 2)

                elif obs.kind == 'photometry':
                    prediction = jnp.reshape(prediction, obs._F.shape) / f_band
                    chi2 += jnp.sum(((obs._F - prediction) / obs._s) ** 2)

                elif obs.kind == 'radialprofile':
                    prediction = jnp.reshape(I2Tb(obs._nu, prediction), obs._Tb.shape) / f_band
                    residual = obs._Tb - prediction
                    whitened = jax.scipy.linalg.solve_triangular(obs.L_Cov, residual, lower=True)
                    chi2 += jnp.sum(whitened ** 2)

        return chi2

    def _calc_chi2_duplicated( self, MAP_results ):
        '''Compute chi-squared for cross-validation.
        
        This method calculates the chi-squared value for cross-validation using the MAP results. Note that the MAP estimate should be done with the data splitted from the same dataset.

        Args:
            MAP_results (results): results object containing MAP samples and related information.
        
        Returns:
            chi2 (float): chi-squared value for the cross-validation.
        
        '''

        Vmod = self._calc_model_cv_duplicated( MAP_results )

        chi2 = 0.0
        
        for band in self._bands:
            obs = self._observations[band]
            
            for _obs in obs: 

                chi2 += jnp.sum( ( _obs._V - Vmod[band][_obs._name] )**2 / _obs._s**2  )

        return chi2

    
    def calc_model_tau( self, parameters ):
        """Compute model optical depths

        This method calculates the model optical depths for all observations based on the provided parameters.

        Args:
            parameters (dict): dictionary of model parameters. Should contain keys corresponding to free and fixed parameters.

        Returns:
            tau_res (dict): dictionary of model optical depths for each band and observation.

        """

        f_latents = parameters


        tau_res = {}
        
        
        for band in self._bands:
            
            tau_res[band] = {}
            
            obs = self._observations[band]

            for _obs in obs:


                _tau_res = self._expansion_model( f_latents, _obs, dryrun=True, output_tau = True )


                tau_res[band][_obs._name] = jnp.array(_tau_res)

        return tau_res

    def _g_to_f( self, g_samples ):

        # convert g_samples to f_samples using the sigmoid transform for GP parameters and identity for non-GP parameters
        f_samples = {}
        for param_name, priors in self._free_parameters.items():
            if priors['GP'] == False:
                f_samples[param_name] = g_samples[f'{param_name}']
            else:
                f_samples[param_name] = sigmoid_transform(
                    g_samples[f'g_{param_name}'], 
                    min_val=priors['f_min'], max_val=priors['f_max'])
        return f_samples


class observation:
    '''class for containing observation data

    '''

    def __init__(self, name, nu, kind='visibility' ):
        '''initialize observation data
        
        Args:
            name (str): name of the observation
            nu (float): frequency in Hz
            kind (str, optional): type of observation ('visibility', 'photometry', or 'radialprofile'). Defaults to 'visibility'.
        '''
        self.kind = kind
        self._name = name
        self._nu = nu

    @property
    def nu(self):
        """Observing frequency [Hz]."""
        return self._nu

    def _visibility(self, q, V, s):
        '''set visibility data
        
        Args:
            q (array): spatial frequencies in 1/arcsec
            V (array): observed visibilities in Jy
            s (array): uncertainties in Jy
        '''
        self._q =  jax.device_put(jnp.asarray(q))
        self._V =  jax.device_put(jnp.asarray(V))
        self._s =  jax.device_put(jnp.asarray(s))

    def _photometry(self, F, s):
        '''set unresolved photometry data
            
            Args:
            F (float): observed flux in Jy
            s (float): uncertainty in Jy
        '''
        self._F =  jax.device_put(jnp.asarray(F))
        self._s =  jax.device_put(jnp.asarray(s))
        self._q = jnp.zeros(1) # this is only for the user_def model (if it takes q dependence)

    def _radialprofile( self, r, Tb, FWHM, Cov):
        '''set radial profile data [experimental]
            
        Args:
            r (array): radial distances in arcsec
            Tb (array): brightness temperatures in K
            FWHM (float): full width at half maximum of the beam in arcsec
            Cov (array): covariance matrix
        '''

        self._r = jax.device_put(jnp.asarray(r))
        self._Tb = jax.device_put(jnp.asarray(Tb))
        self._FWHM = FWHM


        ## Covariance matrix
        sigma_beam = FWHM / (2 * jnp.sqrt(2 * jnp.log(2)))
        

        self.Cov = Cov #s[:, None] * R_dist * s[None, :]
                    
        

        self.L_Cov = jnp.linalg.cholesky(self.Cov)


        ## beam kernel

        dr = r[1] - r[0] # assuming uniform spacing in observations
        
        kernel_r = jnp.arange(-5 * sigma_beam, 5 * sigma_beam + dr, dr)
        kernel = jnp.exp(-(kernel_r**2) / (2.0 * sigma_beam**2))
        kernel = kernel / jnp.sum(kernel)

        self._beam_kernel = kernel
    
        

class inference:
    '''class for inference methods
    
    Attributes:
        prior: method to show prior distributions
        SVI_MAP: method to run SVI for MAP estimation
        MCMC: method to run MCMC sampling

    '''

    def __init__(self, model ):

        self.model = model


    def prior(self, num_samples = 1, seed = None, calc_cond_num = False):
        '''Show prior distributions for the latent parameters.

        This method generates samples from the prior distributions of the latent parameters defined in the model.

        Args:
            num_samples (int, optional): number of prior samples to generate. Defaults to 1.
            seed (int, optional): random seed for reproducibility. Defaults to None.

        Returns:
            prior (results): results object containing prior samples and related information.

        '''

        if seed is None:
            seed = np.random.randint(0, 1e6)

        def prior_model():
            f_latents = self.model._set_latent_params(calc_cond_num=calc_cond_num)

            return f_latents


        

        prior_predictive = Predictive(prior_model, num_samples=num_samples)


        rng_key = jax.random.PRNGKey(seed)


        f_func_all = {}
        g_func_all = {}


        for i, (param_name, priors) in enumerate(self.model._free_parameters.items()):

            rng_key, rng_key2 = jax.random.split(rng_key)
            prior_predictions = prior_predictive(rng_key)[f'g_{param_name}']

            if priors['GP'] == True:

                for j, g_func in enumerate(prior_predictions):

                    f_func = sigmoid_transform(g_func, 
                                            min_val=priors['f_min'], 
                                            max_val=priors['f_max'])
                    
                    
                    if j == 0:
                        _f_func_all = f_func
                        _g_func_all = g_func
                    else:
                        _f_func_all = jnp.vstack((_f_func_all, f_func))
                        _g_func_all = jnp.vstack((_g_func_all, g_func))
                    

                f_func_all[param_name] = jnp.array([_f_func_all])
                g_func_all[param_name] = jnp.array([_g_func_all])

            else:

                f_func = prior_predictions * priors['f_s'] + priors['f_mean']

                f_func_all[param_name] = jnp.array([f_func])
                g_func_all[param_name] = jnp.array([prior_predictions])


        self._prior = results(  r = self.model._r_GP, 
                                    sample = { 'prior_f': f_func_all,
                                               'prior_g': g_func_all,
                                            },
                                    logP = { 'prior': None },
                                    param_set = self.model._free_parameters
                                    )
        

        return self._prior
    
    def SVI_MAP(self, num_iterations=1000, num_particles=1, adam_lr=0.01, uniform_radius = 0.1, seed = None):
        '''Run Stochastic Variational Inference (SVI) to find the Maximum A Posteriori (MAP) estimate of the latent parameters.
        
        This method uses SVI with an AutoDelta guide to estimate the MAP of the latent parameters defined in the model.

        Args:
            num_iterations (int, optional): number of SVI iterations. Defaults to 1000
            num_particles (int, optional): number of particles for ELBO estimation. Defaults to 1.
            adam_lr (float, optional): learning rate for the Adam optimizer. Defaults to 0.01.
            uniform_radius (float, optional): radius for uniform initialization of parameters. Defaults to 0.1.
            seed (int, optional): random seed for reproducibility. Defaults to None.

        Returns:
            svi_map_results (results): results object containing MAP estimates and related information.

        '''

        if seed is None:
            seed = np.random.randint(0, 1e6)
        
        rng_key = jax.random.PRNGKey(seed)
        
        _init_strategy = init_to_uniform(radius=uniform_radius)

         
        guide = AutoDelta(self.model._GP_sample, init_loc_fn=_init_strategy)
        optimizer = Adam(adam_lr)
        elbo = Trace_ELBO(num_particles)
        svi = SVI(self.model._GP_sample, guide, optimizer, elbo)
        
        # run svi
        rng_key, rng_key_svi = jax.random.split(rng_key)
        svi_result = svi.run(
                rng_key_svi, 
                num_iterations, 
                progress_bar=True
            )
        
        self.svi_result = svi_result
        params = svi_result.params
        loss = svi_result.losses[-1]


        # 1. Remap internal AutoDelta names to the original sampling names in the model
        map_samples_fixed = {}
        for k, v in params.items():
            name_in_model = k.replace("_auto_loc", "")
            map_samples_fixed[name_in_model] = v[None, ...]

        # 2. Recalc the model outputs at the MAP estimate
        predictive = Predictive(self.model._GP_sample, posterior_samples=map_samples_fixed, num_samples=1)
        rng_key, rng_key_pred = jax.random.split(rng_key)
        map_outputs = predictive(rng_key_pred)

        #self.map_outputs = map_outputs
        
        
        map_estimates = {k: jnp.array([[jnp.squeeze(v, axis=0)]]) for k, v in map_outputs.items()}
        

        self.delta_medians = {}
        for param_name, priors in self.model._free_parameters.items():
            if priors['GP'] == False:
                self.delta_medians[param_name] = map_estimates[param_name][0,0]
            else:
                g_predictions = map_estimates[f'g_{param_name}']
                
                
                f_predictions = sigmoid_transform(
                    g_predictions, 
                    min_val=priors['f_min'], 
                    max_val=priors['f_max']
                )
                self.delta_medians[param_name] = f_predictions[0,0]

        self.svi_map_results = results(  r = self.model._r_GP, 
                                    sample = { 'MAP_g': map_estimates,
                                               'MAP_f': self.delta_medians,
                                               'MAP_model' : map_outputs
                                            },
                                    logP = { 'MAP': -loss },
                                    param_set = self.model._free_parameters
                                    )

        return self.svi_map_results


    def MCMC(self, num_warmup, num_samples, step_size = 1.0, num_chains = 1, max_tree_depth=10, adapt_step_size=True, uniform_radius = 0.1, seed = None):
        '''Run MCMC sampling using the NUTS algorithm.
        
        This method performs MCMC sampling to estimate the posterior distributions of the latent parameters defined in the model.

        Args:
            num_warmup (int): number of warmup iterations.
            num_samples (int): number of MCMC samples to draw.
            step_size (float, optional): initial step size for the NUTS sampler. Defaults to 1.0.
            num_chains (int, optional): number of MCMC chains to run in parallel. Defaults to 1.
            max_tree_depth (int, optional): maximum tree depth for the NUTS sampler. Defaults to 10.
            adapt_step_size (bool, optional): whether to adapt the step size during warmup. Defaults to True.
            uniform_radius (float, optional): radius for uniform initialization of parameters. Defaults to 0.1.
            seed (int, optional): random seed for reproducibility. Defaults to None.

        Returns:
            mcmc_results (results): results object containing posterior samples and related information.
        '''

        if seed is None:
            seed = np.random.randint(0, 1e6)
        
        rng_key = jax.random.PRNGKey(seed)

        '''
        if init_strategy == 'value':
            _init_strategy = init_to_value( values=medians )
            init_params = medians
        elif init_strategy == 'uniform':
            _init_strategy = init_to_uniform(radius=radius)
            init_params = None
        elif init_strategy == 'sample':
            _init_strategy = init_to_sample()
            init_params = None
        elif init_strategy == 'median':
            _init_strategy = init_to_median()
            init_params = None
        '''

        _init_strategy = init_to_uniform(radius=uniform_radius)

        kernel = NUTS(self.model._GP_sample,
                        step_size=step_size,
                        adapt_step_size=adapt_step_size,
                        dense_mass=True,
                        init_strategy = _init_strategy,
                        max_tree_depth=max_tree_depth)
            
        mcmc = numpyro_MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=True,
            #chain_method='vectorized',
            chain_method='parallel'
        )

        # --- warmup ---
        mcmc.warmup(
            rng_key,
            extra_fields=('diverging', 'accept_prob', 'energy', 'potential_energy'),
            collect_warmup=True
        )
        
        g_warmup_samples = mcmc.get_samples(group_by_chain=True)
        warmup_logP = -mcmc.get_extra_fields(group_by_chain=True)['potential_energy']

        # --- main sampling ---
        mcmc.run(
            mcmc.post_warmup_state.rng_key,
            init_params=None,
            extra_fields=('diverging', 'accept_prob', 'energy', 'potential_energy')
        )
        
        mcmc.print_summary()

        g_posterior_samples = mcmc.get_samples(group_by_chain=True)
        posterior_logP = -mcmc.get_extra_fields(group_by_chain=True)['potential_energy']


        f_warmup_samples = self.model._g_to_f( g_warmup_samples )
        f_posterior_samples = self.model._g_to_f( g_posterior_samples )
        

        self.mcmc_results = results(  r = self.model._r_GP, 
                                    sample = { 'warmup_f': f_warmup_samples,
                                                  'posterior_f': f_posterior_samples,
                                                   'warmup_g': g_warmup_samples,
                                                   'posterior_g': g_posterior_samples,
                                                 },
                                    logP = { 'warmup': warmup_logP,
                                             'posterior': posterior_logP },

                                    param_set = self.model._free_parameters
                                    
                                    )
        
        return self.mcmc_results
    

    



class results:
    '''class for storing inference results

    Attributes:
        r: radial grid points
        sample: dictionary of samples
        logP: dictionary of log probabilities

    
    '''

    def __init__(self, r, sample, logP, param_set):
        self.r = r
        self.sample = sample
        self.logP = logP
        self.param_set = param_set


class plot:
    '''class for plotting results

    Attributes:
        sample_paths: method to plot sample paths

    ''' 

    def __init__(self, results ):
        self.r =  results.r
        self.sample =  results.sample

        print('saved keys: ', self.sample.keys() )

    def _plot_axes(  self, ax, x, y, nskip, twod = True, xlabel=None, ylabel=None, plot_kwargs = {}, scatter_kwargs = {}):

        for j in range(0, y.shape[0], nskip):


            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            if twod:

                ax.plot( x, y[j,:], **plot_kwargs )

            else:


                ax.scatter( np.median(x), y[j], **scatter_kwargs )




    def sample_paths( self, key, nskip = 100, plot_kwargs = {'alpha': 0.5, 'lw': 1.0, 'color':'royalblue'}, scatter_kwargs = {'alpha': 0.5, 'color':'royalblue'} ):
        '''Plot sample paths for a given parameter key.

        Args:
            key (str): parameter key to plot
            nskip (int, optional): number of samples to skip for plotting. Defaults to 100.
            plot_kwargs (dict, optional): keyword arguments for line plots. Defaults to {'alpha': 0.5, 'lw': 1.0, 'color':'roayalblue'}.
            scatter_kwargs (dict, optional): keyword arguments for scatter plots. Defaults to {'alpha': 0.5, 'color':'royalblue'}.  
            
        Returns:
            None
        ''' 

        _samples = self.sample[key]

        N_panel = len( _samples.keys() )

        fig, axes = plt.subplots(N_panel,1, figsize=(10, 4*N_panel), sharex=True )

        axes = np.atleast_1d(axes)

        for i, param_name in enumerate( _samples.keys() ):

            _sample = _samples[param_name]

            if len(np.shape(_sample)) == 3:
                _sample = _sample.reshape( (_sample.shape[0]*_sample.shape[1], _sample.shape[2]) )

                
                self._plot_axes( axes[i], self.r, _sample, nskip, twod = True, xlabel='Radius (arcsec)', ylabel=f'{param_name}', plot_kwargs = plot_kwargs, scatter_kwargs = scatter_kwargs )

            else:
                _sample = _sample[0]

                self._plot_axes( axes[i], self.r, _sample, nskip, twod = False, xlabel='', ylabel=f'{param_name}', plot_kwargs = plot_kwargs, scatter_kwargs = scatter_kwargs )



    