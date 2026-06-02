
###### Utiities #########

import jax
import jax.numpy as jnp
import numpy as np
from ._constants import *
from astropy import units as u
import warnings
import numpy as np

import jax.numpy as jnp


from scipy.ndimage import gaussian_filter1d

from scipy.special import i0

from scipy.special import j1



from scipy.interpolate import interp1d as scipy_interp1d

def check_range_old(x, lower, upper, alpha=10.0):
    # Apply exponential penalty if x is outside [lower, upper]
    penalty_lower = -jnp.exp(-alpha * (x - lower))
    penalty_upper = -jnp.exp(alpha * (x - upper))
    penalty = jnp.where(x < lower, penalty_lower,
                jnp.where(x > upper, penalty_upper, 0.0))
    return penalty

def check_range(x, lower, upper, alpha=100.0):
    """
    勾配消失を防ぎつつ、数値的に安定したペナルティを与える
    """
    # 境界から外れた距離を計算
    dist_lower = lower - x
    dist_upper = x - upper
    
    # 範囲外（dist > 0）の場合のみ、その距離の2乗をペナルティとする
    penalty_lower = -alpha * jnp.square(jnp.maximum(0.0, dist_lower))
    penalty_upper = -alpha * jnp.square(jnp.maximum(0.0, dist_upper))
    
    return penalty_lower + penalty_upper


def term_pbcor( nu, incl, r, D ):

    lam = c / nu # cm

    FoV = np.rad2deg(1.13 * lam * 1e-2 / D)*3600

    Theta = FoV / (2*np.sqrt(2 * np.log(2)))


    a1 = np.exp( -(r**2 / 4 / Theta**2) * ( 1 + np.cos(incl)**2 ) )
    a2 = i0( r**2/4/Theta**2 * np.sin(incl)**2 )

    return a1 * a2



def hankel_transform_0_jax(f, r, k, bessel):
    '''
    Perform the Hankel transform of order 0 using JAX.
    f: jnp.ndarray, function values at radial distances r (shape: [n_r])
    r: jnp.ndarray, radial distances (shape: [n_r])
    k: jnp.ndarray, spatial frequencies (shape: [n_k])
    bessel: jnp.ndarray, precomputed Bessel function values (shape: [n_k, n_r])
    Returns the Hankel transform values at spatial frequencies k (shape: [n_k]).
    '''
    
    dr = jnp.gradient(r)
    fr = f * r

    #def integrate(ki):
    #    integrand = fr * j0(k * r) 
    #    return jnp.sum(integrand * dr)
    
    return jnp.sum( 2*np.pi * fr * bessel * dr, axis=1)

def rbf_kernel(X1, X2, variance, lengthscale):
    '''
    Compute the Radial Basis Function (RBF) kernel between two sets of input points using JAX.
    X1: jnp.ndarray, first set of input points (shape: [n1, d])
    X2: jnp.ndarray, second set of input points (shape: [n2 , d])
    variance: float, variance parameter of the RBF kernel
    lengthscale: float, lengthscale parameter of the RBF kernel
    Returns the RBF kernel matrix (shape: [n1, n2]).
    '''
    
    sq_dist = jnp.sum(X1**2, 1)[:, None] + jnp.sum(X2**2, 1)[None, :] - 2 * jnp.dot(X1, X2.T)
    
    return variance**2 * jnp.exp(-0.5 / lengthscale**2 * sq_dist)

def B(nu, T):
    '''
    Calculate the Planck function B(nu, T).
    nu: jnp.ndarray or float, frequency in Hz
    T: jnp.ndarray or float, temperature in Kelvin
    Returns the Planck function values.
    ''' 

    return 2*h*nu**3/c**2 / ( jnp.exp(h*nu/k_B/T) - 1 )
    

def I2Tb(nu, I):

    '''
    Convert intensity I(nu) to brightness temperature Tb using the Reighleigh-Jeans approximation.
    nu: jnp.ndarray or float, frequency in Hz
    I: jnp.ndarray or float, intensity in cgs units (erg/s/cm^2/Hz/sr)
    Returns the brightness temperature Tb in Kelvin.
    ''' 
    return c**2 / (2*k_B*nu**2) * I

def sigmoid_transform(x, min_val=0.0, max_val=1.0, leak = 0.01):
    '''
    Apply a sigmoid transformation to the input array x.
    The transformed values will be in the range [min_val, max_val].
    x: jnp.ndarray, input array to be transformed
    min_val: float, minimum value of the transformed output
    max_val: float, maximum value of the transformed output
    Returns the transformed array with values in the range [min_val, max_val].
    '''
    
    return min_val + (max_val - min_val) / (1 + jnp.exp(-x))

def sigmoid_transform_old(x, min_val=0.0, max_val=1.0):
    y = (2.0 / jnp.pi) * jnp.arctan(x)   # 範囲は (-1, 1)
    y01 = 0.5 * (y + 1.0)
    return min_val + (max_val - min_val) * y01


def F(tau, omega):
    '''
    Calculate the function F(tau, omega) used in radiative transfer.
    tau: jnp.ndarray or float, optical depth
    omega: jnp.ndarray or float, single scattering albedo
    Returns the computed values of F(tau, omega).
    Ref. Miyake & Nakagawa 1993, Icarus, 106, 20; Sierra et al. 2020, ApJ, 892, 136
    '''
    
    w = omega
    
    term1 = (jnp.sqrt(1 - w) - 1.0) * jnp.exp(-jnp.sqrt(3.0 / (1.0 - w)) * tau)
    
    A_num = 1.0 - jnp.exp(-(jnp.sqrt(3.0 * (1.0 - w)) + 1.0) * tau / (1.0 - w))
    A_den = jnp.sqrt(3.0 * (1.0 - w)) + 1.0
    A = A_num / A_den
    
    B_num = jnp.exp(-tau / (1.0 - w)) - jnp.exp(-jnp.sqrt(3.0 / (1.0 - w)) * tau)
    B_den = jnp.sqrt(3.0 * (1.0 - w)) - 1.0
    B = B_num / B_den
    
    term2 = (jnp.sqrt(1 - w) + 1.0)
    
    denom = term1 - term2
    
    return  (A + B) / denom


def f_I(nu, incl, T, Sigma_d, k_abs_tot, k_sca_eff_tot, output_tau = False):
    '''
    Calculate the intensity I(nu) using radiative transfer with scattering.
    nu: jnp.ndarray or float, frequency in Hz
    incl: jnp.ndarray or float, inclination angle in radians
    T: jnp.ndarray or float, temperature in Kelvin
    Sigma_d: jnp.ndarray or float, dust surface density in g/cm^2
    dust_params: list of jnp.ndarray or float, dust parameters (e.g., maximum grain size). Assuming the order matches the interpolators.
    f_log10_ka: function, interpolator for log10 of absorption opacity
    f_log10_ks: function, interpolator for log10 of scattering opacity
    Returns the computed intensity I(nu).
    ''' 

    ka = k_abs_tot
    ks = k_sca_eff_tot

    chi = ka + ks
    omega = ks / chi
    
    tau = ka * Sigma_d / jnp.cos(incl)

    if output_tau:
        return tau

    else:
        return B(nu, T) * (  1 - jnp.exp( -tau/(1-omega) ) + omega*F(tau, omega)  )


def size_average_opacity( log10_a, log10_k_abs, log10_k_sca_eff, log10_a_max, log10_a_min, q):

    a = 10**log10_a
    a_max = 10**log10_a_max
    a_min = 10**log10_a_min

    mask = jnp.where( (a <= a_max) & (a >= a_min), 1.0, 0.0)
    n = a**(4.0-q) * mask #* jnp.exp(-(a / a_max)**gamma)  * jnp.exp(-(a_min/a)**gamma)

    sum_n = jnp.sum(n)
    
    log10_k_abs_tot = jnp.log10(jnp.dot(n, 10**log10_k_abs) /sum_n)
    log10_k_sca_eff_tot = jnp.log10(jnp.dot(n, 10**log10_k_sca_eff) /sum_n)

    return log10_k_abs_tot, log10_k_sca_eff_tot


def create_opacity_table(lam, a, k_abs, k_sca_eff, lam0, log10_a_dense, q_dense, smooth=True, log10_a_smooth=0.1, log10_a_min = -5.0):
    '''
    Create opacity tables by interpolating and averaging over grain size distributions.
    lam: np.ndarray, wavelengths in microns (shape: [M])
    a: np.ndarray, grain sizes in microns (shape: [N])
    k_abs: np.ndarray, absorption opacities (shape: [N, M])
    k_sca_eff: np.ndarray, effective scattering opacities (shape: [N, M])
    lam0: np.ndarray, target wavelengths for interpolation in microns (shape: [L])
    log10_a_dense: jnp.ndarray, dense grid of log10 grain sizes in microns (shape: [P])
    q_dense: jnp.ndarray, dense grid of size distribution indices (shape: [Q])
    smooth: bool, whether to apply smoothing to the interpolated opacities
    log10_a_smooth: float, smoothing scale in log10 grain size
    Returns:
    log10_k_abs_tot: jnp.ndarray, size-averaged log10 absorption opacities (shape: [P, Q])
    log10_k_sca_eff_tot: jnp.ndarray, size-averaged log10 effective scattering opacities (shape: [P, Q])
    ''' 

    log10_k_abs_itp_lam = scipy_interp1d( np.log10(lam), np.log10(k_abs), axis=1, kind="cubic")( np.log10(lam0) )
    log10_k_sca_eff_itp_lam = scipy_interp1d( np.log10(lam), np.log10(k_sca_eff), axis=1, kind="cubic")( np.log10(lam0) )

    log10_k_abs_itp = jnp.array( scipy_interp1d( np.log10(a), log10_k_abs_itp_lam , kind='cubic', bounds_error=False, fill_value="extrapolate")(log10_a_dense))
    log10_k_sca_eff_itp = jnp.array( scipy_interp1d( np.log10(a), log10_k_sca_eff_itp_lam, kind='cubic', bounds_error=False, fill_value="extrapolate")(log10_a_dense))
                
    # smoothing to avoid Mie interference wiggles

    if smooth:
        sigma_a = log10_a_smooth/ ( log10_a_dense[1] - log10_a_dense[0])
                
        log10_k_abs_itp  = jnp.array( gaussian_filter1d( np.array(log10_k_abs_itp), sigma=sigma_a ) )
        log10_k_sca_eff_itp = jnp.array( gaussian_filter1d( np.array(log10_k_sca_eff_itp), sigma=sigma_a ) )



                
        vmap_over_q = jax.vmap(
                    size_average_opacity,
                    in_axes=(None, None, None, None, None, 0)  # q だけが配列
                )

               
        vmap_over_a_and_q = jax.vmap(
                    vmap_over_q,
                    in_axes=(None, None, None, 0, None, None)  # log10_a_max だけが配列
                )

                
        log10_k_abs_tot, log10_k_sca_eff_tot = vmap_over_a_and_q(
                    log10_a_dense,
                    log10_k_abs_itp,
                    log10_k_sca_eff_itp,
                    log10_a_dense,
                    log10_a_min,
                    q_dense
                )
        
        return log10_k_abs_tot, log10_k_sca_eff_tot
    


def pbcor_fac_ALMA_12m( r, nu, D = 10.7e2, d = 0.75e2 ):
    '''
    This is the same equation implemented in CASA. Note that this is not the same as the physically correct equation. 
    See the CASA documentation for details.
    r: radius in arcsec
    nu: frequeny in Hz
    '''

    eps = d/D
    
    k = np.pi * nu/c 
    x = k*D*np.sin(np.deg2rad(r/3600))

    t1 = 2.0 * j1(x)/x
    t2 = 2.0 * eps * j1( x /eps )/ x 

    V = 1/(1-eps**2)**2 * ( t1 -  eps**2 * t2 )**2
    
    return V


def obscured_airy_pattern_CASA(theta, D, d, nu):
    '''
    In a function in CASA (PBMath1DAiry::fillPBArray()), the obscured Airy disk is implimented as follows.
    I think this is likely not correct, as the "lengthRatio" parameter is multiplied to D,
    resulting in larger effective blockage than the aparture itself.
    
    theta : offset from the center (arcsec)
    D : aparture diameter (cm)
    d : blockage diameter (cm)
    nu : frequency (Hz)
    '''

    areaRatio = (D/d)**2
    areaNorm = areaRatio - 1.0
    lengthRatio = D/d
    
    k = np.pi * nu/c 
    x = k*D*np.sin(np.deg2rad(theta/3600))

    v = ( areaRatio * 2.0 * j1(x)/x - 2.0 * j1( x * lengthRatio)/(x*lengthRatio) )/ areaNorm
    
    V = v**2
    
    return V

def beam_func(rho, nu):

    D = 10.7*100
    d = 0.75*100    
    return obscured_airy_pattern_CASA(rho, D, d, nu)

    
def get_deprojected_beam_arcsec(r_grid, nu, incl, pa=0.0, dx=0.0, dy=0.0, n_theta=360):
    
    i_rad = incl
    pa_rad = pa
    
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    
    R, Theta = np.meshgrid(r_grid, theta, indexing='ij')
    
    x_d = R * np.cos(Theta)
    y_d = R * np.sin(Theta)
    
    x_s = x_d 
    y_s = y_d * np.cos(i_rad)

    
    cos_pa, sin_pa = np.cos(pa_rad), np.sin(pa_rad)
    
    
    ra_off  = x_s * cos_pa - y_s * sin_pa + dx
    dec_off = x_s * sin_pa + y_s * cos_pa + dy

    
    
    rho = np.sqrt(ra_off**2 + dec_off**2)

    
    beam_values = beam_func(rho, nu)
    A_eff = np.mean(beam_values, axis=1)
    
    return A_eff

#f_A = get_deprojected_beam_arcsec(r_GP, beam_func, incl=6.0, pa=6.0, dx=-1.0, dy=2.0, n_theta=360)



#plt.plot(r_GP, f_A)

