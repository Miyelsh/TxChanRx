import numpy as np

def convert_linear_to_db(x):
    return 20*np.log10(np.abs(x))

def est_symbol_radial_esno(symbols):
    avg_radius = np.mean( np.abs(symbols) )
    rms        = np.sqrt( np.mean(np.abs(symbols*symbols.conj())) )

    if rms <= avg_radius:
        radial_esno_linear = np.inf
    else:
        # radial_esno_linear = ( 0.5 * avg_radius**2 / (rms**2 - avg_radius**2) ) - 1
        radial_esno_linear = ( 0.5 * avg_radius**2 / (rms**2 - avg_radius**2) ) - 1

    return convert_power_to_db(radial_esno_linear)

def est_symbol_power(x):
    return np.mean(np.abs(x*np.conj(x)))

def convert_power_to_db(x):
    return 10*np.log10(np.abs(x))
