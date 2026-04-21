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

def est_symbol_tx_rx_esno(symbols_tx,symbols_rx,normalize_symbols):
    symbols_diff = symbols_tx - symbols_rx
    noise_power = est_symbol_power(symbols_diff)
    symbol_power = est_symbol_power(symbols_rx) - noise_power

    if normalize_symbols == True:
        symbols_tx_scale_factor = np.mean(np.abs(symbols_tx))
        symbols_rx_scale_factor = np.mean(np.abs(symbols_rx))
        # symbols_tx_scale_factor = np.sqrt(np.mean(np.abs(symbols_tx*symbols_tx.conj())))
        # symbols_rx_scale_factor = np.sqrt(np.mean(np.abs(symbols_rx*symbols_rx.conj())))
        # print(f"symbols_tx_scale_factor = {symbols_tx_scale_factor}")
        # print(f"symbols_rx_scale_factor = {symbols_rx_scale_factor}")
        symbols_tx_normalized = symbols_tx/symbols_tx_scale_factor
        symbols_rx_normalized = symbols_rx/symbols_rx_scale_factor
        # plot_const(symbols_tx_normalized, symbols_rx_normalized)
        # plt.show()
        symbols_diff = symbols_tx_normalized - symbols_rx_normalized
        # plot_const(symbols_rx_normalized, symbols_diff)
        noise_power = est_symbol_power(symbols_diff)
        symbol_power = est_symbol_power(symbols_rx_normalized) - noise_power
        # print(f"noise_power  = {noise_power}")
        # print(f"symbol_power = {symbol_power}")

    # print(f"symbol_power = {symbol_power}")
    esno_linear = symbol_power/noise_power
    return convert_power_to_db(esno_linear)

def est_symbol_power(x):
    return np.mean(np.abs(x*np.conj(x)))

def convert_power_to_db(x):
    return 10*np.log10(np.abs(x))
