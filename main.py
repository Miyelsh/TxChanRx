import signal
import numpy as np
import matplotlib.pyplot as plt

NUM_SYMBOLS = 2**18
BITS_PER_SYMBOL = 2
NUM_BITS = NUM_SYMBOLS*BITS_PER_SYMBOL
SYMBOL_POWER = 1.0
NOISE_POWER = 0.1

def generate_bits(size):
    return np.random.randint(2, size=size)

def convert_bits_to_symbols(bits, bits_per_symbol, average_power):
    num_bits = len(bits)

    if (num_bits % bits_per_symbol != 0):
        print(f"ERROR: bits_per_symbol does not divide into num_bits!")
        exit(1)

    symbols = np.zeros(num_bits//bits_per_symbol, dtype=complex)

    if bits_per_symbol == 2: # QPSK)
        for bit_idx in np.arange(0, num_bits, 2):
            symbol_idx = bit_idx//2
            symbol = complex(bits[bit_idx], bits[bit_idx+1])
            symbol -= 0.5 + 0.5j
            symbol *= np.sqrt(2)*average_power

            symbols[symbol_idx] = symbol

    else:
        print(f"ERROR: {bits_per_symbol} is not a valid value for bits_per_symbol!")
        exit(1)

    return symbols

def convert_symbols_to_bits(symbols, bits_per_symbol, expected_num_symbols):
    num_symbols = len(symbols)

    bits = np.zeros(expected_num_symbols*bits_per_symbol, dtype=int)

    if bits_per_symbol == 2: # QPSK)
        for symbol_idx in np.arange(expected_num_symbols-num_symbols, num_symbols, 1):
            bit_idx = symbol_idx*2
            symbol = symbols[symbol_idx]
            if (symbol.real > 0.0):
                bits[bit_idx] = 1
            else:
                bits[bit_idx] = 0

            if (symbol.imag > 0.0):
                bits[bit_idx+1] = 1
            else:
                bits[bit_idx+1] = 0

    else:
        print(f"ERROR: {bits_per_symbol} is not a valid value for bits_per_symbol!")
        exit(1)

    return bits

def awgn(size, symbol_power, osnr_db):
    osnr_linear = np.pow(10, osnr_db/10.0)
    noise_power = symbol_power/osnr_linear
    return np.sqrt(noise_power/2)*(np.random.randn(size) + 1j*np.random.randn(size))

def plot_const(symbols, symbols_received):
    figs, axs = plt.subplots(1,2)

    axs[0].grid()
    axs[0].set_aspect("equal", "box")
    axs[0].scatter(symbols.real, symbols.imag)
    axs[0].set_xlim(-2,2)
    axs[0].set_ylim(-2,2)

    axs[1].grid()
    axs[1].set_aspect("equal", "box")
    axs[1].scatter(symbols_received.real, symbols_received.imag)
    axs[1].set_xlim(-2,2)
    axs[1].set_ylim(-2,2)

    plt.tight_layout()

def upsample2x(input):
    # Insert zeros between samples
    input_2x = np.zeros(2*len(input), dtype=complex)
    for sample_idx in range(len(input_2x)):
        input_2x[sample_idx] = input[sample_idx//2]

    # Shift x over by 1
    x      = input_2x[0:-1]
    next_x = input_2x[1:  ]

    # Linear interpolate
    y = (x + next_x)/2
    return y

def downsample2x(input):
    y = input[::2] # capture every other
    return y

def upsample_sps_x(input, sps):
    # Insert zeros between samples
    input_sps_x = np.zeros(sps*len(input), dtype=complex)
    for sample_idx in range(len(input)):
        input_sps_x[sps*sample_idx] = input[sample_idx]

    # Linear interpolate
    filter_kernel = np.ones(sps)
    filter = np.convolve(filter_kernel,filter_kernel)/sps
    # print(f"SPS: {sps}, Filter Power = {np.sum(filter*filter)}")
    y = np.convolve(filter, input_sps_x, mode="same")
    return y

def downsample_sps_x(input, sps):
    y = input[::sps] # capture every other
    return y

def channel_model(symbols, symbol_power, osnr_db, sps):
    symbols_plus_awgn = symbols + awgn(len(symbols), symbol_power, osnr_db)

    return downsample_sps_x(upsample_sps_x(symbols_plus_awgn, sps), sps)

def est_symbol_power(x):
    return np.mean(np.abs(x*np.conj(x)))

def est_symbol_qpsk_esno(symbols, enable_abs_correction):
    symbols_first_quadrant = np.array([ complex(np.abs(symbol.real), np.abs(symbol.imag)) for symbol in symbols ])
    # if enable_abs_correction:
    #     for idx in range(len(symbols_first_quadrant)):
    #         symbol = symbols_first_quadrant[idx]
    #         scale_thresh = 1.66 # 1.66
    #         scale_mult = 1.35 # 1.35
    #         if symbol.real > scale_thresh:
    #             symbols_first_quadrant[idx] = complex(scale_mult*symbol.real, symbol.imag)
    #         if symbol.imag > scale_thresh:
    #             symbols_first_quadrant[idx] = complex(symbol.real, scale_mult*symbol.imag)
    if enable_abs_correction:
        for idx in range(len(symbols_first_quadrant)):
            symbol = symbols_first_quadrant[idx]
            scale_thresh = 1.92 # 1.92
            scale_mult = 1.34 # 1.34
            if np.abs(symbol) > scale_thresh:
                symbols_first_quadrant[idx] = scale_mult*symbol
    symbols_dc_removed = np.array(symbols_first_quadrant) - np.mean(symbols_first_quadrant)
    # plot_const(symbols_first_quadrant, symbols_dc_removed)
    noise_power = est_symbol_power(symbols_dc_removed)
    symbol_power = est_symbol_power(symbols) - noise_power
    # symbol_power = est_symbol_power(symbols) # slightly less accurate
    qpsk_esno_linear = symbol_power/noise_power
    return 10*np.log10(qpsk_esno_linear)

def est_symbol_radial_esno(symbols):
    avg_radius = np.mean( np.abs(symbols) )
    rms        = np.sqrt( np.mean(np.abs(symbols*symbols.conj())) )

    if rms <= avg_radius:
        radial_esno_linear = np.inf
    else:
        # radial_esno_linear = ( 0.5 * avg_radius**2 / (rms**2 - avg_radius**2) ) - 1
        radial_esno_linear = ( 0.5 * avg_radius**2 / (rms**2 - avg_radius**2) ) - 1

    return 10*np.log10(radial_esno_linear)

def est_symbol_tx_rx_esno(symbols_tx,symbols_rx):
    symbols_tx_normalized = symbols_tx#/est_symbol_power(symbols_tx)
    symbols_rx_normalized = symbols_rx#/est_symbol_power(symbols_rx)
    symbols_diff = symbols_tx_normalized - symbols_rx_normalized
    symbol_power = 1.0 # est_symbol_power(symbols_rx_normalized) - est_symbol_power(symbols_diff)
    noise_power = est_symbol_power(symbols_diff)
    esno_linear = symbol_power/noise_power
    return 10*np.log10(esno_linear)

def bit_error_rate(bits_tx, bits_rx):
    return np.mean(np.abs(bits_tx-bits_rx))

def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL) # Make Ctrl-C actually close plots
    plt.rcParams.update({"figure.max_open_warning" : 0}) # Disable max open warning

    bits = generate_bits(NUM_BITS)

    symbols = convert_bits_to_symbols(bits, BITS_PER_SYMBOL, SYMBOL_POWER)
    # h_symbols = convert_bits_to_symbols(bits, BITS_PER_SYMBOL, SYMBOL_POWER)
    # v_symbols = convert_bits_to_symbols(bits, BITS_PER_SYMBOL, SYMBOL_POWER)

    num_symbols = len(symbols)

    # print(symbols)
    # print(symbols_received_resampled_2x)
    # print(bits)
    # print(bits_receieved)
    # print(bit_errors)

    sps = 1
    osnr_db_sweep = np.arange(0, 20, 2)
    # osnr_db_sweep = np.arange(1)
    osnr_linear_sweep = np.pow(10, osnr_db_sweep/10.0)
    noise_power_sweep = SYMBOL_POWER/osnr_linear_sweep
    noise_power_db_sweep = -10*np.log10(noise_power_sweep)
    # print(noise_powers)
    # print(noise_powers_db)
    symbols_received_noise_sweep = [
            channel_model(symbols, SYMBOL_POWER, osnr_db, sps)
            for osnr_db in osnr_db_sweep ]

    bits_received_noise_sweep = [
            convert_symbols_to_bits(x, BITS_PER_SYMBOL, NUM_SYMBOLS)
            for x in symbols_received_noise_sweep ]

    # print("Received Symbol Power")
    # [ print(est_symbol_power(x)) for x in symbols_received_noise_sweep ]

    qpsk_esnos = np.array([ est_symbol_qpsk_esno(x, True) for x in symbols_received_noise_sweep ])
    qpsk_esnos_no_abs_correction = np.array([ est_symbol_qpsk_esno(x, False) for x in symbols_received_noise_sweep ])
    # print("Received Symbol QPSK EsNo")
    # print(qpsk_esnos)

    radial_esnos = np.array([ est_symbol_radial_esno(x) for x in symbols_received_noise_sweep ])

    tx_rx_esnos = [ est_symbol_tx_rx_esno(symbols,x) for x in symbols_received_noise_sweep ]
    # print("Received Symbol TX RX EsNo")
    # print(tx_rx_esnos)

    plt.figure()
    plt.grid()
    plt.title("Channel SNR vs Estimated EsNo")
    plt.plot(osnr_db_sweep, noise_power_db_sweep)
    plt.plot(osnr_db_sweep, qpsk_esnos)
    plt.plot(osnr_db_sweep, qpsk_esnos_no_abs_correction)
    plt.plot(osnr_db_sweep, radial_esnos)
    plt.plot(osnr_db_sweep, tx_rx_esnos)
    plt.legend(["True EsNo", "QPSK EsNo", "QPSK EsNo (no abs correction)", "Radial EsNo", "TX RX EsNo"])

    plt.figure()
    plt.grid()
    plt.title("Channel SNR vs Estimated EsNo Error")
    plt.plot(osnr_db_sweep, noise_power_db_sweep - osnr_db_sweep)
    plt.plot(osnr_db_sweep, qpsk_esnos - osnr_db_sweep)
    plt.plot(osnr_db_sweep, qpsk_esnos_no_abs_correction - osnr_db_sweep)
    plt.plot(osnr_db_sweep, radial_esnos - osnr_db_sweep)
    plt.plot(osnr_db_sweep, tx_rx_esnos - osnr_db_sweep)
    plt.legend(["True EsNo", "QPSK EsNo", "QPSK EsNo (no abs correction)", "Radial EsNo", "TX RX EsNo"])

    bit_error_rates_noise_sweep = [ bit_error_rate(bits,x) for x in bits_received_noise_sweep ]
    # print("Error Rates")

    qpsk_esnos_rms_error = np.sqrt(np.mean((qpsk_esnos-osnr_db_sweep)*(qpsk_esnos-osnr_db_sweep)))
    qpsk_esnos_abs_error = np.mean(np.abs(qpsk_esnos-osnr_db_sweep))

    print(qpsk_esnos_rms_error)
    print(qpsk_esnos_abs_error)

    plt.figure()
    plt.grid()
    plt.semilogy(noise_power_db_sweep, bit_error_rates_noise_sweep)

    # plot_const(symbols, symbols_received_noise_sweep[0])

    plt.show()

if __name__ == "__main__":
    main()
