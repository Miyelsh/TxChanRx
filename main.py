import signal
import numpy as np
import matplotlib.pyplot as plt

np.random.seed = 1234

# Constants
NUM_SYMBOLS = 2**14
BITS_PER_SYMBOL = 2
NUM_BITS = NUM_SYMBOLS*BITS_PER_SYMBOL
SYMBOL_POWER = 1.0
CHANNEL_SPS = 1
CHANNEL_SNR_DB = 10
FILTER_NUM_COEFS=512

signal.signal(signal.SIGINT, signal.SIG_DFL) # Make Ctrl-C actually close plots
plt.rcParams.update({"figure.max_open_warning" : 0}) # Disable max open warning

def generate_bits(size):
    return np.random.randint(2, size=size)

def convert_bits_to_symbols(bits, bits_per_symbol, average_power):
    num_bits = len(bits)

    if (num_bits % bits_per_symbol != 0):
        print(f"ERROR: bits_per_symbol does not divide into num_bits!")
        exit(1)

    symbols = np.zeros(num_bits//bits_per_symbol, dtype=complex)

    if bits_per_symbol == 2: # QPSK
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

def add_awgn(symbols, symbol_power, osnr_db):
    osnr_linear = np.pow(10, osnr_db/10.0)
    noise_power = symbol_power/osnr_linear
    
    # print(f"symbol_power = {symbol_power}")
    
    num_symbols = len(symbols)
    awgn_noise = np.sqrt(noise_power/2)*(np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols))
    return symbols + awgn_noise

def plot_const(symbols, symbols_received, title=""):
    figs, axs = plt.subplots(1,2)

    plt.suptitle(title)

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
    y = np.convolve(filter, input_sps_x, mode="same")
    return y

def downsample_sps_x(input, sps):
    y = input[::sps] # capture every other
    return y

def channel_model(h_symbols_tx, v_symbols_tx, h2h_filter, h2v_filter, v2h_filter, v2v_filter, symbol_power, osnr_db, sps):
    h_symbols_filtered  = np.convolve(h_symbols_tx, h2h_filter, mode="same")
    h_symbols_filtered += np.convolve(v_symbols_tx, v2h_filter, mode="same")
    v_symbols_filtered  = np.convolve(v_symbols_tx, v2v_filter, mode="same")
    v_symbols_filtered += np.convolve(h_symbols_tx, h2v_filter, mode="same")

    average_filtered_symbols_power = est_symbol_power(h_symbols_filtered) + est_symbol_power(v_symbols_filtered)

    h_symbols_plus_awgn = add_awgn(h_symbols_filtered, symbol_power, osnr_db)
    v_symbols_plus_awgn = add_awgn(v_symbols_filtered, symbol_power, osnr_db)

    h_symbols_resampled = downsample_sps_x(upsample_sps_x(h_symbols_plus_awgn, sps), sps)
    v_symbols_resampled = downsample_sps_x(upsample_sps_x(v_symbols_plus_awgn, sps), sps)
    h_symbols_resampled = h_symbols_plus_awgn
    v_symbols_resampled = v_symbols_plus_awgn

    return (h_symbols_resampled, v_symbols_resampled)

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

def est_symbol_tx_rx_esno(symbols_tx,symbols_rx,normalize_symbols):
    symbols_diff = symbols_tx - symbols_rx
    symbol_power = est_symbol_power(symbols_rx) - est_symbol_power(symbols_diff)

    if normalize_symbols == True:
        symbols_tx_scale_factor = np.mean(np.abs(symbols_tx))
        symbols_rx_scale_factor = np.mean(np.abs(symbols_rx))
        # symbols_tx_scale_factor = np.sqrt(np.mean(np.abs(symbols_tx*symbols_tx.conj())))
        # symbols_rx_scale_factor = np.sqrt(np.mean(np.abs(symbols_rx*symbols_rx.conj())))
        # print(f"symbols_tx_scale_factor = {symbols_tx_scale_factor}")
        print(f"symbols_rx_scale_factor = {symbols_rx_scale_factor}")
        symbols_tx_normalized = symbols_tx/symbols_tx_scale_factor
        symbols_rx_normalized = symbols_rx/symbols_rx_scale_factor
        # plot_const(symbols_tx_normalized, symbols_rx_normalized)
        # plt.show()
        symbols_diff = symbols_tx_normalized - symbols_rx_normalized
        # plot_const(symbols_rx_normalized, symbols_diff)
        noise_power = est_symbol_power(symbols_diff)
        symbol_power = est_symbol_power(symbols_rx_normalized) - noise_power
        print(f"noise_power  = {noise_power}")
        print(f"symbol_power = {symbol_power}")

    # print(f"symbol_power = {symbol_power}")
    noise_power = est_symbol_power(symbols_diff)
    esno_linear = symbol_power/noise_power
    return 10*np.log10(esno_linear)

def est_phase(x):
    # returns estimated angle in range (-pi/4,pi/4)
    x_squared = x*x
    x_fourth = x_squared*x_squared
    neg_mean = -np.mean(x_fourth)
    est_angle = np.angle(neg_mean)/4
    return est_angle

def bit_error_rate(bits_tx, bits_rx):
    return np.mean(np.abs(bits_tx-bits_rx))

def calculate_filter_rms(h2h_filter, h2v_filter, v2h_filter, v2v_filter):
    filter_power  = np.sum(h2h_filter*h2h_filter)
    filter_power += np.sum(h2v_filter*h2v_filter)
    filter_power += np.sum(v2h_filter*v2h_filter)
    filter_power += np.sum(v2v_filter*v2v_filter)

    filter_rms = np.sqrt(filter_power/2)
    return filter_rms

def invert_filters(h2h, h2v, v2h, v2v):
    N = len(h2h)
    H2H = np.fft.fft(h2h)
    H2V = np.fft.fft(h2v)
    V2H = np.fft.fft(v2h)
    V2V = np.fft.fft(v2v)

    H2H_inv = np.zeros(N, dtype=complex)
    H2V_inv = np.zeros(N, dtype=complex)
    V2H_inv = np.zeros(N, dtype=complex)
    V2V_inv = np.zeros(N, dtype=complex)

    for freq_idx in range(N):
        a = H2H[freq_idx]
        b = V2H[freq_idx]
        c = H2V[freq_idx]
        d = V2V[freq_idx]

        det = a*d - b*c
        
        H2H_inv[freq_idx] =  d/det
        H2V_inv[freq_idx] = -c/det
        V2H_inv[freq_idx] = -b/det
        V2V_inv[freq_idx] =  a/det

    delay_exp = np.exp(2*2j*np.pi*np.arange(N)/(N))
    # print(delay_exp)
    H2H_inv *= delay_exp
    H2V_inv *= delay_exp
    V2H_inv *= delay_exp
    V2V_inv *= delay_exp

    plt.figure()
    plt.title("H2H filter comparison")
    plt.plot(np.log10(np.abs(np.fft.fftshift(H2H))))
    plt.plot(np.log10(np.abs(np.fft.fftshift(H2H_inv))))
    plt.plot(np.log10(np.abs(np.fft.fftshift((H2H*H2H_inv)))))
    plt.legend(["H2H", "H2H inverted", "H2H*H2H inverted"])

    h2h_inv = np.fft.ifft(H2H_inv)
    h2v_inv = np.fft.ifft(H2V_inv)
    v2h_inv = np.fft.ifft(V2H_inv)
    v2v_inv = np.fft.ifft(V2V_inv)

    print(f"h2h = {np.abs(h2h)}")
    print(f"H2H = {np.abs(H2H)}")
    print(f"H2H_inv = {np.abs(H2H_inv)}")
    print(f"h2h_inv = {np.abs(h2h_inv)}")

    return (h2h_inv, h2v_inv, v2h_inv, v2v_inv)

def equalize_rx_symbols(h_symbols, v_symbols, h2h_filter, h2v_filter, v2h_filter, v2v_filter):
    (h2h_filter_inverted,
     h2v_filter_inverted,
     v2h_filter_inverted,
     v2v_filter_inverted) = invert_filters( h2h_filter
                                          , h2v_filter
                                          , v2h_filter
                                          , v2v_filter )

    plt.figure()
    fig,axs = plt.subplots(2,2)
    axs[0][0].set_title("h2h_filter")
    axs[0][0].plot(h2h_filter.real)
    axs[0][0].plot(h2h_filter.imag)
    axs[0][1].set_title("v2h_filter")
    axs[0][1].plot(v2h_filter.real)
    axs[0][1].plot(v2h_filter.imag)
    axs[1][0].set_title("h2v_filter")
    axs[1][0].plot(h2v_filter.real)
    axs[1][0].plot(h2v_filter.imag)
    axs[1][1].set_title("v2v_filter")
    axs[1][1].plot(v2v_filter.real)
    axs[1][1].plot(v2v_filter.imag)

    plt.figure()
    fig,axs = plt.subplots(2,2)
    axs[0][0].set_title("h2h_filter_inverted")
    axs[0][0].plot(h2h_filter_inverted.real)
    axs[0][0].plot(h2h_filter_inverted.imag)
    axs[0][1].set_title("v2h_filter_inverted")
    axs[0][1].plot(v2h_filter_inverted.real)
    axs[0][1].plot(v2h_filter_inverted.imag)
    axs[1][0].set_title("h2v_filter_inverted")
    axs[1][0].plot(h2v_filter_inverted.real)
    axs[1][0].plot(h2v_filter_inverted.imag)
    axs[1][1].set_title("v2v_filter_inverted")
    axs[1][1].plot(v2v_filter_inverted.real)
    axs[1][1].plot(v2v_filter_inverted.imag)

    plt.figure()
    plt.title("h2h_filter and h2h_filter_inverted")
    h2h_filter_convolved = np.convolve(h2h_filter, h2h_filter_inverted)
    plt.plot(h2h_filter_convolved.real)
    plt.plot(h2h_filter_convolved.imag)
    # plt.plot(np.convolve(h2v_filter, h2v_filter_inverted))
    # plt.plot(np.convolve(v2h_filter, v2h_filter_inverted))
    # plt.plot(np.convolve(v2v_filter, v2v_filter_inverted))
    # plt.show()np.log10(np.abs(np.fft.fftshift(np.fft.fft(np.pad(h2h_filter,16384))))))

    h2h_filter_spectrum = np.log10(np.abs(np.fft.fftshift(np.fft.fft(zero_pad(h2h_filter,16384)))))
    h2h_filter_inverted_spectrum = np.log10(np.abs(np.fft.fftshift(np.fft.fft(zero_pad(h2h_filter_inverted,16384)))))
    h2h_filter_convolved_spectrum = np.log10(np.abs(np.fft.fftshift(np.fft.fft(zero_pad(h2h_filter_convolved,16384)))))

    print(len(h2h_filter_spectrum))
    print(len(h2h_filter_inverted_spectrum))
    print(len(h2h_filter_convolved_spectrum))

    fig, axs = plt.subplots(3,1)
    plt.title("Channel and Inverted Filter Spectrum")
    axs[0].plot(h2h_filter_spectrum)
    axs[1].plot(h2h_filter_inverted_spectrum)
    axs[2].plot(h2h_filter_convolved_spectrum)
    axs[2].plot(h2h_filter_spectrum + h2h_filter_inverted_spectrum)
    plt.legend(["H2H", "H2H inverted", "H2H convolved with H2H inverted"])

    h_symbols_inverted = np.zeros([len(h_symbols), len(h_symbols[0])], dtype=complex)
    v_symbols_inverted = np.zeros([len(v_symbols), len(v_symbols[0])], dtype=complex)

    for snr_idx in range(len(h_symbols)):
        h_symbols_inverted[snr_idx]  = np.convolve(h_symbols[snr_idx], h2h_filter_inverted, mode="same")
        h_symbols_inverted[snr_idx] += np.convolve(v_symbols[snr_idx], v2h_filter_inverted, mode="same")
        v_symbols_inverted[snr_idx]  = np.convolve(v_symbols[snr_idx], v2v_filter_inverted, mode="same")
        v_symbols_inverted[snr_idx] += np.convolve(h_symbols[snr_idx], h2v_filter_inverted, mode="same")

    return h_symbols_inverted, v_symbols_inverted

def zero_pad(array,length_after_padding,equal_left_right=False):
    if equal_left_right == True:
        return np.pad(array, (length_after_padding//2-len(array)//2,length_after_padding//2-len(array)//2))
    else:
        return np.pad(array, (0,length_after_padding-len(array)))

def test_snr_sweep(h_bits_tx, v_bits_tx, h_symbols_tx, v_symbols_tx):
    snr_db_sweep = np.arange(0, 60, 5)
    # snr_db_sweep = [100]

    h_symbols_rx_noise_sweep = np.zeros([len(snr_db_sweep), len(h_symbols_tx)], dtype=complex)
    v_symbols_rx_noise_sweep = np.zeros([len(snr_db_sweep), len(v_symbols_tx)], dtype=complex)

    single_pol_filter = np.array([0,0,0,1,0,0,0,0],dtype=complex)
    h2h_filter = single_pol_filter # np.array([0,0,0,1,0,0,0,0])
    h2v_filter = np.array([0,0,0,0,0,0,0,0])
    v2h_filter = np.array([0,0,0,0,0,0,0,0])
    v2v_filter = single_pol_filter # np.array([0,0,0,1,0,0,0,0])

    filter_noise_power = 0.0 # 0.2
    filter_rand_scale = np.sqrt(filter_noise_power/2)
    h2h_filter = h2h_filter + filter_rand_scale*(    np.random.randn(len(h2h_filter))
                                                + 1j*np.random.randn(len(h2h_filter)))
    h2v_filter = h2v_filter + filter_rand_scale*(    np.random.randn(len(h2v_filter))
                                                + 1j*np.random.randn(len(h2v_filter)))
    v2h_filter = v2h_filter + filter_rand_scale*(    np.random.randn(len(v2h_filter))
                                                + 1j*np.random.randn(len(v2h_filter)))
    v2v_filter = v2v_filter + filter_rand_scale*(    np.random.randn(len(v2v_filter))
                                                + 1j*np.random.randn(len(v2v_filter)))

    h2h_filter = zero_pad(h2h_filter, FILTER_NUM_COEFS, True)
    h2v_filter = zero_pad(h2v_filter, FILTER_NUM_COEFS, True)
    v2h_filter = zero_pad(v2h_filter, FILTER_NUM_COEFS, True)
    v2v_filter = zero_pad(v2v_filter, FILTER_NUM_COEFS, True)

    # Normalize filters
    filter_rms = calculate_filter_rms( h2h_filter, h2v_filter
                                     , v2h_filter, v2v_filter)
    h2h_filter_normalized = h2h_filter/filter_rms
    h2v_filter_normalized = h2v_filter/filter_rms
    v2h_filter_normalized = v2h_filter/filter_rms
    v2v_filter_normalized = v2v_filter/filter_rms

    # print(f"filter_rms = {filter_rms}")
    # print(h2h_filter_normalized)
    # print(h2v_filter_normalized)
    # print(v2h_filter_normalized)
    # print(v2v_filter_normalized)

    # all_pass_filter = np.array([1])
    for snr_idx in range(len(snr_db_sweep)):
        snr_db = snr_db_sweep[snr_idx]
        (h_symbols_rx_noise_sweep[snr_idx], v_symbols_rx_noise_sweep[snr_idx]) = channel_model(h_symbols_tx, v_symbols_tx, h2h_filter_normalized, h2v_filter_normalized, v2h_filter_normalized, v2v_filter_normalized, SYMBOL_POWER, snr_db, CHANNEL_SPS)

    (h_symbols_inverted_rx_noise_sweep,
     v_symbols_inverted_rx_noise_sweep) = \
             equalize_rx_symbols( h_symbols_rx_noise_sweep
                                , v_symbols_rx_noise_sweep
                                , h2h_filter_normalized, h2v_filter_normalized
                                , v2h_filter_normalized, v2v_filter_normalized)

    # print(h_symbols_tx)
    # print(h_symbols_rx_noise_sweep[-1])
    # print(h_symbols_inverted_rx_noise_sweep[-1][-32:])

    plot_const(h_symbols_tx, h_symbols_rx_noise_sweep[-1][:-32], "Constellation before equalization")
    plot_const(h_symbols_tx, h_symbols_inverted_rx_noise_sweep[-1][:-32], "Constellation after equalization")

    fig, axs = plt.subplots(3,1)
    plt.title("TX and RX Spectrum")
    axs[0].plot(np.log10(np.abs(np.fft.fftshift(np.fft.fft(h_symbols_tx)))))
    axs[1].plot(np.log10(np.abs(np.fft.fftshift(np.fft.fft(h_symbols_rx_noise_sweep[-1])))))
    axs[2].plot(np.log10(np.abs(np.fft.fftshift(np.fft.fft(h_symbols_inverted_rx_noise_sweep[-1])))))

    fig, axs = plt.subplots(3,1)
    plt.title("TX and RX Symbols")
    axs[0].plot(h_symbols_tx[:20])
    axs[1].plot(h_symbols_rx_noise_sweep[-1][:20])
    axs[2].plot(h_symbols_inverted_rx_noise_sweep[-1][:20])

    # plot_sweep(h_bits_tx, v_bits_tx, h_symbols_tx[:-FILTER_NUM_COEFS], v_symbols_tx[:-FILTER_NUM_COEFS], h_symbols_rx_noise_sweep[:,:-FILTER_NUM_COEFS], v_symbols_rx_noise_sweep[:,:-FILTER_NUM_COEFS], snr_db_sweep, "Before Equalization: Channel SNR (dB)")

    plot_sweep(h_bits_tx, v_bits_tx, h_symbols_tx[:-FILTER_NUM_COEFS], v_symbols_tx[:-FILTER_NUM_COEFS], h_symbols_inverted_rx_noise_sweep[:,:-FILTER_NUM_COEFS], v_symbols_inverted_rx_noise_sweep[:,:-FILTER_NUM_COEFS], snr_db_sweep, "After Equalization: Channel SNR (dB)")

def test_phase_sweep(h_bits_tx, v_bits_tx, h_symbols_tx, v_symbols_tx,):
    phase_sweep = np.arange(0, 360, 10)

    h_symbols_rx_phase_sweep = np.zeros([len(phase_sweep), len(h_symbols_tx)], dtype=complex)
    v_symbols_rx_phase_sweep = np.zeros([len(phase_sweep), len(v_symbols_tx)], dtype=complex)
    h2h_filter = np.array([0,0,0,1,0,0,0,0])
    h2v_filter = np.array([0,0,0,0,0,0,0,0])
    v2h_filter = np.array([0,0,0,0,0,0,0,0])
    v2v_filter = np.array([0,0,0,1,0,0,0,0])
    for phase_idx in range(len(phase_sweep)):
        phase = phase_sweep[phase_idx]
        (h_symbols_rx_phase_sweep[phase_idx], v_symbols_rx_phase_sweep[phase_idx]) = channel_model(h_symbols_tx, v_symbols_tx, h2h_filter, h2v_filter, v2h_filter, v2v_filter, SYMBOL_POWER, CHANNEL_SNR_DB, CHANNEL_SPS)
        h_symbols_rx_phase_sweep[phase_idx] *= np.exp(1j*np.pi*phase/180)
        v_symbols_rx_phase_sweep[phase_idx] *= np.exp(1j*np.pi*phase/180)

    h_est_phases = np.array([ est_phase(symbols) for symbols in h_symbols_rx_phase_sweep ])
    print("Estimated H-Pol Phases")
    print(h_est_phases*180/np.pi)
    print("H-Pol Phase Error")
    print(h_est_phases*180/np.pi - phase_sweep)

    plot_sweep(h_bits_tx, v_bits_tx, h_symbols_tx, v_symbols_tx, h_symbols_rx_phase_sweep, v_symbols_rx_phase_sweep, phase_sweep, "Phase Offset (degrees)")


def plot_sweep(h_bits_tx, v_bits_tx, h_symbols_tx, v_symbols_tx, h_symbols_rx_sweep, v_symbols_rx_sweep, sweep_params, sweep_param_name):

    h_bits_received_noise_sweep = [
            convert_symbols_to_bits(x, BITS_PER_SYMBOL, NUM_SYMBOLS)
            for x in h_symbols_rx_sweep ]
    v_bits_received_noise_sweep = [
            convert_symbols_to_bits(x, BITS_PER_SYMBOL, NUM_SYMBOLS)
            for x in v_symbols_rx_sweep ]

    h_qpsk_esnos = np.array([ est_symbol_qpsk_esno(x, True) for x in h_symbols_rx_sweep ])
    v_qpsk_esnos = np.array([ est_symbol_qpsk_esno(x, True) for x in v_symbols_rx_sweep ])
    average_qpsk_esnos = (np.array(h_qpsk_esnos) + np.array(v_qpsk_esnos))/2.0

    h_qpsk_esnos_no_abs_correction = np.array([ est_symbol_qpsk_esno(x, False) for x in h_symbols_rx_sweep ])
    v_qpsk_esnos_no_abs_correction = np.array([ est_symbol_qpsk_esno(x, False) for x in v_symbols_rx_sweep ])
    average_qpsk_esnos_no_abs_correction = (np.array(h_qpsk_esnos_no_abs_correction) + np.array(v_qpsk_esnos_no_abs_correction))/2.0

    h_radial_esnos = np.array([ est_symbol_radial_esno(x) for x in h_symbols_rx_sweep ])
    v_radial_esnos = np.array([ est_symbol_radial_esno(x) for x in v_symbols_rx_sweep ])
    average_radial_esnos = (np.array(h_radial_esnos) + np.array(v_radial_esnos))/2.0

    h_tx_rx_esnos = [ est_symbol_tx_rx_esno(h_symbols_tx,x,normalize_symbols=True) for x in h_symbols_rx_sweep ]
    v_tx_rx_esnos = [ est_symbol_tx_rx_esno(v_symbols_tx,x,normalize_symbols=True) for x in v_symbols_rx_sweep ]
    average_tx_rx_esnos = (np.array(h_tx_rx_esnos) + np.array(v_tx_rx_esnos))/2.0

    # plt.figure()
    # plt.grid()
    # plt.title(f"{sweep_param_name} vs Estimated H-Pol EsNo (dB)")
    # # plt.plot(sweep_params, sweep_params)
    # plt.plot(sweep_params, h_qpsk_esnos)
    # plt.plot(sweep_params, h_qpsk_esnos_no_abs_correction)
    # plt.plot(sweep_params, h_radial_esnos)
    # plt.plot(sweep_params, h_tx_rx_esnos)
    # plt.xlabel(sweep_param_name)
    # plt.legend(["QPSK EsNo", "QPSK EsNo (no abs correction)", "Radial EsNo", "TX RX EsNo"])
    # # plt.legend(["True EsNo", "QPSK EsNo", "QPSK EsNo (no abs correction)", "Radial EsNo", "TX RX EsNo"])

    # plt.figure()
    # plt.grid()
    # plt.title(f"{sweep_param_name} vs Estimated V-Pol EsNo (dB)")
    # # plt.plot(sweep_params, sweep_params)
    # plt.plot(sweep_params, v_qpsk_esnos)
    # plt.plot(sweep_params, v_qpsk_esnos_no_abs_correction)
    # plt.plot(sweep_params, v_radial_esnos)
    # plt.plot(sweep_params, v_tx_rx_esnos)
    # plt.xlabel(sweep_param_name)
    # plt.legend(["QPSK EsNo", "QPSK EsNo (no abs correction)", "Radial EsNo", "TX RX EsNo"])

    plt.figure()
    plt.grid()
    plt.title(f"{sweep_param_name} vs Estimated EsNo (dB)")
    plt.plot(sweep_params, sweep_params)
    plt.plot(sweep_params, average_qpsk_esnos)
    plt.plot(sweep_params, average_qpsk_esnos_no_abs_correction)
    plt.plot(sweep_params, average_radial_esnos)
    plt.plot(sweep_params, average_tx_rx_esnos)
    plt.xlabel(sweep_param_name)
    plt.legend(["Channel SNR", "QPSK EsNo (no abs correction)", "Radial EsNo", "TX RX EsNo"])
    # plt.legend(["QPSK EsNo", "QPSK EsNo (no abs correction)", "Radial EsNo", "TX RX EsNo"])

    # plt.figure()
    # plt.grid()
    # plt.title(f"{sweep_param_name} vs Estimated EsNo - TX/RX EsNo (dB)")
    # plt.plot(sweep_params, average_qpsk_esnos - average_tx_rx_esnos)
    # plt.plot(sweep_params, average_qpsk_esnos_no_abs_correction - average_tx_rx_esnos)
    # plt.plot(sweep_params, average_radial_esnos - average_tx_rx_esnos)
    # plt.xlabel(sweep_param_name)
    # plt.legend(["QPSK EsNo", "QPSK EsNo (no abs correction)", "Radial EsNo", "TX/RX EsNo"])

    h_bit_error_rates = [ bit_error_rate(h_bits_tx,x) for x in h_bits_received_noise_sweep ]
    v_bit_error_rates = [ bit_error_rate(v_bits_tx,x) for x in v_bits_received_noise_sweep ]
    average_bit_error_rates = (np.array(h_bit_error_rates) + np.array(v_bit_error_rates))/2.0

    # plt.figure()
    # plt.grid()
    # plt.title(f"{sweep_param_name} vs Bit Error Rate")
    # plt.semilogy(sweep_params, h_bit_error_rates)
    # plt.semilogy(sweep_params, v_bit_error_rates)
    # plt.semilogy(sweep_params, average_bit_error_rates)
    # plt.xlabel(sweep_param_name)
    # plt.legend(["H-Pol Bit Error Rate", "V-Pol Bit Error Rate", "Average Bit Error Rate"])

def main():
    h_bits_tx    = generate_bits(NUM_BITS)
    v_bits_tx    = generate_bits(NUM_BITS)
    h_symbols_tx = convert_bits_to_symbols(h_bits_tx, BITS_PER_SYMBOL, SYMBOL_POWER)
    v_symbols_tx = convert_bits_to_symbols(v_bits_tx, BITS_PER_SYMBOL, SYMBOL_POWER)

    test_snr_sweep(h_bits_tx, v_bits_tx, h_symbols_tx, v_symbols_tx)

    # test_phase_sweep(h_bits_tx, v_bits_tx, h_symbols_tx, v_symbols_tx)

    plt.show()

if __name__ == "__main__":
    main()
