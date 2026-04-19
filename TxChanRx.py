import signal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import dpae
import helper_functions

matplotlib.rcParams.update({"axes.grid" : True})

# Constants
# QPSK
# BITS_PER_SYMBOL = 2
# SYMBOL_POWER = 2.0 # +-1 with QPSK

# 16QAM
BITS_PER_SYMBOL = 4
SYMBOL_POWER = 5.0/(2*np.sqrt(5)) # +-1 with 16QAM

CHANNEL_SPS = 1
CHANNEL_SNR_DB = 10

signal.signal(signal.SIGINT, signal.SIG_DFL) # Make Ctrl-C actually close plots
plt.rcParams.update({"figure.max_open_warning" : 0}) # Disable max open warning
plt.rcParams.update({"figure.figsize" : [8,6]})

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
            symbol *= np.sqrt(2*average_power)

            symbols[symbol_idx] = symbol
    elif bits_per_symbol == 4: # 16QAM
        for bit_idx in np.arange(0, num_bits, 4):
            symbol_idx = bit_idx//4
            symbol_real = 0
            symbol_imag = 0
            # Pick real part from first 2 bits
            if bits[bit_idx+0] == 1:
                symbol_real = 2
            if bits[bit_idx+1] == 1:
                symbol_real += 1
            # Pick imag part from last 2 bits
            if bits[bit_idx+2] == 1:
                symbol_imag = 2
            if bits[bit_idx+3] == 1:
                symbol_imag += 1

            # Scale so power is 2
            symbol = complex(symbol_real,symbol_imag)
            symbol -= (1.5 + 1.5j)
            symbol *= np.sqrt(average_power*2/5)

            symbols[symbol_idx] = symbol

        # For symbols to bits
        # Look at I sample. Assume power is 2, so decision regions are made to be at midpoints. Eyeball it.
        # Decide on first two bits from region in I
        # Decide on second two bits from region in Q
        # Voila
        # Phase is important for this. DPAE dececision needs to turn 4 samples on each axis to a single one.
        # Square or absolute value of QPSK: parabola
        # Square or absolute value of 16QAM: Two points.
        # Shift by mean, do square again. Should be collapesed to a single point.
        # Try this error function with real data. Symbol to bits don't need to be implemented for this anyway.

    else:
        print(f"ERROR: {bits_per_symbol} is not a valid value for bits_per_symbol!")
        exit(1)

    # print(f"{helper_functions.est_symbol_power(symbols)}")

    return symbols

def convert_symbols_to_bits(symbols, bits_per_symbol, expected_num_symbols):
    num_symbols = len(symbols)

    bits = np.zeros(expected_num_symbols*bits_per_symbol, dtype=int)

    if bits_per_symbol == 2: # QPSK
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

def add_awgn(symbols, symbol_power, snr_db):
    snr_linear = np.power(10, snr_db/10.0)
    noise_power = symbol_power/snr_linear
    
    # print(f"symbol_power = {symbol_power}")
    
    num_symbols = len(symbols)
    awgn_noise = np.sqrt(noise_power/2)*(np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols))
    return symbols + awgn_noise

def plot_const(symbols_0, symbols_1, title=""):
    figs, axs = plt.subplots(2,2)

    plt.suptitle(title)

    axs[0][0].set_aspect("equal", "box")
    axs[0][0].scatter(symbols_0.real, symbols_0.imag)
    axs[0][0].set_xlim(-4,4)
    axs[0][0].set_ylim(-4,4)

    axs[0][1].set_aspect("equal", "box")
    axs[0][1].scatter(symbols_1.real, symbols_1.imag)
    axs[0][1].set_xlim(-4,4)
    axs[0][1].set_ylim(-4,4)

    axs[1][0].hist(symbols_0.real, bins=256 )
    axs[1][0].hist(symbols_0.imag, bins=256 )
    axs[1][0].set_xlim(-4,4)

    axs[1][1].hist(symbols_1.real, bins=256 )
    axs[1][1].hist(symbols_1.imag, bins=256 )
    axs[1][1].set_xlim(-4,4)

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

def channel_model(h_symbols_tx, v_symbols_tx, h2h_filter, h2v_filter, v2h_filter, v2v_filter, symbol_power, snr_db, sps):
    h_symbols_filtered  = np.convolve(h_symbols_tx, h2h_filter, mode="same")
    h_symbols_filtered += np.convolve(v_symbols_tx, v2h_filter, mode="same")
    v_symbols_filtered  = np.convolve(v_symbols_tx, v2v_filter, mode="same")
    v_symbols_filtered += np.convolve(h_symbols_tx, h2v_filter, mode="same")

    average_filtered_symbols_power = helper_functions.est_symbol_power(h_symbols_filtered) + helper_functions.est_symbol_power(v_symbols_filtered)

    h_symbols_plus_awgn = add_awgn(h_symbols_filtered, symbol_power, snr_db)
    v_symbols_plus_awgn = add_awgn(v_symbols_filtered, symbol_power, snr_db)

    h_symbols_resampled = downsample_sps_x(upsample_sps_x(h_symbols_plus_awgn, sps), sps)
    v_symbols_resampled = downsample_sps_x(upsample_sps_x(v_symbols_plus_awgn, sps), sps)
    h_symbols_resampled = h_symbols_plus_awgn
    v_symbols_resampled = v_symbols_plus_awgn

    return (h_symbols_resampled, v_symbols_resampled)

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
            scale_thresh = np.sqrt(SYMBOL_POWER)*1.92 # 1.92
            scale_mult = 1.34 # 1.34
            if np.abs(symbol) > scale_thresh:
                symbols_first_quadrant[idx] = scale_mult*symbol
    symbols_dc_removed = np.array(symbols_first_quadrant) - np.mean(symbols_first_quadrant)
    # plot_const(symbols, symbols_first_quadrant)
    # plot_const(symbols_first_quadrant, symbols_dc_removed)
    noise_power = helper_functions.est_symbol_power(symbols_dc_removed)
    symbol_power = helper_functions.est_symbol_power(symbols) - noise_power
    # symbol_power = helper_functions.est_symbol_power(symbols) # slightly less accurate
    qpsk_esno_linear = symbol_power/noise_power
    return helper_functions.convert_power_to_db(qpsk_esno_linear)

def est_symbol_tx_rx_esno(symbols_tx,symbols_rx,normalize_symbols):
    symbols_diff = symbols_tx - symbols_rx
    noise_power = helper_functions.est_symbol_power(symbols_diff)
    symbol_power = helper_functions.est_symbol_power(symbols_rx) - noise_power

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
        noise_power = helper_functions.est_symbol_power(symbols_diff)
        symbol_power = helper_functions.est_symbol_power(symbols_rx_normalized) - noise_power
        # print(f"noise_power  = {noise_power}")
        # print(f"symbol_power = {symbol_power}")

    # print(f"symbol_power = {symbol_power}")
    esno_linear = symbol_power/noise_power
    return helper_functions.convert_power_to_db(esno_linear)

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
    H2H_inv *= delay_exp
    H2V_inv *= delay_exp
    V2H_inv *= delay_exp
    V2V_inv *= delay_exp
    h2h_inv = np.fft.ifft(H2H_inv)
    h2v_inv = np.fft.ifft(H2V_inv)
    v2h_inv = np.fft.ifft(V2H_inv)
    v2v_inv = np.fft.ifft(V2V_inv)

    return (h2h_inv, h2v_inv, v2h_inv, v2v_inv)

def equalize_rx_symbols(h_symbols, v_symbols, h2h_filter, h2v_filter, v2h_filter, v2v_filter):
    (h2h_filter_inverted,
     h2v_filter_inverted,
     v2h_filter_inverted,
     v2v_filter_inverted) = invert_filters( h2h_filter
                                          , h2v_filter
                                          , v2h_filter
                                          , v2v_filter )

    # print(f"h2h_filter_inverted = {h2h_filter_inverted}")
    # print(f"h2v_filter_inverted = {h2v_filter_inverted}")
    # print(f"v2h_filter_inverted = {v2h_filter_inverted}")
    # print(f"v2v_filter_inverted = {v2v_filter_inverted}")
    # print(f"h2h_filter_inverted_power = {helper_functions.est_symbol_power(h2h_filter_inverted)}")
    # print(f"h2v_filter_inverted_power = {helper_functions.est_symbol_power(h2v_filter_inverted)}")
    # print(f"v2h_filter_inverted_power = {helper_functions.est_symbol_power(v2h_filter_inverted)}")
    # print(f"v2v_filter_inverted_power = {helper_functions.est_symbol_power(v2v_filter_inverted)}")

    fig,axs = plt.subplots(2,2)
    plt.suptitle("Inverted Filter Coefficients")
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
    plt.tight_layout()

    fig,axs = plt.subplots(2,2)
    plt.suptitle("Channel Filter Spectrum (dB), relative frequency [-pi,pi]")
    x = np.linspace(-np.pi,np.pi,len(h2h_filter_inverted))
    axs[0][0].set_title("H2H")
    axs[0][0].plot(x, helper_functions.convert_linear_to_db(np.fft.fftshift(np.fft.fft(h2h_filter))))
    axs[0][1].set_title("V2H")
    axs[0][1].plot(x, helper_functions.convert_linear_to_db(np.fft.fftshift(np.fft.fft(v2h_filter))))
    axs[1][0].set_title("H2V")
    axs[1][0].plot(x, helper_functions.convert_linear_to_db(np.fft.fftshift(np.fft.fft(h2v_filter))))
    axs[1][1].set_title("V2V")
    axs[1][1].plot(x, helper_functions.convert_linear_to_db(np.fft.fftshift(np.fft.fft(v2v_filter))))
    plt.tight_layout()

    # fig,axs = plt.subplots(2,2)
    # plt.suptitle("Inverted Filter Spectrum (dB), relative frequency [-pi,pi]")
    # x = np.linspace(-np.pi,np.pi,len(h2h_filter_inverted))
    # axs[0][0].set_title("H2H_inv")
    # axs[0][0].plot(x, helper_functions.convert_linear_to_db(np.fft.fftshift(np.fft.fft(h2h_filter_inverted))))
    # axs[0][1].set_title("V2H_inv")
    # axs[0][1].plot(x, helper_functions.convert_linear_to_db(np.fft.fftshift(np.fft.fft(v2h_filter_inverted))))
    # axs[1][0].set_title("H2V_inv")
    # axs[1][0].plot(x, helper_functions.convert_linear_to_db(np.fft.fftshift(np.fft.fft(h2v_filter_inverted))))
    # axs[1][1].set_title("V2V_inv")
    # axs[1][1].plot(x, helper_functions.convert_linear_to_db(np.fft.fftshift(np.fft.fft(v2v_filter_inverted))))
    # plt.tight_layout()

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

def test_snr_sweep(random_seed=1234,num_symbols=2**18,num_chan_filter_coefs=8,num_eq_filter_coefs=1024,chan_filter_noise_power=0.1, snr_db_sweep=np.arange(0, 30, 1), test_dpae=True, dpae_mu=0.0005):
    np.random.seed(random_seed)

    # Generate Channel Filters
    h2h_filter = np.zeros(num_chan_filter_coefs)
    h2v_filter = np.zeros(num_chan_filter_coefs)
    v2h_filter = np.zeros(num_chan_filter_coefs)
    v2v_filter = np.zeros(num_chan_filter_coefs)

    # Set middle coefficient to 1, making an all-pass filter
    h2h_filter[num_chan_filter_coefs//2-1] = 1
    v2v_filter[num_chan_filter_coefs//2-1] = 1

    # h2v_filter[num_chan_filter_coefs//2-1] = 0.5
    # v2h_filter[num_chan_filter_coefs//2-1] = 0.5

    filter_rand_scale = np.sqrt(chan_filter_noise_power/2)
    h2h_filter = h2h_filter + filter_rand_scale*(    np.random.randn(len(h2h_filter))
                                                + 1j*np.random.randn(len(h2h_filter)))
    h2v_filter = h2v_filter + filter_rand_scale*(    np.random.randn(len(h2v_filter))
                                                + 1j*np.random.randn(len(h2v_filter)))
    v2h_filter = v2h_filter + filter_rand_scale*(    np.random.randn(len(v2h_filter))
                                                + 1j*np.random.randn(len(v2h_filter)))
    v2v_filter = v2v_filter + filter_rand_scale*(    np.random.randn(len(v2v_filter))
                                                + 1j*np.random.randn(len(v2v_filter)))
    
    num_bits = num_symbols*BITS_PER_SYMBOL
    h_bits_tx    = generate_bits(num_bits)
    v_bits_tx    = generate_bits(num_bits)
    h_symbols_tx = convert_bits_to_symbols(h_bits_tx, BITS_PER_SYMBOL, SYMBOL_POWER)
    v_symbols_tx = convert_bits_to_symbols(v_bits_tx, BITS_PER_SYMBOL, SYMBOL_POWER)

    # plot_const(h_symbols_tx, v_symbols_tx)
    # plt.show()
    
    h_symbols_rx_noise_sweep = np.zeros([len(snr_db_sweep), len(h_symbols_tx)], dtype=complex)
    v_symbols_rx_noise_sweep = np.zeros([len(snr_db_sweep), len(v_symbols_tx)], dtype=complex)

    # fig,axs = plt.subplots(2,2)
    # plt.suptitle("Channel Filter Coefficients")
    # axs[0][0].set_title("h2h_filter")
    # axs[0][0].plot(h2h_filter.real)
    # axs[0][0].plot(h2h_filter.imag)
    # axs[0][1].set_title("v2h_filter")
    # axs[0][1].plot(v2h_filter.real)
    # axs[0][1].plot(v2h_filter.imag)
    # axs[1][0].set_title("h2v_filter")
    # axs[1][0].plot(h2v_filter.real)
    # axs[1][0].plot(h2v_filter.imag)
    # axs[1][1].set_title("v2v_filter")
    # axs[1][1].plot(v2v_filter.real)
    # axs[1][1].plot(v2v_filter.imag)
    # plt.tight_layout()

    h2h_filter = zero_pad(h2h_filter, num_eq_filter_coefs, True)
    h2v_filter = zero_pad(h2v_filter, num_eq_filter_coefs, True)
    v2h_filter = zero_pad(v2h_filter, num_eq_filter_coefs, True)
    v2v_filter = zero_pad(v2v_filter, num_eq_filter_coefs, True)

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
        (h_symbols_rx_noise_sweep[snr_idx], v_symbols_rx_noise_sweep[snr_idx]) \
                = channel_model(h_symbols_tx, v_symbols_tx,
                                h2h_filter_normalized, h2v_filter_normalized,
                                v2h_filter_normalized, v2v_filter_normalized,
                                SYMBOL_POWER, snr_db, CHANNEL_SPS)

    (h_symbols_dpae_rx_noise_sweep, v_symbols_dpae_rx_noise_sweep) = (np.zeros((1,1)),np.zeros((1,1)))
    if (test_dpae):
        (h_symbols_dpae_rx_noise_sweep,
         v_symbols_dpae_rx_noise_sweep) = \
                 dpae.compute_dpae(h_symbols_rx_noise_sweep,
                                   v_symbols_rx_noise_sweep,
                                   num_eq_filter_coefs,
                                   dpae_mu)

    (h_symbols_inverted_rx_noise_sweep,
     v_symbols_inverted_rx_noise_sweep) = \
             equalize_rx_symbols(h_symbols_rx_noise_sweep,
                                 v_symbols_rx_noise_sweep,
                                 h2h_filter_normalized, h2v_filter_normalized,
                                 v2h_filter_normalized, v2v_filter_normalized )
    # print(h_symbols_tx)
    # print(h_symbols_rx_noise_sweep[-1])
    # print(h_symbols_inverted_rx_noise_sweep[-1][-32:])

    h_bits_tx_trimmed = h_bits_tx[num_eq_filter_coefs*BITS_PER_SYMBOL:-num_eq_filter_coefs*BITS_PER_SYMBOL]
    v_bits_tx_trimmed = v_bits_tx[num_eq_filter_coefs*BITS_PER_SYMBOL:-num_eq_filter_coefs*BITS_PER_SYMBOL]
    v_symbols_tx_trimmed = v_symbols_tx[num_eq_filter_coefs:-num_eq_filter_coefs]
    h_symbols_tx_trimmed = h_symbols_tx[num_eq_filter_coefs:-num_eq_filter_coefs]
    v_symbols_tx_trimmed = v_symbols_tx[num_eq_filter_coefs:-num_eq_filter_coefs]
    h_symbols_rx_noise_sweep_trimmed = h_symbols_rx_noise_sweep[:,num_eq_filter_coefs:-num_eq_filter_coefs] 
    v_symbols_rx_noise_sweep_trimmed = v_symbols_rx_noise_sweep[:,num_eq_filter_coefs:-num_eq_filter_coefs] 
    h_symbols_inverted_rx_noise_sweep_trimmed = h_symbols_inverted_rx_noise_sweep[:,num_eq_filter_coefs:-num_eq_filter_coefs] 
    v_symbols_inverted_rx_noise_sweep_trimmed = v_symbols_inverted_rx_noise_sweep[:,num_eq_filter_coefs:-num_eq_filter_coefs] 
    h_symbols_dpae_rx_noise_sweep_trimmed = h_symbols_dpae_rx_noise_sweep[:,num_eq_filter_coefs:-num_eq_filter_coefs] 
    v_symbols_dpae_rx_noise_sweep_trimmed = v_symbols_dpae_rx_noise_sweep[:,num_eq_filter_coefs:-num_eq_filter_coefs] 

    # plot_const(h_symbols_rx_noise_sweep_trimmed[-1], v_symbols_rx_noise_sweep_trimmed[-1], "Constellation before equalization")
    # plot_const(h_symbols_inverted_rx_noise_sweep_trimmed[-1], v_symbols_inverted_rx_noise_sweep_trimmed[-1], "Constellation after equalization")
    plot_const(h_symbols_dpae_rx_noise_sweep_trimmed[-1], v_symbols_dpae_rx_noise_sweep_trimmed[-1], "Constellation after DPAE")
    plot_const(h_symbols_rx_noise_sweep_trimmed[-1], h_symbols_dpae_rx_noise_sweep_trimmed[-1], "Constellation Before/After DPAE")

    fig, axs = plt.subplots(3,2)
    plt.suptitle("H-Pol TX and RX Spectrum (dB), relative frequency [-pi,pi]")
    x = np.linspace(-np.pi,np.pi,len(h_symbols_tx_trimmed))
    axs[0][0].set_title("H-Pol TX Spectrum")
    axs[0][0].plot(x, helper_functions.convert_linear_to_db(np.fft.fftshift(np.fft.fft(h_symbols_tx_trimmed))))
    axs[1][0].set_title("H-Pol RX Spectrum Before Equalization")
    axs[1][0].plot(x, helper_functions.convert_linear_to_db(np.fft.fftshift(np.fft.fft(h_symbols_rx_noise_sweep_trimmed[-1]))))
    axs[2][0].set_title("H-Pol RX Spectrum After Equalization")
    axs[2][0].plot(x, helper_functions.convert_linear_to_db(np.fft.fftshift(np.fft.fft(h_symbols_inverted_rx_noise_sweep_trimmed[-1]))))
    axs[0][1].set_title("V-Pol TX Spectrum")
    axs[0][1].plot(x, helper_functions.convert_linear_to_db(np.fft.fftshift(np.fft.fft(v_symbols_tx_trimmed))))
    axs[1][1].set_title("H-Pol RX Spectrum Before Equalization")
    axs[1][1].plot(x, helper_functions.convert_linear_to_db(np.fft.fftshift(np.fft.fft(h_symbols_rx_noise_sweep_trimmed[-1]))))
    axs[2][1].set_title("V-Pol RX Spectrum After Equalization")
    axs[2][1].plot(x, helper_functions.convert_linear_to_db(np.fft.fftshift(np.fft.fft(v_symbols_inverted_rx_noise_sweep_trimmed[-1]))))
    plt.tight_layout()

    # fig, axs = plt.subplots(3,1)
    # plt.suptitle("First 32 H-Pol Symbols")
    # axs[0].set_title("TX Symbols")
    # axs[0].plot(h_symbols_tx_trimmed[:32].real)
    # axs[0].plot(h_symbols_tx_trimmed[:32].imag)
    # axs[1].set_title("RX Symbols")
    # axs[1].plot(h_symbols_rx_noise_sweep_trimmed[-1][:32].real)
    # axs[1].plot(h_symbols_rx_noise_sweep_trimmed[-1][:32].imag)
    # axs[2].set_title("RX Symbols After Equalization")
    # axs[2].plot(h_symbols_inverted_rx_noise_sweep_trimmed[-1][:32].real)
    # axs[2].plot(h_symbols_inverted_rx_noise_sweep_trimmed[-1][:32].imag)
    # plt.tight_layout()

    # plot_sweep(h_bits_tx_trimmed, v_bits_tx_trimmed, h_symbols_tx_trimmed, v_symbols_tx_trimmed, h_symbols_rx_noise_sweep_trimmed, v_symbols_rx_noise_sweep_trimmed, snr_db_sweep, "Before Equalization: Channel SNR (dB)")
    plot_sweep(h_bits_tx_trimmed, v_bits_tx_trimmed, h_symbols_tx_trimmed, v_symbols_tx_trimmed, h_symbols_inverted_rx_noise_sweep_trimmed, v_symbols_inverted_rx_noise_sweep_trimmed, snr_db_sweep, "After Equalization: Channel SNR (dB)")
    if (test_dpae):
        plot_sweep(h_bits_tx_trimmed, v_bits_tx_trimmed, h_symbols_tx_trimmed, v_symbols_tx_trimmed, h_symbols_dpae_rx_noise_sweep_trimmed, v_symbols_dpae_rx_noise_sweep_trimmed, snr_db_sweep, "After DPAE: Channel SNR (dB)")

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

def compute_theoretical_qpsk_ber(snr_array_db):
    snr_array_linear = np.power(10, snr_array_db/10)
    theoretical_qpsk_ber = [0.5*math.erfc(np.sqrt(snr_linear/BITS_PER_SYMBOL)) for snr_linear in snr_array_linear]
    return theoretical_qpsk_ber

def plot_sweep(h_bits_tx, v_bits_tx, h_symbols_tx, v_symbols_tx, h_symbols_rx_sweep, v_symbols_rx_sweep, sweep_params, sweep_param_name):

    # h_bits_received_noise_sweep = [
    #         convert_symbols_to_bits(x, BITS_PER_SYMBOL, len(x))
    #         for x in h_symbols_rx_sweep ]
    # v_bits_received_noise_sweep = [
    #         convert_symbols_to_bits(x, BITS_PER_SYMBOL, len(x))
    #         for x in v_symbols_rx_sweep ]

    h_qpsk_esnos = np.array([ est_symbol_qpsk_esno(x, True) for x in h_symbols_rx_sweep ])
    v_qpsk_esnos = np.array([ est_symbol_qpsk_esno(x, True) for x in v_symbols_rx_sweep ])
    average_qpsk_esnos = (np.array(h_qpsk_esnos) + np.array(v_qpsk_esnos))/2.0

    h_qpsk_esnos_no_abs_correction = np.array([ est_symbol_qpsk_esno(x, False) for x in h_symbols_rx_sweep ])
    v_qpsk_esnos_no_abs_correction = np.array([ est_symbol_qpsk_esno(x, False) for x in v_symbols_rx_sweep ])
    average_qpsk_esnos_no_abs_correction = (np.array(h_qpsk_esnos_no_abs_correction) + np.array(v_qpsk_esnos_no_abs_correction))/2.0

    h_radial_esnos = np.array([ helper_functions.est_symbol_radial_esno(x) for x in h_symbols_rx_sweep ])
    v_radial_esnos = np.array([ helper_functions.est_symbol_radial_esno(x) for x in v_symbols_rx_sweep ])
    average_radial_esnos = (np.array(h_radial_esnos) + np.array(v_radial_esnos))/2.0

    h_tx_rx_esnos = [ est_symbol_tx_rx_esno(h_symbols_tx,x,normalize_symbols=True) for x in h_symbols_rx_sweep ]
    v_tx_rx_esnos = [ est_symbol_tx_rx_esno(v_symbols_tx,x,normalize_symbols=True) for x in v_symbols_rx_sweep ]
    average_tx_rx_esnos = (np.array(h_tx_rx_esnos) + np.array(v_tx_rx_esnos))/2.0

    print(f"average_tx_rx_esnos with {sweep_params[-1]} dB SNR = {average_tx_rx_esnos[-1]}")

    # plt.figure()
    # plt.title(f"{sweep_param_name} vs Estimated H-Pol EsNo (dB)")
    # plt.plot(sweep_params, sweep_params)
    # plt.plot(sweep_params, h_qpsk_esnos)
    # plt.plot(sweep_params, h_qpsk_esnos_no_abs_correction)
    # plt.plot(sweep_params, h_radial_esnos)
    # plt.plot(sweep_params, h_tx_rx_esnos)
    # plt.xlabel(sweep_param_name)
    # plt.legend(["Channel SNR", "QPSK EsNo", "QPSK EsNo (no abs correction)", "Radial EsNo", "TX RX EsNo"])

    # plt.figure()
    # plt.title(f"{sweep_param_name} vs Estimated V-Pol EsNo (dB)")
    # plt.plot(sweep_params, sweep_params)
    # plt.plot(sweep_params, v_qpsk_esnos)
    # plt.plot(sweep_params, v_qpsk_esnos_no_abs_correction)
    # plt.plot(sweep_params, v_radial_esnos)
    # plt.plot(sweep_params, v_tx_rx_esnos)
    # plt.xlabel(sweep_param_name)
    # plt.legend(["Channel SNR", "QPSK EsNo", "QPSK EsNo (no abs correction)", "Radial EsNo", "TX RX EsNo"])

    plt.figure()
    plt.title(f"{sweep_param_name} vs Estimated EsNo (dB)")
    plt.plot(sweep_params, sweep_params)
    plt.plot(sweep_params, average_qpsk_esnos)
    plt.plot(sweep_params, average_qpsk_esnos_no_abs_correction)
    plt.plot(sweep_params, average_radial_esnos)
    plt.plot(sweep_params, average_tx_rx_esnos)
    plt.xlabel(sweep_param_name)
    plt.legend(["Channel SNR", "QPSK EsNo", "QPSK EsNo (no abs correction)", "Radial EsNo", "TX RX EsNo"])
    # plt.legend(["QPSK EsNo", "QPSK EsNo (no abs correction)", "Radial EsNo", "TX RX EsNo"])

    # plt.figure()
    # plt.title(f"{sweep_param_name} vs Estimated EsNo - TX/RX EsNo (dB)")
    # plt.plot(sweep_params, average_qpsk_esnos - average_tx_rx_esnos)
    # plt.plot(sweep_params, average_qpsk_esnos_no_abs_correction - average_tx_rx_esnos)
    # plt.plot(sweep_params, average_radial_esnos - average_tx_rx_esnos)
    # plt.xlabel(sweep_param_name)
    # plt.legend(["QPSK EsNo", "QPSK EsNo (no abs correction)", "Radial EsNo", "TX/RX EsNo"])

    # h_bit_error_rates = [ bit_error_rate(h_bits_tx,x) for x in h_bits_received_noise_sweep ]
    # v_bit_error_rates = [ bit_error_rate(v_bits_tx,x) for x in v_bits_received_noise_sweep ]
    # average_bit_error_rates = (np.array(h_bit_error_rates) + np.array(v_bit_error_rates))/2.0

    # theoretical_qpsk_ber = compute_theoretical_qpsk_ber(sweep_params)

    # plt.figure()
    # plt.title(f"{sweep_param_name} vs Bit Error Rate")
    # plt.semilogy(sweep_params, theoretical_qpsk_ber)
    # plt.semilogy(sweep_params, h_bit_error_rates)
    # plt.semilogy(sweep_params, v_bit_error_rates)
    # plt.semilogy(sweep_params, average_bit_error_rates)
    # plt.xlabel(sweep_param_name)
    # plt.xticks(np.arange(-0,16))
    # plt.xlim((-0,15))
    # plt.ylim((1e-8, 1e-0))
    # plt.legend(["Theoretical QPSK Bit Error Rate", "H-Pol Bit Error Rate", "V-Pol Bit Error Rate", "Average of H-Pol and V-Pol Bit Error Rate"])

def main():
    snr_db_sweep = np.arange(0, 31, 5)

    import time

    start_time = time.clock_gettime(0)

    test_snr_sweep(random_seed=3,num_symbols=2**18,num_chan_filter_coefs=2,num_eq_filter_coefs=32,chan_filter_noise_power=0.5, snr_db_sweep=snr_db_sweep, test_dpae=True, dpae_mu=0.0001)

    end_time = time.clock_gettime(0)

    print(f"Run time = {int(end_time-start_time)} seconds")

    # test_phase_sweep(h_bits_tx, v_bits_tx, h_symbols_tx, v_symbols_tx)

    plt.show()

if __name__ == "__main__":
    main()
