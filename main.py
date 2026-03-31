import numpy as np

NUM_SYMBOLS = 2**12
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

import matplotlib.pyplot as plt
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

def est_symbol_power(x):
    return np.mean(np.abs(x*np.conj(x)))

def est_symbol_qpsk_esno(symbols):
    symbols_first_quadrant = [ complex(np.abs(symbol.real), np.abs(symbol.imag)) for symbol in symbols ]
    symbols_dc_removed = np.array(symbols_first_quadrant) - np.mean(symbols_first_quadrant)
    symbol_power = est_symbol_power(symbols) - est_symbol_power(symbols_dc_removed)
    # symbol_power = est_symbol_power(symbols) # slightly less accurate
    noise_power = est_symbol_power(symbols_dc_removed)
    esno_linear = symbol_power/noise_power
    return 10*np.log10(esno_linear)

def est_symbol_radial_esno(x):
    return 0

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
    osnr_db_sweep = np.arange(-10, 20, 0.1)
    osnr_linear_sweep = np.pow(10, osnr_db_sweep/10.0)
    noise_power_sweep = SYMBOL_POWER/osnr_linear_sweep
    noise_power_db_sweep = -10*np.log10(noise_power_sweep)
    # print(noise_powers)
    # print(noise_powers_db)
    symbols_received_noise_sweep = [
            downsample_sps_x(upsample_sps_x(symbols + awgn(num_symbols, SYMBOL_POWER, osnr_db),sps),sps)
            for osnr_db in osnr_db_sweep ]

    bits_received_noise_sweep = [
            convert_symbols_to_bits(x, BITS_PER_SYMBOL, NUM_SYMBOLS)
            for x in symbols_received_noise_sweep ]

    # print("Received Symbol Power")
    # [ print(est_symbol_power(x)) for x in symbols_received_noise_sweep ]

    qpsk_esnos = [ est_symbol_qpsk_esno(x) for x in symbols_received_noise_sweep ]
    # print("Received Symbol QPSK EsNo")
    # print(qpsk_esnos)

    # print("Received Symbol Radial EsNo")
    # [ print(est_symbol_radial_esno(x)) for x in symbols_received_noise_sweep ]

    tx_rx_esnos = [ est_symbol_tx_rx_esno(symbols,x) for x in symbols_received_noise_sweep ]
    # print("Received Symbol TX RX EsNo")
    # print(tx_rx_esnos)

    plt.figure()
    plt.grid()
    plt.plot(osnr_db_sweep, noise_power_db_sweep)
    plt.plot(osnr_db_sweep, qpsk_esnos)
    plt.plot(osnr_db_sweep, tx_rx_esnos)
    plt.legend(["True EsNo", "QPSK EsNo", "TX RX EsNo"])

    bit_error_rates_noise_sweep = [ bit_error_rate(bits,x) for x in bits_received_noise_sweep ]
    # print("Error Rates")

    plt.figure()
    plt.grid()
    plt.semilogy(noise_power_db_sweep, bit_error_rates_noise_sweep)

    plot_const(symbols, symbols_received_noise_sweep[0])
    plot_const(symbols, symbols_received_noise_sweep[-1])


    plt.show()

if __name__ == "__main__":
    main()
