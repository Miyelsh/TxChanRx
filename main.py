import numpy as np

NUM_SYMBOLS = 2**12
BITS_PER_SYMBOL = 2
NUM_BITS = NUM_SYMBOLS*BITS_PER_SYMBOL
SYMBOL_POWER = 1.0
NOISE_POWER = 0.5

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

def awgn(size, noise_power):
    return noise_power*(np.random.randn(size) + 1j*np.random.randn(size))

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
    # filter = filter / np.sum(filter*filter)
    print(f"SPS: {sps}, Filter Power = {np.sum(filter*filter)}")
    y = np.convolve(filter, input_sps_x, mode="same")
    return y

def downsample_sps_x(input, sps):
    y = input[::sps] # capture every other
    return y

def main():
    bits = generate_bits(NUM_BITS)

    symbols = convert_bits_to_symbols(bits, BITS_PER_SYMBOL, SYMBOL_POWER)
    # h_symbols = convert_bits_to_symbols(bits, BITS_PER_SYMBOL, SYMBOL_POWER)
    # v_symbols = convert_bits_to_symbols(bits, BITS_PER_SYMBOL, SYMBOL_POWER)
    #const_plot(symbols)

    num_symbols = len(symbols)
    # symbols_received_awgn      = symbols + awgn(num_symbols, NOISE_POWER)
    # symbols_received_resampled_2x = downsample2x(upsample2x(symbols + awgn(num_symbols, NOISE_POWER)))
    # symbols_received_resampled_3x = downsample_sps_x(upsample_sps_x(symbols + awgn(num_symbols, NOISE_POWER),3),3)
    # bits_receieved_awgn = convert_symbols_to_bits(symbols_received_awgn, BITS_PER_SYMBOL, len(symbols))
    # bits_receieved_resampled_2x = convert_symbols_to_bits(symbols_received_resampled_2x, BITS_PER_SYMBOL, len(symbols))
    # bits_receieved_resampled_3x = convert_symbols_to_bits(symbols_received_resampled_3x, BITS_PER_SYMBOL, len(symbols))

    # bit_errors_awgn      = np.abs(bits_receieved_awgn - bits)
    # bit_errors_resampled_2x = np.abs(bits_receieved_resampled_2x - bits)
    # bit_errors_resampled_3x = np.abs(bits_receieved_resampled_3x - bits)

    # num_bit_errors_awgn      = np.sum(bit_errors_awgn)
    # num_bit_errors_resampled_2x = np.sum(bit_errors_resampled_2x)
    # num_bit_errors_resampled_3x = np.sum(bit_errors_resampled_3x)

    # print(symbols)
    # print(symbols_received_resampled_2x)
    # print(bits)
    # print(bits_receieved)
    # print(bit_errors)

    # plot_const(symbols, symbols_received_awgn)
    # plot_const(symbols, symbols_received_resampled_2x)
    # plot_const(symbols, symbols_received_resampled_3x)

    symbols_received_sps_1_to_16 = [
            downsample_sps_x(upsample_sps_x(symbols + awgn(num_symbols, NOISE_POWER),sps),sps)
            for sps in range(1,1+16) ]

    bits_received_sps_1_to_16 = [
            convert_symbols_to_bits(x, BITS_PER_SYMBOL, NUM_SYMBOLS)
            for x in symbols_received_sps_1_to_16 ]

    bits_errors_sps_1_to_16 = [
            np.sum(np.abs(bits-x))
            for x in bits_received_sps_1_to_16 ]

    # plot_const(symbols, symbols_received_sps_1_to_16[1])
    # print(symbols_received_sps_1_to_16[1])
    # plot_const(symbols, symbols_received_sps_1_to_16[15])

    print("Received Symbol Power")
    [ print(np.mean(x*np.conj(x))) for x in symbols_received_sps_1_to_16 ]

    # [ print(x) for x in bits_received_sps_1_to_16 ]
    print("Error Rates")
    [ print(x/NUM_BITS) for x in bits_errors_sps_1_to_16 ]


    # print(len(symbols_received_sps_1_to_16[0]))
    # print(len(symbols_received_resampled_2x))

    # print(num_bit_errors_awgn/NUM_BITS)
    # print(num_bit_errors_resampled_2x/NUM_BITS)
    # print(num_bit_errors_resampled_3x/NUM_BITS)
    plt.show()

if __name__ == "__main__":
    main()
