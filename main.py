import numpy as np

NUM_SYMBOLS = 2**8
BITS_PER_SYMBOL = 2
NUM_BITS = NUM_SYMBOLS*BITS_PER_SYMBOL
SYMBOL_POWER = 1.0
NOISE_POWER = 0.25

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

def convert_symbols_to_bits(symbols, bits_per_symbol):
    num_symbols = len(symbols)

    bits = np.zeros(num_symbols*bits_per_symbol, dtype=int)

    if bits_per_symbol == 2: # QPSK)
        for symbol_idx in np.arange(0, num_symbols, 1):
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

def main():
    bits = generate_bits(NUM_BITS)

    symbols = convert_bits_to_symbols(bits, BITS_PER_SYMBOL, SYMBOL_POWER)
    # h_symbols = convert_bits_to_symbols(bits, BITS_PER_SYMBOL, SYMBOL_POWER)
    # v_symbols = convert_bits_to_symbols(bits, BITS_PER_SYMBOL, SYMBOL_POWER)
    #const_plot(symbols)

    num_symbols = len(symbols)
    symbols_received = symbols + awgn(num_symbols, NOISE_POWER)
    bits_receieved = convert_symbols_to_bits(symbols_received, BITS_PER_SYMBOL)

    bit_errors = np.abs(bits_receieved - bits)

    num_bit_errors = np.sum(bit_errors)

    # print(symbols)
    # print(symbols_received)
    # print(bits)
    # print(bits_receieved)
    # print(bit_errors)

    plot_const(symbols, symbols_received)

    print(num_bit_errors/NUM_BITS)
    plt.show()

if __name__ == "__main__":
    main()
