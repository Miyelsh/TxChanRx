import numpy as np
import matplotlib.pyplot as plt
import helper_functions

def signed_square_root(x):
    # return x
    return np.sign(x)*np.sqrt(np.abs(x))

def error_function(i,bits_per_symbol,idx,num_symbols):
    # QPSK
    if (bits_per_symbol == 2):
        if idx < num_symbols/2:
            return (1-i*i)
        else:
            return 0.01*(1-i*i)
    #     return signed_square_root(1 - (i*i))

    # return (2 - (i*i + q*q))

    # power = i*i + q*q
    # power = i*i
    # if power > 2.0:
    #     return -1
    # elif power < 2.0:
    #     return +1
    # else:
    #     return 0


    # 16 QAM
    if (bits_per_symbol == 4):
        if idx < num_symbols/2:
            # return 0.01*(1 - (i*i))**3
            # return (1 - (i*i))
            # return (1 - (i*i))

            # power = i*i
            # if power < 0.9:
            #     return +1
            # elif power > 1.1:
            #     return -1
            # else:
            #     return 0

            return 0.5*(2.2/2.4-i*i)# signed_square_root(1-i*i)

        # 16 QAM
        power = i*i
        if power < 5/9:
            power += 8/9
        return 0.5*(1 - power)

        # power = i*i
        # if power < 5/9:
        #     return 0
        # return (1 - power)


def compute_invert_filters(h_symbols_tx, v_symbols_tx, h_symbols, v_symbols, bits_per_symbol, num_eq_filter_coefs, dpae_mu):

    # iir_step_size = 0.002
    # dpae_mu = 0.01
    iir_step_size = 0.01
    max_error = 0.1
    print(f"{dpae_mu = }")
    print(f"{iir_step_size = }")
    print(f"{max_error = }")

    h2h = np.zeros(num_eq_filter_coefs, dtype=complex)
    h2v = np.zeros(num_eq_filter_coefs, dtype=complex)
    v2h = np.zeros(num_eq_filter_coefs, dtype=complex)
    v2v = np.zeros(num_eq_filter_coefs, dtype=complex)

    hi2hi = np.zeros(num_eq_filter_coefs)
    hi2hq = np.zeros(num_eq_filter_coefs)
    hi2vi = np.zeros(num_eq_filter_coefs)
    hi2vq = np.zeros(num_eq_filter_coefs)
    vi2hi = np.zeros(num_eq_filter_coefs)
    vi2hq = np.zeros(num_eq_filter_coefs)
    vi2vi = np.zeros(num_eq_filter_coefs)
    vi2vq = np.zeros(num_eq_filter_coefs)

    # Initialize H2H and V2V as all-pass, H2V and V2H as zeros
    h2h[num_eq_filter_coefs//2-1] = 1.0
    v2v[num_eq_filter_coefs//2-1] = 1.0

    hi2hi[num_eq_filter_coefs//2-1] = 1.0
    vi2vi[num_eq_filter_coefs//2-1] = 1.0

    # h2h = h2h_filter_inverted
    # h2v = h2v_filter_inverted
    # v2h = v2h_filter_inverted
    # v2v = v2v_filter_inverted

    # hi2hi = h2h_filter_inverted.real
    # hi2hq = h2h_filter_inverted.imag
    # hi2vi = h2v_filter_inverted.real
    # hi2vq = h2v_filter_inverted.imag
    # vi2hi = v2h_filter_inverted.real
    # vi2hq = v2h_filter_inverted.imag
    # vi2vi = v2v_filter_inverted.real
    # vi2vq = v2v_filter_inverted.imag

    # hq2hi = -hi2hq
    # hq2hq =  hi2hi 
    # vq2hi = -vi2hq
    # vq2hq =  vi2hi 
    #                  
    # hq2vi = -hi2vq
    # hq2vq =  hi2vi 
    # vq2vi = -vi2vq
    # vq2vq =  vi2vi

    if (len(h_symbols) != len(v_symbols)):
        print("ERROR: h_symbols and v_symbols must be same length!")
        exit(1)

    num_symbols = len(h_symbols)
    h_symbols_filtered = np.zeros(num_symbols, dtype=complex)
    v_symbols_filtered = np.zeros(num_symbols, dtype=complex)

    # Convolve symbols with 
    N = num_eq_filter_coefs - 1

    power_h = np.zeros(len(h_symbols)-num_eq_filter_coefs)
    radial_esno_h  = np.zeros(len(h_symbols)-num_eq_filter_coefs)
    tx_rx_esno_h = np.zeros(len(h_symbols)-num_eq_filter_coefs)
    power_v = np.zeros(len(v_symbols)-num_eq_filter_coefs)
    radial_esno_v  = np.zeros(len(v_symbols)-num_eq_filter_coefs)
    tx_rx_esno_v = np.zeros(len(h_symbols)-num_eq_filter_coefs)

    hi2hi_error_avg = np.zeros(len(v_symbols)-num_eq_filter_coefs)
    hi2hq_error_avg = np.zeros(len(v_symbols)-num_eq_filter_coefs)
    hi2vi_error_avg = np.zeros(len(v_symbols)-num_eq_filter_coefs)
    hi2vq_error_avg = np.zeros(len(v_symbols)-num_eq_filter_coefs)
    vi2hi_error_avg = np.zeros(len(v_symbols)-num_eq_filter_coefs)
    vi2hq_error_avg = np.zeros(len(v_symbols)-num_eq_filter_coefs)
    vi2vi_error_avg = np.zeros(len(v_symbols)-num_eq_filter_coefs)
    vi2vq_error_avg = np.zeros(len(v_symbols)-num_eq_filter_coefs)

    for t in range(num_symbols-num_eq_filter_coefs):
        # h_in = h_symbols[t:t+num_eq_filter_coefs]
        # v_in = v_symbols[t:t+num_eq_filter_coefs]

        hi_in = h_symbols.real[t:t+num_eq_filter_coefs]
        hq_in = h_symbols.imag[t:t+num_eq_filter_coefs]
        vi_in = v_symbols.real[t:t+num_eq_filter_coefs]
        vq_in = v_symbols.imag[t:t+num_eq_filter_coefs]

        # h_out  = np.convolve(h_in, h2h, mode="valid")[0]
        # h_out += np.convolve(v_in, v2h, mode="valid")[0]
        # v_out  = np.convolve(v_in, v2v, mode="valid")[0]
        # v_out += np.convolve(h_in, h2v, mode="valid")[0]

        hi_out  = np.convolve(hi_in,  hi2hi, mode="valid")[0]
        hi_out += np.convolve(hq_in, -hi2hq, mode="valid")[0] # hq2hi = -hi2hq
        hi_out += np.convolve(vi_in,  vi2hi, mode="valid")[0]
        hi_out += np.convolve(vq_in, -vi2hq, mode="valid")[0] # vq2hi = -vi2hq

        hq_out  = np.convolve(hi_in,  hi2hq, mode="valid")[0]
        hq_out += np.convolve(hq_in,  hi2hi, mode="valid")[0] # hq2hq = hi2hi
        hq_out += np.convolve(vi_in,  vi2hq, mode="valid")[0]
        hq_out += np.convolve(vq_in,  vi2hi, mode="valid")[0] # vq2hq = vi2hi
        # hq_out  = np.convolve(hi_in, hi2hq, mode="valid")[0]
        # hq_out += np.convolve(hq_in, hq2hq, mode="valid")[0]
        # hq_out += np.convolve(vi_in, vi2hq, mode="valid")[0]
        # hq_out += np.convolve(vq_in, vq2hq, mode="valid")[0]

        vi_out  = np.convolve(hi_in,  hi2vi, mode="valid")[0]
        vi_out += np.convolve(hq_in, -hi2vq, mode="valid")[0] # hq2vi = -hi2vq
        vi_out += np.convolve(vi_in,  vi2vi, mode="valid")[0]
        vi_out += np.convolve(vq_in, -vi2vq, mode="valid")[0] # vq2vi = -vi2vq

        vq_out  = np.convolve(hi_in,  hi2vq, mode="valid")[0]
        vq_out += np.convolve(hq_in,  hi2vi, mode="valid")[0] # hq2vq = hi2hi
        vq_out += np.convolve(vi_in,  vi2vq, mode="valid")[0]
        vq_out += np.convolve(vq_in,  vi2vi, mode="valid")[0] # vq2vq = vi2vi
        # vq_out  = np.convolve(hi_in, hi2vq, mode="valid")[0]
        # vq_out += np.convolve(hq_in, hq2vq, mode="valid")[0]
        # vq_out += np.convolve(vi_in, vi2vq, mode="valid")[0]
        # vq_out += np.convolve(vq_in, vq2vq, mode="valid")[0]

        # Save filtered symbols
        # h_symbols_filtered[t] = h_out
        # v_symbols_filtered[t] = v_out

        h_symbols_filtered[t] = complex(hi_out,hq_out)
        v_symbols_filtered[t] = complex(vi_out,vq_out)

        # if (t%100000 == 256):
        if (False):
            # print(f"h2h_power = {helper_functions.est_symbol_power(h2h)}")
            # print(f"h2v_power = {helper_functions.est_symbol_power(h2v)}")
            # print(f"v2h_power = {helper_functions.est_symbol_power(v2h)}")
            # print(f"v2v_power = {helper_functions.est_symbol_power(v2v)}")
            # print(f"h2h = {h2h}")
            # print(f"h2v = {h2v}")
            # print(f"v2h = {v2h}")
            # print(f"v2v = {v2v}")
            # print(f"hi2hi = {hi2hi}")
            # print(f"hi2hq = {hi2hq}")
            # print(f"hi2vi = {hi2vi}")
            # print(f"hi2vq = {hi2vq}")
            # print(f"vi2hi = {vi2hi}")
            # print(f"vq2hi = {vi2hq}")
            # print(f"vi2vi = {vi2vi}")
            # print(f"vq2vi = {vi2vq}")
            # print(f"h2h.shape = {h2h.shape}")
            fig,axs = plt.subplots(1,2)
            axs[0].set_title(f"h_out[{t-256}:{t}]")
            axs[0].set_aspect("equal", "box")
            axs[0].scatter(h_symbols_filtered[t-256:t].real, h_symbols_filtered[t-256:t].imag)
            axs[1].set_title(f"v_out[{t-256}:{t}]")
            axs[1].scatter(v_symbols_filtered[t-256:t].real, v_symbols_filtered[t-256:t].imag)
            axs[1].set_aspect("equal", "box")
            # axs[0].plot(h2h.real)
            # axs[0].plot(h2h.imag)
            print(f"radial_esno_h  = {radial_esno_h}")
            print(f"power_h = {power_h}")
            print(f"tx_rx_esno_h  = {tx_rx_esno_h}")
            print(f"radial_esno_v  = {radial_esno_v}")
            print(f"power_v = {power_v}")
            print(f"tx_rx_esno_v  = {tx_rx_esno_v}")

            plt.show()

        if t < 256+num_eq_filter_coefs:
            continue

        # plt.figure()
        # plt.plot(h_symbols_tx[t-256:t])
        # plt.plot(h_symbols_filtered[t-256:t])
        # plt.show()

        power_h[t] = helper_functions.est_symbol_power(h_symbols_filtered[t-256:t])
        radial_esno_h[t] = helper_functions.est_symbol_radial_esno(h_symbols_filtered[t-256:t])
        tx_rx_esno_h[t] = helper_functions.est_symbol_tx_rx_esno(h_symbols_tx[t-256:t], h_symbols_filtered[t-256:t], True)
        power_v[t] = helper_functions.est_symbol_power(v_symbols_filtered[t-256:t])
        radial_esno_v[t] = helper_functions.est_symbol_radial_esno(v_symbols_filtered[t-256:t])
        tx_rx_esno_v[t] = helper_functions.est_symbol_tx_rx_esno(v_symbols_tx[t-256:t], v_symbols_filtered[t-256:t], True)

        iir_step_size = 0.002
    
        # Add tap error terms to existing filters
        for n in range(num_eq_filter_coefs):
            hi2hi_error = error_function(hi_out, bits_per_symbol, t, num_symbols) \
                        * hi_out \
                        * hi_in[N-n]

            hi2hq_error = error_function(hq_out, bits_per_symbol, t, num_symbols) \
                        * hq_out \
                        * hi_in[N-n]

            hi2vi_error = error_function(vi_out, bits_per_symbol, t, num_symbols) \
                        * vi_out \
                        * hi_in[N-n]

            hi2vq_error = error_function(vq_out, bits_per_symbol, t, num_symbols) \
                        * vq_out \
                        * hi_in[N-n]

            vi2hi_error = error_function(hi_out, bits_per_symbol, t, num_symbols) \
                        * hi_out \
                        * vi_in[N-n]

            vi2hq_error = error_function(hq_out, bits_per_symbol, t, num_symbols) \
                        * hq_out \
                        * vi_in[N-n]

            vi2vi_error = error_function(vi_out, bits_per_symbol, t, num_symbols) \
                        * vi_out \
                        * vi_in[N-n]

            vi2vq_error = error_function(vq_out, bits_per_symbol, t, num_symbols) \
                        * vq_out \
                        * vi_in[N-n]

            hi2hi_error = np.clip(hi2hi_error, -max_error, max_error)
            hi2hq_error = np.clip(hi2hq_error, -max_error, max_error)
            hi2vi_error = np.clip(hi2vi_error, -max_error, max_error)
            hi2vq_error = np.clip(hi2vq_error, -max_error, max_error)
            vi2hi_error = np.clip(vi2hi_error, -max_error, max_error)
            vi2hq_error = np.clip(vi2hq_error, -max_error, max_error)
            vi2vi_error = np.clip(vi2vi_error, -max_error, max_error)
            vi2vq_error = np.clip(vi2vq_error, -max_error, max_error)

            hi2hi[n]    = hi2hi[n] + hi2hi_error*dpae_mu
            hi2hq[n]    = hi2hq[n] + hi2hq_error*dpae_mu
            hi2vi[n]    = hi2vi[n] + hi2vi_error*dpae_mu
            hi2vq[n]    = hi2vq[n] + hi2vq_error*dpae_mu
            vi2hi[n]    = vi2hi[n] + vi2hi_error*dpae_mu
            vi2hq[n]    = vi2hq[n] + vi2hq_error*dpae_mu
            vi2vi[n]    = vi2vi[n] + vi2vi_error*dpae_mu
            vi2vq[n]    = vi2vq[n] + vi2vq_error*dpae_mu

            h2h[n] = complex(hi2hi[n], hi2hq[n])
            h2v[n] = complex(hi2vi[n], hi2vq[n])
            v2h[n] = complex(vi2hi[n], vi2hq[n])
            v2v[n] = complex(vi2vi[n], vi2vq[n])

            vi2hi_error_avg[t] += iir_step_size*np.abs(vi2hi_error)/num_eq_filter_coefs
            hi2hq_error_avg[t] += iir_step_size*np.abs(hi2hq_error)/num_eq_filter_coefs
            hi2vi_error_avg[t] += iir_step_size*np.abs(hi2vi_error)/num_eq_filter_coefs
            hi2vq_error_avg[t] += iir_step_size*np.abs(hi2vq_error)/num_eq_filter_coefs
            hi2hi_error_avg[t] += iir_step_size*np.abs(hi2hi_error)/num_eq_filter_coefs
            vi2hq_error_avg[t] += iir_step_size*np.abs(vi2hq_error)/num_eq_filter_coefs
            vi2vi_error_avg[t] += iir_step_size*np.abs(vi2vi_error)/num_eq_filter_coefs
            vi2vq_error_avg[t] += iir_step_size*np.abs(vi2vq_error)/num_eq_filter_coefs

        hi2hi_error_avg[t] += (1-iir_step_size)*hi2hi_error_avg[t-1]
        hi2hq_error_avg[t] += (1-iir_step_size)*hi2hq_error_avg[t-1]
        hi2vi_error_avg[t] += (1-iir_step_size)*hi2vi_error_avg[t-1]
        hi2vq_error_avg[t] += (1-iir_step_size)*hi2vq_error_avg[t-1]
        vi2hi_error_avg[t] += (1-iir_step_size)*vi2hi_error_avg[t-1]
        vi2hq_error_avg[t] += (1-iir_step_size)*vi2hq_error_avg[t-1]
        vi2vi_error_avg[t] += (1-iir_step_size)*vi2vi_error_avg[t-1]
        vi2vq_error_avg[t] += (1-iir_step_size)*vi2vq_error_avg[t-1]

    fig,axs = plt.subplots(1,2)
    plt.suptitle("TX/RX EsNo over time")
    axs[0].set_title("H-Pol TX/RX EsNo")
    axs[0].plot(tx_rx_esno_h)
    axs[1].set_title("V-Pol TX/RX EsNo")
    axs[1].plot(tx_rx_esno_v)
    plt.tight_layout()

    fig,axs = plt.subplots(1,2)
    fig.suptitle(f"Power over time")
    axs[0].set_title("H-Pol Power")
    axs[0].plot(power_h[256:])
    axs[1].set_title("V-Pol Power")
    axs[1].plot(power_v[256:])
    plt.tight_layout()

    fig,axs = plt.subplots(2,4)
    fig.suptitle(f"Error")
    axs[0][0].set_title("hi2hi_error")
    axs[0][0].plot(hi2hi_error_avg[256:])
    axs[0][1].set_title("hi2hq_error")
    axs[0][1].plot(hi2hq_error_avg[256:])
    axs[0][2].set_title("hi2vi_error")
    axs[0][2].plot(hi2vi_error_avg[256:])
    axs[0][3].set_title("hi2vq_error")
    axs[0][3].plot(hi2vq_error_avg[256:])
    axs[1][0].set_title("vi2hi_error")
    axs[1][0].plot(vi2hi_error_avg[256:])
    axs[1][1].set_title("vi2hq_error")
    axs[1][1].plot(vi2hq_error_avg[256:])
    axs[1][2].set_title("vi2vi_error")
    axs[1][2].plot(vi2vi_error_avg[256:])
    axs[1][3].set_title("vi2vq_error")
    axs[1][3].plot(vi2vq_error_avg[256:])
    plt.tight_layout()

    # plt.figure()
    # plt.hist([error_function(x,bits_per_symbol,1,3)**2 for x in h_symbols.real], bins=256 )
    # plt.hist([error_function(x,bits_per_symbol,1,3)**2 for x in h_symbols.imag], bins=256 )

    # plt.figure()
    # plt.hist([error_function(x,bits_per_symbol,2,3)**2 for x in h_symbols.real], bins=256 )
    # plt.hist([error_function(x,bits_per_symbol,2,3)**2 for x in h_symbols.imag], bins=256 )

    # plt.figure()
    # plt.hist([error_function(x,bits_per_symbol,1,3)**2 for x in h_symbols_filtered.real], bins=256 )
    # plt.hist([error_function(x,bits_per_symbol,1,3)**2 for x in h_symbols_filtered.imag], bins=256 )

    # plt.figure()
    # plt.hist([error_function(x,bits_per_symbol,2,3)**2 for x in h_symbols_filtered.real], bins=256 )
    # plt.hist([error_function(x,bits_per_symbol,2,3)**2 for x in h_symbols_filtered.imag], bins=256 )
    # plt.hist([x for x in h_symbols_filtered.real],bins=256 )

    # plt.show()

    # plt.figure()
    # plt.title("Unfiltered Symbols & Cost Function (squared)")
    # test_range = np.arange(-2,2,0.01)
    # plt.plot(test_range, [error_function(x,bits_per_symbol,1,3)**2 for x in test_range ] )
    # plt.plot(test_range, [error_function(x,bits_per_symbol,2,3)**2 for x in test_range ] )
    # plt.hist(h_symbols.real, bins=1024, density=True)
    # plt.xlim([-2,2])
    # plt.ylim([0,4])

    # plt.figure()
    # plt.title("Filtered Symbols & Cost Function (squared)")
    # test_range = np.arange(-2,2,0.01)
    # plt.plot(test_range, [error_function(x,bits_per_symbol,1,3)**2 for x in test_range ] )
    # plt.plot(test_range, [error_function(x,bits_per_symbol,2,3)**2 for x in test_range ] )
    # plt.hist(h_symbols_filtered.real, bins=1024, density=True)
    # plt.xlim([-2,2])
    # plt.ylim([0,4])

    # plt.figure()
    # plt.plot(1 - h_symbols_filtered.real*h_symbols_filtered.real, ".")

    # h_symbols_shifted = h_symbols_filtered.real
    # for symbol_idx in range(len(h_symbols_shifted)):
    #     symbol = h_symbols_shifted[symbol_idx]
    #     if symbol > 2/3:
    #         symbol -= 2/3
    #     if symbol < -2/3:
    #         symbol += 2/3
    #     h_symbols_shifted[symbol_idx] = symbol
    # plt.figure()
    # plt.plot(1.0/9.0 - h_symbols_shifted*h_symbols_shifted, ".")

    # h_symbols_squared = h_symbols_filtered.real*h_symbols_filtered.real
    # for symbol_idx in range(len(h_symbols_squared)):
    #     symbol_squared = h_symbols_squared[symbol_idx]
    #     if symbol_squared < 5/9:
    #         symbol_squared += 8/9
    #     h_symbols_squared[symbol_idx] = symbol_squared

    # plt.figure()
    # plt.plot(1 - h_symbols_squared, ".")

    print(f"power_h = {power_h[-1]}")
    print(f"power_v = {power_v[-1]}")
    # print(f"radial_esno_h  = {radial_esno_h[-1]}")
    # print(f"radial_esno_v  = {radial_esno_v[-1]}")
    print(f"tx_rx_esno_h  = {tx_rx_esno_h[-1]}")
    print(f"tx_rx_esno_v  = {tx_rx_esno_v[-1]}")

    # fig,axs = plt.subplots(1,2)
    # fig.suptitle(f"Constellation Plots, mu = {mu}")
    # axs[0].set_title(f"h_out[{t-256}:{t}]")
    # axs[0].set_aspect("equal", "box")
    # axs[0].scatter(h_symbols_filtered[t-256:t].real, h_symbols_filtered[t-256:t].imag)
    # axs[1].set_title(f"v_out[{t-256}:{t}]")
    # axs[1].scatter(v_symbols_filtered[t-256:t].real, v_symbols_filtered[t-256:t].imag)
    # axs[1].set_aspect("equal", "box")

    # plt.show()
    # plt.show(block=False)
    # plt.pause(0.001)
        
    return (h2h,
            h2v,
            v2h,
            v2v)

def compute_dpae(h_symbols_tx, v_symbols_tx, h_symbols, v_symbols, bits_per_symbol, num_eq_filter_coefs, dpae_mu):
    num_sweeps = h_symbols.shape[0]

    h_symbols_inverted = np.zeros([len(h_symbols), len(h_symbols[0])], dtype=complex)
    v_symbols_inverted = np.zeros([len(v_symbols), len(v_symbols[0])], dtype=complex)

    compute_filters_for_each_snr = False

    if compute_filters_for_each_snr == False:
        (h2h_filter_inverted,
         h2v_filter_inverted,
         v2h_filter_inverted,
         v2v_filter_inverted) = compute_invert_filters(h_symbols_tx,
                                                       v_symbols_tx,
                                                       h_symbols[-1],
                                                       v_symbols[-1],
                                                       bits_per_symbol,
                                                       num_eq_filter_coefs,
                                                       dpae_mu)

    for snr_idx in range(len(h_symbols)):
        if compute_filters_for_each_snr == True:
            print(f"SNR = {snr_idx}")
            (h2h_filter_inverted,
             h2v_filter_inverted,
             v2h_filter_inverted,
             v2v_filter_inverted) = compute_invert_filters(h_symbols_tx,
                                                           v_symbols_tx,
                                                           h_symbols[snr_idx],
                                                           v_symbols[snr_idx],
                                                           bits_per_symbol,
                                                           num_eq_filter_coefs,
                                                           dpae_mu)

        h_symbols_inverted[snr_idx]  = np.convolve(h_symbols[snr_idx], h2h_filter_inverted, mode="same")
        h_symbols_inverted[snr_idx] += np.convolve(v_symbols[snr_idx], v2h_filter_inverted, mode="same")
        v_symbols_inverted[snr_idx]  = np.convolve(v_symbols[snr_idx], v2v_filter_inverted, mode="same")
        v_symbols_inverted[snr_idx] += np.convolve(h_symbols[snr_idx], h2v_filter_inverted, mode="same")

        if snr_idx != num_sweeps-1:
            continue

        fig,axs = plt.subplots(2,2)
        plt.suptitle("DPAE Filter Coefficients")
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

    return h_symbols_inverted, v_symbols_inverted
