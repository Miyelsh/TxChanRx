import numpy as np
import matplotlib.pyplot as plt
import helper_functions

def equalize_rx_symbols(h_symbols, v_symbols, h2h_filter, h2v_filter, v2h_filter, v2v_filter):
    (h2h_filter_inverted,
     h2v_filter_inverted,
     v2h_filter_inverted,
     v2v_filter_inverted) = invert_filters( h2h_filter
                                          , h2v_filter
                                          , v2h_filter
                                          , v2v_filter )

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

    fig,axs = plt.subplots(2,2)
    plt.suptitle("Inverted Filter Spectrum (dB), relative frequency [-pi,pi]")
    x = np.linspace(-np.pi,np.pi,len(h2h_filter_inverted))
    axs[0][0].set_title("H2H_inv")
    axs[0][0].plot(x, helper_functions.convert_linear_to_db(np.fft.fftshift(np.fft.fft(h2h_filter_inverted))))
    axs[0][1].set_title("V2H_inv")
    axs[0][1].plot(x, helper_functions.convert_linear_to_db(np.fft.fftshift(np.fft.fft(v2h_filter_inverted))))
    axs[1][0].set_title("H2V_inv")
    axs[1][0].plot(x, helper_functions.convert_linear_to_db(np.fft.fftshift(np.fft.fft(h2v_filter_inverted))))
    axs[1][1].set_title("V2V_inv")
    axs[1][1].plot(x, helper_functions.convert_linear_to_db(np.fft.fftshift(np.fft.fft(v2v_filter_inverted))))
    plt.tight_layout()

    h_symbols_inverted = np.zeros([len(h_symbols), len(h_symbols[0])], dtype=complex)
    v_symbols_inverted = np.zeros([len(v_symbols), len(v_symbols[0])], dtype=complex)

    for snr_idx in range(len(h_symbols)):
        h_symbols_inverted[snr_idx]  = np.convolve(h_symbols[snr_idx], h2h_filter_inverted, mode="same")
        h_symbols_inverted[snr_idx] += np.convolve(v_symbols[snr_idx], v2h_filter_inverted, mode="same")
        v_symbols_inverted[snr_idx]  = np.convolve(v_symbols[snr_idx], v2v_filter_inverted, mode="same")
        v_symbols_inverted[snr_idx] += np.convolve(h_symbols[snr_idx], h2v_filter_inverted, mode="same")

    return h_symbols_inverted, v_symbols_inverted

def clipped_power(x):

    power = np.abs(x*x.conj())
    # print(power)
    # breakpoint()
    if power > 2.5:
        return +1
    elif power < 1.5:
        return -1
    else:
        return 0

def compute_invert_filters(h_symbols, v_symbols, num_eq_filter_coefs):
    h2h = np.zeros(num_eq_filter_coefs, dtype=complex)
    h2v = np.zeros(num_eq_filter_coefs, dtype=complex)
    v2h = np.zeros(num_eq_filter_coefs, dtype=complex)
    v2v = np.zeros(num_eq_filter_coefs, dtype=complex)

    # Initialize H2H and V2V as all-pass, H2V and V2H as zeros
    h2h[num_eq_filter_coefs//2-1] = 1.0
    v2v[num_eq_filter_coefs//2-1] = 1.0

    if (len(h_symbols) != len(v_symbols)):
        print("ERROR: h_symbols and v_symbols must be same length!")
        exit(1)

    num_symbols = len(h_symbols)
    h_symbols_filtered = np.zeros(num_symbols, dtype=complex)
    v_symbols_filtered = np.zeros(num_symbols, dtype=complex)

    # Convolve symbols with 
    N = num_eq_filter_coefs - 1
    mu = 0.0005

    for t in range(num_symbols-num_eq_filter_coefs):
        h_in = h_symbols[t:t+num_eq_filter_coefs]
        v_in = v_symbols[t:t+num_eq_filter_coefs]

        h_out  = np.convolve(h_in, h2h, mode="valid")[0]
        h_out += np.convolve(v_in, v2h, mode="valid")[0]
        v_out  = np.convolve(v_in, v2v, mode="valid")[0]
        v_out += np.convolve(h_in, h2v, mode="valid")[0]
    
        for n in range(num_eq_filter_coefs):
            # hi2hi_error[n]   = clipped_power(h_out[t])
            #                  * hi_out[t]
            #                  * hi_in[t-n+N]
            # hi2hi_updated[n] = hi2hi[n] + hi2hi_error*mu

            # h2h_error[n]   = clipped_power(h_out[t])
            #                * h_out[t]
            #                * h_in[t-n+N]
            # h2h_updated[n] = h2h[n] + h2h_error*mu

            # h2v_error[n]   = clipped_power(v_out[t])
            #                * v_out[t]
            #                * h_in[t-n+N]
            # h2v_updated[n] = h2v[n] + h2v_error*mu

            # v2h_error[n]   = clipped_power(h_out[t])
            #                * h_out[t]
            #                * v_in[t-n+N]
            # v2h_updated[n] = v2h[n] + v2h_error*mu

            # v2v_error[n]   = clipped_power(v_out[t])
            #                * v_out[t]
            #                * v_in[t-n+N]
            # v2v_updated[n] = v2v[n] + v2v_error*mu

            h2h_error = clipped_power(h_out) * h_out * h_in[N-n]
            # print(f"h2h_power = {clipped_power(h_out)}")
            # print(f"h2h_error = {h2h_error}")
            h2h[n]    = h2h[n] + h2h_error*mu

            h2v_error = clipped_power(v_out) * v_out * h_in[N-n]
            h2v[n]    = h2v[n] + h2v_error*mu

            v2h_error = clipped_power(h_out) * h_out * v_in[N-n]
            v2h[n]    = v2h[n] + v2h_error*mu

            v2v_error = clipped_power(v_out) * v_out * v_in[N-n]
            v2v[n]    = v2v[n] + v2v_error*mu

        # Update CMA error IIR

        # Compute tap error terms


        # Save filtered symbols
        h_symbols_filtered[t] = h_out
        v_symbols_filtered[t] = v_out

        # Add tap error terms to existing filters
        if (t%10000 == 0):
            print(f"h2h = {h2h}")
            print(f"h2v = {h2v}")
            print(f"v2h = {v2h}")
            print(f"v2v = {v2v}")
            # print(f"h2h.shape = {h2h.shape}")
            fig,axs = plt.subplots(1,2)
            axs[0].set_title(f"h2h[{t}]")
            axs[0].plot(h2h.real)
            axs[0].plot(h2h.imag)
            axs[1].set_title(f"h2h_out[{t-256}:{t}]")
            axs[1].set_aspect("equal", "box")
            axs[1].scatter(h_symbols_filtered[t-256:t].real, h_symbols_filtered[t-256:t].imag)
            radial_esno = helper_functions.est_symbol_radial_esno(h_symbols_filtered[t-256:t])
            print(f"radial_esno = {radial_esno}")

            plt.show()
        
    return (h2h,
            h2v,
            v2h,
            v2v)

def compute_dpae(h_symbols, v_symbols, num_eq_filter_coefs):
    # (h2h_filter_inverted,
    #  h2v_filter_inverted,
    #  v2h_filter_inverted,
    #  v2v_filter_inverted) = invert_filters( h2h_filter
    #                                       , h2v_filter
    #                                       , v2h_filter
    #                                       , v2v_filter )

    # TODO sweep across list instead of just using last
    (h2h_filter_inverted,
     h2v_filter_inverted,
     v2h_filter_inverted,
     v2v_filter_inverted) = compute_invert_filters(h_symbols[-1],
                                                   v_symbols[-1],
                                                   num_eq_filter_coefs)


    # fig,axs = plt.subplots(2,2)
    # plt.suptitle("Inverted Filter Coefficients")
    # axs[0][0].set_title("h2h_filter_inverted")
    # axs[0][0].plot(h2h_filter_inverted.real)
    # axs[0][0].plot(h2h_filter_inverted.imag)
    # axs[0][1].set_title("v2h_filter_inverted")
    # axs[0][1].plot(v2h_filter_inverted.real)
    # axs[0][1].plot(v2h_filter_inverted.imag)
    # axs[1][0].set_title("h2v_filter_inverted")
    # axs[1][0].plot(h2v_filter_inverted.real)
    # axs[1][0].plot(h2v_filter_inverted.imag)
    # axs[1][1].set_title("v2v_filter_inverted")
    # axs[1][1].plot(v2v_filter_inverted.real)
    # axs[1][1].plot(v2v_filter_inverted.imag)
    # plt.tight_layout()

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
