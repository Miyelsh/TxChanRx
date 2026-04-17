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

def error_function(i,q):
    return (1 - (i*i))

    # return (2 - (i*i + q*q))

    power = i*i + q*q
    # power = i*i
    if power > 2.0:
        return -1
    elif power < 2.0:
        return +1
    else:
        return 0


def compute_invert_filters(h_symbols, v_symbols, num_eq_filter_coefs):
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
    mu = 0.0001

    radial_power_h = np.zeros(len(h_symbols)-num_eq_filter_coefs)
    radial_esno_h  = np.zeros(len(h_symbols)-num_eq_filter_coefs)
    radial_power_v = np.zeros(len(v_symbols)-num_eq_filter_coefs)
    radial_esno_v  = np.zeros(len(v_symbols)-num_eq_filter_coefs)

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

        radial_power_h[t] = helper_functions.est_symbol_power(h_symbols_filtered[t-256:t])
        radial_esno_h[t] = helper_functions.est_symbol_radial_esno(h_symbols_filtered[t-256:t])
        radial_power_v[t] = helper_functions.est_symbol_power(v_symbols_filtered[t-256:t])
        radial_esno_v[t] = helper_functions.est_symbol_radial_esno(v_symbols_filtered[t-256:t])

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
            print(f"radial_power_h = {radial_power_h}")
            print(f"radial_esno_v  = {radial_esno_v}")
            print(f"radial_power_v = {radial_power_v}")

            plt.show()

        if t < 256:
            continue
    
        # Add tap error terms to existing filters
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

            # h2h_error = clipped_power(h_out) * h_out.conj() * h_in[N-n]
            # # print(f"h2h_power = {clipped_power(h_out)}")
            # # print(f"h2h_error = {h2h_error}")
            # # print(f"h_in = {h_in[0]}")
            # # print(f"h_out = {h_out}")
            # # print(f"h_in*h_out.conj() = {h_in*h_out.conj()}")
            # h2h[n]    = h2h[n] + h2h_error*mu

            # h2v_error = clipped_power(v_out) * v_out.conj() * h_in[N-n]
            # h2v[n]    = h2v[n] + h2v_error*mu
            # # print(f"h2v_error = {h2v_error}")
            # # print(f"h_in = {h_in[0]}")
            # # print(f"v_out = {v_out}")
            # # print(f"h_in*v_out.conj() = {h_in*v_out.conj()}")

            # v2h_error = clipped_power(h_out) * h_out.conj() * v_in[N-n]
            # v2h[n]    = v2h[n] + v2h_error*mu

            # v2v_error = clipped_power(v_out) * v_out.conj() * v_in[N-n]
            # v2v[n]    = v2v[n] + v2v_error*mu

            # hi2hi_error = (1 - (hi_out*hi_out)) \
            # hi2hi_error = (2 - (hi_out*hi_out + hq_out*hq_out)) \
            hi2hi_error = error_function(hi_out, hq_out) \
                        * hi_out \
                        * hi_in[N-n]
            hi2hi[n]    = hi2hi[n] + hi2hi_error*mu

            # hi2hq_error = (1 - (hq_out*hq_out)) \
            # hi2hq_error = (2 - (hi_out*hi_out + hq_out*hq_out)) \
            hi2hq_error = error_function(hq_out, hi_out) \
                        * hq_out \
                        * hi_in[N-n]
            hi2hq[n]    = hi2hq[n] + hi2hq_error*mu

            # hi2vi_error = (1 - (vi_out*vi_out)) \
            # hi2vi_error = (2 - (vi_out*vi_out + vq_out*vq_out)) \
            hi2vi_error = error_function(vi_out, vq_out) \
                        * vi_out \
                        * hi_in[N-n]
            hi2vi[n]    = hi2vi[n] + hi2vi_error*mu

            # hi2vq_error = (1 - (vq_out*vq_out)) \
            # hi2vq_error = (2 - (vi_out*vi_out + vq_out*vq_out)) \
            hi2vq_error = error_function(vq_out, vi_out) \
                        * vq_out \
                        * hi_in[N-n]
            hi2vq[n]    = hi2vq[n] + hi2vq_error*mu

            # vi2hi_error = (1 - (hi_out*hi_out)) \
            # vi2hi_error = (2 - (hi_out*hi_out + hq_out*hq_out)) \
            vi2hi_error = error_function(hi_out, hq_out) \
                        * hi_out \
                        * vi_in[N-n]
            vi2hi[n]    = vi2hi[n] + vi2hi_error*mu

            # vi2hq_error = (1 - (hq_out*hq_out)) \
            # vi2hq_error = (2 - (hi_out*hi_out + hq_out*hq_out)) \
            vi2hq_error = error_function(hq_out, hi_out) \
                        * hq_out \
                        * vi_in[N-n]
            vi2hq[n]    = vi2hq[n] + vi2hq_error*mu

            # vi2vi_error = (1 - (vi_out*vi_out)) \
            # vi2vi_error = (2 - (vi_out*vi_out + vq_out*vq_out)) \
            vi2vi_error = error_function(vi_out, vq_out) \
                        * vi_out \
                        * vi_in[N-n]
            vi2vi[n]    = vi2vi[n] + vi2vi_error*mu

            # vi2vq_error = (1 - (vq_out*vq_out)) \
            # vi2vq_error = (2 - (vi_out*vi_out + vq_out*vq_out)) \
            vi2vq_error = error_function(vq_out, vi_out) \
                        * vq_out \
                        * vi_in[N-n]
            vi2vq[n]    = vi2vq[n] + vi2vq_error*mu

            h2h[n] = complex(hi2hi[n], hi2hq[n])
            h2v[n] = complex(hi2vi[n], hi2vq[n])
            v2h[n] = complex(vi2hi[n], vi2hq[n])
            v2v[n] = complex(vi2vi[n], vi2vq[n])

        # breakpoint()

        # Update CMA error IIR

        # Compute tap error terms

    fig,axs = plt.subplots(1,2)
    fig.suptitle(f"Power and EsNo, mu = {mu}")
    axs[0].set_title("Radial Power")
    axs[0].plot(radial_power_h)
    axs[0].plot(radial_power_v)
    axs[1].set_title("Radial EsNo")
    axs[1].plot(radial_esno_h)
    axs[1].plot(radial_esno_v)

    # fig,axs = plt.subplots(1,2)
    # fig.suptitle(f"Constellation Plots, mu = {mu}")
    # axs[0].set_title(f"h_out[{t-256}:{t}]")
    # axs[0].set_aspect("equal", "box")
    # axs[0].scatter(h_symbols_filtered[t-256:t].real, h_symbols_filtered[t-256:t].imag)
    # axs[1].set_title(f"v_out[{t-256}:{t}]")
    # axs[1].scatter(v_symbols_filtered[t-256:t].real, v_symbols_filtered[t-256:t].imag)
    # axs[1].set_aspect("equal", "box")

    # plt.show()
        
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
