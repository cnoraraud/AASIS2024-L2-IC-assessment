import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fft as fft
import numpy as np

def stereo_to_mono(data):
    # For some audios, one of the channels might be usable for a noise profile
    # At the moment just adding the two channels together
    return np.sum(data, axis=1)

def print_wav(wav):
    sr, data, speakers, mics = wav
    data = stereo_to_mono(data)
    data = sig.wiener(data)
    plt.plot(data)
    plt.title(f"sr: {sr} sec: {data.shape[0]/sr} speakers: {speakers} mics: {mics}")
    print(len(speakers))
    plt.show()

def normalize(data):
    data_c = np.mean(data)
    c_data = data - data_c
    c_data_s = np.max(np.abs(c_data))
    cs_data = c_data / c_data_s
    return cs_data

def data_to_mfcc(data, window_n, hop_n, rate, first_cep = 1, num_ceps = 12):
    pad_n = window_n - hop_n
    bg = np.mean(data)
    padded_data = np.pad(data, (pad_n, 0), mode='constant', constant_values=(bg, bg))
    total_n = padded_data.shape[0]
    window = sig.windows.hann(window_n, sym=True)
    fbank = get_fbank(window_n, rate)
    
    mfccs = []
    i = 0
    while i + window_n <= total_n:
        frame = padded_data[i : i + window_n]
        frame = frame * window
        dct = fft.dct(frame, n=window_n)
        magnitude = np.absolute(dct)
        power = np.power(magnitude, 2) * (1 / window_n)
        filter_banks = get_filterbanks(power, fbank)
        mfcc = fft.dct(filter_banks, type=2, norm='ortho')[first_cep : num_ceps + first_cep]
        mfccs.append(mfcc)
        i += hop_n
    return np.array(mfccs)

def find_wavs_ms(wavs):
    ms_proposals = []
    for wav in wavs:
        sr, data_raw, speakers, mics = wav
        R = sr
        H = round(R / 1000)
        W = round(data_raw.shape[0] / H)
        ms_proposals.append(W)
    ms_proposals = set(ms_proposals)
    if len(ms_proposals) != 1:
        print(f"Invalid number of ms proposals: {len(ms_proposals)}")
    ms = list(ms_proposals)[0]
    return ms

def data_to_ms(data, ms, samples_per_ms, aggregate='mean'):
    data_stack = data.reshape(ms, samples_per_ms)
    data_ms = None
    if aggregate == 'mean':
        data_ms = np.mean(data_stack, axis=1)
    if aggregate == 'energy':
        data_ms = np.sum(np.abs(data_stack), axis=1) / samples_per_ms
    return data_ms

def wavs_to_ms(wavs, ms):
    #ms ignored currently
    R = None
    H = None
    W = None
    N = len(wavs)

    datas = []
    labels = []
    
    for wav in wavs:
        sr, data_raw, speakers, mics = wav
        if R is None:
            R = sr
        if H is None:
            H = round(R / 1000)
        if W is None:
            W = round(data_raw.shape[0] / H)

        if len(speakers) > 1 and len(mics) > 1:
            continue

        name = "-".join(speakers)
        if (len(speakers) > 1):
            name += f"mic:{mics[0]}"
    
        data = normalize(data_raw)
        data = stereo_to_mono(data)

        window_n = 1000
        hop_n = H

        first_cep = 1
        num_ceps = 12
        ceps = data_to_mfcc(data, window_n, hop_n, sr, first_cep = first_cep, num_ceps = num_ceps)
        energy = data_to_ms(data, W, H, aggregate='energy')
        vad = energy_to_vad(energy)

        for i in range(num_ceps):
            datas.append(ceps[:, i])
            labels.append(f"{name}_cep:{i + first_cep}")
        datas.append(energy)
        datas.append(vad)
        labels.append(f"{name}_energy")
        labels.append(f"{name}_vad")

    return np.array(datas).T, labels

def get_fbank(nfft, rate, nfilt = 40):
    # Cite for MFCC: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bins = np.floor((nfft + 1) * hz_points / rate)
    
    fbank = np.zeros((nfilt, nfft))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bins[m - 1])   # left
        f_m = int(bins[m])             # center
        f_m_plus = int(bins[m + 1])    # right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    return fbank

def get_filterbanks(power, fbank):
    # Cite for MFCC: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    filter_banks = np.dot(power, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    return filter_banks

def ma(a, n) :
    bg = np.mean(a)
    return np.convolve(np.pad(a, (n//2, n - n//2 - 1), mode='constant', constant_values=(bg, bg)), np.ones(n), 'valid') / n

def energy_to_vad(data):
    dbs = DB(ma(data, 300))
    mean_db = np.mean(dbs)
    std_db = np.std(dbs)
    th_db = mean_db + std_db
    ms = 2000
    ms_th = 0.15
    return ma(dbs > th_db, ms) > ms_th

def DB(data):
    return 20*np.log10(data)