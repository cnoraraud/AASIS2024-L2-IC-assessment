import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fft as fft
from scipy.io import wavfile
import numpy as np
import filtering as filt
import io_tools as iot
import naming_tools as nt
import numpy_wrapper as npw
import traceback

def list_wavs():
    wavs = []
    for name in iot.list_dir(iot.wavs_path(), "wav"):
        wavs.append(name)
    return wavs

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
    ms_proposals = [-1]
    for wav in wavs:
        data = wav["data"]
        sr = wav["sr"]
        R = int(sr)
        H = round(R / 1000)
        W = round(data.shape[0] / H)
        ms_proposals.append(W)
    ms_proposals = set(ms_proposals)
    if len(ms_proposals) != 1:
        print(f"Invalid number of ms proposals: {len(ms_proposals)}")
    ms = max(list(ms_proposals))
    if ms <= 0: ms = None
    return ms

def data_to_ms(data, ms, samples_per_ms, aggregate='mean'):
    data_stack = data.reshape(ms, samples_per_ms)
    data_ms = None
    if aggregate == 'mean':
        data_ms = np.mean(data_stack, axis=1)
    if aggregate == 'energy':
        data_ms = np.sum(np.abs(data_stack), axis=1) / samples_per_ms
    return data_ms

def wavs_to_ms(wavs, window_n = 1000, first_cep = 1, num_ceps = 12, left_mics = ["mic1"], right_mics = ["mic2"], gen_mics = ["mic3", "mic4", "mic5"]):
    datas = []
    labels = []

    hop = -1
    width = -1
    for wav in wavs:
        data = wav["data"]
        sr = wav["sr"]
        hop = max(hop, round(sr / window_n))
        width = max(width, round(data.shape[0] / hop))
        
    # SC, MC, Outcome
    #  1,  1, - wav belongs to single speaker
    #  2,  2, - shared microphone
    #  2,  1, - depends on microphone name
    #         * mic3, mic4, mic5: wav belongs to [all]
    #         * mic1, mic2: wav belongs to speaker 1 or speaker 2
    for wav in wavs:
        data = wav["data"]
        sr = wav["sr"]
        speakers = wav["speakers"]
        mics = wav["mics"]
        SC = len(speakers)
        MC = len(mics)

        if SC == 1 and MC == 1:
            speaker = speakers[0]
        elif SC == 2 and MC == 2:
            speaker = nt.ALL_SOURCE
        elif SC == 2 and MC == 1:
            mic = mics[0]
            if mic in left_mics:
                speaker = min(speakers)
            elif mic in right_mics:
                speaker = max(speakers)
            elif mic in gen_mics:
                speaker = nt.ALL_SOURCE
            else:
                raise ValueError(f"Unknown mic {mic}. Don't know how to assign it to sources ({SC}).")
        else:
            raise ValueError(f"Strange speaker ({SC}) and mic counts ({MC}). {speakers} {mics}")


        data = filt.norm(data)
        data = stereo_to_mono(data)
        ceps = data_to_mfcc(data, window_n, hop, sr, first_cep = first_cep, num_ceps = num_ceps)
        energy = data_to_ms(data, width, hop, aggregate='energy')
        vad = energy_to_vad(energy)

        for i in range(num_ceps):
            datas.append(ceps[:, i])
            cep_name = f"cep:{i + first_cep}"
            labels.append(nt.create_label(speaker, nt.EXTRACTION_TAG, cep_name))
        datas.append(energy)
        labels.append(nt.create_label(speaker, nt.EXTRACTION_TAG, "energy"))
        datas.append(vad)
        labels.append(nt.create_label(speaker, nt.EXTRACTION_TAG, "vad"))

    return np.array(datas).T, npw.string_array(labels)

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

def energy_to_vad(data):
    dbs = DB(filt.ma(data, {"n": 300}))
    mean_db = np.mean(dbs)
    std_db = np.std(dbs)
    th_db = mean_db + std_db
    ms = 2000
    ms_th = 0.15
    return filt.ma(dbs > th_db, {"n": ms}) > ms_th

def resample_wav(wav_name, sr):
    sr0, data0 = wavfile.read((iot.wavs_path() / wav_name).as_posix())
    t0 = np.shape(data0)[0]
    sr1 = sr
    t1 = round(sr1/sr0 * t0)
    data1 = sig.resample(data0, t1) # design choice
    iot.create_wavs_sr_folder(sr)
    wavfile.write((iot.wavs_path(sr) / wav_name).as_posix(), sr1, data1)
    print(f"Resampled from {t0} to {t1}")
    return sr1, data1

def read_wav(wav_path, sr):
    wav_name = nt.file_swap(nt.get_name(wav_path),"wav")
    data = None
    if (iot.wavs_path(sr) / wav_name).exists():
        new_sr, data = wavfile.read((iot.wavs_path(sr) / wav_name).as_posix())
    else:
        new_sr, data  = resample_wav(wav_name, sr)
    return data, new_sr

def read_wavs(wav_paths, sr):
    wavs = []
    for wav_path in wav_paths:
        try:
            data, new_sr = read_wav(wav_path, sr=sr)
            speakers = nt.speakers_to_sources(nt.find_speakers(wav_path))
            mics = nt.find_mics(wav_path)
            wav = {"data": data,
                    "sr": new_sr,
                    "speakers": speakers,
                    "mics": mics}
            wavs.append(wav)
        except Exception as e:
            print(f"Could not read {wav_path} due to {e}")
            print(traceback.format_exc())
    return wavs

def DB(data):
    return 20*np.log10(data)