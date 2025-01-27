import sys
import numpy as np
from scipy.io import wavfile
import scipy.signal as sig
def resample_wavs(wav_path, wav_path_sr, sr):
    sr0, data0 = wavfile.read(wav_path.as_posix())
    t0 = np.shape(data0)[0]
    sr1 = sr
    t1 = round(sr1/sr0 * t0)
    data1 = sig.resample(data0, t1) # design choice
    print(f"Resampled from {t0} to {t1}")
    wavfile.write(wav_path_sr.as_posix(), sr1, data1)
    return sr1, data1

if __name__ == '__main__':
    globals()[sys.argv[1]]()