from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
import numpy as np


def calcH(n, f, fs):
    w = 2 * np.pi * f / fs
    h = 1.0 / n * np.sin(w * n / 2) / np.sin(w / 2)
    return h


def searchN(f, H, fs):
    for n in range(1, 1000):
        HThis = calcH(n, f, fs)
        if HThis < H:
            break
    if np.abs(H - calcH(n, f, fs)) > np.abs(H - calcH(n - 1, f, fs)):
        return n - 1
    else:
        return n


def getNs(fs=22050, H=0.5, fArray=np.linspace(100, 5000, 50), verbose=0):
    resultDict = {}
    for i in range(len(fArray)):
        f = fArray[i]
        n = searchN(f, H, fs)
        if verbose:
            print('f: ' + str(f) + '\tN: ' + str(n) + '\t real H: ' + str(calcH(n, f, fs)))
        if n not in resultDict.values():
            resultDict[f] = n
    return resultDict


def running_mean(x, windowSize):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize


# scale to -1.0 -- 1.0
# if x.dtype == 'int16':
#     nb_bits = 16  # -> 16-bit wav files
# elif x.dtype == 'int32':
#     nb_bits = 32  # -> 32-bit wav files
# max_nb_bit = float(2 ** (nb_bits - 1))

# x = x / (max_nb_bit + 1.0)  # samples is a numpy array of float representing the samples
# wavwrite('noisedfiltered1.wav',22050, x)

def getFilteredDataList(save = False):
    asd = getNs()
    result = []
    [sampleRate, original] = wavread("midiTeszt22.05k.wav")
    for key in asd:
        x = running_mean(original, asd[key]).astype('int16')
        x = x + np.random.normal(0, 10, len(x)).astype('int16') # to add noise.
        if save:
            string = 'noisedfiltered' + str(key) + '.wav'
            wavwrite(string, 22050, x)
            print(string)
        result.append(x)
    return result
