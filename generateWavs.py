from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
import numpy as np
import os.path


# Calculations to determine length of kernel at running_mean for a given cut-off frequency.
def calcH(n, f, fs): # calculates transfer function
    w = 2 * np.pi * f / fs
    h = 1.0 / n * np.sin(w * n / 2) / np.sin(w / 2)
    return h


# gets filter width
def searchN(f, H, fs):
    for n in range(1, 1000):
        HThis = calcH(n, f, fs)
        if HThis < H:
            break
    if np.abs(H - calcH(n, f, fs)) > np.abs(H - calcH(n - 1, f, fs)):
        return n - 1
    else:
        return n

# gets filter width for various frequencies.
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
    padded = np.zeros((len(x)+windowSize-1,))
    padded[int(np.floor((windowSize-1)/2)):-int(np.ceil((windowSize-1)/2))] = x # pads input with 0-s, so output will have the same lengthas input
    cumsum = np.cumsum(padded,dtype = float)
    cumsum[windowSize:] = cumsum[windowSize:] - cumsum[:-windowSize]
    return cumsum[windowSize-1:] / windowSize


# scale to -1.0 -- 1.0
# if x.dtype == 'int16':
#     nb_bits = 16  # -> 16-bit wav files
# elif x.dtype == 'int32':
#     nb_bits = 32  # -> 32-bit wav files
# max_nb_bit = float(2 ** (nb_bits - 1))

# x = x / (max_nb_bit + 1.0)  # samples is a numpy array of float representing the samples
# wavwrite('noisedfiltered1.wav',22050, x)


def getFilteredDataList(inputFileName, load = True, save = True):
    frequencyFilterWidthMap = getNs()
    result = []
    [originalSampleRate, original] = wavread(inputFileName)
    for cutOffFrequency in frequencyFilterWidthMap:
        outputFileName = inputFileName + '_noisedfiltered_' + str(cutOffFrequency) + '.wav'
        if os.path.isfile(outputFileName) and load:
            print("Loading file ", outputFileName, " from disk." )
            [sampleRate, x] = wavread(outputFileName)
            if sampleRate != originalSampleRate:
                raise ValueError("Sample rate of file ", outputFileName, " does not eaqual the sample rate of",
                                                                         " the original file " , inputFileName)
        else:
            windowSize =  frequencyFilterWidthMap[cutOffFrequency]
            print('Generating noisedfiltered ', cutOffFrequency, ' data, with windowSize ', windowSize)
            x = running_mean(original, windowSize).astype('int16')
            x = x + np.random.normal(0, 10, len(x)).astype('int16') # to add noise.
            if save:
                wavwrite(outputFileName, originalSampleRate, x)
                print("saved: ", outputFileName)
        if len(x) != len(original):
            raise ValueError("Filtering the wav file failed. Original input is ", len(original), " long",
                             "but the filtered data is " , len(x) , " long.")
        result.append(x)
    return (result ,originalSampleRate)
