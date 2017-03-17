from scipy.io.wavfile import read as wavread
import numpy as np
import midi


# t = sampleNum / samplerate = (tick /resoltuion) * (60/bpm), index = sampleNum / sampleLength
# => index = tick * (1/resoltuion)* (60/bpm) * (sampleRate )* (sampleLength)
# resoltion is ticks per beat


def tickToSampleIndex(tick, resolution, bpm, sampleRate, sampleLength):
    return float((tick) / float(resolution)) * (float(60) / float(bpm)) * (float(sampleRate) / float(sampleLength))


def createNoteTimeline(midiTrack):
    noteTimeline = []
    currentNotes = []
    currentTime = 0
    for index, item in enumerate(midiTrack):
        if (item.name != 'End of Track'):
            if (item.name == 'Note On'):
                currentNotes.append({'pitch': item.pitch, 'velocity': item.velocity})
            if (item.name == 'Note Off'):
                currentNotes.remove({'pitch': item.pitch, 'velocity': item.velocity})
            currentTime += item.tick
            endTime = currentTime + midiTrack[index+1].tick - 1
            noteTimeline.append({'notes': currentNotes[:], 'beginTimeInTicks': currentTime, 'endTimeInTicks': endTime})
    return noteTimeline


pattern = midi.read_midifile("MIDIproba.mid")
noteTimeLine = createNoteTimeline(pattern[1])

[sampleRate, x] = wavread("midiTeszt22.05k.wav")
# used variables
sampleLength = 512
resolution = pattern.resolution
bpm = 120  # this depends on sound rendering/recording bpm

# scale to -1.0 -- 1.0
if x.dtype == 'int16':
    nb_bits = 16  # -> 16-bit wav files
elif x.dtype == 'int32':
    nb_bits = 32  # -> 32-bit wav files
max_nb_bit = float(2 ** (nb_bits - 1))

samples = x / (max_nb_bit + 1.0)  # samples is a numpy array of float representing the samples
emptyArray = np.empty(int(samples.size / 512))
noteDataArray= [{'index': i, 'data': samples[i * sampleLength: (i + 1) * sampleLength]} for i, x in
                           enumerate(emptyArray)]  # could be ordered dict but i dont care

percent = 0
for index, note in enumerate(noteTimeLine): # fill up the noteDataArray's elements with notes
    if len(note['notes']):
        newPercent = int(float(index)*100.0/float(len(noteTimeLine)))
        if newPercent != percent:
            percent = newPercent
        beginIndex = int(np.ceil(tickToSampleIndex(note['beginTimeInTicks'], resolution, bpm, sampleRate, sampleLength)))
        endIndex = int(round(tickToSampleIndex(note['endTimeInTicks'], resolution, bpm, sampleRate, sampleLength)))
        for arrayIndecies in range(beginIndex, endIndex):
            noteDataArray[arrayIndecies]['notes'] = note['notes']

trainData = filter(lambda x: 'notes' in x, noteDataArray)
print("end")
