## Created midi from Wav using the trained network.

from scipy.io.wavfile import read as wavread
import numpy as np
import librosa.core
import sklearn.preprocessing
from keras.models import load_model
from midiutil import MIDIFile


fileName = "bozzguitar1.wav"
sampleLength = 512
timeSteps = 516
modelName = "model_merged_network_0625.hd5"

# Cuts the Wav file so the network's predict function can be called on the output matrix.
#fileName: wav file to read
# sampleLength: calculate CQT on each sampleLength long interval.
# TimeSteps : timesteps in a single row of the output matrix.
def prepareSamplesFromWavFile(fileName, sampleLength, timeSteps):
    [sampleRate, samples] = wavread(fileName)
    cqt = librosa.core.cqt(samples.astype(np.float),sampleRate,sampleLength)[:,:-1]
    cqt = librosa.core.logamplitude(cqt).transpose()
    cqts = sklearn.preprocessing.normalize(cqt,axis=1)
    cqts = cqts.reshape(-1,timeSteps, cqts.shape[1])
    return (cqts, sampleRate)


def predictNotes(cqts, modelName):
    print("Predicting notes")
    model = load_model(modelName)
    predictions = model.predict(cqts)
    predictions = predictions.reshape(predictions.shape[0]*predictions.shape[1],predictions.shape[2])
    return predictions

def writeMidiFromPredictions(predictions, sampleRate, sampleLength,  midiFileName = "predicted_notes.midi", minMIDINumber = 40, useNumberOfNotesInfo = True):
    print("Writing MIDI file...")
    tempo = 120 # bpm
    noteDurationInSec = float(sampleLength) / float(sampleRate)
    noteDurationInBeat = noteDurationInSec / 60.0 * tempo
    MyMIDI = MIDIFile(1)
    MyMIDI.addTempo(0,0, tempo)
    for i in range(predictions.shape[0]):
        numberOfNotesVector = predictions[i,0:3]
        pitchesVector = predictions[i,4:]
        pitches = set()
        if useNumberOfNotesInfo:
            numberOfNotes = np.argmax(numberOfNotesVector)
            for n in range(numberOfNotes):
                thisNote = np.argmax(pitchesVector)
                pitches.add(thisNote + minMIDINumber)
                pitchesVector[thisNote] = -1
        else:
            pitchesVector = np.round(pitchesVector)
            noteIndices = np.nonzero(pitchesVector)[0]
            for noteIndex in noteIndices:
                pitches.add(noteIndex + minMIDINumber)

        for pitch in pitches:
             MyMIDI.addNote(0,0,pitch,noteDurationInBeat*i,noteDurationInBeat,127)
    with open(midiFileName, "wb") as output_file:
        MyMIDI.writeFile(output_file)



(cqts, sampleRate) = prepareSamplesFromWavFile(fileName, sampleLength, timeSteps)
predictions = predictNotes(cqts, modelName)
writeMidiFromPredictions(predictions, sampleRate, sampleLength,  midiFileName = "predicted_notes_using_number_of_notes.midi", useNumberOfNotesInfo = True)
writeMidiFromPredictions(predictions, sampleRate, sampleLength,  midiFileName = "predicted_notes_not_using_number_of_notes.midi", useNumberOfNotesInfo = False) 

