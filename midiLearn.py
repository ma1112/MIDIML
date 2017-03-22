from scipy.io.wavfile import read as wavread
import numpy as np
import midi
#from librosa.core import cqt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping # first saves the best model, second loggs the training, third stops the training of the result does not improve
from keras.models import load_model # to load saved model.
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


def train_model(transformed_data, pitches_onehot, lr):  # trains the model and returns the saved file's filename. (could also return the model, which is not necessarily identical to the saved one.)
	model_name = 'model.hd5'
	log_dir = 'TB_log'
	callbacks = []
	callbacks.append(TensorBoard(log_dir=log_dir)) # logs into the TB_log directory
	callbacks.append(ModelCheckpoint(filepath=model_name,   verbose=1, save_best_only=True, period=1)) # saves the model if the result is improved at the current epoch
	callbacks.append(EarlyStopping(patience = 3,verbose = 1))
	model = Sequential()
	#Maybe scaling the input? Or the activated layers?
	model.add(Dense(input_dim=transformed_data.shape[1], activation='relu', output_dim =10))
	model.add(Dense(activation='softmax', output_dim =pitches_onehot.shape[1]))
	print("Model Summary: \n" + str(model.summary()))
	sgd = SGD(lr=lr,  momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy',  metrics=['accuracy'])
	model.fit(transformed_data, pitches_onehot, nb_epoch=15, batch_size=32, shuffle=True,validation_split = 0.2, callbacks = callbacks)
	return model_name


def generate_random_filter(length_filter = 257, minsupressed = 8, maxsupressed = 26):
	# returns a random noisy high pass filter
	filter = np.ones(length_filter)
	sup_max = np.random.randint(minsupressed,maxsupressed+1)
	supress_part = np.linspace(0.25,sup_max,sup_max) * (0.5 + 0.5 * np.random.rand(sup_max))
	supress_part[np.random.randint(0,sup_max,3)] = 1

	filter[0:len(supress_part)] = supress_part
	filter = filter + np.abs( 0.8 + np.random.randn(len(filter)) /10)
	return filter
	
	


def generate_distorsed_ffts(fft_data,pitches_onehot):

#	distorted_pitches_onehot = np.matlib.repmat(pitches_onehot,num_filters,1)
	filters = np.array([generate_random_filter(fft_data.shape[1]) for i in range(fft_data.shape[0])])
	distorted_data = fft_data * filters
	return (distorted_data, pitches_onehot)
	
	


np.random.seed(13002) # for reproductivity. (fyi: '13' is 'B', '0' is 'O' and '2' is 'Z')

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
emptyArray = np.empty(int(samples.size / sampleLength))
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
        for arrayIndecies in range(beginIndex, endIndex): # maybe its endIndex +1 ? close enough....
            noteDataArray[arrayIndecies]['notes'] = note['notes']

trainData = filter(lambda x: 'notes' in x, noteDataArray)
note_samples = np.array([each['data'] for each in trainData])
pitches = np.array([each['notes'][0]['pitch'] for each in trainData])
pitch_range = [np.min(pitches),np.max(pitches)]
frequency_range = [2.0**((d-69)/12.0)*440.0 for d in pitch_range] # for cqt...
num_octave = (pitch_range[1]-pitch_range[0])/12.0 # for cqt...
pitches_onehot = np.eye(pitch_range[1]-pitch_range[0]+1)[pitches-pitch_range[0]] #one-hot encoded pitches rescaled from 0 to max(pitches)-min(pitches)

#cqt_transform = np.array([cqt(sample,fmin = frequency_range[0], bins_per_octave = 36, n_bins = int(np.ceil(num_octave*36)), sr = sampleRate, hop_length = 2 * sampleLength ) for sample in note_samples]) # I have no idea why we need the hop_length to be 2* sampleLength...
fourier_transform = np.abs(np.fft.rfft(note_samples))


# Crating a lot of training examples by distorting the input.
print('Distorting training data') # for debug 
(fourier_transform, pitches_onehot) = generate_distorsed_ffts(fourier_transform, pitches_onehot)

# Splitting the dataset to train (inc. validation) and test set.
test_split = 0.15
test_num = int(np.ceil(test_split * fourier_transform.shape[0]))
test_indexes = np.random.randint(fourier_transform.shape[0],size = (test_num,))
test_pitches_onehot = pitches_onehot[test_indexes]
test_fourier_transform = fourier_transform[test_indexes]

pitches_onehot = np.delete(pitches_onehot,test_indexes,0)
fourier_transform = np.delete(fourier_transform,test_indexes,0)

#Train  ( chu - chu )
print("Training")
trained_model_path = train_model(fourier_transform,pitches_onehot, 0.005)

#Load saved model and test on the test data
print("Reloadig the model from the hard drive and testing it using the test dataset.")
model = load_model(trained_model_path)
test_accuracy = model.evaluate(test_fourier_transform,test_pitches_onehot)[1] # returns the loss and accuracy in this order.
print("\nTest accuracy: " + str(test_accuracy))

#Calculating the accuracy after using predict() function, which should has the same result

test_predictions = model.predict(test_fourier_transform)
test_prediction_midinote = np.argmax(test_predictions,1) # not actual midi note, the lowest note in the training set is at the 0-th index. Add pitch_range[0] to get the actual midi note.
test_real_midinote = np.argmax(test_pitches_onehot,1) # not actual midi note, but has the same scaling as test_prediction_midinote.
test_matching_notes = np.sum(test_real_midinote == test_prediction_midinote)
test_accuracy_by_hand = float(test_matching_notes) / test_num
print("Test accuracy calculated using the  predicted notes " + str(test_accuracy_by_hand))

if np.abs(test_accuracy - test_accuracy_by_hand) <0.01:
	print("Test accuracy is indeed the same when using built-in evaluation or predicting the notes and evaluating ourselves. Math works then...")
else: # should never ever happen
	print("Test accuracy was not the same using built-in evaluation and predicting the accuracy ourselves. What could happen?") 
print("end")
print("See the training log by executing $ tensorboard --logdir='.'")
