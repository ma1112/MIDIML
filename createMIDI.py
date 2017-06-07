from midiutil import MIDIFile
import itertools
import numpy as np
import pandas as pd

degrees  = range(40,73)  # MIDI note number. Sould be positive. [Silence is added by the code]
track    = 0
maxNotesTogether = 3 # Determines maximal number of notes played together.
midiTime     = 0    # In beats
noteDuration = 4    # In beats
tempo    = 120   # In BPM
volumes =  np.linspace(20,120,3).astype(int)
#volume   = 19  # 0-127, as per the MIDI standard

# Returns a matrix with shape (?,maxNotesTogether+1), where the first element of the -1 axis is: number of notes played
# at the row, then this ammount of notes are present in the matrix in desc order. Other values are 0.
def getNoteCombinations(degrees, maxNotesTogether):
    result = np.zeros((1,maxNotesTogether+1)) # last dimension: how many notes are not zero, and then the notes.
    #result already includes all-0 row. (silecnce).
    for numberOfNotes in range(1,maxNotesTogether+1):
        thisCombination = getNoteCombinationForGivenNumberOfNotes(numberOfNotes,degrees)
        #Extends the matrix with 0-s so each have the same size in axis -1
        thisRightSize = np.zeros((thisCombination.shape[0],maxNotesTogether+1))
        thisRightSize[:,1:1+numberOfNotes] = thisCombination
        thisRightSize[:,0] = numberOfNotes
        result = np.vstack((result,thisRightSize))
    return result.astype(int)

#Returns a matrix with shape (?,numberOfNotes). -1 axis shows which notes are played in the current row.
def getNoteCombinationForGivenNumberOfNotes(numberOfNotes, degrees):
    print("Getting combination matrix with " , numberOfNotes , " notes played together.")
    if numberOfNotes ==1:
        combinations = np.zeros((len(degrees),1))
        combinations[:,0] = degrees
    else:
        combinations = itertools.product(degrees,repeat = numberOfNotes)
        combinations = np.flip(np.sort(np.asarray(list(combinations)),axis=-1),-1) # desc sort.
        combinations = pd.DataFrame(combinations)
        combinations = combinations.drop_duplicates()
        combinations = combinations[combinations.apply(lambda x:  len(np.unique(x)) == len(x), 1)] # keeps rows where values are different ( or zero only)
        combinations = combinations.values
    return combinations # returns an array where each row is different, and shows which note should be played. Size is (?,numberOfNotes)

#Gets cobination matrix ( a kind of extended piano-roll)
combinationMatrix = getNoteCombinations(degrees,maxNotesTogether)

#creates MIDI file


#Actually generates MIDI music
print("Generating MIDI from matrix.")
for volume in volumes: # Same volume for every notes. TODO: can be imrpoved?
    MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
    # automatically)
    MyMIDI.addTempo(track, midiTime, tempo)
    print("Generating Midi with volume " , volume)
    numRows = combinationMatrix.shape[0]
    for rowNumber in range(numRows): # for each row in the matrix ,i.e. for each pitch combination
        numPitches = combinationMatrix[rowNumber,0] # how many pithces are actually there in the current combination
        for numPitch in range(numPitches): # for each pitch, add the pitch to the same timestamp
            MyMIDI.addNote(track, numPitch, combinationMatrix[rowNumber,numPitch+1], noteDuration   * rowNumber, noteDuration, volume)
    #saves output
    with open("MIDIproba_volume_" + str(volume) + ".mid", "wb") as output_file:
        MyMIDI.writeFile(output_file)
np.save("combinationMatrix.npy",combinationMatrix)
print("Created and saved MIDI files and combination matrix with ", combinationMatrix.shape[0], " different note combinations.")
