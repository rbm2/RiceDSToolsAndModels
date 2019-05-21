import numpy as np
import tensorflow as tf

# the number of iterations to train for
numTrainingIters = 10000

# the number of hidden neurons that hold the state of the RNN
hiddenUnits = 1000

# the number of classes that we are learning over
numClasses = 5

# the number of data points in a batch
batchSize = 100

# the learning rate
learningRate = 0.02

# this function takes a dictionary (called data) which contains 
# of (dataPointID, (classNumber, matrix)) entries.  Each matrix
# is a sequence of vectors; each vector has a one-hot-encoding of
# an ascii character, and the sequence of vectors corresponds to
# one line of text.  classNumber indicates which file the line of
# text came from.  
# 
# The argument maxSeqLen is the maximum length of a line of text
# seen so far.  fileName is the name of a file whose contents
# we want to add to data.  classNum is an indicator of the class
# we are going to associate with text from that file.  linesToUse
# tells us how many lines to sample from the file.
#
# The return val is the new maxSeqLen, as well as the new data
# dictionary with the additional lines of text added
def addToData (maxSeqLen, data, fileName, classNum, linesToUse):
    #
    # open the file and read it in
    with open(fileName) as f:
        content = f.readlines()
    #
    # sample linesToUse numbers; these will tell us what lines
    # from the text file we will use
    # [Note] random_integers genetate a vector with size "linesToUse", rand from 0 to len(content)
    myInts = np.random.randint (0, len(content) - 1, linesToUse)
    #
    # i is the key of the next line of text to add to the dictionary
    # [Note] dictionary is called "data" in this case, so i is the length of dictionary
    i = len(data)
    #
    # loop thru and add the lines of text to the dictionary
    for whichLine in myInts.flat: # myInts.flat is a 1-D interator over myInts
        #
        # get the line and ignore it if it has nothing in it
        line = content[whichLine]
        if line.isspace () or len(line) == 0:
            continue;
        #
        # take note if this is the longest line we've seen
        if len (line) > maxSeqLen:
            maxSeqLen = len (line)
        #
        # create the matrix that will hold this line
        temp = np.zeros((len(line), 256))
        #
        # j is the character we are on
        j = 0
        # 
        # loop thru the characters
        for ch in line:
            #
            # non-ascii? ignore
            if ord(ch) >= 256: # ord(c) gives the unicode of c
                continue
            #
            # one hot!
            temp[j][ord(ch)] = 1 # mark the ascii index 
            # 
            # move onto the next character
            j = j + 1
            #
        # remember the line of text
        # add this (class number, matrix_of_line) to end of data (dictionary)
        data[i] = (classNum, temp)
        #
        # move onto the next line
        i = i + 1
    #
    # and return the dictionary with the new data
    return (maxSeqLen, data) # (max length of the line in file, and the dictionary)

# this function takes as input a data set encoded as a dictionary
# (same encoding as the last function) and pre-pends every line of
# text with empty characters so that each line of text is exactly
# maxSeqLen characters in size
def pad (maxSeqLen, data):
   #
   # loop thru every line of text
   for i in data:
        #
        # access the matrix and the label
        temp = data[i][1]
        label = data[i][0]
        # 
        # get the number of chatacters in this line
        len = temp.shape[0]
        #
        # and then pad so the line is the correct length
        padding = np.zeros ((maxSeqLen - len,256)) 
        data[i] = (label, np.transpose (np.concatenate ((padding, temp), axis = 0)))
   #
   # return the new data set
   return data

# this generates a new batch of training data of size batchSize from the
# list of lines of text data. This version of generateData is useful for
# an RNN because the data set x is a NumPy array with dimensions
# [batchSize, 256, maxSeqLen]; it can be unstacked into a series of
# matrices containing one-hot character encodings for each data point
# using tf.unstack(inputX, axis=2)
def generateDataRNN (maxSeqLen, data):
    #
    # randomly sample batchSize lines of text
    myInts = np.random.randint (0, len(data) - 1, batchSize)
    #
    # stack all of the text into a matrix of one-hot characters
    x = np.stack (data[i][1] for i in myInts.flat)
    #
    # and stack all of the labels into a vector of labels
    y = np.stack (np.array((data[i][0])) for i in myInts.flat)
    #
    # return the pair
    return (x, y)

# this also generates a new batch of training data, but it represents
# the data as a NumPy array with dimensions [batchSize, 256 * maxSeqLen]
# where for each data point, all characters have been appended.  Useful
# for feed-forward network training
def generateDataFeedForward (maxSeqLen, data):
    #
    # randomly sample batchSize lines of text
    myInts = np.random.randint (0, len(data) - 1, batchSize)
    #
    # stack all of the text into a matrix of one-hot characters
    x = np.stack (data[i][1].flatten () for i in myInts.flat) # flatten turns matrix into 1-D form
    #
    # and stack all of the labels into a vector of labels
    y = np.stack (np.array((data[i][0])) for i in myInts.flat)
    #
    # return the pair
    return (x, y)

# create the data dictionary
maxSeqLen = 0
data = {}

# load up the five data sets
(maxSeqLen, data) = addToData (maxSeqLen, data, "biochemistry_processed.txt", 0, 10000)
(maxSeqLen, data) = addToData (maxSeqLen, data, "cancerResearch_processed.txt", 1, 10000)
(maxSeqLen, data) = addToData (maxSeqLen, data, "jama_processed.txt", 2, 10000)
(maxSeqLen, data) = addToData (maxSeqLen, data, "nature_processed.txt", 3, 10000)
(maxSeqLen, data) = addToData (maxSeqLen, data, "plosOne_processed.txt", 4, 10000)

# pad each entry in the dictionary with empty characters as needed so
# that the sequences are all of the same length
data = pad (maxSeqLen, data)
        
# now we build the TensorFlow computation... there are two inputs, 
# a batch of text lines and a batch of labels
inputX = tf.placeholder(tf.float32, [batchSize, 256, maxSeqLen])
inputY = tf.placeholder(tf.int32, [batchSize])

# this is the inital state of the RNN, before processing any data
initialState = tf.placeholder(tf.float32, [batchSize, hiddenUnits])

# the weight matrix that maps the inputs and hidden state to a set of values
W = tf.Variable(np.random.normal(0, 0.05, (hiddenUnits + 256, hiddenUnits)), dtype=tf.float32)

# biaes for the hidden values
b = tf.Variable(np.zeros((1, hiddenUnits)), dtype=tf.float32)

# weights and bias for the final classification
W2 = tf.Variable(np.random.normal (0, 0.05, (hiddenUnits, numClasses)),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,numClasses)), dtype=tf.float32)

# unpack the input sequences so that we have a series of matrices,
# each of which has a one-hot encoding of the current character from
# every input sequence
sequenceOfLetters = tf.unstack(inputX, axis=2)

# now we implement the forward pass
currentState = initialState
for timeTick in sequenceOfLetters:
    #
    # concatenate the state with the input, then compute the next state
    inputPlusState = tf.concat([timeTick, currentState], 1)  
    next_state = tf.tanh(tf.matmul(inputPlusState, W) + b) 
    currentState = next_state

# compute the set of outputs
outputs = tf.matmul(currentState, W2) + b2 # matmul

predictions = tf.nn.softmax(outputs) # softmax

# compute the loss
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=inputY)
totalLoss = tf.reduce_mean(losses)

# use gradient descent to train
#trainingAlg = tf.train.GradientDescentOptimizer(learningRate).minimize(totalLoss)
trainingAlg = tf.train.AdagradOptimizer(learning_rate=learningRate).minimize(totalLoss)

# and train!!
with tf.Session() as sess:
    #
    # initialize everything
    sess.run(tf.global_variables_initializer())
    #
    # and run the training iters
    _accLoss = 0.0
    _accCount = 0
    for epoch in range(numTrainingIters):
        # 
        # get some data
        x, y = generateDataRNN (maxSeqLen, data)
        #
        # do the training epoch
        _currentState = np.zeros((batchSize, hiddenUnits))
        _totalLoss, _trainingAlg, _currentState, _predictions, _outputs = sess.run(
                [totalLoss, trainingAlg, currentState, predictions, outputs],
                feed_dict={
                    inputX:x,
                    inputY:y,
                    initialState:_currentState
                })
        #
        # just FYI, compute the number of correct predictions
        numCorrect = 0
        for i in range (len(y)):
            maxPos = -1
            maxVal = 0.0
            for j in range (numClasses):
                if maxVal < _predictions[i][j]:
                    maxVal = _predictions[i][j]
                    maxPos = j
            if maxPos == y[i]:
                numCorrect = numCorrect + 1
        #
        # print out to the screen
        if epoch%100 == 0: 
            print("Step", epoch, "Loss", _totalLoss, "Correct", numCorrect, "out of", batchSize)

        if epoch >= numTrainingIters - 10:
            _accLoss += _totalLoss
            _accCount += numCorrect 
    
    print("Average loss for the last 10 mini-batches is", _accLoss / 10, 
          "average correct labels is", _accCount / 10, "out of 100.")