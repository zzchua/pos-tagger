import json
import sys
import numpy as np
import math
import time

"""
Parts of Speech Tagger using Trigram Hidden Markov Model(HMM)
Interpolated with Bigram HMM and Add-K Smoothing.
This program outputs the tags in an output json in the same 
format as the input json.

zzchua@cs.washington.edu

You may adjust various parameters below:
"""

# ADD-K Smoothing
K = 0.001

# Set to True to account for Twitter special characters
ACCOUNT_SPECIALS = True

# Input files
TRAIN_SET = "twt.train.json"
TEST_SET = "twt.dev.json"
OUTPUT = "bigram.results.json"

# Start, Stop labels
START_LABEL = "<s>"
STOP_LABEL = "</s>"

# Special Emission Types
UNK = "<UNK>"
HANDLE = "<@>"
HASHTAG = "<#>"
URL = "<link>"


# Builds the Bigram Transition and Emission Probabilities for the given input file.
# Returns a list [transitionPr, emissionPr]
#
# emissionPr:a nested map, outerkeys are the tags, inner keys are they words and their corresponding probabilities
#
# transitionPr: a nested map, outerkeys are the "from-tag", 
# inner keys are the "transition-to-tag" and corresponding probabilities 
def buildTransitionEmissionPr(jsonFileName):
    print "[1] Building transmission/emission probabilities..."
    
    # construct the dictionary:
    transitionPr = {}
    transitionCountTotals = {}
    
    emissionPr = {}
    emissionCountTotals = {}
    

    jsonFile = open('dataset/' + jsonFileName)
    for line in jsonFile:
        # each line is a tweet:
        jsonTweet = json.loads(line)
        for i in range(-1, len(jsonTweet), 1):
            
            fromTag = ''
            # Handle the case of starting start symbol:
            # From: start to first tag
            if (i == -1):
                fromTag = START_LABEL
            else:
                fromTag = jsonTweet[i][1]
                currentWord = jsonTweet[i][0]
                
                # Account for @mentions as the same thing
                if ACCOUNT_SPECIALS:
                    if (currentWord[0] == u"@"):
                        currentWord = HANDLE
                    elif (currentWord[0] == u"#"):
                        currentWord = HASHTAG
                    elif (currentWord[0:4] == u"http" or currentWord[0:5] == u"https"):
                        currentWord = URL
                
                # only if its not the first start word, create an emission entry
                # if word is not the start word: add it to emission counts:
                if (fromTag not in emissionPr):
                    emissionPr[fromTag] = {}
                innerEmissionDict = emissionPr[fromTag]
                # check if the word is already inside, if it is ++, if not default is K
                innerEmissionDict[currentWord] = innerEmissionDict.get(currentWord, K) + 1
                emissionCountTotals[fromTag] = emissionCountTotals.get(fromTag, 0) + 1
            
            nextTag = ''
            # Handle the case of ending stop symbol
            if (i + 1 == len(jsonTweet)):
                # last currentWord, next tag is stop:
                nextTag = STOP_LABEL
            else:
                nextTag = jsonTweet[i + 1][1]
            
            # add entry to dict:
            if (fromTag not in transitionPr):
                transitionPr[fromTag] = {}
            
            innerToDict = transitionPr[fromTag]
            # increment the transition count
            innerToDict[nextTag] = innerToDict.get(nextTag, 0) + 1
            
            # update the total count for that current tag:
            transitionCountTotals[fromTag] = transitionCountTotals.get(fromTag, 0) + 1
    
    
    
    # apply add-k smoothing to the emission counts:
    for key in emissionCountTotals:
        # add unk:
        emissionPr[key][UNK] = K
        emissionCountTotals[key] += len(emissionPr[key]) * K
        
    
    
    # Turn the transition counts into a probability distribution for each From Tag
    normalize(transitionPr, transitionCountTotals)
    normalizeLog(emissionPr, emissionCountTotals)
    
    print "\t done \n"
    return [transitionPr, emissionPr]

# Helper method to normalize the emission counts to a valid log-probability distribution
def normalizeLog(nestedMap, totalCounts):
    # in each entry: get the total count:
    for fromKey in nestedMap:
        innerDict = nestedMap[fromKey]
        for toKey in innerDict:
            innerDict[toKey] = math.log(innerDict[toKey]) - math.log(float(totalCounts[fromKey]))

# Helper method to normalize the emission counts to a valid probability distribution
def normalize(nestedMap, totalCounts):
    # in each entry: get the total count:
    for fromKey in nestedMap:
        innerDict = nestedMap[fromKey]
        for toKey in innerDict:
            innerDict[toKey] = innerDict[toKey]/float(totalCounts[fromKey])

# Computes the viterbi table for the given seq
# seq: list of words
# bigramTransition: map of the bigram transition probabilities
# emissionPr: map of emission probabilities
#
# returns a list in the form: 
# [scores, labelIndex, labels]
# scores: a 2D matrix of the viterbi prefix scores
# labelIndex: a 2D matrix of the backtrace for the labels
# labels: a list of labels in the label set
def computeViterbiTable(transitionPr, emissionPr, seq):
    labelSize = len(emissionPr)
    seqSize = len(seq)
    
    # scores will be log scores
    scores = np.zeros((labelSize, seqSize), dtype=float, order='C')
    labelIndex = np.zeros((labelSize, seqSize), dtype=int, order='C')
    
    #enumerate the labels
    labels = emissionPr.keys()
    
    for wIndex in range(0, seqSize, 1):
        for lIndex in range(0, labelSize, 1):
            word = seq[wIndex]
            label = labels[lIndex]
            
            # handling @, #tag
            if ACCOUNT_SPECIALS:
                if (word[0] == u"@"):
                    word = HANDLE
                elif (word[0] == u"#"):
                    word = HASHTAG
                elif (word[0:4] == u"http" or word[0:5] == u"https"):
                    word = URL
                        
            # handling UNKS
            if word not in emissionPr[label]:
                word = UNK
                
            score = float("-inf")
            
            
            # find the emission for label given word
            emitPr = emissionPr[label][word]
            
            
            # special case for first word
            if (wIndex == 0):
                # check if the transition exists:
                if label not in transitionPr[START_LABEL]:
                    score = float("-inf")
                    labelIndex[lIndex][wIndex] = -2 #represent invalid label
                else:
                    score = emitPr + math.log(transitionPr[START_LABEL][label])
                    labelIndex[lIndex][wIndex] = -1 #-1 will represent the start label
                
            else:
                bestScore = float("-inf")
                
                # find the best score from previous
                for lP in range(0, labelSize, 1):
                    lPLabel = labels[lP]
                    
                    if label not in transitionPr[lPLabel]:
                        currentScore = float("-inf")
                    else:
                        currentScore = math.log(transitionPr[lPLabel][label]) + scores[lP][wIndex - 1]
                    
                    if (currentScore > bestScore):
                        bestScore = currentScore
                        # store the backpointer
                        labelIndex[lIndex][wIndex] = lP
                
                score = emitPr + bestScore
            
            # add the best score to scores
            scores[lIndex][wIndex] = score            
    
    for i in range(0, labelSize, 1):
        stopPr = 0
        label = labels[i]
        if STOP_LABEL not in transitionPr[label]:
            score = float("-inf")
        else:
            stopPr = math.log(transitionPr[label][STOP_LABEL])
            
        scores[i][len(seq) - 1] += stopPr 
    
    return [scores, labelIndex, labels]

# Predicts the labels for a given seq of words
# scores: viterbi prefix scores
# backtrace: backtrace of labels
# labels: a list of labels in the label set
# returns a list of predicted labels corresponding to the seq
def predictPOS(scores, backtrace, seq, labels):
    predictedLabels = ["" for x in range(len(seq))]
    # predict last tag:
    # go through all scores for the last word and find the max
    maxScore = float("-inf")
    predictedLabel = ""
    for i in range(0, len(scores), 1):
        score = scores[i][len(seq) - 1]
        if (score > maxScore):
            predictedLabel = labels[i]
            maxScore = score
            
    predictedLabels[len(seq)-1] = predictedLabel
    
    currLabel = labels.index(predictedLabel)
    for i in range(len(seq) - 1, 0, -1):
        backPointer = backtrace[currLabel][i]
        predictedLabels[i - 1] = labels[backPointer]
        currLabel = backPointer
    
    return predictedLabels

# Tests the accuracy of the bigram HMM and outputs the predicted labels for the given input file in a json
# jsonFileName: input of sequences to be tested on
# outputFileName: output json of the predicted labels for the given input
# emissionPr: emission probbabilities
# transitionPr: trigram transition probabilities
def testAccuracy(jsonFileName, outputFileName, transitionPr, emissionPr):
    print "[2] Testing the accuracy..."
    totalWordCount = 0
    totalCorrectCount = 0
    
    lineCount = 1
    
    output = open(outputFileName, 'w')
    
    jsonFile = open('dataset/' + jsonFileName)
    for line in jsonFile:
        print "\tProcessing " + str(lineCount)
        # each line is a tweet:
        jsonTweet = json.loads(line)
        tweetSeq = []
        actualTaglist = []
        for i in range(0, len(jsonTweet), 1):
            tweetSeq.append(jsonTweet[i][0])
            actualTaglist.append(jsonTweet[i][1])
            totalWordCount += 1
        
        
        # compute viterbi for that tweet:
        viterbiResult = computeViterbiTable(transitionPr, emissionPr, tweetSeq)
        backtrace = viterbiResult[1]
        scores = viterbiResult[0]
        labels = viterbiResult[2]
        
        # process the tweet:
        predictedLabels = predictPOS(scores, backtrace, tweetSeq, labels)
        outputLine = []
        for i in range(len(predictedLabels)):
            outputLine.append((tweetSeq[i], predictedLabels[i]))
            
        result = json.dumps(outputLine)
        output.write(result + "\n")
        lineCount += 1
    
    output.close()
        
def main(args):
    tupleResult = buildTransitionEmissionPr(TRAIN_SET)
    transitionPr = tupleResult[0]
    emissionPr = tupleResult[1]
    testAccuracy(TEST_SET, OUTPUT, transitionPr, emissionPr)


main(sys.argv)