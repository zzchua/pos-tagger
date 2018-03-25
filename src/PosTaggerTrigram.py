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

# Trigram HMM Params
LAMBDA_1 = 0.3
LAMBDA_2 = 0.7

# ADD-K Smoothing
K = 0.001

# Set to True to account for Twitter special characters
ACCOUNT_SPECIALS = True

# Input/Output Files
INPUT_TRAIN = "twt.train.json"
INPUT_DEV = "twt.dev.json"
OUTPUT = "trigram.results.json"

# DO NOT CHANGE THESE
# Start, Stop labels
START_LABEL = "<s>"
STOP_LABEL = "</s>"

# Special Emission Types
UNK = "<UNK>"
HANDLE = "<@>"
HASHTAG = "<#>"
URL = "<link>"


# Builds the trigram transition probabilities for the given input training set
# Returns a Map representing the transition probabilities. Outer Keys are the "from-tag",
# Inner keys represent the "transition-to-tag" and their corresponding probabilities.
def buildTrigramTransitionPr(jsonFileName):
    # construct the dictionary:
    transitionPr = {}
    transitionCountTotals = {}
    
    
    jsonFile = open('dataset/' + jsonFileName)
    for line in jsonFile:
        # each line is a tweet:
        jsonTweet = json.loads(line)
        for i in range(-2, len(jsonTweet) - 1, 1):
            
            # set the from tag
            fromTag = None
            if (i == -2):
                fromTag = (START_LABEL, START_LABEL)
            elif (i == -1):
                fromTag = (START_LABEL, jsonTweet[i + 1][1])
            else:    
                fromTag = (jsonTweet[i][1], jsonTweet[i + 1][1])
            
            
            nextTag = ''
            # Handle the case of ending stop symbol
            if (i + 2 == len(jsonTweet)):
                # last currentWord, next tag is stop:
                nextTag = STOP_LABEL
            else:
                nextTag = jsonTweet[i + 2][1]
            
            # add entry to dict:
            if (fromTag not in transitionPr):
                transitionPr[fromTag] = {}
            
            innerToDict = transitionPr[fromTag]
            innerToDict[nextTag] = innerToDict.get(nextTag, 0) + 1
            
            # update the total count for that current tag:
            transitionCountTotals[fromTag] = transitionCountTotals.get(fromTag, 0) + 1
                
    # Turn the transition counts into a probability distribution for each From Tag
    normalize(transitionPr, transitionCountTotals)
    return transitionPr

# Builds the Bigram Transition and Emission Probabilities for the given input file.
# Returns a list [transitionPr, emissionPr]
#
# emissionPr is a Map, outerkeys are the tags, inner keys are they words and their corresponding probabilities
#
# transitionPr is a Map, outerkeys are the "from-tag", 
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
            
            fromTag = None
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




# Returns the interpolated log-transition probability p(label|labelP, labelPP)
# label, labelP, labelPP are the labels of the transition probability
# trigramTransition: map of the trigram transition probabilities
# bigramTransition: map of the bigram transition probabilities
def getInterpolatedPr(label, labelP, labelPP, trigramTransition, bigramTransition):
    p1 = 0.0
    p2 = 0.0
    
    labelPair = (labelPP, labelP)
    if labelPair in trigramTransition and label in trigramTransition[labelPair]:
        p1 = LAMBDA_1 * trigramTransition[labelPair][label]
    
    if labelP in bigramTransition and label in bigramTransition[labelP]:
        p2 = LAMBDA_2 * bigramTransition[labelP][label]
    
    if (p1 + p2 == 0):
        return float("-inf")
    else:
        return math.log(p1 + p2)


# Computes the viterbi table for the given seq
# seq: list of words
# trigramTransition: map of the trigram transition probabilities
# bigramTransition: map of the bigram transition probabilities
# emissionPr: map of emission probabilities
#
# returns a list in the form: 
# [scores, labelIndex, bigramLabels, singleLabels]
# scores: a 2D matrix of the viterbi prefix scores
# labelIndex: a 2D matrix of the backtrace for the labels
# bigramLabels: a list of tuples corresponding to tag-pairs (y, y')
# singleLabels: a list of labels in the label set
def computeViterbiTableTrigram(trigramTransition, emissionPr, bigramTransition, seq):
    # create every pairwise combination of tags:
    bigramLabels = []
    for l1 in emissionPr:
        for l2 in emissionPr:
            bigramLabels.append((l1, l2))
    
    seqSize = len(seq)
    labelSetSize = len(bigramLabels)
    singleLabels = emissionPr.keys()
    singleLabelsSize = len(singleLabels)
    
    scoreIndex = {}
    # way to retrieve score index for s(y', y'')
    for i in range(0, labelSetSize, 1):
        scoreIndex[bigramLabels[i]] = i
        
    
    # scores will be log scores
    scores =  np.full((labelSetSize, seqSize), float("-inf"), dtype=float, order='C')
    labelIndex = np.full((labelSetSize, seqSize), -1,dtype=int, order='C')
    
    # special case for 1 word:
    if (seqSize == 1):
        for i in range(0, labelSetSize, 1):
            label = bigramLabels[i][0]
            word = seq[0]
            if word not in emissionPr[label]:
                word = UNK
            
            emitPr = emissionPr[label][word]
            transPr = getInterpolatedPr(label, START_LABEL, START_LABEL, trigramTransition, bigramTransition)
            stopPr = getInterpolatedPr(STOP_LABEL, label, START_LABEL, trigramTransition, bigramTransition)
            scores[i][0] = emitPr + transPr + stopPr
            
    else:
        for wIndex in range(1, seqSize, 1):
            for lIndex in range(0, labelSetSize, 1):
                word = seq[wIndex]
                bigramLabel = bigramLabels[lIndex]
                label = bigramLabel[0]
                labelP = bigramLabel[1]
                
                
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
                
                
                # starting at the 2nd word
                if (wIndex == 1):
                    # smoothing:
                    transPr = getInterpolatedPr(label, labelP, START_LABEL, trigramTransition, bigramTransition)
                    labelIndex[lIndex][wIndex] = -1 #-1 will represent the start label
                    
                    # Calculate S1:
                    word = seq[wIndex - 1]
                    # handling @, #tag
                    if ACCOUNT_SPECIALS:
                        if (word[0] == u"@"):
                            word = HANDLE
                        elif (word[0] == u"#"):
                            word = HASHTAG
                        elif (word[0:4] == u"http" or word[0:5] == u"https"):
                            word = URL
                    if word not in emissionPr[labelP]:
                        word = UNK
                    
                    emitPrS1 = emissionPr[labelP][word]
                    s1TransPr = getInterpolatedPr(labelP, START_LABEL, START_LABEL, trigramTransition, bigramTransition)
                    s1 = emitPrS1 + s1TransPr
        
                    score = emitPr + transPr + s1

                else:
                    # 3rd word
                    bestScore = float("-inf")
                    
                    # find the best score from previous
                    for lP in range(0, singleLabelsSize, 1):
                        currentScore = 0
                        labelPP = singleLabels[lP]
                        transPr = getInterpolatedPr(label, labelP, labelPP, trigramTransition, bigramTransition)
                        # get score if si-1:
                        prevScoreIndex = scoreIndex[(labelP, labelPP)]
                        currentScore = transPr + scores[prevScoreIndex][wIndex - 1]
                                            
                        if (currentScore > bestScore):
                            bestScore = currentScore
                            # store the backpointer
                            labelIndex[lIndex][wIndex] = lP
                    score = emitPr + bestScore
                
                # add the best score to scores
                scores[lIndex][wIndex] = score 
                   
    
        # add last stop score:
        for i in range(0, labelSetSize, 1):
            bigramLabel = bigramLabels[i]
            label = bigramLabel[0]
            labelP = bigramLabel[1]
            stopPr = getInterpolatedPr(STOP_LABEL, label, labelP, trigramTransition, bigramTransition)
            scores[i][len(seq) - 1] += stopPr 
    

    return [scores, labelIndex, bigramLabels, singleLabels]

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

# Predicts the labels for a given seq of words
# scores: viterbi prefix scores
# backtrace: backtrace of labels
# bigramLabels: a list of tuples of all label-pairs
# singleLabels: a list of labels in the label set
# returns a list of predicted labels corresponding to the seq
def predictPOSTrigram(scores, backtrace, seq, bigramLabels, singleLabels):
    predictedLabels = [None for x in range(len(seq))]
    # predict last 2 tags:
    maxScore = float("-inf")
    predictedLabel = None
    index = -1
    for i in range(0, len(scores), 1):
        score = scores[i][len(seq) - 1]
        if (score > maxScore):
            predictedLabel = bigramLabels[i]
            maxScore = score
            index = i
    # special case for 1 word
    if (len(seq) == 1):
        predictedLabels[0] = predictedLabel[0]
    else:
        predictedLabels[len(seq)-1] = predictedLabel[0]
        predictedLabels[len(seq)-2] = predictedLabel[1]
    
    currLabel = index
    for i in range(len(seq) - 1, 1, -1):
        backPointer = backtrace[currLabel][i]
        predictedLabels[i - 2] = singleLabels[backPointer]
        currLabel = backPointer
    
    return predictedLabels

# Tests the accuracy of the trigram HMM and outputs the predicted labels for the given input file in a json
# jsonFileName: input of sequences to be tested on
# outputFileName: output json of the predicted labels for the given input
# emissionPr: emission probbabilities
# transitionPr: trigram transition probabilities
# transitionSinglePr: bigram transition probabilities
def testAccuracyTrigram(jsonFileName, outputFileName, transitionPr, emissionPr, transitionSinglePr):
    print "[2] Testing the accuracy..."
    totalWordCount = 0
    totalCorrectCount = 0
    
    output = open(outputFileName, 'w')
    
    jsonFile = open('dataset/' + jsonFileName)
    lineCount = 1
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
        # [scores, labelIndex, bigramLabels, singleLabels, scoreIndex]
        viterbiResult = computeViterbiTableTrigram(transitionPr, emissionPr, transitionSinglePr, tweetSeq)
        backtrace = viterbiResult[1]
        scores = viterbiResult[0]
        bigramLabels = viterbiResult[2]
        singleLabels = viterbiResult[3]
        
        # process the tweet:
        predictedLabels = predictPOSTrigram(scores, backtrace, tweetSeq, bigramLabels, singleLabels)
        outputLine = []
        for i in range(len(predictedLabels)):
            outputLine.append((tweetSeq[i], predictedLabels[i]))
            
        result = json.dumps(outputLine)
        output.write(result + "\n")
        lineCount += 1
        
    output.close()

# This is the main method:
def main(args):
    tupleResult = buildTransitionEmissionPr(INPUT_TRAIN)
    transitionSingePr = tupleResult[0]
    emissionPr = tupleResult[1]
    transitionPr = buildTrigramTransitionPr(INPUT_TRAIN)
    start = time.time()
    testAccuracyTrigram(INPUT_DEV, OUTPUT, transitionPr, emissionPr, transitionSingePr)
    end = time.time()
    print "Time elapsed: " + str(end - start)


main(sys.argv)
