import json
import sys
from itertools import izip
import csv
import heapq

"""
Python script to analyze the output of PosTaggerTrigram/PosTaggerBigram
Outputs a confusion matrix and the worst performing 20 tweets in the test set

zzchua@cs.washington.edu

You may adjust various parameters below:
"""

# Set to False for Trigram, True for Bigram
BIGRAM = False

# Set to True to see the worst performing tweets
WORST_PERFORMING = False

# ENSURE THIS CORRESPONDS TO THE INPUT FILE USED TO PRODUCE THE RESULTS
TEST_SET = "twt.dev.json"

#Name of results file:
BIGRAM_RESULTS = "bigram.results.json"
TRIGRAM_RESULTS = "trigram.results.json" 

#Confusion Matrix Output:
BIGRAM_OUTPUT = "confusion-matrix-bigram.csv"
TRIGRAM_OUTPUT = "confusion-matrix-trigram.csv"

# DO NOT CHANGE THIS
TAG_LIST = ["N", "O", "S", "^", "Z",
            "L", "M", "V", "A", "R", 
            "!", "D", "P", "&", "T", 
            "X", "Y", "#", "@", "~", 
            "U", "E", "$", ",", "G"]




def evaluateResults(benchmarkFileName, resultsFileName, confusionMatrix, tagIndex):
    
    totalWordCount = 0
    answerRight = 0
    lowestSeq = []
    heapq.heapify(lowestSeq)
    
    for lineAnswer, lineResult in izip(open('dataset/' + benchmarkFileName), open(resultsFileName)):
        answersLine = json.loads(lineAnswer)
        resultsLine = json.loads(lineResult)
        
        
        localCorrect = 0
        for i in range(len(answersLine)):
            totalWordCount += 1
            rightTag = answersLine[i][1]
            predictedTag = resultsLine[i][1]
            
            rightTagIndex = tagIndex[rightTag]
            predictedIndex = tagIndex[predictedTag]
            
            # update the confusion matrix
            confusionMatrix[rightTagIndex][predictedIndex] += 1
            if predictedTag == rightTag:
                answerRight += 1                
                localCorrect += 1
                
        localScore = localCorrect/float(len(answersLine))
        tuple = (localScore, answersLine, resultsLine)
        heapq.heappush(lowestSeq, tuple)
    
    if (WORST_PERFORMING): 
        print "\t Worst Performing: "
        for seq in heapq.nsmallest(20, lowestSeq):
            for tuple in seq[1]:
                print tuple[0].encode("utf-8") + ' ' + tuple[1].encode("utf-8") + ' ',
            print ''
            for tuple in seq[2]:
                print tuple[0].encode("utf-8") + ' ' + tuple[1].encode("utf-8") + ' ',
            print '\n'
    
    print "\t Results: "
    print "\t Total Word Count: \t" + str(totalWordCount)
    print "\t Total Tags Right: \t" + str(answerRight)
    print "\t Accuracy: \t" + str(answerRight/float(totalWordCount)) 
    

def main(args):

    tagIndex = {}
    for i in range(len(TAG_LIST)):
        tagIndex[TAG_LIST[i]] = i
    confusionMatrix = [[0 for x in range(25)] for y in range(25)]
    
    if BIGRAM:
        evaluateResults(TEST_SET, BIGRAM_RESULTS, confusionMatrix, tagIndex)
        with open(BIGRAM_OUTPUT, "wb") as f:
            writer = csv.writer(f)
            writer.writerows(confusionMatrix)
    else:
        evaluateResults(TEST_SET, TRIGRAM_RESULTS, confusionMatrix, tagIndex)
        with open(TRIGRAM_OUTPUT, "wb") as f:
            writer = csv.writer(f)
            writer.writerows(confusionMatrix)

    
main(sys.argv)