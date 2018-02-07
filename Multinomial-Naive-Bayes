import os
import random
import numpy as np
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#For prototyping MNB
# trainInstances = [["Chinese", "Beijing", "Chinese"], 
#             ["Chinese", "Chinese", "Shanghai"],
#             ["Chinese", "Macao"], 
#             ["Tokyo", "Japan", "Chinese"]]
# trainClassLabels = ["YES", "YES", "YES", "NO"]

# testInstances = [["Chinese", "Chinese", "Chinese", "Tokyo", "Japan"]]
# actualTestLabels = ["NO"]



#creates vector of training and testing instances
def createData(selected5Classes, path):
    instances = []
    classLabels = []
    
    for file in selected5Classes:
        filename = file
        allfiles = os.listdir(path+"\\"+filename)
        
        for mainfile in allfiles:
            size = len(allfiles)
            currentFilePath = path+"\\"+filename+"\\"+mainfile
            vector = getVectorOfFile(currentFilePath)
            instances.append(vector)
            classLabels.append(filename)
        
    return instances, classLabels
    
    
#remove special characters from data
def removeNonAplhabet(vector):
    return [w.lower() for w in vector if w.isalpha()]
    
 
 
#get the vocabulary of training instances
def extractVocabulary(trainInstances):
    totalTrainInstances = len(trainInstances)
    vocabulary = []
    for i in range(totalTrainInstances):
        vocabulary = vocabulary + trainInstances[i]
    vocabulary = list(set(vocabulary))
    return vocabulary
    
    
    
#removes header from file and converts text to tokens
def getVectorOfFile(currentFilePath):
    flag = 0
    tokens = []

    currentFile = open(currentFilePath, 'r')

    for line in currentFile:
        if line.find("Lines") == -1:
            if(flag != 0):
                word_tokens = word_tokenize(line)
                filteredLine = [w for w in word_tokens if not w in stop_words]
                tokens.extend(filteredLine)
        else:
            flag = 1
            
    currentFile.close()        
    tokens = removeNonAplhabet(tokens)
    tokens.sort()
    return tokens
    
    
#calculates the accuracy of MNB
def getAccuracy(testInstances, actualTestLabels, C, V, prior, conditionalProb):
    correctCount = 0
    size = len(testInstances)
    
    for i in range(size):
        percentage = (i+1)*100/size
        d = testInstances[i]
        prediction = applyMultinomialNB(C, V, prior, conditionalProb, d)
        if(prediction == actualTestLabels[i]):
            correctCount = correctCount + 1
            
    return (correctCount/size)
    

#count total docs in training set
def countDocs(trainInstances):
    return len(trainInstances)
    
    
#get the unique class labels
def getClassLabels(classLabels):
    return list(set(classLabels))
    
    
    
#count total number of documents in a given class c
def countDocsInClass(classLabels, c):
    return classLabels.count(c)
    
    
#make a vector of all tokens in all instances of class c
def concatenateTextOfAllDocsInClass(trainInstances, classLabels, c):
    text_c = []
    for i in range(len(trainInstances)):
        if(classLabels[i]==c):
            for j in range(len(trainInstances[i])):
                text_c.append(trainInstances[i][j])
            
    return text_c
    
    
    
#counts occurences
def countTokensOfTerm(text_c, t):
    return text_c.count(t)
    
    
    
#gets all tokens in class classLabel
def getTotalFeaturesInClass(classLabel, trainInstances, classLabels):
    count = 0
    for i in range(len(trainInstances)):
        if(classLabels[i] == classLabel):
            count = count + len(trainInstances[i])
    return count
    
    
    
#for testing, test on only those which words exist in the vocabulary
def extractTokensFromDoc(V, d):
    W = []
    for i in range(len(d)):
        t = d[i]
        if(t in V):
            W.append(t)
    return W
    
    
#returns index of token t in vocabulary
def getIndexOfTermInV(t, V):
    return V.index(t)
    
    
    
#algorithm to train classifier
def trainMultinomialNB(C, D, classLabels):
    V = extractVocabulary(D)
    N = countDocs(D)
    prior = []
    conditionalProb = np.zeros((len(V), len(C)))
    B = len(V)
    
    for index_of_c in range(len(C)):
        c = C[index_of_c]
        Nc = countDocsInClass(classLabels, c)
        prior.append(Nc/N)
        text_c = concatenateTextOfAllDocsInClass(D, classLabels, c)
        Sum_Tct_prime = getTotalFeaturesInClass(c, D, classLabels)
        size = len(V)
        
        for index_of_t in range(size): 
            t = V[index_of_t]
            Tct = countTokensOfTerm(text_c, t)
            conditionalProb[index_of_t][index_of_c] =  (Tct+1)/(Sum_Tct_prime+B)
    
    return V, prior, conditionalProb
    
    
    
    
 #argmax computation, maximize the probablity of class c given instance x
def applyMultinomialNB(C, V, prior, conditionalProb, d):
    W = extractTokensFromDoc(V, d)
    #print(W)
    score = []
    for index_of_c in range(len(C)):
        c = C[index_of_c]
        #print("Current Class:", c)
        score.append(math.log(prior[index_of_c]))
    
        for index_of_t in range(len(W)):
            t = W[index_of_t]
            index_of_t_in_V = getIndexOfTermInV(t, V)
            score[index_of_c] = score[index_of_c] + math.log(conditionalProb[index_of_t_in_V][index_of_c])
        #print(c,score[index_of_c])
    
    index_prediction = np.argmax(score)
    prediction = C[index_prediction]
    return prediction
    
    
 #input from user
trainPath = input("Enter the path to the Training Dataset: ")
testPath = input("Enter the path to the Testing Dataset: ")   


#Enter the path to the Training Dataset: 20news-bydate\\20news-bydate-test
#Enter the path to the Testing Dataset: 20news-bydate\\20news-bydate-train



# trainPath = "20news-bydate\\20news-bydate-test"
# testPath = "20news-bydate\\20news-bydate-train"

labels = os.listdir(trainPath)

#select 5 classes randomly
random.shuffle(labels)

selected5Classes = labels[0:5]
#print("The 5 selected classes:", selected5Classes)

stop_words = set(stopwords.words('english'))




#the creation of vectors, training and testing takes time.

#get the tokens from files of various classes into vectors
trainInstances, trainClassLabels = createData(selected5Classes, trainPath)
testInstances, actualTestLabels = createData(selected5Classes, testPath)



#training the classifier
C = getClassLabels(trainClassLabels)
D = trainInstances




V, prior, conditionalProb = trainMultinomialNB(C, D, trainClassLabels)


accuracy = getAccuracy(testInstances, actualTestLabels, C, V, prior, conditionalProb) 


print("The 5 selected classes:\n", selected5Classes)


print("Accuracy is: {:.3f}".format(accuracy*100))

    
