import math
import random
import pandas as pd


# X_Y_dict = pd.read_excel('dataset_x_y.xlsx', index_col=0).to_dict()
# X = list(X_Y_dict['Y'].keys())
# Y = list(X_Y_dict['Y'].values())
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#
# le = preprocessing.LabelEncoder()
# le.fit(X_train)
# X_train = le.transform(X_train).reshape(-1, 1)
#
# enc = preprocessing.OneHotEncoder(dtype=float, handle_unknown='ignore').fit(X_train)
#
# X_train = enc.transform(X_train).toarray()
# X_test = le.fit_transform(X_test).reshape(-1, 1)
# X_test = enc.transform(X_test).toarray()
#
# gnb = GaussianNB()
# gnb.
# y_pred = gnb.fit(X_train, y_train)
#
# print('Accuracy of GNB classifier on training set: {:.2f}'
#       .format(gnb.score(X_train, y_train)))
# print('Accuracy of GNB classifier on test set: {:.2f}'
#       .format(gnb.score(X_test, y_test)))
# print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], y_test != y_pred))


def loadDataset():
    dataframe = pd.read_excel('dataset_x_y.xlsx', index_col=0)
    dataset = list(dataframe)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset
#
# def preProcessForTest(data_list):
#     le = preprocessing.LabelEncoder()
#     le.fit(data_list)
#     data_list = le.transform(data_list).reshape(-1, 1)
#     enc = preprocessing.OneHotEncoder(dtype=float, handle_unknown='ignore')
#     return enc.transform(data_list)

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
            separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0

if __name__ == "__main__":
    dataset = loadDataset()
    trainingSet, testSet = splitDataset(dataset, 0.8)
    print('Split {0} rows into train = {1} and test = {2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
    # prepare model
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%'.format(accuracy))