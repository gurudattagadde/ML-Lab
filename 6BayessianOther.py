import csv
import random
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def loadCsv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = [list(map(float, line)) for line in lines]
    return dataset

def main():
    filename = 'NaiveBayesDiabetes.csv'
    dataset = loadCsv(filename)
    trainingSet = dataset
    testSet = loadCsv('NaiveBayesDiabetes1.csv')
    print('Records in training data={} and test data={} rows'.format(len(dataset), len(trainingSet), len(testSet)))
    
    X_train = [row[:-1] for row in trainingSet]
    y_train = [row[-1] for row in trainingSet]
    
    X_test = [row[:-1] for row in testSet]
    y_test = [row[-1] for row in testSet]
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print(predictions)
    
    accuracy = accuracy_score(y_test, predictions) * 100.0
    print("Accuracy:", accuracy, "%")

if __name__ == "__main__":
    main()
