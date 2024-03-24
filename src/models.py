import numpy
import pandas

class NaiveBayesClassifier:
    def __init__(self)->None:
        # P(Y|X) = P(X|Y) * P(Y) / P(X) 
        pass

    def predict(self, train:pandas.DataFrame, X_test:pandas.DataFrame, label:str):
        # calculating prior
        prior = self.calculate_prior(train, label)

        # calculating likelihood
        likelihood = self.calculate_likelihood(train, X_test, label)

    def calculate_prior(self, train:pandas.DataFrame, label:str)->list[float]:
        labels = sorted(list(train[:,label].unique()))
        prior = []
        for i in labels:
            prior.append(len(train[train[:,label] == i]) / len(train))
        return prior

    def calculate_likelihood(self, train:pandas.DataFrame, X_test:pandas.DataFrame, label:str)->list[float]:
        labels = sorted(list(train[:,label].unique()))
        features = list(train.columns)[:-1]

        likelihood = [1]*len(labels)
        
        for x in X_test:
            for j in range(len(labels)):
                for i in range(len(features)):
                    train_ = train[train[label] == labels[j]]
                    likelihood[j] *= (len(train_[train_[features[i]] == x[i]]) / len(train_))

        return likelihood








