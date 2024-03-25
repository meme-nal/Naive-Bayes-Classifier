import numpy as np
import pandas as pd

class NaiveBayesClassifier:
    def __init__(self)->None:
        pass

    def predict(self, train:pd.DataFrame, X_test:pd.DataFrame, label:str)->np.ndarray:
        # get features name
        features = list(train.columns)[:-1] 

        # calculating prior
        prior = self.calculate_prior(train, label)

        predictions = []
        X_test = X_test.to_numpy()
        for x in X_test:
            labels = sorted(list(train[label].unique())) # our classes
            likelihood = [1]*len(labels)

            for j in range(len(labels)):
                for i in range(len(features)):
                    likelihood[j] *= self.calculate_likelihood(train, feature_name=features[i], feature_val=x[i], y=labels[j], label=label)

            # calculating posteriors
            posteriors = []
            for k in range(len(labels)):
                posteriors.append(likelihood[k]*prior[k])

            predictions.append(np.argmax(posteriors)) # get the number of the biggest probability
        return np.array(predictions)


    def calculate_prior(self, train:pd.DataFrame, label:str)->list[float]:
        labels = sorted(list(train[label].unique()))
        prior = []
        for i in labels:
            prior.append(len(train[train[label] == i]) / len(train))
        return prior
    
    def calculate_likelihood(self, train:pd.DataFrame, feature_name:str, feature_val:int, y:int, label:str)->float:
        # get dataset according to label and feature of test sample
        count_x_get_y = train[(train[label] == y) & (train[feature_name] == feature_val)]
        likelihood = len(count_x_get_y) / len(train[train[label] == y])
        return likelihood








