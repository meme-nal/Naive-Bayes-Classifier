import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import NaiveBayesClassifier


dataset = pd.read_csv("./data/preprocessed/Iris_preprocessed.csv")
dataset = dataset.sample(frac=1) # shuffling

### Simple EDA

# check if predictable classes are unbalanced
fig = plt.figure(figsize=(8,8))
plt.hist(dataset['Species'])
plt.title("count of labels")
plt.savefig('./results/balanced_data.jpg')

# check correlation between features
fig = plt.figure(figsize=(8,8))
corr_matrix = dataset.iloc[:,:-1].corr(method='pearson')
sns.heatmap(corr_matrix, annot=True)
plt.title("Correlation matrix")
plt.savefig('./results/correlations.jpg')

# check distributions of all features
features = list(dataset.columns)[:-1]
for feature in features:
    fig = plt.figure(figsize=(8,8))
    plt.hist(dataset[feature])
    plt.title(feature)
    plt.savefig('./results/'+str(feature)+'.jpg')



# splitting into train and test data
#train_size = 0.8

#train = dataset.iloc[:int(train_size*dataset.shape[0]),:]
#test = dataset.iloc[int(train_size*dataset.shape[0]):,:]




# end splitting

#model = NaiveBayesClassifier()
#model.train(train, 'Species')
#model.predict()








