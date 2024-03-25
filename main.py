import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import NaiveBayesClassifier


dataset = pd.read_csv("./data/preprocessed/Iris_preprocessed.csv")
dataset = dataset.sample(frac=1) # shuffling

### Simple EDA

# check if there are missings in data
print(dataset.isna().mean().sort_values(ascending=False))

# check if predictable classes are unbalanced
fig = plt.figure(figsize=(8,8))
plt.hist(dataset['Species'])
plt.title("count of labels")
plt.savefig('./results/eda/balanced_data.jpg')

# check correlation between features
fig = plt.figure(figsize=(8,8))
corr_matrix = dataset.iloc[:,:-1].corr(method='pearson')
sns.heatmap(corr_matrix, annot=False)
plt.title("Correlation matrix")
plt.savefig('./results/eda/correlations.jpg')

# check distributions of all features
features = list(dataset.columns)[:-1]
for feature in features:
    fig = plt.figure(figsize=(8,8))
    plt.hist(dataset[feature])
    plt.title(feature)
    plt.savefig('./results/eda/'+str(feature)+'.jpg')

# convert continuous features to categorical features
dataset['cat_SepalLengthCm'] = pd.cut(dataset['SepalLengthCm'].values, bins=4, labels=[0,1,2,3])
dataset['cat_SepalWidthCm'] = pd.cut(dataset['SepalWidthCm'].values, bins=4, labels=[0,1,2,3])
dataset['cat_PetalLengthCm'] = pd.cut(dataset['PetalLengthCm'].values, bins=4, labels=[0,1,2,3])
dataset['cat_PetalWidthCm'] = pd.cut(dataset['PetalWidthCm'].values, bins=4, labels=[0,1,2,3])

dataset = dataset.drop(columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
dataset = dataset[['cat_SepalLengthCm', 'cat_SepalWidthCm', 'cat_PetalLengthCm', 'cat_PetalWidthCm', 'Species']]

# splitting into train and test data
train_size = 0.8

#train = dataset.iloc[:int(train_size*dataset.shape[0]),:]
#test = dataset.iloc[int(train_size*dataset.shape[0]):,:]




# end splitting

#model = NaiveBayesClassifier()
#model.train(train, 'Species')
#model.predict()








