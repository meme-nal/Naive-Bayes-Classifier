import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import seaborn as sns

from src.models import NaiveBayesClassifier
from src import evaluations
from src import results


dataset = pd.read_csv("./data/preprocessed/Iris_preprocessed.csv")
dataset = dataset.sample(frac=1, random_state=19) # shuffling

### Simple EDA

# check if there are missings in data
#print(dataset.isna().mean().sort_values(ascending=False))

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

train = dataset.iloc[:int(train_size*dataset.shape[0]),:]
test = dataset.iloc[int(train_size*dataset.shape[0]):,:]
X_test = test.iloc[:,:-1]
y_test = test.iloc[:,-1].to_numpy()


# model
model = NaiveBayesClassifier()
predictions = model.predict(train, X_test, label='Species')


model_results = {}

# Evaluations 
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
confusion_matrix = evaluations.get_confusion_matrix(predictions, y_test)

precisions = evaluations.get_precision(confusion_matrix, labels)
model_results['precisions'] = precisions

recalls = evaluations.get_recall(confusion_matrix, labels)
model_results['recalls'] = recalls

f1_scores = evaluations.get_f1_score(precisions, recalls, labels)
print(f"f1-scores: {f1_scores}")


#results.generate_results(model_results)

