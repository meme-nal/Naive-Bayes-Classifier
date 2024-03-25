import pandas as pd

data = pd.read_csv("./data/raw/Iris.csv")
data = data.drop(columns='Id')

species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
data = data.replace(to_replace=species, value=(0,1,2))

data.to_csv("./data/preprocessed/Iris_preprocessed.csv", index=False)