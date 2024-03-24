import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

data = pd.read_csv("./data/raw/Iris.csv")
data = data.drop(columns='Id')

species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
data = data.replace(to_replace=species, value=(1,2,3))

data.to_csv("./data/preprocessed/Iris_preprocessed.csv")