import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Loading the iris dataset

iris_dataset = pd.read_csv('iris.csv')

#Checking for missing values

print(iris_dataset.isnull().sum())


# Renaming columns for better understanding
iris_dataset = iris_dataset.rename(columns={
    'SepalLengthCm': 'Sepal Length',
    'SepalWidthCm': 'Sepal Width',
    'PetalLengthCm': 'Petal Length',
    'PetalWidthCm': 'Petal Width',
    'Species': 'Species'
})


#Plotting Histogram for all the features


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
colors = ['blue', 'green', 'red', 'purple']

for i, ax in enumerate(axes.flat):
    sns.histplot(iris_dataset[features[i]], kde=True, color=colors[i], ax=ax)
    ax.set_title(f'Histogram showing distribution of {features[i]}')
    ax.set_xlabel(features[i])
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

#Using countplot() for Species feature since it is categorical

plt.figure(figsize=(8,6))
sns.countplot(x=iris_dataset['Species'], palette='Set2')
plt.title('Count of Each Iris Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()
