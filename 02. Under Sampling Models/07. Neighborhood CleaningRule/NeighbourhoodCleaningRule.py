#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.datasets import make_classification
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

def plot_dataset(X, y, label):
    X = pd.DataFrame(X, columns = ["Feature_1", "Feature_2"])
    y = pd.Series(y)

    print(y.value_counts())

    markers = {1: "X", 0: "v"}
    sns.scatterplot(
        data = X, 
        x = "Feature_1", y = "Feature_2", 
        hue = y, 
        style = y, 
        markers = markers
    )

    plt.title(label)
    plt.ylim(-5, 4)
    plt.xlim(-3, 3)

    plt.show()
    
    
    
X, y = make_classification(
    n_samples = 10000,
    n_features = 2,
    n_redundant = 0,
    n_classes = 2,
    flip_y = 0,
    n_clusters_per_class = 2,
    class_sep = 0.79,
    weights = [0.99],
    random_state = 81,
)

plot_dataset(X, y, "Dataset with Imbalance Ratio: 1:99")


# ###  NeighbourhoodCleaningRule :

# `NeighbourhoodCleaningRule` (NCR) is a machine learning technique designed to address imbalanced datasets. It employs the k-nearest neighbors algorithm to identify and remove instances from the majority class that are considered noisy or potentially mislabeled. The primary objective is to clean the neighborhood around the decision boundary, enhancing the quality of the training dataset. By selectively eliminating instances close to the boundary between classes, `NeighbourhoodCleaningRule` aims to improve the generalization and performance of machine learning models, particularly in scenarios with imbalanced class distributions.

# In[2]:


from imblearn.under_sampling import NeighbourhoodCleaningRule

ncr = NeighbourhoodCleaningRule(
    sampling_strategy="auto", n_neighbors=200, threshold_cleaning=0.5
)
X_res, y_res = ncr.fit_resample(X, y)
print("Resampled dataset shape %s" % Counter(y_res))

plot_dataset(X_res, y_res, "Dataset undersampled using NeighbourhoodCleaningRule")


# In[ ]:




