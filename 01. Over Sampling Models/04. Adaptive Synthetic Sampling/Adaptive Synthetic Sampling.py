#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def create_dataset():
    X, y = make_classification(n_samples = 10000,
                               n_features = 2,
                               n_redundant = 0,
                               n_classes = 2,
                               flip_y = 0,
                               n_clusters_per_class = 2,
                               class_sep = 0.79,
                               weights = [0.99],
                               random_state = 81)
    return pd.DataFrame(X, columns = ['feature_1', 'feature_2']), pd.Series(y)


# In[4]:


def plot_dataset(X, y, label):
    print(y.value_counts())

    markers = {1: "X", 0: "v"}
    sns.scatterplot(data = X, x = 'feature_1', y = 'feature_2', hue = y, style = y, markers = markers)

    plt.title(label)
    plt.ylim(-5, 4)
    plt.xlim(-3, 3)

    plt.show()


# In[5]:


X, y = create_dataset()

plot_dataset(X, y, "Dataset")


# ### ADASYN (Adaptive Synthetic Sampling) :

# ADASYN (Adaptive Synthetic Sampling) is a data augmentation technique used in machine learning to address imbalanced datasets. It generates synthetic examples for the minority class, helping to balance the dataset and improve model performance on underrepresented classes. ADASYN is an extension of SMOTE (Synthetic Minority Over-sampling Technique) and adapts its synthetic sample generation based on the density distribution of the minority class instances.

# In[7]:


from imblearn.over_sampling import ADASYN

X_resampled, y_resampled = ADASYN().fit_resample(X, y)
print(sorted(Counter(y_resampled).items()))

plot_dataset(X_resampled, y_resampled, "Dataset oversampled using ADASYN")


# In[ ]:




