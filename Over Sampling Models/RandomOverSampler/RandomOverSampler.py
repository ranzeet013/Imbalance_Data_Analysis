#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


# In[2]:


import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# 1. ** Dataset:**
#    - Using `make_classification` from scikit-learn to generate a synthetic dataset with the following characteristics:
#      - `n_samples`: 10,000 samples
#      - `n_features`: 2 features
#      - `n_redundant`: 0 redundant features
#      - `n_classes`: 2 classes
#      - `flip_y`: 0 (no flipping of labels)
#      - `n_clusters_per_class`: 2 clusters per class
#      - `class_sep`: 0.79 separation between classes
#      - `weights`: [0.99] weights for each class
#      - `random_state`: 81 for reproducibility

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


# ## RandomOverSampler :

# The RandomOverSampler is a technique used in imbalanced classification problems to address the issue of having significantly more samples in one class than another. It oversamples the minority class by randomly duplicating samples from that class until the class distribution is more balanced.

# 1. **`sampling_strategy`**: Sets the desired ratio of minority to majority class samples after resampling.
# 
# 2. **`random_state`**: Ensures reproducibility by setting the seed for the random number generator.

# In[9]:


from imblearn.over_sampling import RandomOverSampler
from collections import Counter

ros = RandomOverSampler(sampling_strategy = 0.1, 
                        random_state = 42)

X_res, y_res = ros.fit_resample(X, y)


# In[11]:


print('Resampled dataset shape %s' % Counter(y_res))


# In[12]:


plot_dataset(X_res, y_res, 'Dataset undersampled using RandomOverSampler')


# 1. **`sampling_strategy`**: Sets the desired ratio of minority to majority class samples after resampling.
# 
# 2. **`random_state`**: Ensures reproducibility by setting the seed for the random number generator.
# 
# 3. **`shrinkage`**: Controls the amount of shrinkage applied during random sample size adjustment.

# In[13]:


from imblearn.over_sampling import RandomOverSampler
from collections import Counter

ros = RandomOverSampler(sampling_strategy = 0.1, 
                        random_state = 42, 
                        shrinkage = 0.2)

X_res, y_res = ros.fit_resample(X, y)


# In[14]:


print('Resampled dataset shape %s' % Counter(y_res))


# In[15]:


plot_dataset(X_res, y_res, 'Dataset undersampled using RandomOverSampler')


# In[ ]:




