# -*- coding: utf-8 -*-
__author__ = 'ivanvallesperez'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from ELM import ELMRegressor

plt.style.use('ggplot')
sns.set_context("talk")

## DATA PREPROCESSING

lines = [line.rstrip('\n').split(',') for line in open('abalone.txt')];

genders = [row[0] for row in lines];
for i, n in enumerate(genders):
    if n == 'M':
        lines[i][0] = 2
    elif n == 'F':
        lines[i][0] = 0
    elif n == 'I':
        lines[i][0] = 1
y = np.array([row[8] for row in lines]).astype(float);
X = np.delete(lines,8,1).astype(float);
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

stdScaler_data = StandardScaler()
X_train = stdScaler_data.fit_transform(X_train)
X_test = stdScaler_data.transform(X_test)

stdScaler_target = StandardScaler()
y_train = stdScaler_target.fit_transform(np.reshape(y_train,(-1,1)))  # /max(y_train)
y_test = stdScaler_target.transform(np.reshape(y_test,(-1,1)))  # /max(y_train)
y_min = min([min(y_train),min(y_test)])
y_max = max([max(y_train),max(y_test)])
y_range = y_max[0] - y_min[0];

## ELM TRAINING
MAE_TRAIN_MINS = []
MAE_TEST_MINS = []

for M in range(1, 120, 1):
    MAES_TRAIN = []
    MAES_TEST = []
    # print "Training with %s neurons..."%M
    for i in range(20):
        ELM = ELMRegressor(M)
        ELM.fit(X_train, y_train)
        prediction = ELM.predict(X_train)
        MAES_TRAIN.append(mean_absolute_error(y_train,prediction))

        prediction = ELM.predict(X_test)
        MAES_TEST.append(mean_absolute_error(y_test,prediction))
    MAE_TEST_MINS.append((sum(MAES_TEST)/float(len(MAES_TEST)))/y_range)
    MAE_TRAIN_MINS.append((sum(MAES_TRAIN)/float(len(MAES_TRAIN)))/y_range)

print "Minimum MAE ELM =", min(MAE_TEST_MINS)
print "at: ", MAE_TEST_MINS.index(min(MAE_TEST_MINS))

#############################################################################################
## PLOTTING THE RESULTS
df = pd.DataFrame()
df["test"] = MAE_TEST_MINS
df["train"] = MAE_TRAIN_MINS
ax = df.plot()
ax.set_xlabel("Number of Neurons in the hidden layer")
ax.set_ylabel("Mean Absolute Error")
ax.set_title("Average ELM error for Abalone dataset per number of neurons in the hidden layer")
plt.show()
