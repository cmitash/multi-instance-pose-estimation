import numpy as np
from sklearn import svm, datasets,linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import os,sys
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from math import sqrt
from sklearn.neural_network import MLPRegressor

if __name__=="__main__":
  data_path = 'train_data.txt'
  X=[]
  Y=[]
  with open(data_path,'r') as ff:
    lines = ff.readlines()
  for line in lines:
    line = line.strip('\n').split()
    line = list(map(float,line))
    if np.all(line[1:6]==np.zeros((5))):
      continue
    if line[1] < 10 or line[2] < 10:
      continue
    X.append(line[:6])
    Y.append(line[6])

  X = np.array(X)
  Y = np.array(Y)
  print("Y min/max are",np.amin(Y),np.amax(Y))
  print(X.shape)
  print(Y.shape)

  # GTB - Start  
  X, Y = shuffle(X, Y,random_state=0)
  model = GradientBoostingRegressor(loss='ls',tol=1e-8,learning_rate=0.005,n_estimators=1100,max_depth=5)
  model.fit(X,Y)
  pred = model.predict(X)
  mse = sqrt(((pred-Y)**2).mean())
  print(mse)
  with open('GTB.p','wb') as ff:
    pickle.dump(model, ff)
  # GTB - End 