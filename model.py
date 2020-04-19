# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import pickle
#import word2number

data = pd.read_csv("D:\Data-Science-Practice\April 2020 Practice\Heruko Deploy using flask\hiring.csv")
data["experience"].fillna(0,inplace =True)
data["test_score"].fillna(data["test_score"].mean(),inplace=True)

X = data.iloc[:,:3]

def convert_to_int(word):
    word_dict = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
                 "eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"zero":0,0:0}
    return word_dict[word]
X["experience"] = X["experience"].apply(lambda x: convert_to_int(x))

y = data.iloc[:,-1]

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X,y)

pickle.dump(reg, open("model.pkl","wb"))