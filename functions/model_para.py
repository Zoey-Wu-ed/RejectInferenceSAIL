# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 21:36:10 2024

@author: wzxia
"""

from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
import itertools
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.svm import SVC


def gen_paradicts(cl_type, ran_seed):
    list_para = []
    if cl_type == "LR":
        par_strings = ["random_state" ,"C"] # ‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’
        value_ranges = [[ran_seed], [0.01,0.1,1,2,3]] # , 'liblinear', 'sag', 'saga'],['l1','l2','none']
        for element in itertools.product(*value_ranges):
            dict_element = {}
            for i in range(len(par_strings)):
                dict_element[par_strings[i]] = element[i]
            list_para.append(dict_element)
        
    elif cl_type == "DT":
        par_strings = ["random_state", "max_depth", "min_samples_split","min_samples_leaf","class_weight"]
        value_ranges = [[ran_seed], [2,6,8,10], [2,6,8,10], [2,6,8,10],["balanced"]]
        for element in itertools.product(*value_ranges):
            dict_element = {}
            for i in range(len(par_strings)):
                dict_element[par_strings[i]] = element[i]
            list_para.append(dict_element)

    elif cl_type == "RF":
        par_strings = ["random_state", "max_depth", "min_samples_split","min_samples_leaf", "n_estimators"]
        value_ranges = [[ran_seed], [2,6,8,10], [2,6,8,10], [2,6,8,10], [30,50,70,90,110,140]]
        for element in itertools.product(*value_ranges):
            dict_element = {}
            for i in range(len(par_strings)):
                dict_element[par_strings[i]] = element[i]
            list_para.append(dict_element)
        
    elif cl_type == "GBDT":
        par_strings = ["random_state" ,"learning_rate", "max_depth", "min_samples_split","min_samples_leaf", "n_estimators"]
        value_ranges = [[ran_seed], [0.01, 0.1], [2,6,8,10], [2,6,8,10], [2,6,8,10], [30,50,70]]
        for element in itertools.product(*value_ranges):
            dict_element = {}
            for i in range(len(par_strings)):
                dict_element[par_strings[i]] = element[i]
            list_para.append(dict_element)

    elif cl_type == "LGBM":
        par_strings = ["random_state" ,"learning_rate", "max_depth", "n_estimators"]
        value_ranges = [[ran_seed], [0.01, 0.1], [1,2,3,6], [10, 30, 70, 90]]
        # value_ranges = [[int(args.ran_seed)], [0.01, 0.1], [2,4,6,8,10], [30,50,70,90], [30,50,70]]
        for element in itertools.product(*value_ranges):
            dict_element = {}
            for i in range(len(par_strings)):
                dict_element[par_strings[i]] = element[i]
            list_para.append(dict_element)
            
    elif cl_type == "XGB":
        par_strings = ["seed", "learning_rate", "max_depth", "n_estimators", 'lambda']
        value_ranges = [[ran_seed], [0.01, 0.1], [2,6,8,10], [10, 30, 50, 70],[10,20]]
        for element in itertools.product(*value_ranges):
            dict_element = {}
            for i in range(len(par_strings)):
                dict_element[par_strings[i]] = element[i]
            list_para.append(dict_element)
            
    elif cl_type == "SVM":
        par_strings = ["random_state", "C", "class_weight", "kernel", "gamma"]
        value_ranges = [[ran_seed],[0.1,1,2,3,4], [None, "balanced"], ["rbf","poly"],["scale", "auto"]]
        for element in itertools.product(*value_ranges):
            dict_element = {}
            for i in range(len(par_strings)):
                dict_element[par_strings[i]] = element[i]
            list_para.append(dict_element)
            
    elif cl_type == "MLP":
        par_strings = ["random_state", "hidden_layer_sizes", "learning_rate_init"]
        value_ranges = [[ran_seed], [5,10,25,50], [0.0001,0.001,0.01]] 
        for element in itertools.product(*value_ranges):
            dict_element = {}
            for i in range(len(par_strings)):
                dict_element[par_strings[i]] = element[i]
            list_para.append(dict_element)
    else:
        print("ERROR: UNKNOWN CLS")
    return list_para 