# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:43:22 2024

@author: wzxia
"""

import argparse
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
import itertools
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.svm import SVC
import xlwt
from sklearn.neural_network import MLPClassifier
from functions import model_para

parser = argparse.ArgumentParser(description='Set Args')
parser.add_argument('--model', default="LR", type=str)
parser.add_argument('--reject', default="All", type=str)
parser.add_argument('--ran_seed',default='42',type=str)

args = parser.parse_args()

test = pd.read_csv('../data/A_test scaled.csv')
test=test.drop(['Unnamed: 0'],axis=1)

train = pd.read_csv('../data/A_train scaled.csv')
train=train.drop(['Unnamed: 0'],axis=1)

train_X = pd.DataFrame(train.drop(['Y'],axis=1))
train_Y = train['Y']

test_X = pd.DataFrame(test.drop(['Y'],axis=1))
test_Y = test['Y']

reject = pd.read_csv(f'../data/R_all_SAIL.csv')
reject = reject.drop(['Unnamed: 0'],axis=1)

reject_bad = pd.read_csv(f'../data/R_bad_SAIL.csv')
reject_bad=reject_bad.drop(['Unnamed: 0'],axis=1)

reject_X = np.array(reject.drop(['Y'],axis=1))
reject_Y = np.array(reject['Y'])

reject_bad_X = np.array(reject_bad.drop(['Y'],axis=1))
reject_bad_Y = np.array(reject_bad['Y'])

# cross validation
random.seed(int(args.ran_seed))
np.random.seed(int(args.ran_seed))

list_para = model_para.gen_paradicts(args.model)
scores = [0] * len(list_para)
best_dict = None
best_score = 0

train_val_features = train_X.values
train_val_labels = train_Y.values

test_features = test_X
test_labels = test_Y

good_idxs = np.where(train_val_labels==0)[0] 
bad_idxs = np.where(train_val_labels==1)[0]

good_idxs = list(np.random.permutation(good_idxs))
bad_idxs = list(np.random.permutation(bad_idxs))

# indexs = list(np.random.permutation(np.arange(len(train_val_features))))
cross_val_k = 5
one_kth_idx_good = len(good_idxs) // cross_val_k
one_kth_idx_bad = len(bad_idxs) // cross_val_k
for i in range(cross_val_k):
    start_idx_good = i * one_kth_idx_good 
    end_idx_good = min((i+1) * one_kth_idx_good, len(good_idxs))
    
    start_idx_bad = i * one_kth_idx_bad 
    end_idx_bad = min((i+1) * one_kth_idx_bad, len(bad_idxs))
    
    # val_idxs = indexs[start_idx:end_idx]
    # train_idxs = indexs[:start_idx] + indexs[end_idx:]
    
    val_idxs_good = good_idxs[start_idx_good:end_idx_good]
    val_idxs_bad = bad_idxs[start_idx_bad:end_idx_bad]
    
    train_idxs_good = good_idxs[:start_idx_good] + good_idxs[end_idx_good:]
    train_idxs_bad = bad_idxs[:start_idx_bad] + bad_idxs[end_idx_bad:]
    
    val_idxs = val_idxs_good + val_idxs_bad
    train_idxs = train_idxs_good + train_idxs_bad
    
    train_features = train_val_features[train_idxs].copy()
    train_labels = train_val_labels[train_idxs].copy()
    val_features = train_val_features[val_idxs].copy()
    val_labels = train_val_labels[val_idxs].copy()

    # Adding Reject Data to Train:
    # 需要调整是否要reject
    if args.reject == "None":
        pass
    elif args.reject == "Bad":
        train_features = np.concatenate([train_features,reject_bad_X])
        train_labels = np.concatenate([train_labels,reject_bad_Y])
    elif args.reject == "All":
        train_features = np.concatenate([train_features,reject_X])
        train_labels = np.concatenate([train_labels,reject_Y])

    for j,para_dict in enumerate(list_para):
        # 需要调整clf
        if args.model == "LR":
            clf = LogisticRegression(**para_dict)
        elif args.model == "DT":
            clf = DecisionTreeClassifier(**para_dict)
        elif args.model == "RF":
            clf = RandomForestClassifier(**para_dict)
        elif args.model == "GBDT":
            clf = GradientBoostingClassifier(**para_dict)
        elif args.model == "LGBM":
            clf = LGBMClassifier(**para_dict)
        elif args.model == "XGB":
            clf = XGBClassifier(**para_dict)
        elif args.model == "SVM":
            clf = SVC(**para_dict,probability=True)
        elif args.model == "MLP":
            clf = MLPClassifier(**para_dict)
    
        clf.fit(train_features,train_labels)
        Y_predict = clf.predict_proba(val_features)
        Y_predict=Y_predict[:, 1]
        roc = roc_auc_score(val_labels,Y_predict)  
        scores[j] += roc        
        
for i,score in enumerate(scores):
    if score > best_score:
        best_score = score 
        best_dict = list_para[i]
        # if roc > best_score:
        #     best_score = roc
        #     best_dict = para_dict
print(best_dict, best_score / cross_val_k)

# 需要调整clf以及是否要reject
if args.model == "LR":
    clf = LogisticRegression(**best_dict)
elif args.model == "DT":
    clf = DecisionTreeClassifier(**best_dict)
elif args.model == "RF":
    clf = RandomForestClassifier(**best_dict)
elif args.model == "GBDT":
    clf = GradientBoostingClassifier(**best_dict)
elif args.model == "LGBM":
    clf = LGBMClassifier(**best_dict)
elif args.model == "XGB":
    clf = XGBClassifier(**best_dict)
elif args.model == "SVM":
    clf = SVC(**best_dict,probability=True)
elif args.model == "MLP":
    clf = MLPClassifier(**best_dict)

if args.reject == "None":
    pass
elif args.reject == "Bad":
    train_val_features = np.concatenate([train_val_features,reject_bad_X])
    train_val_labels = np.concatenate([train_val_labels,reject_bad_Y])
elif args.reject == "All":
    train_val_features = np.concatenate([train_val_features,reject_X])
    train_val_labels = np.concatenate([train_val_labels,reject_Y])

clf.fit(train_val_features,train_val_labels)
Y_predict = clf.predict_proba(test_features)
Y_predict=Y_predict[:, 1]
roc = roc_auc_score(test_labels,Y_predict)
print('ROC= ', roc)
print(best_dict, best_score / cross_val_k)

# Outputs
clf_y_predict=pd.DataFrame()
clf_y_predict['Y_test']=test_labels
clf_y_predict['Y_predict']=Y_predict
clf_y_predict.to_csv('../main_results/SAIL/'+str(args.ran_seed)+  "_" + str(args.model)+'_Metrics.csv')

# file location
fw = open("../main_results/SAIL/AUC_result_SAIL.txt", 'a') 
fw.write("Model: " + str(args.model)) 
fw.write("\n")  
fw.write("Reject: " + str(args.reject))
fw.write("\n") 
fw.write("Random Seed: " + str(args.ran_seed))
fw.write("\n") 
fw.write("Val: " + str(best_score/cross_val_k))
fw.write("\n") 
fw.write("Test: " + str(roc))
fw.write("\n") 
fw.write("Parameters: " + str(best_dict))
fw.write("\n") 
fw.write("\n") 
fw.close()
