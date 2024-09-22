# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:16:55 2024

@author: wzxia
"""

"""
K Cluster
"""

import functools
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y
from sklearn.utils import _safe_indexing
from sklearn.metrics.pairwise import pairwise_distances_chunked
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import sklearn.datasets as ds
import matplotlib
from sklearn.metrics import calinski_harabasz_score
from sklearn.neighbors import KNeighborsClassifier

total = pd.read_csv('../data/A_train+R scaled.csv')
total = total.drop(['Unnamed: 0'],axis=1)

X = total.drop(['Y'],axis=1) # Y is the bad/good label
Y = total['Y']

# 1.  set n_cluster
n_cluster = [2,3,4,5,6,7,8,9,10,11,12]

optimal_n = []
optimal_CH_score = []

for i in n_cluster:
    optimal_n.append(i)        
    model = SpectralClustering(n_clusters=i,affinity='nearest_neighbors').fit_predict(X.values)
    CH_score = metrics.calinski_harabasz_score(X.values,model)
    optimal_CH_score.append(CH_score)
        
optimal = np.array([optimal_n,optimal_CH_score]).T
 
# 2. CH_score and WSS plots
import functools
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y
from sklearn.utils import _safe_indexing
from sklearn.metrics.pairwise import pairwise_distances_chunked
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder

def check_number_of_labels(n_labels, n_samples):
    """Check that number of labels are valid.
    Parameters
    ----------
    n_labels : int
        Number of labels.
    n_samples : int
        Number of samples.
    """
    if not 1 < n_labels < n_samples:
        raise ValueError(
            "Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)"
            % n_labels)
        
def calinski_harabasz_WSS(X, labels):
    """Compute the Calinski and Harabasz score.
    It is also known as the Variance Ratio Criterion.
    The score is defined as ratio between the within-cluster dispersion and
    the between-cluster dispersion.
    Read more in the :ref:`User Guide <calinski_harabasz_index>`.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    Returns
    -------
    score : float
        The resulting Calinski-Harabasz score.
    References
    ----------
    .. [1] `T. Calinski and J. Harabasz, 1974. "A dendrite method for cluster
       analysis". Communications in Statistics
       <https://www.tandfonline.com/doi/abs/10.1080/03610927408827101>`_
    """
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    n_samples, _ = X.shape
    n_labels = len(le.classes_)

    check_number_of_labels(n_labels, n_samples)

    extra_disp, intra_disp = 0.0, 0.0
    mean = np.mean(X, axis=0)
    
    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
        intra_disp += np.sum((cluster_k - mean_k) ** 2)
    return intra_disp # This is WSS

CH_score = []  
WSS_score = []

for k in range(2,13):
    estimator = SpectralClustering(n_clusters=k,random_state=0, affinity="nearest_neighbors").fit_predict(X.values)
    
    CH_score.append(metrics.calinski_harabasz_score(X.values, estimator))
    # 为CH_score添加质量值
    WSS_score.append(calinski_harabasz_WSS(X.values, estimator))
    # 为WSS_score添加质量值
    
optimal = pd.DataFrame()
optimal['CH_score'] = CH_score
optimal['WSS_score'] = WSS_score
i = [2,3,4,5,6,7,8,9,10,11,12]
optimal['K'] = i

# Seaborn绘图-CH_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

color = sns.color_palette("muted",4)[0]
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
# plt.figure(dpi=600) # 设置分辨率
plt.plot(optimal['K'], optimal['CH_score'], marker='o',color=color)
plt.ylabel('CH_score', fontsize = 17,labelpad=8)
plt.xlabel('K', fontsize = 17,labelpad=6)
plt.tick_params(labelsize=13.75)
# plt.show()
plt.savefig('../figures/CH_score.svg',bbox_inches='tight',dpi=600)

# Seaborn绘图-WSS_score
color = sns.color_palette("muted",4)[3]
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
# plt.figure(dpi=600) # 设置分辨率
plt.plot(optimal['K'], optimal['WSS_score'], marker='o',color=color)
plt.ylabel('WSS_score', fontsize = 17,labelpad=8)
plt.xlabel('K', fontsize = 17,labelpad=6)
plt.tick_params(labelsize=13.75)
# plt.show()
plt.savefig('../figures/WSS_score.svg',bbox_inches='tight',dpi=600)

# You can find the best k cluster based on these two figures


"""RI-SAIL"""

"""
Spectual Clustering
"""
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import pandas as pd 

total = pd.read_csv('../data/A_train+R scaled.csv')
total = total.drop(['Unnamed: 0'],axis=1)

X = total.drop(['Y'],axis=1)
Y = total['Y']

estimator = SpectralClustering(n_clusters=5, affinity="nearest_neighbors",random_state=1 ).fit_predict(X.values) 

X['Y_clus'] = estimator
X['Y'] = Y

## t-SNE test on Y_clus 
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd  
import matplotlib.pyplot as plt

all_data = X.drop(['Y','Y_clus'],axis=1)

tsne_em = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1,random_state=1).fit_transform(all_data)

df = pd.DataFrame()
df["y"] = X['Y_clus']
df["t-SNE-x"] = tsne_em[:,0]
df["t-SNE-y"] = tsne_em[:,1]

color_list = [sns.color_palette("hls", 8)[5],
              sns.color_palette("hls", 8)[1],
              sns.color_palette("husl", 8)[0], 
              sns.color_palette("hls", 8)[4], 
              sns.color_palette("hls", 8)[6]]
# plt.figure(dpi=600) # 设置分辨率
sns.set(font = "Times New Roman",rc={'axes.labelsize':17.0,'xtick.labelsize':13.75, 'ytick.labelsize':13.75, 'legend.fontsize':12})
g = sns.scatterplot(x="t-SNE-x", y="t-SNE-y", hue=df.y.tolist(),
                palette=color_list,
                data=df,s=18,alpha=1)
plt.xlim(-95, 95)
plt.ylim(-95, 95)
plt.ylabel('t-SNE-y', labelpad=-5)
plt.xlabel('t-SNE-x', labelpad=6)
plt.legend(g.get_legend_handles_labels()[0][:6],['3','1','2','4','5'],loc='lower right')
plt.savefig('../figures/TSNE_clus.svg',bbox_inches='tight',dpi=600)

# ## PCA test on reject(-1)-bad(1)-good(0)
# from sklearn.decomposition import PCA 
# all_data = X.drop(['Y','Y_clus'],axis=1)

# # Reducing the dimensions of the data 
# pca = PCA(n_components = 2,random_state=1) 
# X_principal = pca.fit_transform(all_data) 
# X_principal = pd.DataFrame(X_principal) 
# X_principal.columns = ['P1', 'P2'] 
  
# tsne_em = X_principal.values

# df = pd.DataFrame()
# df["y"] = X['Y_clus']
# df["PCA-x"] = tsne_em[:,0]
# df["PCA-y"] = tsne_em[:,1]

# color_list = [sns.color_palette("husl", 8)[0], 
#               sns.color_palette("hls", 8)[1], 
#               sns.color_palette("hls", 8)[5], 
#               sns.color_palette("hls", 8)[4], 
#               sns.color_palette("hls", 8)[6]]
# size_list = [300 if i == 1 else 0.11 for i in list(Y.values)]
# plt.figure(dpi=600) # 设置分辨率
# sns.set(font = "Times New Roman",font_scale = 1.15)
# g = sns.scatterplot(x="PCA-x", y="PCA-y", hue=df.y.tolist(),
#                 palette=color_list,
#                 data=df,size=size_list,size_order=[11,30],alpha=1)
# plt.xlim(-1, 1)
# plt.ylim(-0.9, 1.1)
# plt.legend(g.get_legend_handles_labels()[0][:6],['1','2','3','4','5'],loc='upper right')

total['Y_clus'] = X['Y_clus']
total.to_csv('../data/A_train+R culsed.csv')

import pandas as pd
import argparse
import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.linear_model import LogisticRegression
total = pd.read_csv('../data/A_train+R culsed.csv')
total = total.drop(['Unnamed: 0'],axis=1)

clus_1 = total[(total['Y_clus']== 0)]
clus_1.sort_values(by=['Y'],ascending=False,inplace=True)
clus_1.reset_index(inplace=True)
clus_1.drop(['Y_clus','index'],axis=1,inplace=True)

clus_2 = total[(total['Y_clus']== 1)]
clus_2.sort_values(by=['Y'],ascending=False,inplace=True)
clus_2.reset_index(inplace=True)
clus_2.drop(['Y_clus','index'],axis=1,inplace=True)

clus_3 = total[(total['Y_clus']== 2)]
clus_3.sort_values(by=['Y'],ascending=False,inplace=True)
clus_3.reset_index(inplace=True)
clus_3.drop(['Y_clus','index'],axis=1,inplace=True)

clus_4 = total[(total['Y_clus']== 3)]
clus_4.sort_values(by=['Y'],ascending=False,inplace=True)
clus_4.reset_index(inplace=True)
clus_4.drop(['Y_clus','index'],axis=1,inplace=True)

clus_5 = total[(total['Y_clus']== 4)]
clus_5.sort_values(by=['Y'],ascending=False,inplace=True)
clus_5.reset_index(inplace=True)
clus_5.drop(['Y_clus','index'],axis=1,inplace=True)


"""
Anomaly Detection
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

df = pd.DataFrame()

rng = np.random.RandomState(42)
def Out_detection(clus):
    A_train = clus.drop(clus[clus['Y'] == -1].index)
    A_train.reset_index(inplace=True)
    A_train.drop(['index'],axis=1,inplace=True)
    Reject = clus[(clus['Y']== -1)]
    Reject.reset_index(inplace=True)
    Reject.drop(['index'],axis=1,inplace=True)
    X_train = A_train.drop(['Y'],axis=1)
    X_reject = Reject.drop(['Y'],axis=1)
    # fit the model
    clf = IsolationForest(n_estimators=100, random_state=rng, contamination=0.1)
    clf.fit(X_train)
    y_pred_reject = clf.predict(X_reject)
    scores_reject = clf.decision_function(X_reject)
    
    # 将anomaly score还原成论文里面的范围
    # Score range of original paper
    original_paper_score = [-1*s + 0.5 for s in scores_reject]
    Reject['Outlier_score'] = original_paper_score
    
    Reject = Reject[(Reject['Outlier_score'] > 0.2)]  
    Reject = Reject[(Reject['Outlier_score'] < 0.5)]
    
    Reject.sort_values(by=['Outlier_score'],ascending=False,inplace=True)
    Reject.reset_index(inplace=True)
    Reject.drop(['index'],axis=1,inplace=True)
    
    beta_top = int(np.ceil(0 * len(Reject)))
    beta_bottom = int(np.ceil(1 * len(Reject)))
    Reject = Reject.iloc[beta_top:beta_bottom]    
    Reject.reset_index(inplace=True)
    Reject.drop(['index','Outlier_score'],axis=1,inplace=True)
    clus = pd.concat([A_train,Reject],axis=0,ignore_index=True)
    return clus 

clus = [clus_1, clus_2, clus_3, clus_4, clus_5]
new_clus = []

for i in clus:
    new_i = Out_detection(i)
    new_clus.append(new_i)
clus_1 = new_clus[0]
clus_2 = new_clus[1]
clus_3 = new_clus[2]
clus_4 = new_clus[3]
clus_5 = new_clus[4]

alphas = [0.45,0.475,0.5,0.525,0.55] 
n_neighbors = [5,7,9,11]
models = []
for neighbor_i in n_neighbors:
    for alpha_i in alphas:
        model = LabelSpreading(kernel ='knn', # {'knn', 'rbf'} default='rbf'
                        # gamma = gamma_i, # default=20, Parameter for rbf kernel.
                        n_neighbors = neighbor_i, # default=7, Parameter for knn kernel which is a strictly positive integer.
                        alpha = alpha_i, # Clamping factor. A value in (0, 1) that specifies the relative amount that an instance should adopt the information from its neighbors as opposed to its initial label. alpha=0 means keeping the initial label information; alpha=1 means replacing all initial information.
                        max_iter = 100, # default=30, Maximum number of iterations allowed.
                        tol = 0.0001, # default=1e-3, Convergence tolerance: threshold to consider the system at steady state.
                        n_jobs = -1, # default=None, The number of parallel jobs to run. -1 means using all processors.
                        )
        models.append(model)


"""
Iterative Relabelling mechanism
"""
def Self_learning(clus, model):
    R = clus[(clus['Y']== -1)]
    R.reset_index(inplace=True)
    R.drop(['index'],axis=1,inplace=True)
    n = len(R) # 初始有多少个R
    perc = 0.0 # 指定当R中剩余的sample低于perc * R时即停止self-learning
    A = clus.drop(clus[clus['Y'] == -1].index)
    A.sort_values("Y",inplace=True)
    A.reset_index(inplace=True)
    A.drop(['index'],axis=1,inplace = True)
    len_A = len(A)
    for i in range(n):
        if len(R) >= 2:
            if len(R) >= (n * perc): # 当R中剩余的样本数量不足perc*R时 就停止迭代
                m = len(R) 
                A_X = A.drop(['Y'],axis=1).values
                A_Y = A['Y'].values
                R_X = R.drop(['Y'],axis=1).values
                R_Y = R['Y'].values
                X = np.r_[A_X,R_X] # X中包含了A和R
                Y = np.r_[A_Y,R_Y] # 有-1,0,1三种标签
                model.fit(X, Y) 
                Y_pred = model.transduction_
                Y_prob = model.label_distributions_[:,1]
                R_Y_pred = Y_pred[-m:]
                R_Y_prob = Y_prob[-m:]
                R['Y_pred'] = R_Y_pred
                R['Y_prob'] = R_Y_prob
                R.sort_values(by=['Y_prob'],ascending=False,inplace=True,ignore_index=True)
                
                if R['Y_prob'].iloc[0] > 0.5:
                    R['Y'].iloc[0] = R['Y_pred'].iloc[0] 
                    R['Y'].iloc[-1] = R['Y_pred'].iloc[-1] 
                    R.drop(['Y_pred','Y_prob'],axis=1,inplace=True)
                    first_last_rows = pd.concat([R.head(1), R.tail(1)])
                    A = pd.concat([A, first_last_rows], ignore_index=True)
                    R = R.drop([0,len(R)-1])
                    R.reset_index(inplace=True)
                    R.drop(['index'],axis=1,inplace=True)
                else:
                    R['Y'].iloc[-1] = R['Y_pred'].iloc[-1]
                    R['Y'].iloc[-2] = R['Y_pred'].iloc[-2] 
                    R.drop(['Y_pred','Y_prob'],axis=1,inplace=True)
                    last_two_rows = R.tail(2)
                    A = pd.concat([A, last_two_rows], ignore_index=True)
                    R = R.drop([len(R)-1,len(R)-2])
                    R.reset_index(inplace=True)
                    R.drop(['index'],axis=1,inplace=True)
                    
        elif len(R) == 1:
            if len(R) >= (n * perc): # 当R中剩余的样本数量不足perc*R时 就停止迭代
                m = len(R) 
                A_X = A.drop(['Y'],axis=1).values
                A_Y = A['Y'].values
                R_X = R.drop(['Y'],axis=1).values
                R_Y = R['Y'].values
                X = np.r_[A_X,R_X] # X中包含了A和R
                Y = np.r_[A_Y,R_Y] # 有-1,0,1三种标签
                model.fit(X, Y) 
                Y_pred = model.transduction_
                Y_prob = model.label_distributions_[:,1]
                R_Y_pred = Y_pred[-m:]
                R_Y_prob = Y_prob[-m:]
                R['Y_pred'] = R_Y_pred
                R['Y_prob'] = R_Y_prob
                R.sort_values(by=['Y_prob'],ascending=False,inplace=True,ignore_index=True)
                
                if R['Y_prob'].iloc[0] > 0.5:
                    R['Y'].iloc[0] = R['Y_pred'].iloc[0] 
                    R.drop(['Y_pred','Y_prob'],axis=1,inplace=True)
                    A = pd.concat([A, R.head(1)], ignore_index=True)
                    R = R.drop([0])
                    R.reset_index(inplace=True)
                    R.drop(['index'],axis=1,inplace=True)
                else:
                    R['Y'].iloc[-1] = R['Y_pred'].iloc[-1]
                    R.drop(['Y_pred','Y_prob'],axis=1,inplace=True)
                    A = pd.concat([A, R.head(1)], ignore_index=True)
                    R = R.drop([0])
                    R.reset_index(inplace=True)
                    R.drop(['index'],axis=1,inplace=True)
        elif len(R) == 0:   
            break
                    
    reject_all = A.iloc[len_A:,:] 
    return reject_all

clus_all = [clus_1, clus_2, clus_3, clus_4, clus_5] 
reject_alls = []

for j in models: 
    for i in clus_all:
        if len(i.Y.unique()) == 2:
            R_all = i[(i['Y']== -1)]
            R_all.reset_index(inplace=True)
            R_all.drop(['index'],axis=1,inplace=True)
            A = i.drop(i[i['Y'] == -1].index)
            R_all['Y'] = A.Y.unique()[0]
            reject_alls.append(R_all)
        else: 
            R_all = Self_learning(i,j)
            reject_alls.append(R_all)
            
R_all_alls = []
R_all_bads = []

for i in range(100)[0:100:5]: # 5 alphas * 4 n_neighbours = 20
    clus_1_all = reject_alls[i]
    clus_2_all = reject_alls[i+1]
    clus_3_all = reject_alls[i+2]
    clus_4_all = reject_alls[i+3]
    clus_5_all = reject_alls[i+4]
    R_all_all = pd.concat([clus_1_all,clus_2_all,clus_3_all,clus_4_all,clus_5_all],ignore_index=True)
    R_all_all.reset_index(inplace=True)
    R_all_all = R_all_all.drop(['index'],axis=1)
    R_all_alls.append(R_all_all)
    
    R_all_bad = R_all_all.drop(R_all_all[R_all_all['Y'] == 0].index)
    R_all_bad.reset_index(inplace=True)
    R_all_bad = R_all_bad.drop(['index'],axis=1)
    R_all_bads.append(R_all_bad)

# 寻找最优的Label spreading的参数
# Find optimal hyperparameters for label spreading
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
from functions import model_para

test = pd.read_csv('../data/A_test scaled.csv')
test=test.drop(['Unnamed: 0'],axis=1)

train = pd.read_csv('../data/A_train scaled.csv')
train=train.drop(['Unnamed: 0'],axis=1)

train_X = pd.DataFrame(train.drop(['Y'],axis=1))
train_Y = train['Y']

test_X = pd.DataFrame(test.drop(['Y'],axis=1))
test_Y = test['Y']

classifiers = ['LR','DT'] 
# Set your classifiers to select optimal hyperparameters
# Here we use LR and DT to obtain average AUCs

AUC_result = pd.DataFrame()

for l in range(len(classifiers)):
    parser = argparse.ArgumentParser(description='Set Args')
    parser.add_argument('--model', default=classifiers[l], type=str)
    parser.add_argument('--reject', default="All", type=str)
    parser.add_argument('--method',default='Self-learning',type=str)
    args = parser.parse_args()
    
    Para_result = []
    AUC_val_result = []
    AUC_test_result = []
    Number = []

    for n_i in range(len(R_all_bads)):
        Number.append(n_i)
        
        if args.method == "Self-learning":
            reject = R_all_alls[n_i]
            reject_bad = R_all_bads[n_i]
            
        reject_X = np.array(reject.drop(['Y'],axis=1))
        reject_Y = np.array(reject['Y'])
        
        reject_bad_X = np.array(reject_bad.drop(['Y'],axis=1))
        reject_bad_Y = np.array(reject_bad['Y'])
        
        # cross validation
        random.seed(42)
        np.random.seed(42) 
        
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
        
        #indexs = list(np.random.permutation(np.arange(len(train_val_features))))
        cross_val_k = 5
        one_kth_idx_good = len(good_idxs) // cross_val_k
        one_kth_idx_bad = len(bad_idxs) // cross_val_k
        for i in range(cross_val_k):
            start_idx_good = i * one_kth_idx_good 
            end_idx_good = min((i+1) * one_kth_idx_good, len(good_idxs))
            
            start_idx_bad = i * one_kth_idx_bad 
            end_idx_bad = min((i+1) * one_kth_idx_bad, len(bad_idxs))
            
            #val_idxs = indexs[start_idx:end_idx]
            #train_idxs = indexs[:start_idx] + indexs[end_idx:]
            
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
        Para_result.append(best_dict)
        AUC_val_result.append(best_score / cross_val_k)
        
        
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
        AUC_test_result.append(roc)
        
    AUC_result[str(args.model)+'_number'] = Number
    AUC_result[str(args.model)+'_para'] = Para_result
    AUC_result[str(args.model)+"_val"] = AUC_val_result
    AUC_result[str(args.model)+"_test"] = AUC_test_result
    
AUC_result['aver_val'] = (AUC_result['LR_val'] + AUC_result['DT_val']) / 2 
AUC_result['aver_test'] = (AUC_result['LR_test'] + AUC_result['DT_test']) / 2 
index = np.argsort(AUC_result['aver_val']).iloc[-1]

model = models[index]
# LR & DT 2 LabelSpreading(alpha=0.5, kernel='knn', max_iter=100, n_jobs=-1, n_neighbors=5,tol=0.0001)

clus_all = [clus_1, clus_2, clus_3, clus_4, clus_5] 
reject_alls = []
for i in clus_all:
    if len(i.Y.unique()) == 2:
        R_all = i[(i['Y']== -1)]
        R_all.reset_index(inplace=True)
        R_all.drop(['index'],axis=1,inplace=True)
        A = i.drop(i[i['Y'] == -1].index)
        R_all['Y'] = A.Y.unique()[0]
        reject_alls.append(R_all)
    else: 
        R_all = Self_learning(i,model)
        reject_alls.append(R_all)

clus_1_all = reject_alls[0]
clus_2_all = reject_alls[1]
clus_3_all = reject_alls[2]
clus_4_all = reject_alls[3]
clus_5_all = reject_alls[4]
R_all_all = pd.concat([clus_1_all,clus_2_all,clus_3_all,clus_4_all,clus_5_all],ignore_index=True)
R_all_all.reset_index(inplace=True)
R_all_all = R_all_all.drop(['index'],axis=1)
R_all_all.to_csv(f'../data/R_all_SAIL.csv')

R_all_bad = R_all_all.drop(R_all_all[R_all_all['Y'] == 0].index)
R_all_bad.reset_index(inplace=True)
R_all_bad = R_all_bad.drop(['index'],axis=1)
R_all_bad.to_csv(f'../data/R_bad_SAIL.csv')