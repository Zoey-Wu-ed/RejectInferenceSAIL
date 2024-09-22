# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:18:44 2024

@author: wzxia
"""

import pandas as pd

total = pd.read_csv('../data/A_train+R scaled.csv')
total = total.drop(['Unnamed: 0'],axis=1)


## t-SNE test on reject(-1)-bad(1)-good(0)
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd  
import matplotlib.pyplot as plt

X = total.drop(['Y'],axis=1)
Y = total['Y']
all_data = X
tsne_em = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1,random_state=1).fit_transform(all_data)

df = pd.DataFrame()
df["y"] = Y
df["t-SNE-x"] = tsne_em[:,0]
df["t-SNE-y"] = tsne_em[:,1]

size_list = [300 if i == 1 else 0.11 for i in list(Y.values)]
color_list = [sns.color_palette("muted",4)[0],sns.color_palette("muted",4)[2], sns.color_palette("muted",4)[3]]
sns.set(font = "Times New Roman",rc={'axes.labelsize':17.0,'xtick.labelsize':13.75, 'ytick.labelsize':13.75, 'legend.fontsize':12})
g = sns.scatterplot(x="t-SNE-x", y="t-SNE-y", hue=df.y.tolist(),
                palette=color_list,
                data=df,size=size_list,size_order=[11,30],alpha=1)
plt.xlim(-95, 95)
plt.ylim(-95, 95)
plt.ylabel('t-SNE-y', labelpad=-5)
plt.xlabel('t-SNE-x', labelpad=6)
plt.legend(g.get_legend_handles_labels()[0][:3],['Reject','Good','Bad'],loc='upper right')
plt.savefig('../figures/TSNE_all.svg',bbox_inches='tight',dpi=600)


## t-SNE test on bad(1)
all_data = X
tsne_em = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1,random_state=1).fit_transform(all_data)

df = pd.DataFrame()
idx_list = [i for i,j in enumerate(list(Y.values)) if j==1]
df["y"]=Y.values[idx_list]
df["t-SNE-x"] = tsne_em[idx_list,0]
df["t-SNE-y"] = tsne_em[idx_list,1]

color_list = [sns.color_palette("muted",4)[3]]
sns.set(font = "Times New Roman",rc={'axes.labelsize':17.0,'xtick.labelsize':13.75, 'ytick.labelsize':13.75, 'legend.fontsize':12})
g = sns.scatterplot(x="t-SNE-x", y="t-SNE-y", hue=df.y.tolist(),
                palette=color_list,
                data=df,s=75,alpha=1)
plt.xlim(-95, 95)
plt.ylim(-95, 95)
plt.ylabel('t-SNE-y', labelpad=-5)
plt.xlabel('t-SNE-x', labelpad=6)
plt.legend(g.get_legend_handles_labels()[0][:1],['Bad'],loc='upper right')
plt.savefig('../figures/TSNE_bad.svg',bbox_inches='tight',dpi=600)


## t-SNE test on bad(1)-good(0)
all_data = X
tsne_em = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1,random_state=1).fit_transform(all_data)

df = pd.DataFrame()
idx_list = [i for i,j in enumerate(list(Y.values)) if j!=-1]
df["y"] = Y.values[idx_list]
df["t-SNE-x"] = tsne_em[idx_list,0]
df["t-SNE-y"] = tsne_em[idx_list,1]

color_list = [sns.color_palette("muted",4)[2], sns.color_palette("muted",4)[3]]
sns.set(font = "Times New Roman",rc={'axes.labelsize':17.0,'xtick.labelsize':13.75, 'ytick.labelsize':13.75, 'legend.fontsize':12})
g = sns.scatterplot(x="t-SNE-x", y="t-SNE-y", hue=df.y.tolist(),
                palette=color_list,
                data=df,size_order=[11,30],alpha=1)
plt.xlim(-95, 95)
plt.ylim(-95, 95)
plt.ylabel('t-SNE-y', labelpad=-5)
plt.xlabel('t-SNE-x', labelpad=6)
plt.legend(g.get_legend_handles_labels()[0][:2],['Good','Bad'],loc='upper right')
plt.savefig('../figures/TSNE_bad_good.svg',bbox_inches='tight',dpi=600)


## t-SNE test on reject(-1)
all_data = X
tsne_em = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1,random_state=1).fit_transform(all_data)

df = pd.DataFrame()
idx_list = [i for i,j in enumerate(list(Y.values)) if j==-1]
df["y"] = Y.values[idx_list]
df["t-SNE-x"] = tsne_em[idx_list,0]
df["t-SNE-y"] = tsne_em[idx_list,1]

color_list = [sns.color_palette("muted",4)[0]]
sns.set(font = "Times New Roman",rc={'axes.labelsize':17.0,'xtick.labelsize':13.75, 'ytick.labelsize':13.75, 'legend.fontsize':12})
g = sns.scatterplot(x="t-SNE-x", y="t-SNE-y", hue=df.y.tolist(),
                palette=color_list,
                data=df,s=20,alpha=1)
plt.xlim(-95, 95)
plt.ylim(-95, 95)
plt.ylabel('t-SNE-y', labelpad=-5)
plt.xlabel('t-SNE-x', labelpad=6)
plt.legend(g.get_legend_handles_labels()[0][:1],['Reject'],loc='upper right')
plt.savefig('../figures/TSNE_reject.svg',bbox_inches='tight',dpi=600)