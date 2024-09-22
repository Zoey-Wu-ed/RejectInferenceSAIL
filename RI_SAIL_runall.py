# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 21:31:14 2024

@author: wzxia
"""

import os
import sys
import itertools
models =["LR","DT","LGBM","XGB","SVM","MLP"] 
rejects = ["None","All"] 
ran_seeds = ['42','32', '22','12', '2'] 
for i in itertools.product(models, rejects, ran_seeds):
    print (i)
    model = i[0]
    reject = i[1]
    ran_seed = i[2]
    os.system("python RI_SAIL_main.py --model {} --reject {} --ran_seed {}".format(i[0], i[1], i[2]))
    
