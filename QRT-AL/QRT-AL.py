import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing, tree
import scipy
from math import sqrt

# ==========================================================
# 1. ACTIVE LEARNING PARAMETERS
# ==========================================================

maxlabel1 = 20    #initial set of samples labeled by random sampling or model-free AL method
maxlabel4 = [maxlabel1]

maxlabel2=[40]    # total training set size
maxlabel3 = [maxlabel2[0]-maxlabel1]
for i in range(1,len(maxlabel2)):
    maxlabel3 = np.append(maxlabel3,maxlabel2[i]-maxlabel2[i-1])  # number of samples to be labeled in current AL step

from tree_q import Regression_Tree_q

MOFs_selected_rtq=[]
MOF_rt_newq = []

# ==========================================================
# 2. QUANTILE INTERVALS (used for leaf selection)
# ==========================================================

quantile1=0.05
quantile2=0.4
quantile3=0.95
quantile4=1

# ==========================================================
# 3. LOAD TRAINING POOL
# ==========================================================

df = pd.read_csv('racs_train.csv').dropna() #features
X = df.to_numpy()
feat=len(X[0])-1
print('Length of feature vector = ', feat)

# ==========================================================
# 4. LOAD INITIAL LABELED SET
# ==========================================================

# set of initially labeled samples (selected and labeled randomly)
init_samples_path = 'train_init.csv' 
init_samples = np.genfromtxt(init_samples_path, dtype=str, delimiter=',')

# labels of initially labeled samples
Y_path = 'Y_train_init.csv' # path to .csv of id 
Y = np.genfromtxt(Y_path, delimiter=',')
print(Y)
print('Initial samples = ', init_samples)
print('Total training pool =' , len(X))

indices=[]
train_pool=X
initial_training_set=[]
for i,j in enumerate(X[:,0]):
    if j in init_samples:
        indices = np.append(indices,i)
        initial_training_set = np.append(initial_training_set,X[i])
initial_training_set = np.array(initial_training_set)       
initial_training_set = np.hsplit(initial_training_set,len(initial_training_set)/int(feat+1))
print('Initial training set size =',len(initial_training_set))

# current training pool
for i in (-np.sort(-indices)):
    train_pool = np.delete(train_pool,(int(i)),axis=0)
print('Current training pool = ',len(train_pool))
print('Labeled indices = ',indices.astype(int))

# ==========================================================
# 5. VISUALIZE INITIAL LABEL DISTRIBUTION
# ==========================================================

import matplotlib.pyplot as plt
plt.hist(Y,10)

X_train_full = X
X_train = X[:,1:len(X[0])-1]
Y_train = Y

X_train_newq = initial_training_set
Y_train_newq = Y    

# ==========================================================
# 6. TRAIN REGRESSION TREE (RTq)
# ==========================================================

# input the data into the tree and fit it using the current training set
RTq = Regression_Tree_q(seed=treeseed,min_samples_leaf = 5)
RTq.input_data(X_train, indices.astype(int), Y_train[:maxlabel1])
RTq.fit_tree()
d1 = RTq.decision_path()
print(d1)
d2 = RTq.decision_path2()
print(d2)
tree_plot = RTq.plot_tree()
print(tree_plot)
plt.show()
plt.savefig('tree.png', dpi=300)

for p in indices.astype(int):
        MOF_rt_newq = np.append(MOF_rt_newq,X_train_full[p][0])

# ==========================================================
# 7. ACTIVE LEARNING LOOP
# ==========================================================

for i,j in enumerate(maxlabel2):

    # compute the threshold values of the quntiles based on the current training set
    q1 = np.quantile(Y_train_newq,quantile1)
    q2 = np.quantile(Y_train_newq,quantile2)
    q3 = np.quantile(Y_train_newq,quantile3)
    q4 = np.quantile(Y_train_newq,quantile4)
    print(q1,q2,q3,q4)

    # Retrieve labeled and unlabeled indices per leaf

    # returns the full set of labeled and unlabeled indices
    labeled_indices, unlabeled_indices = RTq.return_labeled_and_unlabeled_indices()
    leaves  = labeled_indices.keys()
    
    # returns the unlabeled (or labeled) indices in each leaf
    for leaf in leaves:
        print(X[unlabeled_indices[leaf],0])

    # Compute leaf proportions for AL selection
        
    # compute relavant quantities required to pick new points
    RTq.al_calculate_leaf_proportions(q1,q2,q3,q4)
    
    # pick the new points from the respective leaves
    new_points = RTq.pick_new_points(num_samples = maxlabel3[i])
    print(new_points)

    # add the new points to the training set
    for k in new_points:        
        X_train_newq = np.append(X_train_newq,X_train_full[k])
        MOF_rt_newq = np.append(MOF_rt_newq,X_train_full[k][0])

    X_train_newq = np.hsplit(X_train_newq,len(X_train_newq)/int(feat+1))

# ==========================================================
# 8. SAVE SELECTED SAMPLES
# ==========================================================

MOFs_selected_rtq = MOF_rt_newq
np.savetxt("RTq_selected.csv", MOFs_selected_rtq, fmt="%s",delimiter=",")
np.savetxt("QRTAL_selected_train.csv", X_train_newq, fmt="%s",delimiter=",")

print(new_points, X[new_points,0])
print(leaves)
