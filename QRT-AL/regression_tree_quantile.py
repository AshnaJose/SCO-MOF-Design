"""
Regression_Tree_q

Quantile Active Learning using sklearn's DecisionTreeRegressor.

This class:
- Trains a regression tree on partially labeled data
- Tracks labeled and unlabeled samples
- Computes per-leaf statistics (variance, density, quantile weighting)
- Selects new samples using a quantile-weighted active learning strategy

Dependencies:
- sklearn
- numpy
- var.py (must contain unbiased_variance function)
"""

from sklearn.tree import DecisionTreeRegressor
from collections import Counter
import var as var
import numpy as np
import matplotlib 
#matplotlib.use('AGG')
import matplotlib.pyplot as plt
import copy
from sklearn import tree as tree_export
from sklearn import tree

class Regression_Tree_q:

    def __init__(self, min_samples_leaf=None, seed=None):

        self.points = None
        self.labels = None
        self.labeled_indices = None
        self._num_points = 0
        self._num_labeled = 0

        if seed is None:
            self.seed = 0
        else:
            self.seed = seed

        if min_samples_leaf is None:
            self.min_samples_leaf=1
        else:
            self.min_samples_leaf=min_samples_leaf

        self.tree = DecisionTreeRegressor(random_state=self.seed,min_samples_leaf=self.min_samples_leaf)
        self._leaf_indices = []
        self._leaf_marginal = []
        self._leaf_var = []
        self._al_proportions =[]

        self._leaf_statistics_up_to_date = False
        self._leaf_proportions_up_to_date = False

        self._verbose = False

    def input_data(self, all_data, labeled_indices, labels, copy_data=True):
    
        if copy_data:
            all_data = copy.deepcopy(all_data)
            labeled_indices = copy.deepcopy(labeled_indices)
            labels = copy.deepcopy(labels)

        if len(all_data) < len(labeled_indices):
            raise ValueError('Cannot have more labeled indicies than points')

        if len(labeled_indices) != len(labels):
            raise ValueError('Labeled indicies list and labels list must be same length')

        if str(type(all_data)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting all_data to list of lists internally')
            all_data = all_data.tolist()

        if str(type(labeled_indices)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting labeled_indices to list internally')
            labeled_indices = labeled_indices.tolist()

        if str(type(labels)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting labels to list internally')
            labels = labels.tolist()

        self.points = all_data
        self._num_points = len(self.points)
        self._num_labeled = len(labels)

        # Making a label list, with None in places where we don't have the label (y)

        temp = [None] * self._num_points
        for i,ind in enumerate(labeled_indices):
            temp[ind] = labels[i]
        self.labels = temp
        self.labeled_indices = list(labeled_indices)
        
    def decision_path(self):
        return(tree.plot_tree(self.tree))
    
    def decision_path2(self):
        return(tree_export.export_text(self.tree))
        
    def plot_tree(self):
        return(tree.plot_tree(self.tree))

    def fit_tree(self):
        self.tree.fit(np.array(self.points)[self.labeled_indices,:], 
            np.array(self.labels)[self.labeled_indices])
        self._leaf_indices = self.tree.apply(np.array(self.points)) #return index of leaf for each point, labeled and unlabeled
        self._leaf_statistics_up_to_date = False
        
    def get_depth(self):
        return(self.tree.get_n_leaves())

    def label_point(self, index, value):

        if self.labels is None:
            raise RuntimeError('No data in the tree')

        if len(self.labels) <= index:
            raise ValueError('Index {} larger than size of data in tree'.format(index))

        value = copy.copy(value)
        index = copy.copy(index)

        self.labels[index] = value
        self.labeled_indices.append(index)
        self._num_labeled += 1

    def predict(self, new_points):
        return(self.tree.predict(new_points))
    
    def return_labeled_and_unlabeled_indices(self):
        labeled_indices={}
        unlabeled_indices={}
        for key in np.unique(self._leaf_indices):
            u = [i for i,x in enumerate(self.labels) if x is None and self._leaf_indices[i]==key] 
            l = [i for i,x in enumerate(self.labels) if x is not None and self._leaf_indices[i]==key] 
            labeled_indices['leaf_'+str(key)]=l
            unlabeled_indices['leaf_'+str(key)]=u
        return(labeled_indices,unlabeled_indices)
    
    def calculate_leaf_statistics(self,q1,q2,q3,q4):
        temp = Counter(self._leaf_indices) #get the no. of points in different leaves, thus density
        print(temp)
        self._leaf_marginal = []
        self._leaf_var = []
        self._quantile = []
        for key in np.unique(self._leaf_indices):
            self._leaf_marginal.append(temp[key]/self._num_points)  #proportion of each leaf
            temp_ind = [i for i,x in enumerate(self._leaf_indices) if x == key]

            #temp_labels = [x for x in self.labels if x is not None]
            temp_labels = [x for i,x in enumerate(self.labels) if x is not None and self._leaf_indices[i]==key]   #set of y in leaf
            quant_labels4 = [x for x in temp_labels if x >= q3 and x <= q4]
            quant_labels3 = [x for x in temp_labels if x >= q2 and x <= q3]
            quant_labels2 = [x for x in temp_labels if x >= q1 and x <= q2]
            quant_labels1 = [x for x in temp_labels if x <= q1]
            self._leaf_var.append(var.unbiased_variance(temp_labels))
            self._quantile.append(len(quant_labels4)*0.05 + len(quant_labels3)*0.70 + len(quant_labels2)*0.20 + len(quant_labels1)*0.05/len(temp_labels))
        self._leaf_statistics_up_to_date = True
        

    def al_calculate_leaf_proportions(self,q1,q2,q3,q4):
        if not self._leaf_statistics_up_to_date:
            self.calculate_leaf_statistics(q1,q2,q3,q4)
        al_proportions = []
        for i, val in enumerate(self._leaf_var):
            al_proportions.append(np.sqrt(self._leaf_var[i] * self._leaf_marginal[i] * self._quantile[i] ))
        al_proportions = np.array(al_proportions)/sum(al_proportions)
        self._al_proportions = al_proportions
        self._leaf_proportions_up_to_date = True

    def pick_new_points(self, num_samples = 1): 
        if not self._leaf_proportions_up_to_date:
            self.al_calculate_leaf_proportions()

        temp = Counter(np.array(self._leaf_indices)[[x for x in range(self._num_points
            ) if self.labels[x] is None]])
        point_proportions = {}
        for i,key in enumerate(np.unique(self._leaf_indices)):
            point_proportions[key] = self._al_proportions[i] / max(1,temp[key]) 
        temp_probs = np.array([point_proportions[key] for key in self._leaf_indices])
        temp_probs[self.labeled_indices] = 0
        temp_probs = temp_probs / sum(temp_probs)
        if 'NaN' in temp_probs:
            return(temp,temp_probs,sum(temp_probs))
        # print(sum(temp_probs))
        leaves_to_sample = np.random.choice(self._leaf_indices,num_samples, 
            p=temp_probs, replace = False)     #leaves to be sampled from have been selected
                
        #points to label randomly based on leaf proportions
        points_to_label = []
        for leaf in np.unique(leaves_to_sample):
            points = []
            for j in range(Counter(leaves_to_sample)[leaf]):
            
                possible_points = np.setdiff1d([x for i,x in enumerate(range(self._num_points)
                    ) if self._leaf_indices[i] ==leaf and self.labels[i] is None ], points)
                                               
                point_to_label = np.random.choice(possible_points)
                points_to_label.append(point_to_label)
                points.append(point_to_label)  

        return(points_to_label)
