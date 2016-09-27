import numpy as np
import pandas as pd
from sklearn import datasets as ds
from sklearn.cross_validation import train_test_split as tsp
from sklearn import preprocessing as pre

iris = ds.load_iris()
bc = ds.load_breast_cancer()
# read in data and add a header colum
car = pd.read_csv('./cars.csv', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'target'])
# make all of the row data except the last item
data_car = car.ix[:, :-1]
target_cat = car.target

class DTreeClassifier:

    def make_tree(self, he):
        pass
    

