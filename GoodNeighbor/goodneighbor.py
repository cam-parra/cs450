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


class GoodNeighborClassifier:
    target = None
    data = None
    k = None
    mean = None
    std = None
    cl = None

    def set_cl(self, cla):
        self.cl = cla

    def k_nearest(self, inputs):

        dist = np.sum((self.data - inputs) ** 2, axis=1)
        idx = np.argsort(dist, axis=0)
        classes = np.unique(self.target[idx[:self.k]])
        if len(classes) == 1:
            closest = np.unique(classes)[0]
        else:
            counts = np.zeros(max(classes) + 1)
            for i in range(self.k):
                counts[self.target[idx[i]]] += 1
            closest = np.max(counts)

        return closest
    def train(self, training, train_t):
        self.k =  3 if len(self.cl) != 3 else 5
        self.target = train_t
        x = np.asarray(training)
        self.mean = x.mean()
        self.std = x.std()
        self.data = pre.normalize(training, norm='l2')

good = GoodNeighborClassifier()
