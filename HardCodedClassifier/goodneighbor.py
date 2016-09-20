import numpy as np
from sklearn import datasets as ds

iris = ds.load_iris()


class HardCodedClassifier:
    def __init__(self):
        pass

    @staticmethod
    def train_machine(train_set, train_target):
        print("Machine is trained, me Lord.")

    @staticmethod
    def predict(train_set):
        x = []
        for i in train_set:
            x.append("setsoa")
        percent = (len(x) / 150) * 100
        print(len(train_set), "where tested. ", percent, "% are setosa.")

        return x


    def k_nearest(self, k, data, data_class, inputs):

        n_inputs = inputs.shape[0]
        closest = np.zeros(n_inputs)

        for x in range(n_inputs):

            distances = np.sum((data - inputs[x, :])**2, axis=1)
            indicies = np.argsort(distances, axis=0)
            classes = np.unique(data_class[indicies[:k]])
            if len(classes) == 1:
                closest[x] = np.unique(classes)
            else:
                counts = np.zeros(max(classes)+1)
                for i in range(k):
                    counts[data_class[indicies[i]]] += 1
                closest[x] = np.max(counts)

            return closest


hc = HardCodedClassifier()
randomarray = np.array([[ 5.8, 2.8 , 5.1,  2.4]])
hc.k_nearest(4, iris.data, iris.target, randomarray)

print(hc.k_nearest(4, iris.data, iris.target, randomarray))