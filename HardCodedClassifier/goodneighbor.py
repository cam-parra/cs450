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

    @staticmethod
    def k_nearest(k, data, data_class, inputs):

        size_of_data = int(input("How many dimensions does your array have?\n>> "))

        n_inputs = inputs.shape[0]
        if (size_of_data % 2 == 0 and k % 2 == 0) or ():
            print("Both K and the dimensions are even making k odd\n...")
            k += 1
            print("K is now ", k)


        closest = np.zeros(n_inputs)

        for x in range(n_inputs):

            distances = np.sum((data - inputs[x, :]) ** 2, axis=1)
            indicies = np.argsort(distances, axis=0)
            classes = np.unique(data_class[indicies[:k]])
            if len(classes) == 1:
                closest[x] = np.unique(classes)
            else:
                counts = np.zeros(max(classes) + 1)
                for i in range(k):
                    counts[data_class[indicies[i]]] += 1
                closest[x] = np.max(counts)

            return closest



hc = HardCodedClassifier()
randomarray = np.array([[1.8, 2.8, 4.1, 2.1], [5.8, 2.8, 5.1, 2.4], [5.8, 2.8, 5.1, 2.4]])
hc.k_nearest(4, iris.data, iris.target, randomarray)

print(hc.k_nearest(4, iris.data, iris.target, randomarray))
