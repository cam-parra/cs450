from sklearn import datasets as ds
from sklearn.cross_validation import train_test_split as tts
from random import triangular
from pandas import read_csv as pd
from numpy import array as array
from numpy import asarray as asa
from scipy.special import expit
import numpy as np
import matplotlib.pyplot as plt
from random import randint as rand


class Neurons:
    def __init__(self, size):
        self.weight = [triangular(-1.0, 1.0) for _ in range(size + 1)]
        self.threshold = 0
        self.is_bias = -1

    def output(self, data):
        data = np.append(data, [self.is_bias])
        return self.sigmoid(data)

    def sigmoid(self, item_list):
        return expit(sum(list(map(lambda x, y: x * y, self.weight, item_list))))


class MultiLayerPreceptron:
    def __init__(self):
        self.layers = None
        self.std = None
        self.attributes = None
        self.mean = None
        self.learning_rate = None
        self.classes = None
        self.target = None
        self.data = None

    def fit(self, tdata, ttarget, classes, learning_rate):
        self.mean = tdata.mean()
        self.std = tdata.std()
        self.learning_rate = learning_rate
        # a nifty way to make this function less ugly
        standarize = lambda data: (asa(data) - self.mean) / self.std
        self.data = standarize(tdata)
        self.target = ttarget
        self.attributes = self.data.shape[1]
        self.classes = classes
        self.make_preceptron(int(input("LAYERS: ")))
        self.epoch_learn(int(input("CYCLES: ")))

    def make_neural_row(self, inputs, nodes):
        return [Neurons(inputs) for _ in range(nodes)]

    def input_size(self, node_layer):
        if node_layer > 0:
            return len(self.layers[node_layer - 1])
        else:
            return self.attributes

    def neurons_in_row(self, current_layer, layers_total):
        return int(input("How many Neurons would you like in hidden layer {}?\n>> ".format(current_layer + 1))
                   if current_layer < layers_total else len(self.classes))

    def make_preceptron(self, size):
        self.layers = []
        for idx in range(size + 1):
            self.layers.append(self.make_neural_row(self.input_size(idx), self.neurons_in_row(idx, size)))

    def total(self, inputs):
        totals = []
        for index, layer in enumerate(self.layers):
            totals.append([n.output(totals[index - 1] if index > 0 else inputs) for n in layer])
        return totals

    def update(self, target, f_inputs, results):
        self.update_errors(target, results)
        self.update_all_weights(f_inputs, results)

    def update_errors(self, target, results):
        for i_layer, layer in reversed(list(enumerate(self.layers))):
            for i_neuron, neuron in enumerate(layer):
                neuron.error = self.get_error(i_neuron, i_layer, target, results)

    def get_error(self, i_neuron, i_layer, target, results):
        return self.get_hidden_error(
            results[i_layer][i_neuron], self.get_f_weights(i_neuron, i_layer), self.get_f_errors(i_layer)) \
            if i_layer < len(results) - 1 else self.get_output_error(results[i_layer][i_neuron], i_neuron == target)

    def get_f_weights(self, i_neuron, i_layer):
        return [nn.weight[i_neuron] for nn in self.layers[i_layer + 1]]

    def get_f_errors(self, i_layer):
        return [nn.error for nn in self.layers[i_layer + 1]]

    def update_all_weights(self, f_inputs, results):
        for i, layer in enumerate(self.layers):
            for n in layer:
                self.update_weights(n, results[i - 1] if i > 0 else f_inputs.tolist())

    def update_weights(self, neuron, inputs):
        inputs = inputs + [-1]
        neuron.weight = [w - self.learning_rate * inputs[i] * neuron.error for i, w in enumerate(neuron.weight)]

    def get_output_error(self, result, target):
        return result * (1 - result) * (result - target)

    def get_hidden_error(self, result, f_weights, errors):
        return result * (1 - result) * sum([fw * errors[i] for i, fw in enumerate(f_weights)])

    def epoch_learn(self, num_epochs):
        accuracy = []
        for epoch in range(num_epochs):
            predictions = []
            for d, t in zip(self.data, self.target):
                results = self.total(d)
                predictions.append(np.argmax(results[-1]))
                self.update(t, d, results)
            accuracy.append(100 * sum([self.target[i] == p for i, p in enumerate(predictions)]) / self.target.size)
            print("Accuracy for Epoch {}: {:.4f}%".format(epoch + 1, accuracy[epoch]))
        if input("Plot accuracy graph? (y/n)\n>> ") == 'y':
            plt.plot(range(1, num_epochs + 1), accuracy)
            plt.show()


def run():
    pre = MultiLayerPreceptron()
    choice = int(input("1. Iris 2. Diabetes \n>>"))
    if choice == 1:
        iris = ds.load_iris()
        training, test, training_target, test_target = tts(iris.data, iris.target, train_size=.45,
                                                           random_state=rand(1, 100000))
        pre.fit(training, training_target, iris.target_names, 0.2)
    if choice == 2:
        read_in = pd("../diabetes.csv",
                     names=["pregnant", "plasma_glucose", "blood_pressure", "triceps", "insulin", "mass",
                            "pedigree", "age", "target"], dtype=float)
        diabetes_data = read_in.ix[:, :-1].values
        diabetes_target = read_in.target.values
        training, test, training_target, test_target = tts(diabetes_data, diabetes_target, train_size=.45,
                                                           random_state=rand(1, 100000))
        pre.fit(training, training_target, ["negative", "Positive"], 0.02)


run()
