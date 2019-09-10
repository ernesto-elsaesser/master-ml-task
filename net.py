import csv
import numpy as np
from numpy import array # required for eval in import function

class BMIData:

    def __init__(self, filename = "data_a_2_2016242.csv"):
        self.xs = np.zeros((0,6))
        self.ys = np.zeros((0,2))

        with open(filename, newline='') as file:
            data = csv.reader(file, delimiter=';')
            next(data) # skip first lineexit
            for row in data:
                gender = 1 if row[0] == 'w' else 0
                height = self.normalize(int(row[1]), 140, 200)
                age = self.normalize(int(row[2]), 18, 100)
                weight = self.normalize(int(row[3]), 20, 150)
                strength_sports = 1 if row[4] == 'Kraftsport' else 0
                endurance_sports = 1 if row[4] == 'Ausdauersport' else 0
                underweight = 1 if row[5] == 'Untergewicht' else 0
                overweight = 1 if row[5] == 'Uebergewicht' else 0
                self.xs = np.append(self.xs, [[gender, height, age, weight, strength_sports, endurance_sports]], 0)
                self.ys = np.append(self.ys, [[underweight, overweight]], 0)

    def normalize(self, value, upper, lower):
        normalized = (value - lower) / (upper - lower)
        return min(max(normalized, 0), 1)

class BMINet:

    def __init__(self):
        self.net = FeedForwardNet(6, 15, 2, 0.01, 0.3)

    def load(self):
        with open("trained.net", 'r') as file:
            raw = file.read()
            self.net.deserialize(raw)

    def save(self):
        with open("trained.net", 'w') as file:
            raw = self.net.serialize()
            file.write(raw)

    def train(self, data, range_start, range_end):
        start = np.datetime64('now')
        train_range = range(range_start, range_end)
        self.net.train(data.xs, data.ys, train_range)
        end = np.datetime64('now')
        print("Trainingsdauer: " + str(end - start))

    def test(self, data, range_start, range_end):
        test_range = range(range_start, range_end)
        (correct, _) = self.net.test(data.xs, data.ys, test_range)
        count = len(test_range)
        accuracy = correct / count
        print("{0}/{1} korrekt ({2:.0%})".format(correct, count, accuracy))

    def classify(self, data, index):
        (_, out) = self.net.apply(data.xs[index])
        if (1 - out[0] < self.net.epsilon):
            return "Untergewicht"
        if (1 - out[1] < self.net.epsilon):
            return "Uebergewicht"
        return "Normalgewicht"


class FeedForwardNet:

    def __init__(self, in_count, hidden_count, out_count, epsilon, learning_rate):
        self.in_count = in_count
        self.hidden_count = hidden_count
        self.out_count = out_count
        self.in_to_hidden_weights = 2 * np.random.random((in_count + 1, hidden_count)) - 1
        self.hidden_to_out_weights = 2 * np.random.random((hidden_count + 1, out_count)) - 1
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    def transfer(self, x):
        return (1/(1 + np.exp(-x))) # sigmoid

    def transfer_deriv(self, x):
        return x*(1-x)

    def compute_layer(self, inputs, weights):
        inputs_ext = np.append(inputs, 1)
        outputs = self.transfer(np.dot(inputs_ext, weights))
        return outputs

    def apply(self, x):
        hidden_layer = self.compute_layer(x, self.in_to_hidden_weights)
        out_layer = self.compute_layer(hidden_layer, self.hidden_to_out_weights)
        return (hidden_layer, out_layer)

    def overall_error(self, outputs, y):
        return np.sum(np.square(y - outputs)) / 2

    def adjust_weights(self, inputs, outputs, error, weights):
        scaled_error = error * self.transfer_deriv(outputs)
        delta = self.learning_rate * np.outer(inputs, scaled_error)
        delta_ext = np.append(delta, [np.zeros(outputs.shape[0])], 0)
        weights += hidden_to_out_delta_ext # TODO test

    def backpropagate(self, x, hidden, out, y):
        out_error = y - out
        self.adjust_weights(hidden, out, out_error, self.hidden_to_out_weights)
        scaled_out_error = out_error * self.transfer_deriv(out)
        weights_without_tresholds = np.delete(self.hidden_to_out_weights, self.hidden_count, 0)
        hidden_error= scaled_out_error.dot(weights_without_tresholds.T)
        self.adjust_weights(x, hidden, hidden_error, self.in_to_hidden_weights)

    def train(self, xs, ys, sample_range):
        round = 0
        learned_all = False
        while (not learned_all):
            round += 1

            for i in sample_range:
                x = xs[i]
                y = ys[i]
                (hidden, out) = self.apply(x)
                overall_error = self.overall_error(out, y)
                while overall_error > self.epsilon:
                    self.backpropagate(x, hidden, out, y)
                    (hidden, out) = self.apply(x)
                    overall_error = self.overall_error(out, y)

            (correct, total_error) = self.test(xs, ys, sample_range)
            learned_all = correct == len(sample_range)

            print("Runde " + str(round) + " - Korrekte: " + str(correct) + " Fehler : " + str(total_error))

    def test(self, xs, ys, sample_range):
        correct = 0
        total_error = 0

        for i in sample_range:
            (_, out) = self.apply(xs[i])
            overall_error = self.overall_error(out, y)
            if (overall_error < self.epsilon):
                correct += 1
            total_error += overall_error

        return (correct, total_error)
    
    def serialize(self):
        array = [self.in_to_hidden_weights, self.hidden_to_out_weights]
        return repr(array)

    def deserialize(self, raw):
        array = eval(raw)
        self.in_to_hidden_weights = array[0]
        self.hidden_to_out_weights = array[1]