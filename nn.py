import csv
import numpy as np
from numpy import array # required for deserialization

class WeightClassifier:

    def __init__(self, hidden_neurons = 15, epsilon = 0.01, learning_rate = 0.3):
        self.net = FeedForwardNet(6, hidden_neurons, 2, epsilon, learning_rate)
        self.sample_count = 0

    def load_weights(self, filename = "trained.net"):
        with open("trained.net", 'r') as file:
            raw = file.read()
            self.net.deserialize(raw)
        print("Gewichte aus Datei " + filename + " geladen.")

    def save_weights(self, filename = "trained.net"):
        with open("trained.net", 'w') as file:
            raw = self.net.serialize()
            file.write(raw)
        print("Gewichte in Datei " + filename + " gespeichert.")
    
    def load_data(self, filename = "data_a_2_2016242.csv"):
        self.xs = np.zeros((0,6))
        self.targets = np.zeros((0,2))

        print("Lese CSV-Datei ...")
        with open(filename, newline='') as file:
            data = csv.reader(file, delimiter=';')
            next(data) # skip first line
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
                self.targets = np.append(self.targets, [[underweight, overweight]], 0)

        self.sample_count = self.targets.shape[0]
        print(str(self.sample_count) + " Beispiele aus Datei " + filename + " ausgelesen.")
        
    def normalize(self, value, upper, lower):
        normalized = (value - lower) / (upper - lower)
        return min(max(normalized, 0), 1)

    def train(self, from_index = 0, to_index = 10000):
        start = np.datetime64('now')
        train_range = range(max(from_index, 0), min(to_index, self.sample_count))
        print("Trainiere mit " + str(len(train_range)) + " Beispielen ...")
        self.net.train(self.xs, self.targets, train_range)
        end = np.datetime64('now')
        print("Trainingsdauer: " + str(end - start))

    def test(self, from_index = 0, to_index = 10000, print_classes = False):
        test_range = range(max(from_index, 0), min(to_index, self.sample_count))
        (ys, matches) = self.net.test(self.xs, self.targets, test_range)

        if print_classes:
            for i in test_range:
                class_name = self.classify(ys[i])
                status = "falsch" if matches[i] == 0 else "richtig"
                print("{0}: {1} [{2}]".format(i, class_name, status))

        count = len(test_range)
        correct = np.sum(matches)
        accuracy = correct / count
        print("Test-Ergebnis: {0}/{1} richtig ({2:.0%})".format(correct, count, accuracy))

    def classify(self, y):
        if (1 - y[0] < self.net.epsilon):
            return "Untergewicht"
        if (1 - y[1] < self.net.epsilon):
            return "Uebergewicht"
        return "Normalgewicht"


class FeedForwardNet:

    def __init__(self, in_count, hidden_count, out_count, epsilon, learning_rate):
        self.INPUT = 0
        self.HIDDEN = 1
        self.OUTPUT = 2
        self.neuron_count = [in_count, hidden_count, out_count]
        weights_from_in = 2 * np.random.random((in_count + 1, hidden_count)) - 1
        weights_from_hidden = 2 * np.random.random((hidden_count + 1, out_count)) - 1
        self.weights_from = [weights_from_in, weights_from_hidden]
        sigmoid = lambda x: (1/(1 + np.exp(-x)))
        sigmoid_deriv = lambda x: x * (1-x)
        self.transfer_for = [sigmoid, sigmoid]
        self.transfer_deriv_for = [sigmoid_deriv, sigmoid_deriv]
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    def set_transfer(self, layer, f, fd):
        self.transfer_for[layer] = f
        self.transfer_deriv_for[layer] = fd

    def propagate_layer(self, layer, levels):
        levels_ext = np.append(levels, 1) # treshold
        t = self.transfer_for[layer]
        w = self.weights_from[layer]
        return t(np.dot(levels_ext, w))

    def propagate(self, x):
        h = self.propagate_layer(self.INPUT, x)
        y = self.propagate_layer(self.HIDDEN, h)
        return (h, y)

    def overall_error(self, y, target):
        return np.sum(np.square(target - y)) / 2

    def adjust_weights(self, layer, levels_left, levels_right, error):
        t = self.transfer_deriv_for[layer]
        scaled_error = error * t(levels_right)
        delta = self.learning_rate * np.outer(levels_left, scaled_error)
        delta_ext = np.append(delta, np.zeros((1, error.shape[0])), 0)
        self.weights_from[layer] += delta_ext
        return scaled_error # reuse for next layer

    def backpropagate(self, x, h, y, target):
        error = target - y
        scaled_error = self.adjust_weights(self.HIDDEN, h, y, error)
        weights_from_hidden = self.weights_from[self.HIDDEN]
        hidden_error = scaled_error.dot(weights_from_hidden[:-1].T)
        self.adjust_weights(self.INPUT, x, h, hidden_error)

    def train(self, xs, targets, sample_range):
        round = 0
        pending = 1
        while (pending > 0):
            round += 1

            for i in sample_range:
                x = xs[i]
                target = targets[i]
                (h, y) = self.propagate(x)
                overall_error = self.overall_error(y, target)
                if round == 1 and i % 10 == 0:
                    print("Initiale Backpropagation ({0} - {1})".format(i, i + 9))
                while overall_error > self.epsilon:
                    self.backpropagate(x, h, y, target)
                    (h, y) = self.propagate(x)
                    overall_error = self.overall_error(y, target)

            pending = len(sample_range)
            total_error = 0
            for i in sample_range:
                (_, y) = self.propagate(xs[i])
                overall_error = self.overall_error(y, targets[i])
                if (overall_error < self.epsilon):
                    pending -= 1
                total_error += overall_error

            print("Runde " + str(round) + " - Falsche: " + str(pending) + " Fehler : " + str(total_error))
    
    def test(self, xs, targets, sample_range):
        count = len(sample_range)
        ys = np.zeros((count, self.out_count))
        matches = np.zeros((count, 1), dtype=int)

        for i in sample_range:
            (_, y) = self.propagate(xs[i])
            ys[i] = y
            if (self.overall_error(y, targets[i]) < self.epsilon):
                matches[i] = 1

        return (ys, matches)

    def serialize(self):
        array = [self.in_to_hidden_weights, self.hidden_to_out_weights]
        return repr(array)

    def deserialize(self, raw):
        array = eval(raw)
        self.in_to_hidden_weights = array[0]
        self.hidden_to_out_weights = array[1]