import csv
import numpy as np
from numpy import array # required for eval in import function

class WeightClassifier:

    def __init__(self, epsilon = 0.01, learning_rate = 0.3):
        self.net = FeedForwardNet(6, 15, 2, epsilon, learning_rate)
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
        self.ys = np.zeros((0,2))

        print("Lese CSV-Datei ...")
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

        self.sample_count = self.ys.shape[0]
        print(str(self.sample_count) + " Beispiele aus Datei " + filename + " ausgelesen.")
        
    def normalize(self, value, upper, lower):
        normalized = (value - lower) / (upper - lower)
        return min(max(normalized, 0), 1)

    def train(self, from_index = 0, to_index = 10000):
        start = np.datetime64('now')
        train_range = range(max(from_index, 0), min(to_index, self.sample_count))
        print("Trainiere mit " + str(len(train_range)) + " Beispielen ...")
        self.net.train(self.xs, self.ys, train_range)
        end = np.datetime64('now')
        print("Trainingsdauer: " + str(end - start))

    def test(self, from_index = 0, to_index = 10000, print_classes = False):
        test_range = range(max(from_index, 0), min(to_index, self.sample_count))
        (outputs, matches) = self.net.test(self.xs, self.ys, test_range)

        if print_classes:
            for i in test_range:
                class_name = self.classify(outputs[i])
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

    def propagate_layer(self, inputs, weights):
        inputs_ext = np.append(inputs, 1)
        outputs = self.transfer(np.dot(inputs_ext, weights))
        return outputs

    def propagate(self, x):
        hidden_layer = self.propagate_layer(x, self.in_to_hidden_weights)
        out_layer = self.propagate_layer(hidden_layer, self.hidden_to_out_weights)
        return (hidden_layer, out_layer)

    def overall_error(self, outputs, y):
        return np.sum(np.square(y - outputs)) / 2

    def adjust_weights(self, inputs, outputs, error, weights):
        scaled_error = error * self.transfer_deriv(outputs)
        delta = self.learning_rate * np.outer(inputs, scaled_error)
        delta_ext = np.append(delta, [np.zeros(outputs.shape[0])], 0)
        weights += delta_ext # TODO test

    def backpropagate(self, x, hidden, out, y):
        out_error = y - out
        self.adjust_weights(hidden, out, out_error, self.hidden_to_out_weights)
        scaled_out_error = out_error * self.transfer_deriv(out)
        weights_without_tresholds = np.delete(self.hidden_to_out_weights, self.hidden_count, 0)
        hidden_error= scaled_out_error.dot(weights_without_tresholds.T)
        self.adjust_weights(x, hidden, hidden_error, self.in_to_hidden_weights)

    def train(self, xs, ys, sample_range):
        round = 0
        pending = 1
        while (pending > 0):
            round += 1

            for i in sample_range:
                x = xs[i]
                y = ys[i]
                (hidden, out) = self.propagate(x)
                overall_error = self.overall_error(out, y)
                if round == 1 and i % 10 == 0:
                    print("Initiale Backpropagation ({0} - {1})".format(i, i + 9))
                while overall_error > self.epsilon:
                    self.backpropagate(x, hidden, out, y)
                    (hidden, out) = self.propagate(x)
                    overall_error = self.overall_error(out, y)

            pending = len(sample_range)
            total_error = 0
            for i in sample_range:
                (_, out) = self.propagate(xs[i])
                overall_error = self.overall_error(out, ys[i])
                if (overall_error < self.epsilon):
                    pending -= 1
                total_error += overall_error

            print("Runde " + str(round) + " - Falsche: " + str(pending) + " Fehler : " + str(total_error))
    
    def test(self, xs, ys, sample_range):
        count = len(sample_range)
        outs = np.zeros((count, self.out_count))
        matches = np.zeros((count, 1), dtype=int)

        for i in sample_range:
            (_, out) = self.propagate(xs[i])
            outs[i] = out
            if (self.overall_error(out, ys[i]) < self.epsilon):
                matches[i] = 1

        return (outs, matches)

    def serialize(self):
        array = [self.in_to_hidden_weights, self.hidden_to_out_weights]
        return repr(array)

    def deserialize(self, raw):
        array = eval(raw)
        self.in_to_hidden_weights = array[0]
        self.hidden_to_out_weights = array[1]