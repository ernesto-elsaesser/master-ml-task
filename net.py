import csv
import numpy as np
from numpy import array # required for eval in import function

class NeuralNetwork:

    def __init__(self, in_count, hidden_count, out_count, epsilon, learning_rate):
        self.in_count = in_count
        self.hidden_count = hidden_count
        self.out_count = out_count
        self.syn0 = 2 * np.random.random((in_count + 1, hidden_count)) - 1
        self.syn1 = 2 * np.random.random((hidden_count + 1, out_count)) - 1
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    def transfer(self, x):
        return (1/(1 + np.exp(-x))) # sigmoid

    def transfer_deriv(self, x):
        return x*(1-x)
    
    def feed(self, x, y, backpropagate):
        l0p = x
        l0 = np.append(l0p, 1)
        l1p = self.transfer(np.dot(l0, self.syn0))
        l1 = np.append(l1p, 1)
        l2 = self.transfer(np.dot(l1, self.syn1))
        l2_error = y - l2

        error_sum = np.sum(np.square(l2_error)) / 2

        if backpropagate and error_sum > self.epsilon:
            l2_delta = l2_error * self.transfer_deriv(l2)
            syn1p = np.delete(self.syn1, self.hidden_count, 0)
            l1_error = l2_delta.dot(syn1p.T)
            l1_delta = l1_error * self.transfer_deriv(l1p)
            syn1_delta = self.learning_rate * np.outer(l1p, l2_delta)
            syn0_delta = self.learning_rate * np.outer(l0p, l1_delta)
            self.syn1 += np.append(syn1_delta, [np.zeros(self.out_count)], 0)
            self.syn0 += np.append(syn0_delta, [np.zeros(self.hidden_count)], 0)

        return error_sum

    def train(self, xs, ys, sample_range):
        round = 0
        wrong_classifications = True
        while (wrong_classifications):
            round += 1

            for i in sample_range:
                x = xs[i]
                y = ys[i]
                while self.feed(x, y, True) > self.epsilon:
                    continue

            (correct, total_error) = self.test(xs, ys, sample_range)
            wrong_classifications = correct < len(sample_range)

            print("Runde " + str(round) + " - Korrekte: " + str(correct) + " Fehler : " + str(total_error))

    def test(self, xs, ys, sample_range):
        correct = 0
        total_error = 0

        for i in sample_range:
            error = self.feed(xs[i], ys[i], False)
            total_error += error
            if (error < self.epsilon):
                correct += 1

        return (correct, total_error)
    
    def export(self, filename = "trained.net"):
        with open(filename, 'w') as file:
            file.write(repr(self.syn0).replace("\n", "") + "\n")
            file.write(repr(self.syn1).replace("\n", "") + "\n")

    def import_(self, filename = "trained.net"):
        with open(filename, 'r') as file:
            self.syn0 = eval(file.readline())
            self.syn1 = eval(file.readline())

class BMIData:

    def __init__(self, train_count = 100, test_count = 500, filename = "data_a_2_2016242.csv"):
        self.train_range = range(0, train_count)
        self.test_range = range(train_count, train_count + test_count)
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

    def train(self, net):
        start = np.datetime64('now')
        net.train(self.xs, self.ys, self.train_range)
        end = np.datetime64('now')
        print("Trainingsdauer: " + str(end - start))

    def test(self, net):
        (correct, _) = net.test(self.xs, self.ys, self.test_range)
        count = len(self.test_range)
        accuracy = correct / count
        print("{0}/{1} korrekt ({2:.0%})".format(correct, count, accuracy))

def make():
    return NeuralNetwork(6, 15, 2, 0.01, 0.3)