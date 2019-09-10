# Ernesto Elsaesser - Matrikelnummer 2016424 - fuer Python 3

import csv
import numpy
from numpy import array # required for eval in import function
import backpropagation

input_dim  = 6
hidden_dim = 15
output_dim = 2
epsilon = 0.01
property_names = ["InputLayer", "HiddenLayer", "OutputLayer", "weightsToHidden", "weightsToOutput"]

def parse(filename = "data_a_2_2016242.csv"):
    file = open(filename, newline='')
    data = csv.reader(file, delimiter=';')
    next(data) # skip first line
    in_data = numpy.zeros((0,6))
    out_data = numpy.zeros((0,2))

    def normalize(value, upper, lower):
        normalized = (value - lower) / (upper - lower)
        return min(max(normalized, 0), 1)

    for row in data:
        gender = 1 if row[0] == 'w' else 0
        height = normalize(int(row[1]), 140, 200)
        age = normalize(int(row[2]), 18, 100)
        weight = normalize(int(row[3]), 20, 150)
        strength_sports = 1 if row[4] == 'Kraftsport' else 0
        endurance_sports = 1 if row[4] == 'Ausdauersport' else 0
        underweight = 1 if row[5] == 'Untergewicht' else 0
        overweight = 1 if row[5] == 'Uebergewicht' else 0
        in_data = numpy.append(in_data, [[gender, height, age, weight, strength_sports, endurance_sports]], axis=0)
        out_data = numpy.append(out_data, [[underweight, overweight]], axis=0)

    return (in_data, out_data)

def create():
    net = backpropagation.feedForwardNetwork()
    net.configure(input_dim,hidden_dim,output_dim)
    net.init()
    net.setEpsilon(epsilon)
    net.setLearningRate(0.3)
    return net

def export(net, filename = "trained.net"):
    with open(filename, 'w') as file:
        for name in property_names:
            value = getattr(net, name)
            line = repr(value).replace("\n", "")
            file.write(line + "\n")

def import_(net, filename = "trained.net"):
    with open(filename, 'r') as file:
        for name in property_names:
            line = file.readline()
            value = eval(line)
            setattr(net, name, value)

def train(net, input_data, target_data, sample_start, sample_end):
    sample_range = range(sample_start, sample_end)
    correct_samples = 0
    round = 0

    while (correct_samples < len(sample_range)):
        round += 1

        for i in sample_range:
            outputs = compute_outputs_for_inputs(net, input_data[i])
            targets = target_data[i]
            while True:
                error = get_error(net, targets, outputs)
                if (error < epsilon):
                    break
                net.backpropagate(targets)
                outputs = get_outputs(net)

        (correct_samples, total_error) = test(net, input_data, target_data, sample_start, sample_end)
        print("Runde " + str(round) + " - Korrekte: " + str(correct_samples) + " Fehler : " + str(total_error))

def test(net, input_data, target_data, sample_start, sample_end):
    sample_range = range(sample_start, sample_end)
    correct_samples = 0
    total_error = 0

    for i in sample_range:
        outputs = compute_outputs_for_inputs(net, input_data[i])
        error = get_error(net, target_data[i], outputs)
        total_error += error
        if (error < epsilon):
            correct_samples += 1

    return (correct_samples, total_error)

def classify(net, inputs):
    outputs = compute_outputs_for_inputs(net, inputs)
    error_under = get_error(net, [1,0], outputs)
    error_over = get_error(net, [0,1], outputs)
    if (error_under < epsilon):
        return "Untergewicht"
    if (error_over < epsilon):
        return "Uebergewicht"
    return "Normalgewicht"


def classify(x):
    self.apply(x)
    error_under = self.error([1,0])
    error_over = self.error([0,1])
    if (error_under < epsilon):
        return "Untergewicht"
    if (error_over < epsilon):
        return "Uebergewicht"
    return "Normalgewicht"

def compute_outputs_for_inputs(net, inputs):
    for i in range(0,input_dim):
        net.setInput(i,inputs[i])
    return get_outputs(net)

def get_outputs(net):
    net.apply()
    outputs = numpy.empty(output_dim)
    for i in range(0,output_dim):
        outputs[i] = net.getOutput(i)
    return outputs

def get_error(net, targets, outputs):
    error = net.energy(targets, outputs, output_dim)
    return error
