# Lösung von Matrikelnummer 2016424 (Python 3)

import csv
import json
import numpy
from numpy import array
import backpropagation

input_dim  = 6
hidden_dim = 15
output_dim = 2
property_names = ["InputLayer", "HiddenLayer", "OutputLayer", "weightsToHidden", "weightsToOutput"]

def parse(filename = "data_a_2_2016242.csv", include_output = True):
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
        in_data = numpy.append(in_data, [[gender, height, age, weight, strength_sports, endurance_sports]], axis=0)

        if include_output:
            underweight = 1 if row[5] == 'Untergewicht' else 0
            overweight = 1 if row[5] == 'Uebergewicht' else 0
            out_data = numpy.append(out_data, [[underweight, overweight]], axis=0)

    return (in_data, out_data)

def create():
    net = backpropagation.feedForwardNetwork()
    net.configure(input_dim,hidden_dim,output_dim)
    net.init()
    net.setEpsilon(0.01)
    net.setLearningRate(0.3)
    return net

def export(net, filename = "saved.net"):
    with open(export_filename, 'w') as file:
        for name in property_names:
            value = getattr(net, name)
            line = repr(value).replace("\n", "")
            file.write(line + "\n")

def import_(net, filename = "saved.net"):
    with open(export_filename, 'r') as file:
        for name in property_names:
            line = file.readline()
            value = eval(line)
            setattr(net, name, value)

def train(net, input_data, expected_output, sample_count):
    iterations = 0
    correct = 0
    while (correct < sample_count):
        iterations += 1

        for i in range(0,sample_count):
            outputs = compute_outputs_for_inputs(net, input_data[i])
            expected = expected_output[i]
            while True:
                error = net.energy(expected, outputs, output_dim);
                if (error < net.getEpsilon()):
                    break
                net.backpropagate(expected)
                outputs = get_outputs(net)

        correct = 0
        total_error = 0
        for i in range(0,sample_count):
            outputs = compute_outputs_for_inputs(net, input_data[i])
            error = net.energy(expected_output[i],outputs,output_dim)
            total_error += error
            if (error < net.getEpsilon()):
                correct += 1

        print("Runde " + str(iterations) + " - Korrekte: " + str(correct) + " Fehler : " + str(total_error))

def classify(net, example):
    outputs = compute_outputs_for_inputs(net, input_data[i])
    return "Untergewicht {0:.0%} | Uebergewicht {1:.0%}".format(outputs[0], outputs[1])

def compute_outputs_for_inputs(net,inputs):
    for i in range(0,input_dim):
        net.setInput(i,inputs[i])
    return get_outputs(net)

def get_outputs(net):
    outputs = numpy.empty(output_dim)
    for i in range(0,output_dim):
        outputs[i] net.getOutput(i)
    return outputs
