# TM40507 Pruefungsleistung - Matrikelnummer 2016424
# (c) Ernesto Elsaesser

import classifier

wc = classifier.WeightClassifier()

train_filename = input("Pfad zu den Trainingsdaten (CSV): ")
wc.load_data(train_filename)
wc.train()

test_filename = input("Pfad zu den Testdaten (CSV): ")
wc.load_data(test_filename)
answer = input("Klassen ausgeben (j/n)? ")
verbose = answer == "j"
wc.test(verbose = verbose)
