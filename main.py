# Lösung von Matrikelnummer 2016424 (Python 3)

import lib

train_filename = input("Pfad zu den Trainingsdaten: ")
print("Lade Daten aus CSV-Datei ...")
(train_in, train_out) = lib.parse(train_filename, True)
train_count = train_out.shape[0]
print(str(train_count) + " Beispiele geladen.")

sample_count = input("Anzahl zu nutzender Trainingsbeispiele: ")
if sample_count < 1 || sample_count > train_count:
    sample_count = train_count
print("Es wird mit " + train_count + " Beispielen trainiert.")
print("Trainiere neuronales Netz ...")
net = lib.create()
lib.train(net, train_in, train_out, 0, sample_count)
print("Netz erfolgreich trainiert.")

test_filename = input("Pfad zu den Testdaten: ")
print("Lade Daten aus CSV-Datei ...")
(test_in, test_out) = lib.parse(test_filename, False)
test_count = test_out.shape[0]
print(str(test_count) + " Beispiele geladen.")

print("Klassifiziere Testdaten ...")
(test_correct, test_error) = lib.test(net, test_in, test_out, 0, test_count)
accuracy = test_correct / test_count
print("{0}/{1} Beispiele richtig klassifiziert ({2:.0%})".format(test_correct, test_count, accuracy))
answer = input("Klassifikationsergebnisse ausgeben (j/n)? ")
if answer != "j":
    exit()
print("\n---- ERGEBNISSE ----\n")
for i in range(0,test_count):
    classification = lib.classify(net, test_in[i])
    print(str(i) + ": " + classification)
