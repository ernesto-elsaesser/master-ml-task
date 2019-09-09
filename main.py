# Lösung von Matrikelnummer 2016424 (Python 3)

import lib


print("Lade Trainingsdaten ...")
(train_in, train_out) = lib.load_csv('data_a_2_2016242.csv', True) # TODO prompt for name
train_count = train_out.shape[0]
print(str(train_count) + " Beispiele importiert.")

print("Trainiere ...")
net = lib.create_nn()
lib.train(net, train_in, train_out, 100)
print("Netz erfolgreich trainiert.")

#print("Lade Testdaten ...")
#(test_in, _) = lib.load_csv('data_a_2_2016242.csv', False) # TODO prompt for name
#test_count = test_out.shape[0]
#print(str(test_count) + " Beispiele importiert.")

# Klassifikation Testdaten

test_out = lib.classify(NN, [0,0.33,0.02,0.25,0,0])
print(test_out)
