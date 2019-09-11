import nn
import sys

hidden_neurons = int(sys.argv[1])
epsilon = float(sys.argv[2])
learning_rate = float(sys.argv[3])
n = nn.WeightClassifier(hidden_neurons, epsilon, learning_rate)
n.load_data()
n.train(0,2000)
n.test(2000,10000)

# ------- 0 - 50 ----------

# 15 0.05 0.5 
# Trainingsdauer: 5 seconds
# Test-Ergebnis: 7878/10000 richtig (79%)

# 20 0.05 0.5
# Trainingsdauer: 4 seconds
# Test-Ergebnis: 7773/10000 richtig (78%)

# 25 0.05 0.5
# Trainingsdauer: 4 seconds
# Test-Ergebnis: 7796/10000 richtig (78%)

# 15 0.01 0.5
# Trainingsdauer: 10 seconds
# Test-Ergebnis: 7677/10000 richtig (77%)

# 15 0.05 0.3
# Trainingsdauer: 6 seconds
# Test-Ergebnis: 7786/10000 richtig (78%)

# 15 0.05 0.7
# Trainingsdauer: 4 seconds
# Test-Ergebnis: 7532/10000 richtig (75%)

# 12 0.05 0.5
# Trainingsdauer: 4 seconds
# Test-Ergebnis: 7861/10000 richtig (79%)

# 15 0.1 0.5
# Trainingsdauer: 3 seconds
# Test-Ergebnis: 7759/10000 richtig (78%)


# ------- 100 - 200 ----------

# 15 0.05 0.5
# Trainingsdauer: 48 seconds
# Test-Ergebnis: 7460/10000 richtig (75%)

# 25 0.05 0.5
# Trainingsdauer: 47 seconds
# Test-Ergebnis: 7684/10000 richtig (77%)

# 20 0.1 0.5
# Trainingsdauer: 47 seconds
#Test-Ergebnis: 7887/10000 richtig (79%)
# Trainingsdauer: 45 seconds
# Test-Ergebnis: 7670/10000 richtig (77%)

# 20 0.05 0.5
# Trainingsdauer: 47 seconds
# Test-Ergebnis: 7628/10000 richtig (76%)

# 30 0.05 0.5
# Trainingsdauer: 58 seconds
# Test-Ergebnis: 7453/10000 richtig (75%)

# 20 0.15 0.5
# Trainingsdauer: 60 seconds
# Test-Ergebnis: 8008/10000 richtig (80%)

# 20 0.2 0.5
# Trainingsdauer: 1 seconds
# Test-Ergebnis: 8222/10000 richtig (82%)

# 20 0.25 0.5
# Trainingsdauer: 0 seconds
# Test-Ergebnis: 6300/10000 richtig (63%)

# 15 0.2 0.5
# Trainingsdauer: 1 seconds
# Test-Ergebnis: 8318/10000 richtig (83%)


# ------- 0 - 200 ----------

