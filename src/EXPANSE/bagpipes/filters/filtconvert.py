import numpy as np

file = "F460M_LePhare.dat"

X = np.genfromtxt(file, usecols=(0)) * 1e4
Y = np.genfromtxt(file, usecols=(1))

out = np.column_stack((X, Y))
print("Saving")
np.savetxt("F460M_LePhare.txt", out)
