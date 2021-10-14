import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("density.csv")
r=data[:,0]
rho=data[:,1]

plt.scatter(r, r**2*rho, s=0.1)
plt.show()
