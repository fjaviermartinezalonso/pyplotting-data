import random
import numpy as np
import matplotlib.pyplot as plt

# roll a dice N times and plot an histogram. For a big N, the histogram becomes flat and the
# large numbers law is proved

N = 1000000
rolls = []
for k in range(N):
    rolls.append(random.choice(range(1,7)))

plt.hist(rolls, bins=np.linspace(0.5,6.5,7),density=True)
plt.title("Dice rolls")
plt.xlabel("Roll result")
plt.ylabel("Frequency of results")
plt.show()
