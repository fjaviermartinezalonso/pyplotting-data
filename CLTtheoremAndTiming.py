# Roll 10 different dices N times and plot their combined result as a histogram. 
# For a big N, the histogram has approximately the shape of the prob. distribution of the experiment
# Central Limit Theorem (CLT): the sum of a large number of random variables becomes a normal distribution,
# regardless of their distribution!

# We will also do an execution time comparison between two implementations

import numpy as np
import random
import time
import matplotlib.pyplot as plt

N = 1000000             # Realizations number

# A) Standard Python
st_time = time.perf_counter()
N = 1000000
rolls = []
for k in range(N):
    ysum = 0
    for j in range(10): # roll 10 times a dice
        ysum += random.choice(range(1,7))
    rolls.append(ysum)
std_python_time = time.perf_counter() - st_time

# B) Using NumPy
st_time = time.perf_counter()
realizations = np.random.randint(1,7,size=(N,10))
# axis=1 sums over the columns, so the resulting vector has N dots
sum_realizations = np.sum(realizations,axis=1)       
numpy_time = time.perf_counter() - st_time

# C) Results
print("Standard Python implementation: ", std_python_time, " seconds")
print("NumPy implementation: ", numpy_time, " seconds")
print("Numpy is ", std_python_time/numpy_time, " times faster")

plt.hist(sum_realizations)                                           
plt.title("x10 Dice rolls experiment")
plt.xlabel("Roll result")
plt.ylabel("Frequency of results")
plt.show()
