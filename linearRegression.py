import numpy as np 
import scipy.stats as ss 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split


# 1) Generate example regression data
n = 100
beta_0 = 5
beta_1 = 2
np.random.seed(1) # Use a specific seed from numpy (get identical results as in the mooc)
x = 10 * ss.uniform.rvs(size=n)
y = beta_0 + beta_1 * x + ss.norm.rvs(loc=0, scale=1, size=n)
# loc is location. The location (offset) of a Gaussian is its "mean"

plt.figure()
plt.plot(x, y, "o", ms=5)
xx = np.array([0, 10])
plt.plot(xx, beta_0 + beta_1 * xx)
plt.xlabel("x")
plt.ylabel("y")
plt.show()



# 2) Simple linear regression
# We assume that the real data has the model Y = b0 + b1X + e
# We estimate \hat y = \hat b0 + \hat b1x
# RSS (residual sum of squares) = e0^2 + e1^2 + ... + en^2
# Best estimations for b0 and b1 are those that minimize RSS

# Assume that b0 is known and b1 is to be estimated [toy example]
rss = []
slopes = np.arange(-10,15,0.01)
for slope in slopes:
    rss.append(np.sum((y - beta_0 - slope * x)**2))
ind_min = np.argmin(rss)
print("Estimate for the slope: ", slopes[ind_min])
# And \hat b1 can be estimated then as slopes[ind_min]
plt.figure()
plt.plot(slopes,rss)
plt.xlabel("Slope")
plt.ylabel("RSS")
plt.show()

# Use the FIT method over a OLS model
mod = sm.OLS(y, x) # Ordinary Least Squares
est = mod.fit()
print(est.summary())
# Slope is not too accurate because the data has an offset
X = sm.add_constant(x)  # X includes a column of 1s (estimate the constant too)
mod = sm.OLS(y,X)
est = mod.fit()
print(est.summary())
# Values are much better estimated

# TSS (total sum of squares) = sum[(yi - \hat y)^2]
# RSS = sum[(yi - \hat yi)^2]
# If RSS < TSS, the model is working
# R^2 = (TSS - RSS) / TSS (percentage, between 0 and 1, 1 the better)



# 3) scikit-learn usage for linear regression
n = 500
beta_0 = 5
beta_1 = 2
beta_2 = -1

np.random.seed(1)
x_1 = 10 * ss.uniform.rvs(size=n)
x_2 = 10 * ss.uniform.rvs(size=n)
y = beta_0 + beta_1*x_1 + beta_2*x_2 + ss.norm.rvs(loc=0, scale=1, size=n)
X = np.stack([x_1, x_2], axis=1)
X.shape # (500, 2)
# Use Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:,0], X[:,1], y, c=y)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$y$")
plt.show()

# Fit the model using sklearn LinearRegression
lm = LinearRegression(fit_intercept=True) 
# In this case the data is not aligned with the origin, so there is an intercept
lm.fit(X, y)
beta_0_est = lm.intercept_
beta_1_est = lm.coef_[0]
beta_2_est = lm.coef_[1]
X_0 = np.array([2, 4]) # Select a point where x_1 is 2 and x_2 is 4
lm.predict(X_0.reshape(1,-1)) # Reshape because it is only one sample (if not, it gives a warning)
lm.score(X, y) # X matrix and the y outcome. Give R^2 value



# 4) Model accuracy
# Divide data into training and testing data using sklearn train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1)
lm = LinearRegression(fit_intercept=True)
lm.fit(X_train, y_train)
lm.score(X_test, y_test) # R^2
# Avoid underfit and overfit issues in your models!
