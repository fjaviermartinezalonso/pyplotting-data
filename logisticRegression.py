import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
#%matplotlib notebook
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


output_path = "./output/"

# 1) Generate data for classification: two independent 2D Gaussian
h = 1.5 # X-axis offset. Separates the origin of the distributions
std1 = 1
std2 = 1.5
n = 1000

# Data distributions are X1 (x1, y1) and X2 (x2, y2)
def gen_data(n, h, std1, std2):
    x1 = ss.norm.rvs(-h, std1, n)
    y1 = ss.norm.rvs(0, std1, n)
    x2 = ss.norm.rvs(h, std2, n)
    y2 = ss.norm.rvs(0, std2, n)
    return (x1, y1, x2, y2)

def plot_data(x1, y1, x2, y2):
    plt.figure()
    plt.plot(x1, y1, "o", ms=2)
    plt.plot(x2, y2, "o", ms=2)
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")

(x1, y1, x2, y2) = gen_data(n,h,std1,std2)
plot_data(x1, y1, x2, y2)
plt.show()



# 2) Classify the data using Logistic Regression
clf = LogisticRegression() # Instance of classifier object
X = np.vstack((np.vstack((x1, y1)).T, np.vstack((x2, y2)).T)) 
# Vstack first turns (N,) vectors into (1,N) for row-wise stack.
# Then, the result is (2,N), so we need to transpose it
y = np.hstack((np.repeat(1,n), np.repeat(2,n)))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
# Obtain probabilities for a specific point to be in class 1 or class 2
clf.predict_proba(np.array([-2, 0]).reshape(1,-1)) # Reshape because it is only one data point



# 3) Computing predictive probabilities across the grid
def plot_probs(ax, clf, class_no):
    # ax parameter is needed because of the contour method
    xx1, xx2 = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1))    # range of values for X1 and X2 (-5 to 5)
    probs = clf.predict_proba(np.stack((xx1.ravel(), xx2.ravel()), axis=1)) # (N,2)
    Z = probs[:,class_no] # Extract probs for class_no (class number)
    Z = Z.reshape(xx1.shape) # Change shape to be a matrix. We want to plot a 2D graph!
    CS = ax.contourf(xx1, xx2, Z) # Contour plots the values of Z in the meshgrid xx1, xx2
    plt.colorbar(CS)
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")

plt.figure(figsize=(5,8))
ax = plt.subplot(211)
plot_probs(ax, clf, 0)
plt.title("Pred. prob for class 1")
ax = plt.subplot(212)
plot_probs(ax, clf, 1)
plt.title("Pred. prob for class 2")
plt.savefig(output_path + "logisticRegressionPlot.png")
