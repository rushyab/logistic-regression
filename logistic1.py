import numpy as np

N = 100
D = 2

# 100 x 2 normally distributed data matrix
X = np.random.randn(N,D)
# ones = np.array([[1]*N]).T # old
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)

w = np.random.randn(D + 1)

z = Xb.dot(w)
print("output shape:", z.shape)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# all the values lie between 0 and 1
print(sigmoid(z))
