import numpy as np


class CustomLogisticRegression:
  """
  LogisticRegression
  """

  def __init__(self, learning_rate=0.01, num_iters=3000, normalize=True):
    self.learning_rate = learning_rate
    self.num_iters = num_iters
    self.normalize = normalize

  def __normalize(self, X):
    self.mean = np.mean(X, axis=0)
    self.std = np.std(X, axis=0)
    X_new = (X - self.mean) / self.std
    return X_new

  def __prepare_X(self, X):
    m = X.shape[0]
    ones = np.ones((m, 1))
    X_new = np.column_stack((ones, X))
    return X_new

  def __sigmoid(self, z):
    g = 1 / (1 + np.exp(-z))
    return g

  def __h(self, X):
    z = np.dot(X, self.theta)
    return self.__sigmoid(z)

  def __compute_cost(self, X, y):
    m = X.shape[0]
    if m == 0:
        return None

    h_X = self.__h(X)
    J = - np.sum(y*np.log(h_X) + (1 - y) * np.log(1 - h_X))/m
    return J

  def __derivative_theta(self, X, y):
    m = X.shape[0]
    if m == 0:
        return None

    d_theta = np.dot(X.T, (self.__h(X) - y))/m

    return d_theta

  def __gradient_descent(self, X, y, epsilon=0.001, print_J=True):
    self.J_history = []

    J = self.__compute_cost(X, y)

    if print_J == True:
      print(J)
    self.J_history.append(J)
    for i in range(self.num_iters):
      self.theta = self.theta - self.learning_rate * self.__derivative_theta(X, y)

      J = self.__compute_cost(X, y)
      self.J_history.append(J)

      if i % 1000 == 0 and print_J == True:
        print(J)
      if epsilon != None and abs(J-self.J_history[-2]) < epsilon:
        break

  def fit(self, X, y, print_cost=False, epsilon=None):
    X_new = self.__normalize(X) if self.normalize else X
    
    X_new = self.__prepare_X(X_new)

    y_new = y.values.reshape((X.shape[0], 1))

    self.theta = np.zeros((X_new.shape[1], 1))

    self.__gradient_descent(X_new, y_new, epsilon, print_cost)

  def predict(self, X):
    if(self.normalize):
      X = (X - self.mean)/self.std
    
    predictions = self.__h(self.__prepare_X(X))
    return predictions > 0.5
