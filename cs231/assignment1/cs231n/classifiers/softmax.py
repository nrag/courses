import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  num_classes = W.shape[1]
  num_train = X.shape[0]

  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #compute the loss
  scores = X.dot(W) # scores becomes of size 10 x 1, the scores for each class
  scores -= np.max(scores)
  correct_score = scores[range(num_train),y]
  loss = -np.mean( np.log(np.exp(correct_score)/np.sum(np.exp(scores), axis=1))) + reg * np.sum(W ** 2)

  #compute the gradient
  p = np.exp(scores)/np.repeat(np.sum(np.exp(scores), axis=1).reshape(scores.shape[0],1), W.shape[1], axis=1)
  ind = np.zeros(p.shape)
  ind[range(num_train), y.reshape(1,scores.shape[0])] = 1
  dW = np.dot(X.T, p - ind)/num_train + 2 * reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  
  num_train = X.shape[0]

  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #compute the loss
  scores = X.dot(W) # scores becomes of size 10 x 1, the scores for each class
  scores -= np.max(scores)
  correct_score = scores[range(num_train),y]
  loss = -np.mean( np.log(np.exp(correct_score)/np.sum(np.exp(scores), axis=1))) + reg * np.sum(W ** 2)
  
  #compute the gradient
  p = np.exp(scores)/np.repeat(np.sum(np.exp(scores), axis=1).reshape(scores.shape[0],1), W.shape[1], axis=1)
  ind = np.zeros(p.shape)
  ind[range(num_train), y] = 1
  dW = np.dot(X.T, p - ind)/num_train + 2 * reg * W 

  return loss, dW

