from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i].T
                dW[:,y[i]] -= X[i].T

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 修改了上面的代码，见39~49行

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 1.先求Q=X dot W
    # 2.通过y为索引求出Qy,也就是正确的评分列
    # 3.然后将Q用broadcast减去Qy+1得到dev_Q
    # 4.以Qy为索引访问dev_Q将所在yi位置的置0
    # 5.将小于0的位置也要置0
    # 6.dev_Q中挑选出>0的位置，得到Index矩阵
    # 7.以Index矩阵为索引，访问Q然后求和，进而再经过简单的求和，平均等求出loss

    num_train = X.shape[0]
    Q = X.dot(W)
    Qy = Q[np.arange(num_train), y]
    Qy = np.reshape(Qy, (num_train, 1))
    dev_Q = Q - Qy + 1
    dev_Q[np.arange(num_train), y] = 0
    dev_Q[dev_Q < 0] = 0
    Index = dev_Q[dev_Q > 0]
    loss += np.sum(Index) / num_train
    loss += reg * np.sum(W * W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 1.求梯度需要知道哪些位置大于0，将这些位置赋值为1
    # 2.yi位置牵涉到要减的次数，可以用行的sum来得到需要减几次
    # 3.用X和yi点乘就可以得到grad相关的矩阵，除以N就可以得到核心部分的grad
    # 详情见svm的梯度推导.wmf
    dev_Q[dev_Q > 0] = 1
    row_sum = np.sum(dev_Q, axis=1)
    dev_Q[np.arange(num_train), y] = -row_sum
    dW = X.T.dot(dev_Q) / num_train + 2 * reg * W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
