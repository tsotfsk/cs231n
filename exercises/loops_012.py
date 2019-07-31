import numpy as np

if __name__ == "__main__":
    A = np.array([[1,2,3], [4,5,6], [6,3,4]])
    B = np.array([[3,1,2], [7,8,9]])
    re = np.zeros((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        re[i, :] = np.sqrt(np.sum(abs(A[i]- B) ** 2, axis=1))
    print(re)

    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            re[i, j] = np.sqrt(np.sum(abs(A[i] - B[j]) ** 2))

    print(re)

    a = np.sum(A ** 2, axis=1).reshape(A.shape[0], 1)
    b = 2 * np.dot(A, B.T)
    c = np.sum(B ** 2, axis=1).reshape(B.shape[0], 1).T

    print(np.sqrt(a - b + c))
