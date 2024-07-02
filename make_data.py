import numpy as np
import random
from scipy.spatial.transform import Rotation as R

from utils import stack, TF_random, skew2vec, log, vec2skew, exp



def make_exact_data(n):
    X = TF_random()
    A = [None]*n
    B = [None]*n
    for i in range(n):
        A[i] = TF_random()
        B[i] = np.linalg.inv(X) @ A[i] @ X
    return [X, A, B]


def make_noisy_data(n):
    X = TF_random()
    X[:3,3] *= 1e1
    A = [None]*n
    B = [None]*n
    for i in range(n):
        A[i] = TF_random()
        B[i] = np.linalg.inv(X) @ A[i] @ X
        alpha = skew2vec(log(A[i][:3,:3]))
        beta = skew2vec(log(B[i][:3,:3]))
        alpha += (np.random.rand(3)*2-1)*3.14e-2
        beta += (np.random.rand(3)*2-1)*3.14e-2
        A[i][:3,:3] = exp(vec2skew(alpha))
        B[i][:3,:3] = exp(vec2skew(beta))
        A[i][:3,3] += (np.random.rand(3)*2-1)*1e-1
        B[i][:3,3] += (np.random.rand(3)*2-1)*1e-1
    return [X, A, B]


def check_exact_data(X,A,B):
    n = len(A)
    assert n == len(B)
    for i in range(n):
        assert np.allclose(A[i]@X,X@B[i]), "X=\n{}\nA=\n{}\nB=\n{}\n".format(X, A[i], B[i]) + "AX=\n{}\nXB=\n{}".format(A[i]@X, X@B[i]) + "\nAX-XB=\n{}".format(A[i]@X-X@B[i]) + "\n{}".format(np.isclose(A[i]@X,X@B[i]))

def check_similar_data(X,A,B,rtol=1e-05, atol=1e-08):
    n = len(A)
    assert n == len(B)
    for i in range(n):
        A[i]@X
        X@B[i]
        # assert np.allclose(A[i]@X,X@B[i],rtol=rtol,atol=atol), "X=\n{}\nA=\n{}\nB=\n{}\n".format(X, A[i], B[i]) + "AX=\n{}\nXB=\n{}".format(A[i]@X, X@B[i]) + "\nAX-XB=\n{}".format(A[i]@X-X@B[i]) + "\n{}".format(np.isclose(A[i]@X,X@B[i]))



def save(X,A,B,filename='temp.npy'):
    with open(filename, 'wb') as f:
        np.save(f, X)
        np.save(f, A)
        np.save(f, B)

def load(filename='temp.npy'):
    with open(filename, 'rb') as f:
        X = np.load(f)
        A = np.load(f)
        B = np.load(f)
    return [X, A, B]



if __name__ == "__main__":
    # X, A, B = make_exact_data(100)
    X, A, B = make_noisy_data(10)
    save(X,A,B)
    X,A,B = load()
    print("\n")
    print("X")
    print(X)
    print("A")
    print(A[0])
    print()
    print(A[0]@X)
    print(X@B[0])