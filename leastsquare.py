import numpy as np

from make_data import load, check_exact_data, check_similar_data
from utils import log, skew2vec, stack, sqrtm

def ls_solve(X,A,B):
    # check_exact_data(X,A,B)
    rtol=None
    atol=None
    check_similar_data(X,A,B,rtol=rtol,atol=atol)

    M = np.zeros((3,3))
    C = np.zeros((0,3))
    d = np.zeros((0))
    for i in range(len(A)):
        theta_A = A[i][:3,:3]
        theta_B = B[i][:3,:3]
        alpha = skew2vec(log(theta_A))
        beta = skew2vec(log(theta_B))
        M += beta[:,None]@alpha[None,:]
    theta_X = np.linalg.inv(sqrtm(M.transpose()@M))@M.transpose()

    C = np.zeros((0,3))
    d = np.zeros((0))
    for i in range(len(A)):
        theta_A = A[i][:3,:3]
        theta_B = B[i][:3,:3]
        b_A = A[i][:3,3]
        b_B = B[i][:3,3]
        C = np.vstack((C, np.identity(3)-theta_A))
        d = np.hstack((d, b_A-theta_X@b_B))
    print(C.shape)
    print(d.shape)
    b_X = np.linalg.inv(C.transpose()@C)@C.transpose()@d

    X_est = stack(theta_X, b_X)
    return X_est
    

if __name__ == "__main__":
    X, A, B = load("noisy_data.npy")
    # X, A, B = load("exact_data.npy")
    # A1 = np.array([[-0.989992, -0.14112,  0.000, 0],
    #              [0.141120 , -0.989992, 0.000, 0],
    #              [0.000000 ,  0.00000, 1.000, 0],
    #              [0        ,        0,     0, 1]])

    # B1 = np.array([[-0.989992, -0.138307, 0.028036, -26.9559],
    #                 [0.138307 , -0.911449, 0.387470, -96.1332],
    #                 [-0.028036 ,  0.387470, 0.921456, 19.4872],
    #                 [0        ,        0,     0, 1]])

    # A2 = np.array([[0.07073, 0.000000, 0.997495, -400.000],
    #                 [0.000000, 1.000000, 0.000000, 0.000000],
    #                 [-0.997495, 0.000000, 0.070737, 400.000],
    #                 [0, 0, 0,1]])

    # B2 = np.array([[ 0.070737, 0.198172, 0.997612, -309.543],
    #                 [-0.198172, 0.963323, -0.180936, 59.0244],
    #                 [-0.977612, -0.180936, 0.107415, 291.177],
    #                 [0, 0, 0, 1]])
    # A=[A1,A2]
    # B=[B1,B2]
    # X=np.array([[ 1.          ,0.         , 0.         , 9.99989033],
    #             [ 0.          ,0.98014571 ,-0.19827854 ,50.00001805],
    #             [ 0.          ,0.19827854 , 0.98014571 ,99.99990214],
    #             [ 0.          ,0.         , 0.         , 1.        ]])
    print("data length=",len(A))
    error = []
    for i in range(2,len(A)+1):
        X_est = ls_solve(X,A[:i],B[:i])
        error.append(np.linalg.norm((X-X_est).reshape((-1,))))
    print(error)
    print("X")
    print(X)
    print("X_est")
    print(X_est)
    print("X-X_est")
    print(X-X_est)
    print()
    print("AX")
    print(A[0]@X)
    print("XB")
    print(X@B[0])
    print("AX-XB")
    print(A[0]@X-X@B[0])