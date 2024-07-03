import numpy as np

from make_data import load, check_exact_data, check_similar_data
from AXXB.utils import log, skew2vec, stack

def exact_solve(X,A,B):
    # check_exact_data(X,A,B)
    # rtol=1e-2
    # atol=1e-2
    # check_similar_data(X,A,B,rtol=rtol,atol=atol)

    theta_A0 = A[0][:3,:3]
    theta_A1 = A[1][:3,:3]
    theta_B0 = B[0][:3,:3]
    theta_B1 = B[1][:3,:3]

    alpha0 = skew2vec(log(theta_A0))
    alpha1 = skew2vec(log(theta_A1))
    beta0 = skew2vec(log(theta_B0))
    beta1 = skew2vec(log(theta_B1))

    AA = np.zeros((3,3))
    AA[:,0] = alpha0
    AA[:,1] = alpha1
    AA[:,2] = np.cross(alpha0,alpha1)
    BB = np.zeros((3,3))
    BB[:,0] = beta0
    BB[:,1] = beta1
    BB[:,2] = np.cross(beta0,beta1)
    theta_X = AA @ np.linalg.inv(BB)
    # assert np.allclose(X[:3,:3], theta_X), "\ntheta_X=\n{}\nanswer=\n{}\ntheta_X-answer=\n{}".format(theta_X,X[:3,:3],theta_X-X[:3,:3])

    b_A0 = A[0][:3,3]
    b_A1 = A[1][:3,3]
    b_B0 = B[0][:3,3]
    b_B1 = B[1][:3,3]


    C = np.vstack((theta_A0-np.identity(3),theta_A1-np.identity(3)))
    d = np.hstack((theta_X@b_B0 - b_A0,theta_X@b_B1 - b_A1))
    b_X = np.linalg.inv(C.transpose()@C)@C.transpose()@d

    # assert np.allclose(X[:3,3], b_X), "\n{}\n{}".format(b_X,X[:3,3])

    X_est = stack(theta_X, b_X)
    return X_est
    

if __name__ == "__main__":
    # X, A, B = load("exact_data.npy")
    A1 = np.array([[-0.989992, -0.14112,  0.000, 0],
                 [0.141120 , -0.989992, 0.000, 0],
                 [0.000000 ,  0.00000, 1.000, 0],
                 [0        ,        0,     0, 1]])

    B1 = np.array([[-0.989992, -0.138307, 0.028036, -26.9559],
                    [0.138307 , -0.911449, 0.387470, -96.1332],
                    [-0.028036 ,  0.387470, 0.921456, 19.4872],
                    [0        ,        0,     0, 1]])

    A2 = np.array([[0.07073, 0.000000, 0.997495, -400.000],
                    [0.000000, 1.000000, 0.000000, 0.000000],
                    [-0.997495, 0.000000, 0.070737, 400.000],
                    [0, 0, 0,1]])

    B2 = np.array([[ 0.070737, 0.198172, 0.997612, -309.543],
                    [-0.198172, 0.963323, -0.180936, 59.0244],
                    [-0.977612, -0.180936, 0.107415, 291.177],
                    [0, 0, 0, 1]])
    A=[A1,A2]
    B=[B1,B2]
    X=np.array([[ 1.          ,0.         , 0.         , 9.99989033],
                [ 0.          ,0.98014571 ,-0.19827854 ,50.00001805],
                [ 0.          ,0.19827854 , 0.98014571 ,99.99990214],
                [ 0.          ,0.         , 0.         , 1.        ]])
    X_est = exact_solve(X,A,B)
    print(X)
    print(X_est)
