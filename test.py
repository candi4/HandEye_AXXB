import numpy as np

from AXXB import AXXBsolver
from make_data import load, check_exact_data, check_similar_data

X, A, B = load("noisy_data.npy")
error = []
for i in range(2,len(A)+1):
    axxb = AXXB(A[:i],B[:i])
    X_est = axxb.solve()
    error.append(np.linalg.norm((X-X_est).reshape((-1,))))
print(error)