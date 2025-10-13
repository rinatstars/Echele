import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, brentq

dl = 0.02 # 20 мкм
dlamda = 0.01 # 10 пм
f = 310
lambdaM = 380 # nm
l = 13 # mm

def disp(dl, dlambda):
    return dl/dlambda

def k1(lambdaM, disp, l):
    return lambdaM * disp / l - 1/2

def p(lambdaM, lambdam, k1):
    return (lambdaM / lambdam - 1) * (k1 - 1/2)

def N(disp, f, k1, fi):
    return 1000000 * disp * np.cos(np.radians(fi)) / (f * k1)



fi_val = np.linspace(0, 90, 90)

#plt.figure(figsize=(12, 8))

for dlambda in np.linspace(0.005, 0.015, 11):
    N_val = [N(disp(dl, dlambda), f, k1(lambdaM, disp(dl, dlambda), l), fi) for fi in fi_val]

    #plt.plot(fi_val, N_val, label=f'{np.round(k1(lambdaM, disp(dl, dlambda), l), decimals=0)}, {np.round(p(lambdaM, 160, k1(lambdaM, disp(dl, dlambda), l)))}, {np.round(dlambda, decimals = 3)}')

#plt.grid(True)
#plt.legend()
#plt.show()

print()