import numpy as np


def HMC_Learning_Parameters(dataset):
        
    Pi, A, B = {}, {}, {}
        
    for Z in dataset:

        xt = Z[0][0]
        yt = Z[0][1]

        Pi[xt] = Pi.get(xt, 0) + 1
        
        if xt not in B:
            B[xt] = {}
        B[xt][yt] = B[xt].get(yt, 0) + 1
        
        for xt_plus_1, yt_plus_1 in Z[1:]:
            Pi[xt_plus_1] = Pi.get(xt_plus_1, 0) + 1

            if xt not in A:
                A[xt] = {}
            A[xt][xt_plus_1] = A[xt].get(xt_plus_1, 0) + 1

            if xt_plus_1 not in B:
                B[xt_plus_1] = {}
            B[xt_plus_1][yt_plus_1] = B[xt_plus_1].get(yt_plus_1, 0) + 1

            xt, yt = xt_plus_1, yt_plus_1

            
    sum_Pi = np.sum(list(Pi.values()))
    for x in Pi:
        Pi[x] = Pi[x]/sum_Pi
        
    for x1 in A:
        sum_Ax1 = np.sum(list(A[x1].values()))
        for x2 in A[x1]:
            A[x1][x2] = A[x1][x2]/sum_Ax1
            
    for x in B:
        sum_Bx = np.sum(list(B[x].values()))
        for y in B[x]:
            B[x][y] = B[x][y]/sum_Bx
    
    
    return Pi, A, B