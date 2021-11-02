import numpy as np


class HMC_Viterbi_Restorator:

    def __init__(self, Omega_X, Pi, A, B):
        self.Omega_X = Omega_X
        self.N = len(Omega_X)
        self.Pi = Pi
        self.A = A
        self.B = B

    def restore_X(self, Y):
        T = len(Y)
        V, X_path = self.forward(Y)
        X_hat = self.viterbi_path(T, V, X_path)
        return X_hat

    def forward(self, Y):
        T = len(Y)
        V = np.zeros((T, self.N))
        X_path = np.zeros((T, self.N))

        V[0] = self.compute_V1(Y[0])
        for t in range(T - 1):
            V[t + 1], X_path[t + 1] = self.compute_V_t_plus_1(Y[t + 1], V[t])

        return V, X_path

    def viterbi_path(self, T, V, X_path):
        X = [None] * T
        V_max = 0
        X[T-1] = self.Omega_X[0]
        for i in range(0,self.N-1):
            if V_max <= V[T-1][i]:
                V_max = V[T-1][i]
                X[T-1] = self.Omega_X[i]
        for t in range(T-2, -1, -1):
            if t != 0:
                index_X = self.Omega_X.index(X[t+1])
                X[t] = self.Omega_X[int(X_path[t+1][index_X])]
            else:
                X[t] = X[t+1]
        return X

    ####################
    ### V and X_path ###
    ####################

    def compute_V1(self, y0):
        V1 = np.zeros(self.N)
        temp = 0
        for k in self.Omega_X:
            temp += self.B.get(k, 0).get(y0, 0)
        for ind, i in enumerate(self.Omega_X):
            if temp == 0:
                V1[ind] = self.Pi.get(i, 0)
            else:
                V1[ind] = self.Pi.get(i, 0) * self.B.get(i, 0).get(y0, 0)
        V1 /= np.sum(V1)
        return V1

    def compute_V_t_plus_1(self, yt_plus_1, Vt):
        Vt_plus_1 = np.zeros(self.N)
        X_path_t_plus_1 = np.zeros(self.N)
        temp = 0
        for k in self.Omega_X:
            temp += self.B.get(k, 0).get(yt_plus_1, 0)
        if temp == 0:
            for k in range(0, len(Vt_plus_1)-1):
                for i in range(0, self.N):
                    if (self.Omega_X[k] in self.A[self.Omega_X[i]]) and (Vt_plus_1[k] < self.A[self.Omega_X[i]][self.Omega_X[k]] * Vt[i]):
                        Vt_plus_1[k] = self.A[self.Omega_X[i]][self.Omega_X[k]] * Vt[i]
                        X_path_t_plus_1[k] = i
        else:
            for k in range(0, len(Vt_plus_1)-1):
                if yt_plus_1 in self.B[self.Omega_X[k]]:
                    for i in range(0, self.N):
                        if (self.Omega_X[k] in self.A[self.Omega_X[i]]) and (Vt_plus_1[k] < self.B[self.Omega_X[k]][yt_plus_1] * self.A[self.Omega_X[i]][self.Omega_X[k]] * Vt[i]):
                            Vt_plus_1[k] = self.B[self.Omega_X[k]][yt_plus_1] * self.A[self.Omega_X[i]][self.Omega_X[k]] * Vt[i]
                            X_path_t_plus_1[k] = i
                else:
                    Vt_plus_1[k] = 0
                    X_path_t_plus_1[k] = 0
        Vt_plus_1 /= np.sum(Vt_plus_1)
        return Vt_plus_1, X_path_t_plus_1
