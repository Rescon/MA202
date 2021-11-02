import numpy as np

class HMC_Forward_Backward_Restorator:
    
    def __init__(self, Omega_X, Pi, A, B):
        self.Omega_X = Omega_X
        self.N = len(Omega_X)
        self.Pi = Pi
        self.A = A
        self.B = B
        
    def forward(self, Y):
        T = len(Y)
        alpha = np.zeros((T, len(self.Omega_X)))

        alpha[0] = self.compute_alpha_1(Y[0])
        for t in range(0, T - 1):
            alpha[t + 1] = self.compute_alpha_t_plus_1(Y[t + 1], alpha[t])
     
        return alpha
    
    
    def backward(self, Y, default = 0):
        T = len(Y)
        beta = np.zeros((T, self.N))
        
        beta[T - 1] = self.compute_beta_T()
        for t in range(T - 2, -1, -1):
            beta[t] = self.compute_beta_t(Y[t + 1], beta[t + 1])
        
        return beta
    
    
    def compute_gamma(self, alpha, beta):
        gamma = alpha * beta
        gamma = gamma/np.sum(gamma, axis = 1, keepdims = True)
        return gamma
    
    def restore_X(self, Y):
        alpha = self.forward(Y)
        beta = self.backward(Y)  
        gamma = self.compute_gamma(alpha, beta)
        
        indexes = np.argmax(gamma, axis = 1)
        X = [self.Omega_X[index] for index in indexes]
        return X
    
    
    #######################
    ### COMPUTING ALPHA ###
    #######################
    
    def compute_alpha_1(self, y0):
        alpha_1 = np.zeros(self.N)
        temp = 0
        for k in self.Omega_X:
            temp += self.B.get(k, 0).get(y0, 0)
        for ind, i in enumerate(self.Omega_X):
            if temp == 0:
                alpha_1[ind] = self.Pi.get(i, 0)
            else:
                alpha_1[ind] = self.Pi.get(i, 0) * self.B.get(i, 0).get(y0, 0)
        alpha_1 /= np.sum(alpha_1)
        return alpha_1

    
    def compute_alpha_t_plus_1(self, yt_plus_1, alpha_t):
        alpha_t_plus_1 = np.zeros(self.N)
        temp = 0
        for k in self.Omega_X:
            temp += self.B.get(k, 0).get(yt_plus_1, 0)
        for ind1, i in enumerate(self.Omega_X):
            for ind2, j in enumerate(self.Omega_X):
                alpha_t_plus_1[ind1] += alpha_t[ind2] * self.A.get(j, 0).get(i, 0)
            if temp != 0:
                alpha_t_plus_1[ind1] *= self.B.get(i, 0).get(yt_plus_1, 0)
        alpha_t_plus_1 /= np.sum(alpha_t_plus_1)
        return alpha_t_plus_1
    
    
    ######################
    ### COMPUTING BETA ###
    ######################
    
    def compute_beta_T(self):
        beta_T = np.zeros(self.N)
        for i in range(self.N):
            beta_T[i] = 1
        beta_T = beta_T/np.sum(beta_T)
        return beta_T

    def compute_beta_t(self, yt_plus_1, beta_t_plus_1):
        beta_t = np.zeros(self.N)
        temp = 0
        for k in self.Omega_X:
            temp += self.B.get(k, 0).get(yt_plus_1, 0)
        for ind1, i in enumerate(self.Omega_X):
            for ind2, j in enumerate(self.Omega_X):
                if temp == 0:
                    beta_t[ind1] += beta_t_plus_1[ind2] * self.A.get(i, 0).get(j, 0)
                else:
                    beta_t[ind1] += beta_t_plus_1[ind2] * self.A.get(i, 0).get(j, 0) * self.B.get(j, 0).get(yt_plus_1, 0)
        beta_t /= np.sum(beta_t)
        return beta_t
