####################
### DEPENDENCIES ###
####################

import sys
sys.path.append("..")
sys.path.append("../../Learning_Parameters")
sys.path.append("../../Dataset")

from Code_Sujet.HMC.HMC_Viterbi_Restorator import HMC_Viterbi_Restorator
from Code_Sujet.Learning_Parameters.HMC_Learning_Parameters import HMC_Learning_Parameters
from Code_Sujet.Dataset.UD_English.load_ud_english import load_ud_english
import numpy as np


#######################
### LOADING DATASET ###
#######################

train_set, test_set = load_ud_english(path = "../../Dataset/UD_English/")
Omega_X = list(set([x for sent in train_set for x, y in sent]))

Omega_X_dict = dict(zip(Omega_X, np.arange(0, len(Omega_X))))
vocabulary = list(set([y for sent in train_set for x, y in sent]))

vocabulary_dict = {}
for idw, word in enumerate(vocabulary):
    vocabulary_dict[word] = idw
    

##################################
### STATIONNARY HMC PARAMETERS ###
##################################

Pi, A, B = HMC_Learning_Parameters(train_set)



###############
### TESTING ###
###############

hmc_viterbi = HMC_Viterbi_Restorator(Omega_X, Pi, A, B)

score, total = 0, 0
score_kw, total_kw = 0, 0
score_uw, total_uw = 0, 0

for ids, sent in enumerate(test_set):
    
    X = [x for x, y in sent]
    Y = [y for x, y in sent]
    
    X_hat = hmc_viterbi.restore_X(Y)
    
    for t in range(len(Y)):
        score += 1 * (X[t] == X_hat[t])
            
        if Y[t] in vocabulary_dict:
            total_kw += 1
            score_kw += 1 * (X[t] == X_hat[t])
            
        else:
            total_uw += 1
            score_uw += 1 * (X[t] == X_hat[t])
            
        total += 1
        
    if ids % 100 == 0 and ids != 0:
        print(ids, score/total * 100)
    
print("HMC Viterbi Accuracy POS Tagging UD English:", score/total * 100, "%")

print("KW:", score_kw/total_kw * 100)
print("UW:", score_uw/total_uw * 100)
