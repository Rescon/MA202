####################
### DEPENDENCIES ###
####################

import sys
sys.path.append("..")
sys.path.append("../../Learning_Parameters/")
sys.path.append("../../Dataset")

from Code_Sujet.HMC.HMC_Forward_Backward_Restorator import HMC_Forward_Backward_Restorator
from Code_Sujet.Learning_Parameters.HMC_Learning_Parameters import HMC_Learning_Parameters
from Code_Sujet.Dataset.CoNLL2000.Load_CoNLL2000 import load_conll2000
import numpy as np

#######################
### LOADING DATASET ###
#######################

train_set, test_set = load_conll2000(path = "../../Dataset/CoNLL2000/")
Omega_X = list(set([x for sent in train_set for x, y in sent]))

Omega_X_dict = dict(zip(Omega_X, np.arange(0, len(Omega_X))))
vocabulary = list(set([y for sent in train_set for x, y in sent]))

vocabulary_dict = dict.fromkeys(vocabulary, 0)
for sent in train_set:
    for taf, word in sent:
        vocabulary_dict[word] = vocabulary_dict[word] + 1
    

##################################
### STATIONNARY HMC PARAMETERS ###
##################################

Pi, A, B = HMC_Learning_Parameters(train_set)


###############
### TESTING ###
###############


hmc_fb = HMC_Forward_Backward_Restorator(Omega_X, Pi, A, B)

detected, true, gold = 0, 0, 0
kw_detected, kw_true, kw_gold = 0, 0, 0
uw_detected, uw_true, uw_gold = 0, 0, 0

for ids, sent in enumerate(test_set):
    
    X = [x for x, y in sent]
    Y = [y for x, y in sent]
    
    X_hat = hmc_fb.restore_X(Y)
    
    for t in range(len(Y)):
        gold += 1 * (X[t] == X_hat[t] and X[t] != "O")
        detected += 1 * (X_hat[t] != "O")
        true += 1 * (X[t] != "O")
        
        if Y[t] in vocabulary_dict:
            kw_gold += 1 * (X[t] == X_hat[t] and X[t] != "O")
            kw_detected += 1 * (X_hat[t] != "O")
            kw_true += 1 * (X[t] != "O")
            
        else:
            uw_gold += 1 * (X[t] == X_hat[t] and X[t] != "O")
            uw_detected += 1 * (X_hat[t] != "O")
            uw_true += 1 * (X[t] != "O")
        
    if ids % 100 == 0 and ids != 0:
        print(ids, gold/detected * 100, gold/true * 100)
    
precision = gold/true 
recall = gold/detected
f1 = 2 * precision * recall / (precision + recall)

print("\nResults:")
print("Precision:", precision)
print("Recall:", recall)
print("Chunking F1 CoNLL 2000:", f1, "\n")

kw_precision = kw_gold/kw_true 
kw_recall = kw_gold/kw_detected
kw_f1 = 2 * kw_precision * kw_recall / (kw_precision + kw_recall)

print("Results KW:")
print("Precision:", kw_precision)
print("Recall:", kw_recall)
print("F1:", kw_f1, "\n")

uw_precision = uw_gold/uw_true 
uw_recall = uw_gold/uw_detected
uw_f1 = 2 * uw_precision * uw_recall / (uw_precision + uw_recall)

print("Results UW:")
print("Precision:", uw_precision)
print("Recall:", uw_recall)
print("F1:", uw_f1)
