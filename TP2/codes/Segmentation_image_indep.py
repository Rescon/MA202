'''
On observe que les résultats du premier algorithme ne sont pas toujours très bons, il est lié au bruit.
De plus, nous pouvons voir que la valeur de la condition initiale a une grande influence sur la convergence de l'algorithme EM.
'''

import numpy as np
import cv2
import utils
from sklearn.cluster import KMeans

path = "/Users/wangyu980312/Desktop/MA202/TP2/images/beee2.bmp"
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image / 255
cv2.imshow("Image réelle", image)

cl1 = 0
cl2 = 1
m1 = 0
sig1 = 1
m2 = 1
sig2 = 1
Y = utils.bruit_gauss2(utils.line_transform_img(image), cl1, cl2, m1, sig1, m2, sig2)

kmeans_clusters = 2
kmeans = KMeans(n_clusters=kmeans_clusters).fit(Y.reshape(-1, 1))
X_seg = kmeans.labels_
label1 = np.where(kmeans.labels_ == 0)
label2 = np.where(kmeans.labels_ == 1)

proba_cl1, proba_cl2 = utils.calc_probaprio2(X_seg, cl1, cl2)
m1 = np.mean(label1)
sig1 = np.std(label1)
m2 = np.mean(label2)
sig2 = np.std(label2)

p1_est, p2_est, m1_est, sig1_est, m2_est, sig2_est = utils.estim_param_EM_gm(50, Y, proba_cl1, proba_cl2, m1, sig1, m2, sig2)
image_segmentee = utils.MPM_gm(Y, cl1, cl2, p1_est, p2_est, m1_est, sig1_est, m2_est, sig2_est)
print("Le taux d'erreur est: ", utils.taux_erreur(utils.line_transform_img(image), image_segmentee))
cv2.imshow("Image segmentée", utils.transform_line_in_img(image_segmentee, 256))

cv2.imshow("Image bruitée", utils.transform_line_in_img(Y, 256))
