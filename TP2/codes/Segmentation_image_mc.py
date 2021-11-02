'''
Le deuxième algorithme est beaucoup plus lent que le premier, mais nous avons obtenu de meilleurs résultats.
'''
import numpy as np
import cv2
import utils
from sklearn.cluster import KMeans

path = "/Users/wangyu980312/Desktop/MA202/TP2/images/alfa2.bmp"
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image/255
cv2.imshow("Image réelle", image)

cl1 = 0
cl2 = 1
m1 = 0
sig1 = 1
m2 = 1
sig2 = 1
Y = utils.bruit_gauss2(utils.peano_transform_img(image), cl1, cl2, m1, sig1, m2, sig2)


kmeans_clusters = 2
kmeans = KMeans(n_clusters=kmeans_clusters).fit(Y.reshape(-1,1))
res_seg = kmeans.labels_
a = Y[res_seg == 0]
b = Y[res_seg == 1]


proba_cl1, proba_cl2 = utils.calc_probaprio2(res_seg, cl1, cl2)
A = utils.calc_probatrans2(res_seg, cl1, cl2)
m1 = np.mean(a)
sig1 = np.std(a)
m2 = np.mean(b)
sig2 = np.std(b)


Mat_f = utils.gauss2(Y, 0, m1, sig1, m2, sig2)
A_est, p10_est, p20_est, m1_est, sig1_est, m2_est, sig2_est = utils.estim_param_EM_mc(20, Y, A, proba_cl1, proba_cl2, m1, sig1, m2, sig2)
image_segmentee = utils.MPM_chaines2(Mat_f, 0, cl1, cl2, A_est, p10_est, p20_est)
print("Le taux d'erreur est : ",utils.taux_erreur(utils.peano_transform_img(image), image_segmentee))
cv2.imshow("Image segmentee", utils.transform_peano_in_img(image_segmentee, 256))

cv2.imshow("Image bruitee", utils.transform_peano_in_img(Y,256))
