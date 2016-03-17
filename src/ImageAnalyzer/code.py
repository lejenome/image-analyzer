#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

# import Utils
# import time
# import random
# import csv
# import cv2
# import sys
# if sys.version_info[0] == 2:
#    import cProfile
#    import cPickle
# else:
#    import profile
#    import pickle


import pylab
from pylab import *

# import Image #Python Imaging Library (PIL)
from PIL import Image

import numpy as np
from numpy.random.mtrand import dirichlet

from scipy.misc import fromimage
from scipy.linalg import toeplitz
import scipy.io
from scipy.misc import toimage
import scipy as sp
import scipy.sparse
import scipy.io as spio
from scipy import linalg
from scipy import stats

# from tifffile import imread

from matplotlib import *
import matplotlib.pyplot as plt
from matplotlib import pylab as pl
import matplotlib.font_manager as fm

# from pyhrf.boldsynth.boldsynthold import *
# from pyhrf.vbjde.Utils import *
# from pyhrf.vbjde.Utils import *
from pyhrf.paradigm import restarize_events
from pyhrf.boldsynth.hrf import genBezierHRF, genGaussianSmoothHRF
from pyhrf.boldsynth.scenarios import *
# from pyhrf.boldsynth.boldsynth.boldmodel import *
from pyhrf.graph import *
from pyhrf.boldsynth.field import genPotts, count_homo_cliques
from pyhrf.graph import graph_from_lattice, kerMask2D_4n, kerMask2D_8n
# from pyhrf.vbjde.Utils import roc_curve
from pyhrf.boldsynth.hrf import getCanoHRF

# repartoire des données ######
dataDir = os.getenv('HOME') + 'git/py-gtk3/Paraguay'
#########################
# repertoir des outputs #####
outDir = os.getenv('HOME') + '/ParaguayOut'
if not os.path.isdir(outDir):
    print('creating output directory...')
    os.mkdir(outDir)

def seuillage(image, seuil):
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i, j] > 0.5:
                image[i, j] = 1
            else:
                image[i, j] = 0
    return image

def ConditionalNRLHist(nrls, labels, M, height, width):
    for m in range(0, M):
        q = labels[m, 1, :]
        ind = find(q >= 0.5)
        ind2 = find(q < 0.5)
        r = nrls[ind]
        xmin, xmax = min(nrls), max(nrls)
        lnspc = np.linspace(xmin, xmax, 100)
        m, s = stats.norm.fit(r)
        pdf_g = stats.norm.pdf(lnspc, m, s)
        r = nrls[ind2]
        # xmin, xmax = min(r), max(r)
        lnspc2 = np.linspace(xmin, xmax, 100)  # len(r)
        m, s = stats.norm.fit(r)
        pdf_g2 = stats.norm.pdf(lnspc2, m, s)

        pylab.figure()
        pylab.plot(lnspc, pdf_g / len(pdf_g), label="Norm")
        pylab.hold(True)
        pylab.plot(lnspc2, pdf_g2 / len(pdf_g2), 'k', label="Norm")
        pylab.legend(['Posterior: Activated', 'Posterior: Non Activated'])
        # xmin, xmax = min(xt), max(xt)
        # ind2 = find(q <= 0.5)
    pylab.show()


def lecture_data(dataDir, xmin, xmax, ymin, ymax, bande, start, facteur, end):
    images = os.listdir(dataDir)
    images = sorted(images)
    images = images[start:end]
    signal = []
    for image in images:
        fn = os.path.join(dataDir, image)
        print(fn)
        labels = imread(fn)
        labels = labels[xmin:xmax, ymin:ymax, bande].astype(float)
        if (facteur > 1):
            labels = labels / facteur
        signal.append(labels.flatten())
    Y = np.asarray(signal)
    width = ymax - ymin
    height = xmax - xmin
    return Y, height, width, bande

def gaussian(x, mu, sig):
    return 1. / (sqrt(2. * pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.) / 2)
zones = []
######################
# lecture des données
######################
facteur = 1000.0
Centred = 0
start = 2  # 2
end = 9  # 9
bande = 0
Y, height, width, bande = lecture_data(dataDir, 2660, 2730, 2600, 2680, bande, start, facteur, end)
STD = std(Y, 1)
MM = mean(Y, 1)
TT, NN = Y.shape
if Centred:
    for t in xrange(0, TT):
        Y[t, :] = (Y[t, :] - MM[t]) / STD[t]
        # Y[t,:] = (Y[t,:]  ) /STD[t]
# Y,height,width,bande = lecture_data(dataDir,100,200,150,250,0,start,facteur,end)
# Y = Y.astype(float)

# intialisation des paramètres ######################
beta = 0.1
sigmaH = 0.01
v_h = 0.1 * sigmaH
# beta_Q = 0.5

dt = 1
Thrf = 4
TR = 1

#
# nf=1
# for i in range (0,Y.shape[0]) :
#  figure(nf)
#  PL_img =reshape(Y[i,:],(height,width))
#  matshow(PL_img,cmap=plt.get_cmap('gray'))
#  colorbar()
#  nf=nf+1
#  print('save image ' + str(i))
#  savefig(outDir+'Image'+str(i)+'bande=' +str(bande)+'_inondation.png')
#  #savefig(outDir+'Image'+str(i)+'bande=' +str(bande)+'.png')
#####################################


nItMin = 30
nItMax = 30

K = 2
scale = 1

M = 1
#####################
# flags
#####################

# pl =0 sans PL ,pl =1 avec PL
pl = 1

# estimationSigmah = 0 sans estimation de sigmh , estimationSigmah=1 estimation de sigmah
estimateSigmaH = 0

# estimateBeta = 0 sans estimation de beta , estimateBeta=1 estimation de beta
estimateBeta = 0

# save = 1  les outputs sont sauvgardés
save = 1

# savepl les PL sont sauvgardés dans le repertoir outDir

savepl = 1

shower = 1

#################################################


# construction Onsets #################
# Onsets = {'nuages' : np.array([1,6,7])}
Onsets = {'nuages': np.array([0])}
##################

nf = 1


areas = ['ra']
labelFields = {}
cNames = ['inactiv', 'activ']
spConf = RegularLatticeMapping((height, width, 1))
graph = graph_from_lattice(numpy.ones((height, width, 1), dtype=int))
J = Y.shape[0]
l = int(sqrt(J))


pyhrf.verbose.set_verbosity(2)
# NbIter, nrls_mean, hrf_mean, hrf_covar, labels_proba, noise_var, \
# nrls_class_mean, nrls_class_var, beta, drift_coeffs, drift,CONTRAST, CONTRASTVAR, \
# nrls_criteria, hrf_criteria,labels_criteria, nrls_hrf_criteria, compute_time,compute_time_mean, \
# nrls_covar, stimulus_induced_signal, density_ratio,density_ratio_cano, density_ratio_diff, density_ratio_prod, \
# ppm_a_nrl,ppm_g_nrl, ppm_a_contrasts, ppm_g_contrasts, variation_coeff = \
# jde_vem_bold_fast_python(pl,graph, Y, Onsets, Thrf, K,TR, beta, dt, estimateSigmaH, sigmaH,nItMax,nItMin,estimateBeta)
# FlagZ = 1
# q_Z = numpy.zeros((M,K,J),dtype=numpy.float64)
# NbIter,m_A, m_H, q_Z, sigma_epsilone, mu_k, sigma_k,Beta,L,PL,CONTRAST, CONTRASTVAR,cA,cH,cZ,cAH,cTime,cTimeMean,Sigma_A,XX = \
# Main_vbjde_Extension_TD(height,width,q_Z,FlagZ,pl,graph,Y,Onsets,Thrf,K,TR,beta,dt,scale,estimateSigmaH,sigmaH,nItMax,nItMin,estimateBeta)

# facteur = 1.0
# Y,height,width,bande = lecture_data(dataDir,2660,2730,2600,2680,bande,start,facteur,end)
# FlagZ = 0
# NbIter_f,m_A_f, m_H_f, q_Z_f, sigma_epsilone_f, mu_k_f, sigma_k_f,Beta_f,L_f,PL,CONTRAST, CONTRASTVAR,cA,cH,cZ,cAH,cTime,cTimeMean,Sigma_A,XX = \
# Main_vbjde_Extension_TD(height,width,q_Z,FlagZ,pl,graph,Y,Onsets,Thrf,K,TR,beta,dt,scale,estimateSigmaH,sigmaH,nItMax,nItMin,estimateBeta)
#
# ytild=PL
# matshow(reshape(m_A_f[:,0],(height,width)) - reshape(m_A[:,0],(height,width)),cmap=plt.get_cmap('gray'))
# colorbar()


FlagZ = 1
q_Z0 = numpy.zeros((M, K, J), dtype=numpy.float64)
if not FlagZ:
    q_Z0 = q_Z
FlagH = 1
TT, m_h = getCanoHRF(Thrf - dt, dt)
hrf0 = numpy.array(m_h).astype(numpy.float64)
Sigma_H0 = eye(hrf0.shape[0])
if not FlagH:
    hrf0 = h_H
    Sigma_H0 = Sigma_H

m_A, m_H, q_Z, sigma_epsilone, mu_k, sigma_k, Beta, PL, Sigma_A, XX, Sigma_H = \
    Main_vbjde_Extension_TD(FlagH, hrf0, Sigma_H0, height, width, q_Z0, FlagZ, pl, graph, Y, Onsets, Thrf, K, TR, beta, dt, scale, estimateSigmaH, sigmaH, nItMin, estimateBeta)


ConditionalNRLHist(m_A, q_Z, M, height, width)


MMin = -1.0  # Y.min()
MMax = 1.0  # Y.max()
pas = (MMax - MMin) / 100
xx = np.arange(MMin, MMax, pas)
g0 = gaussian(xx, mu_k[0][0], sigma_k[0][0])
g1 = gaussian(xx, mu_k[0][1], sigma_k[0][1])
#
# figure(77)
# plot(xx,g0*pas, label='m=%.2f;v=%.2f' % (mu_k[0][0],sigma_k[0][0]))
# hold(True)
# plot(xx,g1*pas, label='m=%.2f;v=%.2f' % (mu_k[0][1],sigma_k[0][1]))
# legend()

#
# savgarde des PLs #
# if ((pl != 0 )and (savepl !=0 )) :
#   for i in range (0,ytild.shape[0]) :
#       figure(nf)
#       PL_img =reshape(ytild[i,:],(height,width))
#       matshow(PL_img,cmap=plt.get_cmap('gray'))
#       colorbar()
#       nf=nf+1
#       savefig(outDir+'PL'+str(i)+'bande=' +str(bande)+'.png')

# savefig(outDir+'PL'+str(i)+'bande=' +str(bande)+'beta='+str(beta)+'sigma='+str(sigmaH)+'pl='+str(pl)+'dt='+str(dt)+'thrf'+str(Thrf)+'.png')
#
# figure Hrf #######
figure((nf + 1) * 123)
title("hrf")
plot(m_H)
if save == 1:
    savefig(outDir + 'hrf bande =' + str(bande) + 'beta=' + str(beta) + 'sigma= ' + str(sigmaH) + 'pl=' + str(pl) + 'dt=' + str(dt) + 'thrf' + str(Thrf) + '.png')


# figure(55)
# matshow(reshape(sigma_epsilone,(height,width)))
# colorbar()
for m in range(0, M):
    hh = m_H
    z1 = m_A[:, m]
    z2 = reshape(z1, (height, width))
    figure((nf + 1) * 110)
    # figure Nrl ########,cmap=plt.get_cmap('gray')
    matshow(z2, cmap=plt.get_cmap('gray'))
    colorbar()
    # title("Est: m = " + str(m))
    if save == 1:
        savefig(outDir + 'nrl bande =' + str(bande) + 'beta=' + str(beta) + 'sigma= ' + str(sigmaH) + 'pl=' + str(pl) + 'dt=' + str(dt) + 'thrf' + str(Thrf) + '.png')
    q = q_Z[m, 1, :]
    q2 = reshape(q, (height, width))
    # q2 = seuillage(q2,0.5)
    matshow(q2, cmap=plt.get_cmap('gray'))
    colorbar()
#   for k in range(0,1):
#       q = q_Z[m,k,:]
#       q2 = reshape(q,(height,width))
#       figure ((nf+1) *38)
#       matshow(q2,cmap=plt.get_cmap('gray'))
#       colorbar()
#       # figure Q_sans seuil ###########
#       if save == 1 :
#           savefig(outDir+'Q_Sans_seuil bande =' +str(bande)+'beta='+str(beta)+'sigma= '+str(sigmaH)+'pl='+str(pl)+ 'dt='+str(dt)+'thrf'+str(Thrf)+'.png')
#       q2=seuillage(q2,0.5)
#       figure((nf+1) *102)
#       # figure Q_z avec seuil ###########
#
#       imshow(q2,cmap=cm.gray)
#       colorbar()
#       if save == 1 :
#           savefig(outDir+'Q_avec_seuil bande =' +str(bande)+'beta='+str(beta)+'sigma= '+str(sigmaH)+'pl='+str(pl)+ 'dt='+str(dt)+'thrf'+str(Thrf)+'.png')
if shower == 1:
    show()
