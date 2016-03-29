#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from math import pi, sqrt
from numpy import (
    arange, array, power, exp, asarray, float64, zeros, ones, linspace, eye
)
from pylab import (
    show, legend, hold, matshow, colorbar, reshape, savefig, std, mean, title,
    plot, figure, find
)

from scipy import stats
from matplotlib.pyplot import get_cmap, cm
from pyhrf.boldsynth.hrf import getCanoHRF
from pyhrf.graph import graph_from_lattice
from pyhrf.vbjde.Utils import Main_vbjde_Extension_TD
from pyhrf.boldsynth.boldsynth.scenarios import RegularLatticeMapping
import pyhrf.verbose
from tifffile import imread


class ImageAnalyzer:
    output_dir = 'ParaguayOut/'

    def __init__(self, images,
                 output_dir=output_dir,
                 facteur=1000.0,
                 centred=0,
                 bande=0,
                 verbosity=2
                 ):
        self.output_dir = output_dir
        if not os.path.isdir(output_dir):
            print('creating output directory...')
            os.mkdir(output_dir)
        self.images = images
        self.facteur = facteur
        self.centred = centred
        self.bande = bande
        pyhrf.verbose.set_verbosity(verbosity)

    def seuillage(self, image):  # , seuil):
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                if image[i, j] > 0.5:
                    image[i, j] = 1
                else:
                    image[i, j] = 0
        return image

    def ConditionalNRLHist(self, nrls, labels, M, height, width):
        for m in range(0, M):
            q = labels[m, 1, :]
            ind = find(q >= 0.5)
            ind2 = find(q < 0.5)
            r = nrls[ind]
            xmin, xmax = min(nrls), max(nrls)
            lnspc = linspace(xmin, xmax, 100)
            m, s = stats.norm.fit(r)
            pdf_g = stats.norm.pdf(lnspc, m, s)
            r = nrls[ind2]
            # xmin, xmax = min(r), max(r)
            lnspc2 = linspace(xmin, xmax, 100)  # len(r)
            m, s = stats.norm.fit(r)
            pdf_g2 = stats.norm.pdf(lnspc2, m, s)

            figure()
            plot(lnspc, pdf_g / len(pdf_g), label="Norm")
            hold(True)
            plot(lnspc2, pdf_g2 / len(pdf_g2), 'k', label="Norm")
            legend(['Posterior: Activated', 'Posterior: Non Activated'])
            # xmin, xmax = min(xt), max(xt)
            # ind2 = find(q <= 0.5)
        show()

    def lecture_data(self, xmin, xmax, ymin, ymax, start=0, end=-1):
        images = self.images[start:end]
        signal = []
        for image in images:
            print(image)
            labels = imread(image)
            labels = labels[xmin:xmax, ymin:ymax, self.bande].astype(float)
            if (self.facteur > 1):
                labels = labels / self.facteur
            signal.append(labels.flatten())
        self.Y = asarray(signal)
        self.width = ymax - ymin
        self.height = xmax - xmin

    def gaussian(self, x, mu, sig):
        return 1. / (sqrt(2. * pi) * sig) * exp(-power((x - mu) / sig, 2.) / 2)

    def post_lecture(self):
        STD = std(self.Y, 1)
        MM = mean(self.Y, 1)
        TT, self.NN = self.Y.shape
        if self.centred:
            for t in xrange(0, TT):
                self.Y[t, :] = (self.Y[t, :] - MM[t]) / STD[t]
                # Y[t,:] = (Y[t,:]  ) /STD[t]
        # Y,height,width,bande = lecture_data(dataDir,100,200,150,250,0,start,facteur,end)
        # Y = Y.astype(float)

    ###############################
    # intialisation des paramètres
    ###############################
    def init_params(self):
        self.beta = 0.1
        self.sigmaH = 0.01
        self.v_h = 0.1 * self.sigmaH
        # beta_Q = 0.5

        self.dt = 1
        self.Thrf = 4
        self.TR = 1
        # nf=1
        # for i in range (0,Y.shape[0]) :
        #  figure(nf)
        #  PL_img =reshape(Y[i,:],(height,width))
        #  matshow(PL_img,cmap=get_cmap('gray'))
        #  colorbar()
        #  nf=nf+1
        #  print('save image ' + str(i))
        #  savefig(outDir+'Image'+str(i)+'bande=' +str(bande)+'_inondation.png')
        #  #savefig(outDir+'Image'+str(i)+'bande=' +str(bande)+'.png')
        #####################################
        self.nItMin = 30
        self.nItMax = 30

        self.K = 2
        self.scale = 1

        self.M = 1

    #####################
    # flags
    #####################
    def set_flags(self):
        # pl =0 sans PL ,pl =1 avec PL
        self.pl = 1
        # estimationSigmah = 0 sans estimation de sigmh , estimationSigmah=1 estimation de sigmah
        self.estimateSigmaH = 0
        # estimateBeta = 0 sans estimation de beta , estimateBeta=1 estimation de beta
        self.estimateBeta = 0
        # save = 1  les outputs sont sauvgardés
        self.save = 1
        # savepl les PL sont sauvgardés dans le repertoir outDir
        self.savepl = 1
        self.shower = 1

        # construction Onsets
        # Onsets = {'nuages' : array([1,6,7])}
        self.Onsets = {'nuages': array([0])}
        self.nf = 1

    def gen_hrf(self):
        areas = ['ra']
        labelFields = {}
        cNames = ['inactiv', 'activ']
        spConf = RegularLatticeMapping((self.height, self.width, 1))
        graph = graph_from_lattice(ones((self.height, self.width, 1), dtype=int))
        J = self.Y.shape[0]
        l = int(sqrt(J))

        # NbIter, nrls_mean, hrf_mean, hrf_covar, labels_proba, noise_var, \
        # nrls_class_mean, nrls_class_var, beta, drift_coeffs, drift,CONTRAST, CONTRASTVAR, \
        # nrls_criteria, hrf_criteria,labels_criteria, nrls_hrf_criteria, compute_time,compute_time_mean, \
        # nrls_covar, stimulus_induced_signal, density_ratio,density_ratio_cano, density_ratio_diff, density_ratio_prod, \
        # ppm_a_nrl,ppm_g_nrl, ppm_a_contrasts, ppm_g_contrasts, variation_coeff = \
        # jde_vem_bold_fast_python(pl,graph, Y, Onsets, Thrf, K,TR, beta, dt, estimateSigmaH, sigmaH,nItMax,nItMin,estimateBeta)
        # FlagZ = 1
        # q_Z = zeros((M,K,J),dtype=float64)
        # NbIter,m_A, m_H, q_Z, sigma_epsilone, mu_k, sigma_k,Beta,L,PL,CONTRAST, CONTRASTVAR,cA,cH,cZ,cAH,cTime,cTimeMean,Sigma_A,XX = \
        # Main_vbjde_Extension_TD(height,width,q_Z,FlagZ,pl,graph,Y,Onsets,Thrf,K,TR,beta,dt,scale,estimateSigmaH,sigmaH,nItMax,nItMin,estimateBeta)

        # facteur = 1.0
        # Y,height,width,bande = lecture_data(dataDir,2660,2730,2600,2680,bande,start,facteur,end)
        # FlagZ = 0
        # NbIter_f,m_A_f, m_H_f, q_Z_f, sigma_epsilone_f, mu_k_f, sigma_k_f,Beta_f,L_f,PL,CONTRAST, CONTRASTVAR,cA,cH,cZ,cAH,cTime,cTimeMean,Sigma_A,XX = \
        # Main_vbjde_Extension_TD(height,width,q_Z,FlagZ,pl,graph,Y,Onsets,Thrf,K,TR,beta,dt,scale,estimateSigmaH,sigmaH,nItMax,nItMin,estimateBeta)

        # ytild=PL
        # matshow(reshape(m_A_f[:,0],(height,width)) - reshape(m_A[:,0],(height,width)),cmap=get_cmap('gray'))
        # colorbar()

        FlagZ = 1
        q_Z0 = zeros((self.M, self.K, J), dtype=float64)
        if not FlagZ:
            q_Z0 = q_Z
        FlagH = 1
        TT, m_h = getCanoHRF(self.Thrf - self.dt, self.dt)
        hrf0 = array(m_h).astype(float64)
        Sigma_H0 = eye(hrf0.shape[0])
        if not FlagH:
            hrf0 = h_H
            Sigma_H0 = Sigma_H

        self.m_A, self.m_H, self.q_Z, sigma_epsilone, mu_k, sigma_k, Beta, PL, Sigma_A, XX, Sigma_H = \
            Main_vbjde_Extension_TD(
                FlagH, hrf0, Sigma_H0, self.height, self.width, q_Z0, FlagZ,
                self.pl, graph, self.Y, self.Onsets, self.Thrf, self.K, self.TR,
                self.beta, self.dt, self.scale,
                self.estimateSigmaH, self.sigmaH, self.nItMin, self.estimateBeta)

        self.ConditionalNRLHist(self.m_A, self.q_Z, self.M, self.height, self.width)

        MMin = -1.0  # Y.min()
        MMax = 1.0  # Y.max()
        pas = (MMax - MMin) / 100
        xx = arange(MMin, MMax, pas)
        g0 = self.gaussian(xx, mu_k[0][0], sigma_k[0][0])
        g1 = self.gaussian(xx, mu_k[0][1], sigma_k[0][1])
        #
        # figure(77)
        # plot(xx,g0*pas, label='m=%.2f;v=%.2f' % (mu_k[0][0],sigma_k[0][0]))
        # hold(True)
        # plot(xx,g1*pas, label='m=%.2f;v=%.2f' % (mu_k[0][1],sigma_k[0][1]))
        # legend()

        # savgarde des PLs #
        # if ((pl != 0 )and (savepl !=0 )) :
        #   for i in range (0,ytild.shape[0]) :
        #       figure(nf)
        #       PL_img =reshape(ytild[i,:],(height,width))
        #       matshow(PL_img,cmap=get_cmap('gray'))
        #       colorbar()
        #       nf=nf+1
        #       savefig(outDir+'PL'+str(i)+'bande=' +str(bande)+'.png')

        # savefig(outDir+'PL'+str(i)+'bande=' +str(bande)+'beta='+str(beta)+'sigma='+str(sigmaH)+'pl='+str(pl)+'dt='+str(dt)+'thrf'+str(Thrf)+'.png')
        #
        # figure Hrf #######
        figure((self.nf + 1) * 123)
        title("hrf")
        plot(self.m_H)
        if self.save == 1:
            savefig(self.output_dir + 'hrf bande =' + str(self.bande) + 'beta=' + str(self.beta) + 'sigma= ' +
                    str(self.sigmaH) + 'pl=' + str(self.pl) + 'dt=' + str(self.dt) + 'thrf' + str(self.Thrf) + '.png')

    def gen_nrl(self):
        # figure(55)
        # matshow(reshape(sigma_epsilone,(height,width)))
        # colorbar()
        for m in range(0, self.M):
            hh = self.m_H
            z1 = self.m_A[:, m]
            z2 = reshape(z1, (self.height, self.width))
            figure((self.nf + 1) * 110)
            # figure Nrl ########,cmap=get_cmap('gray')
            matshow(z2, cmap=get_cmap('gray'))
            colorbar()
            # title("Est: m = " + str(m))
            if self.save == 1:
                savefig(self.output_dir + 'nrl bande =' + str(self.bande) + 'beta=' + str(self.beta) + 'sigma= ' +
                        str(self.sigmaH) + 'pl=' + str(self.pl) + 'dt=' + str(self.dt) + 'thrf' + str(self.Thrf) + '.png')
            q = self.q_Z[m, 1, :]
            q2 = reshape(q, (self.height, self.width))
            # q2 = seuillage(q2,0.5)
            matshow(q2, cmap=get_cmap('gray'))
            colorbar()
        #   for k in range(0,1):
        #       q = q_Z[m,k,:]
        #       q2 = reshape(q,(height,width))
        #       figure ((nf+1) *38)
        #       matshow(q2,cmap=get_cmap('gray'))
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
        if self.shower == 1:
            show()


######################
# lecture des données
######################
img_analyzer = ImageAnalyzer(['Paraguay/' + f for f in sorted(os.listdir('Paraguay/'))])
img_analyzer.lecture_data(2660, 2730, 2600, 2680)
img_analyzer.post_lecture()
img_analyzer.init_params()
img_analyzer.set_flags()
img_analyzer.gen_hrf()
img_analyzer.gen_nrl()