#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from math import pi, sqrt
from numpy import (
    arange, array, power, exp, asarray, float64, zeros, ones, linspace, eye
)
from pylab import (
    show, legend, hold, matshow, colorbar, reshape, savefig, std, mean, title,
    plot, figure, find, figtext, suptitle
)

from scipy import stats
from matplotlib.pyplot import get_cmap, cm, subplots
from pyhrf.boldsynth.hrf import getCanoHRF
from pyhrf.graph import graph_from_lattice
from pyhrf.vbjde.Utils import Main_vbjde_Extension_TD
from pyhrf.boldsynth.boldsynth.scenarios import RegularLatticeMapping
import pyhrf.verbose
from tifffile import imread

from .Analyse_Data import Analyse_data
from .Conf import Configuration_InPuts


class Result_Analysis:
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

    def init_params(self,
                    beta=0.1,
                    sigmaH=0.01,
                    v_h_facteur=0.1,
                    dt=1,
                    Thrf=4,
                    TR=1,
                    K=2,
                    M=1,):
        """initialization parameters for the analysis method ConditionalNRLHist

        :param beta: paramétre de regularité spaciale
        :type beta: float
        :param sigmaH: paramétre de lissage de la HRF
        :type sigmaH: float
        :param v_h_facteur:hyper parametre
        :type v_h_facture: float
        :param dt: pas d'echelle Temporel de la HRF
        :type dt: float
        :param Thrf: durée
        :type Thrf: int
        :param TR: temps de repetition
        :type TR: int
        :param K: nombre de class
        :type K: int
        :param M: nombre de coordonnées experimontales
        :type M: int
        """
        self.beta = beta
        self.sigmaH = sigmaH
        self.v_h = v_h_facteur * sigmaH
        self.dt = dt
        self.Thrf = Thrf
        self.TR = TR
        self.K = K
        self.M = M

    def set_flags(self, pl=1, save=0, savepl=1, shower=0, nf=1):
        """
        initialization parameters for saving results

        :param pl: low frequency component
        :type pl: int
        :param save: variable to indicate the state of outputs
        :type save: int
        :param savepl: pl are saved in the directory OUTDIR
        :type savepl: int
        :param shower: show or not images results
        :type shower: int
        :param nf:
        :type nf: int
        """
        # pl =0 sans PL ,pl =1 avec PL
        self.pl = pl
        # save = 1  les outputs sont sauvgardés
        self.save = save
        # savepl les PL sont sauvgardés dans le repertoir outDir
        self.savepl = savepl
        self.shower = shower
        self.nf = nf

    def gen_hrf(self,
                nItMin=30,
                nItMax=30,
                estimateSigmaH=0,
                estimateBeta=0,
                Onsets={'nuages': array([0])},
                scale=1,
                ):
        """
        llow to generate figures

        :param nItMin: Minimum number of iteration
        :type nItMin: int
        :param nItMax: Maximum number of iteration
        :type nItMax: int
        :param estimateSigmaH: estimation of sigmaH
        :type estimateSigmaH: int
        :param estimateBeta: estimation of Beta
        :type estimateBeta: int
        :param scale: scale factor
        :type scale: int
        """
        # estimationSigmah = 0 sans estimation de sigmh , estimationSigmah=1 estimation de sigmah
        # estimateBeta = 0 sans estimation de beta , estimateBeta=1 estimation de beta
        # construction Onsets
        areas = ['ra']
        labelFields = {}
        cNames = ['inactiv', 'activ']
        spConf = RegularLatticeMapping((self.height, self.width, 1))
        graph = graph_from_lattice(ones((self.height, self.width, 1), dtype=int))
        J = self.Y.shape[0]
        l = int(sqrt(J))
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
        Y1, Width1, heigth1 = self.Configuration_InPuts.lecture_data(2658, 2730, 2600, 2680)
        self.m_A, self.m_H, self.q_Z, sigma_epsilone, mu_k, sigma_k, Beta, PL, Sigma_A, XX, Sigma_H = \
            Main_vbjde_Extension_TD(
                FlagH, hrf0, Sigma_H0, self.height1, self.width1, q_Z0, FlagZ,
                self.pl, graph, self.Y1, Onsets, self.Thrf, self.K, self.TR,
                self.beta, self.dt, scale,
                estimateSigmaH, self.sigmaH, nItMin, estimateBeta)

        # fgs = self.ConditionalNRLHist(self.m_A, self.q_Z)
        fgs = self.Analyse_data.ConditionalNRLHist(self.m_A, self.q_Z)

        MMin = -1.0  # Y.min()
        MMax = 1.0  # Y.max()
        pas = (MMax - MMin) / 100
        xx = arange(MMin, MMax, pas)
        g0 = self.gaussian(xx, mu_k[0][0], sigma_k[0][0])
        g1 = self.gaussian(xx, mu_k[0][1], sigma_k[0][1])
        fgs.insert(0, figure((self.nf + 1) * 123))
        title("hrf", fontsize='xx-large')
        figtext(0.4, 0.04,
                'bande = ' + str(self.bande) +
                ' beta =' + str(self.beta) +
                ' sigma = ' + str(self.sigmaH) +
                ' pl = ' + str(self.pl) +
                ' dt = ' + str(self.dt) +
                ' thrf = ' + str(self.Thrf),
                fontsize='x-large')
        plot(self.m_H)
        if self.save == 1:
            savefig(self.output_dir + 'hrf bande =' + str(self.bande) + 'beta=' + str(self.beta) + 'sigma= ' +
                    str(self.sigmaH) + 'pl=' + str(self.pl) + 'dt=' + str(self.dt) + 'thrf' + str(self.Thrf) + '.png')
        if self.shower == 1:
            show()
        return fgs

    def gen_nrl(self):
        """
        generation of nrl figures

        :param hh:
        :type hh:
        :param z1:
        :type z1:
        :param z2:
        :type z2:
        :param fg:
        :type fg:
        :param fig:
        :type fig:
        """
        figures = []
        # figure(55)
        # matshow(reshape(sigma_epsilone,(height,width)))
        # colorbar()
        for m in range(0, self.M):
            hh = self.m_H
            z1 = self.m_A[:, m]
            z2 = reshape(z1, (self.height, self.width))
            fg = figure((self.nf + 1) * 110)
            fig, ax = subplots()
            # figure Nrl ########,cmap=get_cmap('gray')
            data = ax.matshow(z2, cmap=get_cmap('gray'))
            fig.colorbar(data)
            title("nrl", fontsize='xx-large')
            figtext(0.4, 0.04,
                    'bande = ' + str(self.bande) +
                    ' beta =' + str(self.beta) +
                    ' sigma = ' + str(self.sigmaH) +
                    ' pl = ' + str(self.pl) +
                    ' dt = ' + str(self.dt) +
                    ' thrf = ' + str(self.Thrf),
                    fontsize='x-large')
            figures.append(fig)
            # title("Est: m = " + str(m))
            if self.save == 1:
                savefig(self.output_dir + 'nrl bande =' + str(self.bande) + 'beta=' + str(self.beta) + 'sigma= ' +
                        str(self.sigmaH) + 'pl=' + str(self.pl) + 'dt=' + str(self.dt) + 'thrf' + str(self.Thrf) + '.png')
            q = self.q_Z[m, 1, :]
            q2 = reshape(q, (self.height, self.width))
            # q2 = seuillage(q2,0.5)
            fig, ax = subplots()
            data = ax.matshow(q2, cmap=get_cmap('gray'))
            fig.colorbar(data)
            title("nrl", fontsize='xx-large')
            figtext(0.4, 0.04,
                    'bande = ' + str(self.bande) +
                    ' beta =' + str(self.beta) +
                    ' sigma = ' + str(self.sigmaH) +
                    ' pl = ' + str(self.pl) +
                    ' dt = ' + str(self.dt) +
                    ' thrf = ' + str(self.Thrf),
                    fontsize='x-large')
            figures.append(fig)
        if self.shower == 1:
            show()
        return figures

if __name__ == "__main__":
    imgs = ['Paraguay/' + f for f in sorted(os.listdir('Paraguay/'))]
    print(imgs)
    conf = Configuration_InPuts(imgs)
    conf.lecture_data(2658, 2730, 2600, 2680)
    conf.post_lecture()
    Result = Result_Analysis.init_params()
    Result.set_flags(shower=1, save=1)
    Result.gen_hrf()
    Result.gen_nrl()
