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


class Analyse_data:

    def __init__(self, verbosity=2,):
        pyhrf.verbose.set_verbosity(verbosity)

    def ConditionalNRLHist(self, nrls, labels):
        """Analyze method

        :param nrls:
        :type nrls:
        :param labels:
        :type labels:
        :rtype: Resultlist of figures
        """
        figures = []
        for m in range(0, self.M):
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

            fg = figure()
            plot(lnspc, pdf_g / len(pdf_g), label="Norm")
            hold(True)
            plot(lnspc2, pdf_g2 / len(pdf_g2), 'k', label="Norm")
            legend(['Posterior: Activated', 'Posterior: Non Activated'])
            # xmin, xmax = min(xt), max(xt)
            # ind2 = find(q <= 0.5)
            figures.append(fg)
        if self.shower:
            show()
        return figures
