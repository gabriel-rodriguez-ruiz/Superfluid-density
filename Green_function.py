#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 18:46:42 2024

@author: gabriel
"""
import numpy as np
from SOC_superconductivity import SOCSuperconductivity

class GreenFunction:
    r"""
    .. math::
        G_{\nu}(\omega) = [\omega I - H_{\mathbf{k}} + i\Gamma I]^{-1}
    Parameters
    ----------
    omega : float
        Frequency.
    k_x : float
        Momentum in x direction.
    k_y : float
        Momentum in y direction.
    Gamma : float
        Damping.

    Returns
    -------
    ndarray
        Green function.

    """
    def __init__(self, S:SOCSuperconductivity,
                 omega:float, Gamma:float):
        self.Gamma = Gamma
        self.omega = omega
        self.matrix = self._get_matrix(S, omega, Gamma)
    def _get_matrix(self, S:SOCSuperconductivity,
                    omega:float, Gamma:float):
        H = S.matrix
        I = np.eye(np.shape(S.matrix)[0], np.shape(S.matrix)[0])
        return np.linalg.inv((omega+1j*Gamma)*I - H) 