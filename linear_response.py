#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:24:34 2024

@author: gabriel
"""

import numpy as np
from SOC_superconductivity import (SOCSuperconductorKxKy,
                                   SOCSparseSuperconductor,
                                   SOCSuperconductor,
                                   SOCSuperconductorKx)
import matplotlib.pyplot as plt
from pathlib import Path

L_x = 200
L_y = 200
w_0 = 10
Delta = 0.2
mu = -40
theta = np.pi/2
B = 2*Delta
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
B_z = 0
Lambda = 0.56 #5*Delta/k_F
Omega = 0#0.02
superconductor_params = {"w_0":w_0, "Delta":Delta,
                         "mu":mu, "B_x":B_x, 
                         "B_y":B_y, "Lambda":Lambda,
                         "B_z":B_z }

Gamma = 0.1
alpha = 0
beta = 0
Beta = 1000
# k_x_values = 2*np.pi*np.arange(0, L_x)/L_x
# k_y_values = 2*np.pi*np.arange(0, L_y)/L_y
k_x_values = np.pi*np.arange(-L_x, L_x)/L_x
k_y_values = np.pi*np.arange(-L_y, L_y)/L_y


omega_values = np.linspace(-45, 0, 100)

# part = "paramagnetic"
# part = "diamagnetic"
part = "total"
# fermi_function = lambda omega: 1/(1 + np.exp(Beta*omega))
# fermi_function = lambda omega: 1 - np.heaviside(omega, 1)
params = {
    "Gamma":Gamma, "alpha":alpha,
    "beta":beta, "Omega":Omega, "part":part
    }

def fermi_function(omega):
    return np.heaviside(-omega, 1)

S = SOCSuperconductorKxKy(k_x=0, k_y=0, **superconductor_params)
# S = SOCSparseSuperconductor(L_x=2, L_y=2, **superconductor_params)
# S = SOCSuperconductor(L_x=2, L_y=2, **superconductor_params)
# S = SOCSuperconductorKx(k=0, L_y=2, **superconductor_params)
