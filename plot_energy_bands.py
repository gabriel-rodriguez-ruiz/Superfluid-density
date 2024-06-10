#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 18:05:50 2024

@author: gabriel
"""

import numpy as np
from SOC_superconductivity import SOCSuperconductorKxKy
import matplotlib.pyplot as plt
from pathlib import Path

L_x = 200
L_y = 200
w_0 = 10
Delta = 0.2
mu = -40
theta = np.pi/2
B = 0
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
B_z = 0
Lambda = 0#0.56 
superconductor_params = {"w_0":w_0, "Delta":Delta,
                         "mu":mu, "B_x":B_x, 
                         "B_y":B_y, "Lambda":Lambda,
                         "B_z":B_z }

# k_x_values = 2*np.pi*np.arange(0, L_x)/L_x
# k_y_values = 2*np.pi*np.arange(0, L_y)/L_y
k_x_values = np.pi*np.arange(-L_x, L_x)/L_x
k_y_values = np.pi*np.arange(-L_y, L_y)/L_y


S = SOCSuperconductorKxKy(k_x=0, k_y=0, **superconductor_params)

E_k_x = np.zeros((len(k_x_values), 4))
for i, k_x in enumerate(k_x_values):
    for j in range(4):
        S = SOCSuperconductorKxKy(k_x=k_x, k_y=0, **superconductor_params)
        E_k_x[i, j] = np.linalg.eigvalsh(S.matrix)[j]

fig, ax = plt.subplots()
ax.plot(k_x_values, E_k_x)
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$E(k_x, k_y=0)$")