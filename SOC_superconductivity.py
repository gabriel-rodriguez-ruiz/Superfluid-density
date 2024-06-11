#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:36:38 2024

@author: gabriel
"""

import numpy as np
from pauli_matrices import tau_x, sigma_x, tau_z, sigma_0, sigma_y,\
                            tau_0, sigma_z
from hamiltonian import Hamiltonian, PeriodicHamiltonianInY,\
                        SparseHamiltonian, SparsePeriodicHamiltonianInY

class SOCSuperconductivity():
    r"""2D Superconductor with local s-wave pairing, spin-orbit coupling and magnetic field.
    
    Parameters
    ----------
    w_0 : float
        Hopping amplitude in x and y directions. Positive.
    mu : float
        Chemical potential.
    Delta : float
        Local s-wave pairing potential.
    Lambda : float
        Spin-orbit coupling.
    B_x : float
        Magnetic field in x.
    B_y : float
        Magnetic field in y.
    B_z : float
        Magnetic field in z.
        
    .. math ::
       \vec{c_{n,m}} = (c_{n,m,\uparrow},
                        c_{n,m,\downarrow},
                        c^\dagger_{n,m,\downarrow},
                        -c^\dagger_{n,m,\uparrow})^T
       
       H = \frac{1}{2} \sum_n^{L_x} \sum_m^{L_y} \vec{c}^\dagger_{n,m} 
       \left(-\mu 
          \tau_z\sigma_0 -\Delta\tau_x\sigma_0 
          -\tau_0(B_x\sigma_x+B_y\sigma_y+B_z\sigma_z)\right) \vec{c}_{n,m}
       + \frac{1}{2}
       \sum_n^{L_x-1}\sum_m^{L_y}\left[\mathbf{c}^\dagger_{n,m}\left(
           -w_0\tau_z\sigma_0 +
           i\lambda\tau_z\sigma_y \right)\mathbf{c}_{n+1,m}
       + H.c.\right]
       + \frac{1}{2}
       \sum_n^{L_x}\sum_m^{L_y-1}\left[\mathbf{c}^\dagger_{n,m}
       \left(-w_0\tau_z\sigma_0 -
       i\lambda\tau_z\sigma_x \right)\mathbf{c}_{n,m+1}
       + H.c.\right]
    """
    def __init__(self, w_0:float,
                 mu:float, Delta:float,
                 Lambda:float, B_x:float,
                 B_y:float, B_z:float):
        self.w_0 = w_0
        self.mu = mu
        self.Delta = Delta
        self.Lambda = Lambda
        self.B_x = B_x
        self.B_y = B_y
        self.B_z = B_z
    def _get_onsite(self):
        return 1/2 * (-self.mu * np.kron(tau_z, sigma_0)
                      - self.Delta * np.kron(tau_x, sigma_0)
                      - self.B_x * np.kron(tau_0, sigma_x)
                      - self.B_y * np.kron(tau_0, sigma_y)
                      - self.B_z * np.kron(tau_0, sigma_z))
    def _get_hopping_x(self):
        return 1/2 * (-self.w_0 * np.kron(tau_z, sigma_0)
                      + 1j*self.Lambda * np.kron(tau_z, sigma_y))
    def _get_hopping_y(self):
        return 1/2 * (-self.w_0 * np.kron(tau_z, sigma_0)
                      - 1j*self.Lambda * np.kron(tau_z, sigma_x))


class SOCSuperconductor(SOCSuperconductivity, Hamiltonian):
    def __init__(self, L_x:int, L_y: int, w_0:float, mu:float,
                 Delta:float, Lambda:float,
                 B_x:float, B_y:float, B_z:float):
        SOCSuperconductivity.__init__(self, w_0, mu, Delta, Lambda,
                                       B_x, B_y, B_z)
        Hamiltonian.__init__(self, L_x, L_y, self._get_onsite(), 
                                   self._get_hopping_x(),
                                   self._get_hopping_y())


class SOCSparseSuperconductor(SOCSuperconductivity,
                                  SparseHamiltonian):
    def __init__(self, L_x:int, L_y: int, w_0:float, mu:float,
                 Delta:float, Lambda:float,
                 B_x:float, B_y:float, B_z:float):
        SOCSuperconductivity.__init__(self, w_0, mu, Delta, Lambda,
                                       B_x, B_y, B_z)
        SparseHamiltonian.__init__(self, L_x, L_y, self._get_onsite(), 
                                   self._get_hopping_x(),
                                   self._get_hopping_y())


class SOCSuperconductorKx(SOCSuperconductivity, Hamiltonian):
    r"""SOC-superconductor for a given k in the x direction and magnetic field.
    
    .. math::

        H = \frac{1}{2}\sum_k H_{ZKMB,k}
        
        H_{k} = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 - \Delta \tau_x\sigma_0
            +2\lambda sin(k) \tau_z\sigma_y
            -\tau_0(B_x\sigma_x+B_y\sigma_y+B_z\sigma_z)
            \right]\vec{c}_n +
            \sum_n^{L-1}\left(\vec{c}^\dagger_n(-w_0\tau_z\sigma_0 
            +i\lambda \tau_z\sigma_x
            )\vec{c}_{n+1}
            + H.c. \right)
        
        \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},
                   -c^\dagger_{-k,\uparrow})^T
    
        \xi_k = -2tcos(k) - \mu
    """
    def __init__(self,  k:float, L_y:int, w_0:float, mu:float,
                 Delta:float, Lambda:float,
                 B_x:float, B_y:float, B_z:float):
        self.k = k
        SOCSuperconductivity.__init__(self, w_0=w_0, mu=mu, Delta=Delta, 
                                      Lambda=Lambda, B_x=B_x, B_y=B_y,
                                      B_z=B_z)
        Hamiltonian.__init__(self, 1, L_y, self._get_onsite(),
                             np.zeros((4, 4)), self._get_hopping_y())
    def _get_onsite(self):
        chi_k = -2*self.w_0 * np.cos(self.k) - self.mu
        return 1/2 * (chi_k * np.kron(tau_z, sigma_0)
                      - self.Delta * np.kron(tau_x, sigma_0)
                      + 2*self.Lambda * np.sin(self.k) * np.kron(tau_z, sigma_y)
                      - self.B_x * np.kron(tau_0, sigma_x)
                      - self.B_y * np.kron(tau_0, sigma_y)
                      - self.B_z * np.kron(tau_0, sigma_z))
    def _get_hopping_y(self):
        return 1/2 * (-self.w_0 * np.kron(tau_z, sigma_0)
                      + 1j*self.Lambda * np.kron(tau_z, sigma_x))

class SOCSuperconductorKxKy(SOCSuperconductivity, Hamiltonian):
    r""" Periodic Hamiltonian in x and y with magnetioc field.
    
    .. math::

        H = \frac{1}{2}\sum_{\mathbf{k}} \psi_{\mathbf{k}}^\dagger H_{\mathbf{k}} \psi_{\mathbf{k}}
        
        H_{\mathbf{k}} =  
            \xi_k\tau_z\sigma_0 - \Delta \tau_x\sigma_0
            + \lambda_{k_x}\tau_z\sigma_y
            + \lambda_{k_y}\tau_z\sigma_x                
            -B_x\tau_0\sigma_x - B_y\tau_0\sigma_y 
            -B_z\tau_0\sigma_z
            
        \vec{c}_k = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},
                   -c^\dagger_{-k,\uparrow})^T
    
        \xi_k = -2w_0(cos(k_x)+cos(k_y)) - \mu
        
        \lambda_{k_x} = 2\lambda sin(k_x)

        \lambda_{k_y} =  - 2\lambda sin(k_y)
    """
    def __init__(self,  k_x:float, k_y:float,
                 w_0:float, mu:float,
                 Delta:float, Lambda:float,
                 B_x:float, B_y:float, B_z:float):
        self.k_x = k_x
        self.k_y = k_y
        SOCSuperconductivity.__init__(self, w_0=w_0, mu=mu, Delta=Delta, 
                                      Lambda=Lambda, B_x=B_x, B_y=B_y,
                                      B_z=B_z)
        Hamiltonian.__init__(self, 1, 1, self._get_onsite(),
                             np.zeros((4, 4)), self._get_hopping_y())
    def _get_onsite(self):
        chi_k = (-2*self.w_0 * (np.cos(self.k_x) + np.cos(self.k_y))
                 - self.mu)
        return 1 * (chi_k * np.kron(tau_z, sigma_0)
                      - self.Delta * np.kron(tau_x, sigma_0)
                      + 2*self.Lambda * np.sin(self.k_x) * np.kron(tau_z, sigma_y)
                      - 2*self.Lambda * np.sin(self.k_y) * np.kron(tau_z, sigma_x)
                      - self.B_x * np.kron(tau_0, sigma_x)
                      - self.B_y * np.kron(tau_0, sigma_y)
                      - self.B_z * np.kron(tau_0, sigma_z))
