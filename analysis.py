# -*- coding: utf-8 -*-
"""
@author: Adam Milsom

    MultilayerPy - build, run and optimise kinetic multi-layer models for 
    aerosol particles and films.
    
    Copyright (C) 2021  Adam Milsom

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
"""

'''
Analysis module of the MultilayerPy package

'''

import numpy as np




# calculation of surface-to-total loss ratio (STLR)
def STLR(X_surf,Y_surf,X_bulk,Y_bulk,k_SLR,k_BR):
    '''
    Calculate the Surface to Total Loss rate Ratio.
    To determine the amount of surface/bulk dominance of the reaction
    
    X_surf = [X]surf
    Y_surf = [Y]surf
    X_bulk = [X]bulk
    Y_bulk = [Y]bulk
    k_SLR = second order surface layer rate coefficient
    k_BR = second order rate coefficient for bulk reactions
    
    STLR = L_surf / (L_surf + sum_over_k[L_k])
    
    L_surf = k_SLR * [X]surf * [Y]surf
    L_bk = k_BR * [X]bulk_k * [Y]bulk_k (k = bulk layer number)
    
    returns a list of STLR values at each time point
    '''
    
    L_s = k_SLR * X_surf * Y_surf
    L_bk = k_BR * X_bulk * Y_bulk
    
    # loop over all time points and calculate the STLR
    STLR = []
    for t in np.arange(len(X_surf)):
        stlr_t = L_s[t] / (L_s[t] + np.sum(L_bk[t,:]))
        STLR.append(stlr_t)
    
    return STLR




def saturation_ratio(X_b,X_s,X_gs,H_cp,T,alpha_s_0_X=None,w_X=None,
                    Td_X=None,delta_X=None,bulk=True):
    '''
    Calculate either the bulk or surface saturation ratio
    
    Calculate the saturation ratio for a bulk reaction-dominated case (BSR)
    
    BSR = [X]b1 / [X]b,sat (b1 = bulk layer 1)
    
    [X]b,sat = H_cc * [X]g 
    (H_cc = Henry's law coefficient, H_cp * RT; [X]g = gas-phase conc. of X)
    
    returns BSR at each time point
    
    Calculate the saturation ratio for a surface reaction-dominated process (SSR)

    SSR = [X]s / [X]s,sat

    [X]s,sat = (ka/kd) * [X]gs

    ka = (alpha_s,X * w_X / 4) * [X]gs
    kd = 1 / Td,X
    
    '''
    R = 82.0578
    if bulk:
        H_cc = H_cp * R * T
        X_b1 = X_b[:,0]
        X_b_sat = H_cc * X_gs
        
        BSR = X_b1 / X_b_sat
        #print(X_b_sat)
        #print(X_b1[:10])
        return BSR
    
    else:
        surf_cover = delta_X * X_s
        alpha_s_X = alpha_s_0_X * (1-surf_cover)
        ka = (alpha_s_X * w_X / 4) * X_gs
        kd = 1.0 / Td_X
        
        SSR = X_s * (ka/kd)
        
        return SSR


def mixing_parameter(Db_X,Db_Y,k_BR,X_b,Y_b,rp,V_k):
    '''
    Calculate the mixing parameter for a bulk diffusion-limited system BMP
    
    reacto-diffusive length (for X and Y):
    
    l_rd = sqrt(Db / k_BR * [X or Y]eff)
    
    [X]eff = (SUM L_k * V_k * [X]bk) / (SUM L_k * V_k)
    (same eq for [Y]eff)
    
    BMP_X = l_rd_X / (l_rd_X + rp/e)
    (rp = particle radius; same eq for BMP_Y)
    
    BMP_XY = (BMP_X + BMP_Y) / 2
    
    returns BMP_XY at each time point
    '''
    
    # INCLUDE Db EVOLUTION FOR EACH PARAM to get avg Db
    
    # calc loss rate for each layer
    
    L_bk = k_BR * (X_b * Y_b)
    
    # calc numerator for [X] and [Y] eff calculation
    
    Lk_Vk = L_bk * V_k # (allows multiplication over cols) CHECK
    
    l_rd_X_num = np.sum(Lk_Vk * X_b, axis=1)
    l_rd_Y_num = np.sum(Lk_Vk * Y_b, axis=1)
    
    X_eff = l_rd_X_num / np.sum(Lk_Vk, axis=1)
    Y_eff = l_rd_Y_num / np.sum(Lk_Vk, axis=1)
        
    # calc l_rd (using Dx and Dy in Y - may want to change this an an avg D?)
    
   
    
    l_rd_X = np.sqrt(Db_X / (k_BR * Y_eff))
    l_rd_Y = np.sqrt(Db_Y/ (k_BR * X_eff))
    
    # calc BMP X and BMP Y
    
    BMP_X = l_rd_X / (l_rd_X + rp/np.e)
    BMP_Y = l_rd_Y / (l_rd_Y + rp/np.e)
    
    BMP_XY = (BMP_X + BMP_Y) / 2
    
    return BMP_XY



