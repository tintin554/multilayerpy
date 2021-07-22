# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:01:48 2021

@author: Adam
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




def saturation_ratio(X_b,X_s,X_gs,alpha_X_0,w_X,H_cc,Td_X,delta_X,bulk=True):
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
    if bulk:
        X_b1 = X_b[:,0]
        X_b_sat = H_cc * X_gs
        
        BSR = X_b1 / X_b_sat
        #print(X_b_sat)
        #print(X_b1[:10])
        return BSR
    
    else:
        surf_cover = delta_X * X_s
        alpha_s_X = alpha_X_0 * (1-surf_cover)
        ka = (alpha_s_X * w_X / 4) * X_gs
        kd = 1.0 / Td_X
        
        SSR = X_s * (ka/kd)
        
        return SSR






