###############################################
#A KM-SUB model constructed using MultilayerPy

#Created 

#Reaction name: rxn scheme
#Geometry: spherical
#Number of model components: 3
#Diffusion regime: vignes
###############################################

def dydt(t,y,param_dict,V,A):
    """ Function defining ODEs, returns list of dydt"""

    #--------------Unpack parameters---------------

    Db_3 = param_dict["Db_3"]
    Db_1 = param_dict["Db_1"]
    kd_X_2 = param_dict["kd_X_2"]
    Db_2 = param_dict["Db_2"]
    Xgs_2 = param_dict["Xgs_2"]
    w_2/4 = param_dict["w_2/4"]
    k_1_2 = param_dict["k_1_2"]
    k_1_2_surf = k_1_2 * scale_bulk_to_surf
    delta_3 = param_dict["delta_3"]
    delta_2 = param_dict["delta_2"]
    delta_1 = param_dict["delta_1"]
    alpha_s_0_2 = param_dict["alpha_s_0_2"]

    # init dydt array
    dydt = np.zeros(Lorg * 3 + 3)

    #--------------Define surface uptake parameters for gas components---------------

    #calculate surface fraction of each component
    fs_1 = y[1*Lorg+1] / (+ y[2*Lorg+2]+ y[3*Lorg+3])
    fs_2 = y[2*Lorg+2] / (y[1*Lorg+1] + y[3*Lorg+3])
    fs_3 = y[3*Lorg+3] / (y[1*Lorg+1] + y[2*Lorg+2])

    # component 2 surf params

    surf_cover = delta_2**2 * y[2*Lorg+2] 
    alpha_s_2 = alpha_s_0_2 * (1-surf_cover)
    J_coll_X_2 = Xgs_2 * w_2/4
    J_ads_X_2 = alpha_s_2 * J_coll_X_2
    J_des_X_2 = kd_X_2 * y[2*Lorg+2]
    #--------------Bulk Diffusion evolution---------------

    # Db and fb arrays
    fb_1 = y[0*Lorg+1:1*Lorg+0+1] / (+ y[1*Lorg+2:2*Lorg+1+1]+ y[2*Lorg+3:3*Lorg+2+1])
    fb_2 = y[1*Lorg+2:2*Lorg+1+1] / (y[0*Lorg+1:1*Lorg+0+1] + y[2*Lorg+3:3*Lorg+2+1])
    fb_3 = y[2*Lorg+3:3*Lorg+2+1] / (y[0*Lorg+1:1*Lorg+0+1] + y[1*Lorg+2:2*Lorg+1+1])

    Db_1_arr = np.ones(Lorg) * Db_1
    Db_2_arr = np.ones(Lorg) * Db_2
    Db_3_arr = np.ones(Lorg) * Db_3

    # surface diffusion
    Ds_1 = D_1
    Ds_2 = (Db_2**fs_2) * (Db_2_3**fs_3) * (Db_2_1**fs_1) 
    Ds_3 = D_3
    ksb_2 = H_2 * kbs_2 / Td_2 / (W_2*alpha_s_2/4) 
    kbs_2 = (4/pi) * Ds_2 / delta 
    kssb_1 = kbss_1 / delta_1 
    kssb_3 = kbss_3 / delta_3 
    kbss_1 = (8*Db_1_arr[0])/((delta+delta_1)*pi) 
    kbss_3 = (8*Db_3_arr[0])/((delta+delta_3)*pi) 


    # bulk diffusion
    Db_1_arr = D_1_arr
    Db_2_arr = (Db_2_arr**fb_2_arr) * (Db_2_3_arr**fb_3_arr) * (Db_2_1_arr**fb_1_arr) 
    Db_3_arr = D_3_arr
    kbby_1 = (4/pi) * Db_1_arr / delta 
    kbbx_2 = (4/pi) * Db_2_arr / delta 
    kbby_3 = (4/pi) * Db_3_arr / delta 

    #----component number 1, Oleic acid----
    dydt[0] = kbss_1 * y[1] - kssb_1 * y[0] - y[0*Lorg+0] * y[1*Lorg+1] * k_1_2
    dydt[1] = (kssb_1 * y[0] - kbss_1 * y[1]) * (A[0]/V[0]) + kbby_1[0] * (y[2] - y[1]) * (A[1]/V[0]) - y[0*Lorg+1] * y[1*Lorg+2] * k_1_2
        dydt[0*Lorg*1+i] = kbby_1[i] * (y[0*Lorg*1+(i-1)] - y[0*Lorg*1+i]) * (A[i]/V[i]) + kbby_1[i+1] * (y[0*Lorg*1+(i+1)] - y[0*Lorg*1+i]) * (A[i+1]/V[i]) - y[0*Lorg+1+i] * y[1*Lorg+2+i] * k_1_2
    dydt[1*Lorg+0] = kbby_1[-1] * (y[1*Lorg+0-1] - y[1*Lorg+0]) * (A[-1]/V[-1]) - y[1*Lorg+0] * y[1*Lorg+0] * k_1_2

    #----component number 2, Ozone----
    dydt[1*Lorg+1] = kbs_2 * y[1*Lorg+2]] - ksb_2 * y[1*Lorg+1] - y[1*Lorg+1] * y[0*Lorg+0] * k_1_2
    dydt[1*Lorg+2] = (ksb_2 * y[1*Lorg+1] - kbs_2 * y[1*Lorg+2]) * (A[0]/V[0]) + kbbx_2[0] * (y[1*Lorg+3] - y[1*Lorg+2]) * (A[1]/V[0]) - y[1*Lorg+2] * y[0*Lorg+1] * k_1_2
        dydt[1*Lorg*2+i] = kbbx_2[i] * (y[1*Lorg*2+(i-1)] - y[1*Lorg*2+i]) * (A[i]/V[i]) + kbbx_2[i+1] * (y[1*Lorg*2+(i+1)] - y[1*Lorg*2+i]) * (A[i+1]/V[i]) - y[1*Lorg+2+i] * y[0*Lorg+1+i] * k_1_2
    dydt[2*Lorg+1] = kbbx_2[-1] * (y[2*Lorg+1-1] - y[2*Lorg+1]) * (A[-1]/V[-1]) - y[2*Lorg+1] * y[2*Lorg+1] * k_1_2

    #----component number 3, Y3----
    dydt[2*Lorg+2] = kbss_3 * y[2*Lorg+3]] - kssb_3 * y[2*Lorg+2] 
    dydt[2*Lorg+3] = (kssb_3 * y[2*Lorg+2] - kbss_3 * y[2*Lorg+3]) * (A[0]/V[0]) + kbby_3[0] * (y[2*Lorg+4] - y[2*Lorg+3]) * (A[1]/V[0]) 
        dydt[2*Lorg*3+i] = kbby_3[i] * (y[2*Lorg*3+(i-1)] - y[2*Lorg*3+i]) * (A[i]/V[i]) + kbby_3[i+1] * (y[2*Lorg*3+(i+1)] - y[2*Lorg*3+i]) * (A[i+1]/V[i]) 
    dydt[3*Lorg+2] = kbby_3[-1] * (y[3*Lorg+2-1] - y[3*Lorg+2]) * (A[-1]/V[-1]) 

    return dydt