#%% Import necessary libraries
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp

#%% Import initial values reported in Benchmark Simulation Model no. 2 (BSM2)
S_su_init = 0.0124
S_aa_init = 0.0055
S_fa_init = 0.1074
S_va_init = 0.0123
S_bu_init = 0.0140
S_pro_init = 0.0176
S_ac_init = 0.0893
S_h2_init = 2.5055e-7
S_ch4_init = 0.0555
S_IC_init = 0.0951
S_IN_init = 0.0945
S_I_init = 0.1309
X_xc_init = 0.1079
X_ch_init = 0.0205
X_pr_init = 0.0842
X_li_init = 0.0436
X_su_init = 0.3122
X_aa_init = 0.9317
X_fa_init = 0.3384
X_c4_init = 0.3258
X_pro_init = 0.1011
X_ac_init = 0.6772
X_h2_init = 0.2848
X_I_init = 17.2162
S_cat_init = 3.5659e-43
S_an_init = 0.0052
S_hva_init = 0.0123   # is actually Sva-
S_hbu_init = 0.0140   # is actually Sbu-
S_hpro_init = 0.0175  # is actually Spro-
S_hac_init = 0.0890   # is actually Sac-
S_hco3_init = 0.0857
S_nh3_init = 0.0019
S_gas_h2_init = 1.1032e-5
S_gas_ch4_init = 1.6535
S_gas_co2_init = 0.0135
Q_D_init = 178.4674
T_D_init = 35

DIGESTERINIT = np.array([
    S_su_init, S_aa_init, S_fa_init, S_va_init, S_bu_init, S_pro_init,
    S_ac_init, S_h2_init, S_ch4_init, S_IC_init, S_IN_init, S_I_init,
    X_xc_init, X_ch_init, X_pr_init, X_li_init, X_su_init, X_aa_init,
    X_fa_init, X_c4_init, X_pro_init, X_ac_init, X_h2_init, X_I_init,
    S_cat_init, S_an_init, S_hva_init, S_hbu_init, S_hpro_init, S_hac_init,
    S_hco3_init, S_nh3_init, S_gas_h2_init, S_gas_ch4_init, S_gas_co2_init
])

#%% Define the ADM1 Model - System of ODEs
def ADM1_model(t, x):
    # Stoichiometric parameter values from Benchmark Simulation Model no. 2 (BSM2)
    f_sI_xc = 0.1
    f_xI_xc = 0.2
    f_ch_xc = 0.2
    f_pr_xc = 0.2
    f_li_xc = 0.3
    N_xc = 0.0376 / 14
    N_I = 0.06 / 14  # kmole N.kg^-1COD
    N_aa = 0.007  # kmole N.kg^-1COD
    C_xc = 0.02786  # kmole C.kg^-1COD
    C_sI = 0.03  # kmole C.kg^-1COD
    C_ch = 0.0313  # kmole C.kg^-1COD
    C_pr = 0.03  # kmole C.kg^-1COD
    C_li = 0.022  # kmole C.kg^-1COD
    C_xI = 0.03  # kmole C.kg^-1COD
    C_su = 0.0313  # kmole C.kg^-1COD
    C_aa = 0.03  # kmole C.kg^-1COD
    f_fa_li = 0.95
    C_fa = 0.0217  # kmole C.kg^-1COD
    f_h2_su = 0.19
    f_bu_su = 0.13
    f_pro_su = 0.27
    f_ac_su = 0.41
    N_bac = 0.08 / 14  # kmole N.kg^-1COD
    C_bu = 0.025  # kmole C.kg^-1COD
    C_pro = 0.0268  # kmole C.kg^-1COD
    C_ac = 0.0313  # kmole C.kg^-1COD
    C_bac = 0.0313  # kmole C.kg^-1COD
    Y_su = 0.1
    f_h2_aa = 0.06
    f_va_aa = 0.23
    f_bu_aa = 0.26
    f_pro_aa = 0.05
    f_ac_aa = 0.40
    C_va = 0.024  # kmole C.kg^-1COD
    Y_aa = 0.08
    Y_fa = 0.06
    Y_c4 = 0.06
    Y_pro = 0.04
    C_ch4 = 0.0156  # kmole C.kg^-1COD
    Y_ac = 0.05
    Y_h2 = 0.06

    # Biochemical parameter values from BSM 2
    k_dis = 0.5  # d^-1
    k_hyd_ch = 10  # d^-1
    k_hyd_pr = 10  # d^-1
    k_hyd_li = 10  # d^-1
    K_S_IN = 10 ** -4  # M
    k_m_su = 30  # d^-1
    K_S_su = 0.5  # kgCOD.m^-3
    pH_UL_aa = 5.5
    pH_LL_aa = 4
    k_m_aa = 50  # d^-1
    K_S_aa = 0.3  # kgCOD.m^-3
    k_m_fa = 6  # d^-1
    K_S_fa = 0.4  # kgCOD.m^-3
    K_Ih2_fa = 5.0e-6
    k_m_c4 = 20  # d^-1
    K_S_c4 = 0.2  # kgCOD.m^-3
    K_Ih2_c4 = 10 ** -5  # kgCOD.m^-3
    k_m_pro = 13  # d^-1
    K_S_pro = 0.1  # kgCOD.m^-3
    K_Ih2_pro = 3.5 * 10 ** -6  # kgCOD.m^-3
    k_m_ac = 8  # kgCOD.m^-3
    K_S_ac = 0.15  # kgCOD.m^-3
    K_I_nh3 = 0.0018  # M
    pH_UL_ac = 7.0
    pH_LL_ac = 6.0
    k_m_h2 = 35  # d^-1
    K_S_h2 = 7 * 10 ** -6  # kgCOD.m^-3
    pH_UL_h2 = 6
    pH_LL_h2 = 5
    k_dec_Xsu = 0.02  # d^-1
    k_dec_Xaa = 0.02  # d^-1
    k_dec_Xfa = 0.02  # d^-1
    k_dec_Xc4 = 0.02  # d^-1
    k_dec_Xpro = 0.02  # d^-1
    k_dec_Xac = 0.02  # d^-1
    k_dec_Xh2 = 0.02  # d^-1
    ## M is kmole m^-3

    #  Physiochemical parameter values from BSM2
    R = 0.083145  # bar.M^-1.K^-1
    T_base = 298.15  # K
    T_op = 308.15  # k ##T_op #=35 C
    pK_w_base = 14.0
    pK_a_va_base = 4.86
    pK_a_bu_base = 4.82
    pK_a_pro_base = 4.88
    pK_a_ac_base = 4.76
    pK_a_co2_base = 6.35
    pK_a_IN_base = 9.25
    factor = (1.0 / T_base - 1.0 / T_op) / (100.0 * R)    # Temperature adjustment factor
    K_w = 10 ** (-pK_w_base) * np.exp(55900.0 * factor)
    K_a_va = 10 ** (-pK_a_va_base)
    K_a_bu = 10 ** (-pK_a_bu_base)
    K_a_pro = 10 ** (-pK_a_pro_base)
    K_a_ac = 10 ** (-pK_a_ac_base)
    K_a_co2 = 10 ** (-pK_a_co2_base) * np.exp(7646.0 * factor)
    K_a_IN = 10 ** (-pK_a_IN_base) * np.exp(51965.0 * factor)
    K_A_Bva = 10 ** 10  # M^-1 * d^-1
    K_A_Bbu = 10 ** 10  # M^-1 * d^-1
    K_A_Bpro = 10 ** 10  # M^-1 * d^-1
    K_A_Bac = 10 ** 10  # M^-1 * d^-1
    K_A_Bco2 = 10 ** 10  # M^-1 * d^-1
    K_A_BIN = 10 ** 10  # M^-1 * d^-1
    P_atm = 1.013  # bar
    k_P = 5 * 10 ** 4  # m^3.d^-1.bar^-1
    kLa = 200.0  # d^-1
    K_H_h2o_base = 0.0313
    p_gas_h2o = K_H_h2o_base * np.exp(5290.0 * (1.0 / T_base - 1.0 / T_op))
    K_H_co2_base = 0.035
    K_H_ch4_base = 0.0014
    K_H_h2_base = 7.8e-4
    K_H_h2 = K_H_h2_base * np.exp(-4180.0 * factor) # Mliq.bar^-1
    K_H_ch4 = K_H_ch4_base * np.exp(-14240.0 * factor) # Mliq.bar^-1
    K_H_co2 = K_H_co2_base * np.exp(-19410.0 * factor) # Mliq.bar^-1

    # Physical parameter values from BSM1
    V_liq = 3400  # m3, size of BSM2 AD
    V_gas = 300   # m3, size of BSM2 AD
    eps = 0.000001

    # Calculate phi
    phi = (x[24] + (x[10] - x[31]) - x[30] - x[29] / 64.0 -
           x[28] / 112.0 - x[27] / 160.0 - x[26]/208.0 - x[25])

    # Calculate S_H_ion and pH_op
    S_H_ion = -phi * 0.5 + 0.5 * (phi ** 2 + 4.0 * K_w)**0.5

    # Calculate gas partial pressures
    p_gas_h2 = x[32] * R * T_op / 16.0
    p_gas_ch4 = x[33] * R * T_op / 64.0
    p_gas_co2 = x[34] * R * T_op

    # Total gas pressure
    P_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o

    # Total gas flow rate
    q_gas = k_P * (P_gas - P_atm)

    # Hill function calculations
    pHLim_aa = 10 ** (-(pH_UL_aa + pH_LL_aa) / 2.0)
    pHLim_ac = 10 ** (-(pH_UL_ac + pH_LL_ac) / 2.0)
    pHLim_h2 = 10 ** (-(pH_UL_h2 + pH_LL_h2) / 2.0)

    n_aa = 3.0 / (pH_UL_aa - pH_LL_aa)
    n_ac = 3.0 / (pH_UL_ac - pH_LL_ac)
    n_h2 = 3.0 / (pH_UL_h2 - pH_LL_h2)

    I_pH_aa = pHLim_aa ** n_aa / (S_H_ion ** n_aa + pHLim_aa ** n_aa)
    I_pH_ac = pHLim_ac ** n_ac / (S_H_ion ** n_ac + pHLim_ac ** n_ac)
    I_pH_h2 = pHLim_h2 ** n_h2 / (S_H_ion ** n_h2 + pHLim_h2 ** n_h2)

    # Additional inhibition functions
    I_IN_lim = 1.0 / (1.0 + K_S_IN / (x[10]+eps))
    I_h2_fa = 1.0 / (1.0 + x[7] / K_Ih2_fa)
    I_h2_c4 = 1.0 / (1.0 + x[7] / K_Ih2_c4)
    I_h2_pro = 1.0 / (1.0 + x[7] / K_Ih2_pro)
    I_nh3 = 1.0 / (1.0 + x[31] / K_I_nh3)

    eps = 1e-10  # A small constant to avoid division by zero

    # Define inhibition variables
    inhib = [None] * 6

    # Inhibition calculations
    inhib[0] = I_pH_aa * I_IN_lim
    inhib[1] = inhib[0] * I_h2_fa
    inhib[2] = inhib[0] * I_h2_c4
    inhib[3] = inhib[0] * I_h2_pro
    inhib[4] = I_pH_ac * I_IN_lim * I_nh3
    inhib[5] = I_pH_h2 * I_IN_lim

    # Process calculations
    proc1 = k_dis * x[12]
    proc2 = k_hyd_ch * x[13]
    proc3 = k_hyd_pr * x[14]
    proc4 = k_hyd_li * x[15]

    proc5 = k_m_su * x[0] / (K_S_su + x[0]) * x[16] * inhib[0]
    proc6 = k_m_aa * x[1] / (K_S_aa + x[1]) * x[17] * inhib[0]
    proc7 = k_m_fa * x[2] / (K_S_fa + x[2]) * x[18] * inhib[1]

    proc8 = k_m_c4 * x[3] / (K_S_c4 + x[3]) * x[19] * x[3] / (x[3] + x[4] + eps) * inhib[2]
    proc9 = k_m_c4 * x[4] / (K_S_c4 + x[4]) * x[19] * x[4] / (x[3] + x[4] + eps) * inhib[2]

    proc10 = k_m_pro * x[5] / (K_S_pro + x[5]) * x[20] * inhib[3]
    proc11 = k_m_ac * x[6] / (K_S_ac + x[6]) * x[21] * inhib[4]
    proc12 = k_m_h2 * x[7] / (K_S_h2 + x[7]) * x[22] * inhib[5]

    # Decay processes
    proc13 = k_dec_Xsu * x[16]
    proc14 = k_dec_Xaa * x[17]
    proc15 = k_dec_Xfa * x[18]
    proc16 = k_dec_Xc4 * x[19]
    proc17 = k_dec_Xpro * x[20]
    proc18 = k_dec_Xac * x[21]
    proc19 = k_dec_Xh2 * x[22]

    # Additional processes
    procA4 = K_A_Bva * (x[26] * (K_a_va + S_H_ion) - K_a_va * x[3])
    procA5 = K_A_Bbu * (x[27] * (K_a_bu + S_H_ion) - K_a_bu * x[4])
    procA6 = K_A_Bpro * (x[28] * (K_a_pro + S_H_ion) - K_a_pro * x[5])
    procA7 = K_A_Bac * (x[29] * (K_a_ac + S_H_ion) - K_a_ac * x[6])
    procA10 = K_A_Bco2 * (x[30] * (K_a_co2 + S_H_ion) - K_a_co2 * x[9])
    procA11 = K_A_BIN * (x[31] * (K_a_IN + S_H_ion) - K_a_IN * x[10])

    procT8 = kLa * (x[7] - 16.0 * K_H_h2 * p_gas_h2)
    procT9 = kLa * (x[8] - 64.0 * K_H_ch4 * p_gas_ch4)
    procT10 = kLa * ((x[9] - x[30]) - K_H_co2 * p_gas_co2)

    stoich1 = -C_xc + f_sI_xc * C_sI + f_ch_xc * C_ch + f_pr_xc * C_pr + f_li_xc * C_li + f_xI_xc * C_xI
    stoich2 = -C_ch + C_su
    stoich3 = -C_pr + C_aa
    stoich4 = -C_li + (1.0 - f_fa_li) * C_su + f_fa_li * C_fa
    stoich5 = -C_su + (1.0 - Y_su) * (f_bu_su * C_bu + f_pro_su * C_pro + f_ac_su * C_ac) + Y_su * C_bac
    stoich6 = -C_aa + (1.0 - Y_aa) * (f_va_aa * C_va + f_bu_aa * C_bu + f_pro_aa * C_pro + f_ac_aa * C_ac) + Y_aa * C_bac
    stoich7 = -C_fa + (1.0 - Y_fa) * 0.7 * C_ac + Y_fa * C_bac
    stoich8 = -C_va + (1.0 - Y_c4) * 0.54 * C_pro + (1.0 - Y_c4) * 0.31 * C_ac + Y_c4 * C_bac
    stoich9 = -C_bu + (1.0 - Y_c4) * 0.8 * C_ac + Y_c4 * C_bac
    stoich10 = -C_pro + (1.0 - Y_pro) * 0.57 * C_ac + Y_pro * C_bac
    stoich11 = -C_ac + (1.0 - Y_ac) * C_ch4 + Y_ac * C_bac
    stoich12 = (1.0 - Y_h2) * C_ch4 + Y_h2 * C_bac
    stoich13 = -C_bac + C_xc

    reac  = [None]*24
    # Define reactions
    reac[0] = proc2 + (1.0 - f_fa_li) * proc4 - proc5 # S_su
    reac[1] = proc3 - proc6 # S_aa
    reac[2] = f_fa_li * proc4 - proc7 # S_fa
    reac[3] = (1.0 - Y_aa) * f_va_aa * proc6 - proc8 # S_va
    reac[4] = (1.0 - Y_su) * f_bu_su * proc5 + (1.0 - Y_aa) * f_bu_aa * proc6 - proc9 # S_bu
    reac[5] = (1.0 - Y_su) * f_pro_su * proc5 + (1.0 - Y_aa) * f_pro_aa * proc6 + (1.0 - Y_c4) * 0.54 * proc8 - proc10 # S_pro
    reac[6] = (1.0 - Y_su) * f_ac_su * proc5 + (1.0 - Y_aa) * f_ac_aa * proc6 + (1.0 - Y_fa) * 0.7 * proc7 + (1.0 - Y_c4) * 0.31 * proc8 + (1.0 - Y_c4) * 0.8 * proc9 + (1.0 - Y_pro) * 0.57 * proc10 - proc11 # S_ac
    reac[7] = (1.0 - Y_su) * f_h2_su * proc5 + (1.0 - Y_aa) * f_h2_aa * proc6 + (1.0 - Y_fa) * 0.3 * proc7 + (1.0 - Y_c4) * 0.15 * proc8 + (1.0 - Y_c4) * 0.2 * proc9 + (1.0 - Y_pro) * 0.43 * proc10 - proc12 - procT8 # S_h2
    reac[8] = (1.0 - Y_ac) * proc11 + (1.0 - Y_h2) * proc12 - procT9 # S_ch4
    reac[9] = -stoich1 * proc1 - stoich2 * proc2 - stoich3 * proc3 - stoich4 * proc4 - stoich5 * proc5 - stoich6 * proc6 - stoich7 * proc7 - stoich8 * proc8 - stoich9 * proc9 - stoich10 * proc10 - stoich11 * proc11 - stoich12 * proc12 - stoich13 * proc13 - stoich13 * proc14 - stoich13 * proc15 - stoich13 * proc16 - stoich13 * proc17 - stoich13 * proc18 - stoich13 * proc19 - procT10 # S_IC
    reac[10] = (N_xc - f_xI_xc * N_I - f_sI_xc * N_I - f_pr_xc * N_aa) * proc1 - Y_su * N_bac * proc5 + (N_aa - Y_aa * N_bac) * proc6 - Y_fa * N_bac * proc7 - Y_c4 * N_bac * proc8 - Y_c4 * N_bac * proc9 - Y_pro * N_bac * proc10 - Y_ac * N_bac * proc11 - Y_h2 * N_bac * proc12 + (N_bac - N_xc) * (proc13 + proc14 + proc15 + proc16 + proc17 + proc18 + proc19) # S_IN
    reac[11] = f_sI_xc * proc1 # S_IN
    reac[12] = -proc1 + proc13 + proc14 + proc15 + proc16 + proc17 + proc18 + proc19 # X_c
    reac[13] = f_ch_xc * proc1 - proc2 # X_ch
    reac[14] = f_pr_xc * proc1 - proc3 # X_pr
    reac[15] = f_li_xc * proc1 - proc4 # X_li
    reac[16] = Y_su * proc5 - proc13 # X_su
    reac[17] = Y_aa * proc6 - proc14 # X_aa
    reac[18] = Y_fa * proc7 - proc15 # X_fa
    reac[19] = Y_c4 * proc8 + Y_c4 * proc9 - proc16 # X_c4
    reac[20] = Y_pro * proc10 - proc17 # X_pro
    reac[21] = Y_ac * proc11 - proc18 # X_ac
    reac[22] = Y_h2 * proc12 - proc19 # X_h2
    reac[23] = f_xI_xc * proc1 # X_I

    # Import influent values reported in BSM2 (Benchmark Simulation Model No 2)
    S_su_in = 0  # kg COD m^-3
    S_aa_in = 0.04388  # kg COD m^-3
    S_fa_in = 0  # kg COD m^-3
    S_va_in = 0  # kg COD m^-3
    S_bu_in = 0  # kg COD m^-3
    S_pro_in = 0  # kg COD m^-3
    S_ac_in = 0  # kg COD m^-3
    S_h2_in = 0  # kg COD m^-3
    S_ch4_in = 0  # kg COD m^-3
    S_IC_in = 0.0079326  # kmole m^-3
    S_IN_in = 0.0019721  # kmole m^-3
    S_I_in = 0.028067  # kg COD m^-3
    X_xc_in = 0  # kg COD m^-3
    X_ch_in = 3.7236  # kg COD m^-3
    X_pr_in = 15.9235  # kg COD m^-3
    X_li_in = 8.047  # kg COD m^-3
    X_su_in = 0  # kg COD m^-3
    X_aa_in = 0  # kg COD m^-3
    X_fa_in = 0  # kg COD m^-3
    X_c4_in = 0  # kg COD m^-3
    X_pro_in = 0  # kg COD m^-3
    X_ac_in = 0  # kg COD m^-3
    X_h2_in = 0  # kg COD m^-3
    X_I_in = 17.0106  # kg COD m^-3
    S_cat_in = 0
    S_an_in = 0.0052
    Q_in = 178.4674

    # Create the array for Influent composition
    influent_comp = np.array([
        S_su_in, S_aa_in, S_fa_in, S_va_in, S_bu_in,
        S_pro_in, S_ac_in, S_h2_in, S_ch4_in, S_IC_in,
        S_IN_in, S_I_in, X_xc_in, X_ch_in, X_pr_in,
        X_li_in, X_su_in, X_aa_in, X_fa_in, X_c4_in,
        X_pro_in, X_ac_in, X_h2_in, X_I_in, S_cat_in,
        S_an_in, Q_in])
    u = influent_comp

    # Define dx array with size 35 (Python uses 0-based indexing)
    dx = [0] * 35 # Initialize differential equations arrays (dx) with zeros

    # Loop to populate the dx with odes of solubles and particulates
    for i in range(24):
        dx[i] = 1.0 / V_liq * (u[26] * (u[i] - x[i])) + reac[i]

    # Populate dx with odes of cations and anions
    dx[24] = 1.0 / V_liq * (u[26] * (u[24] - x[24]))  # Scat+
    dx[25] = 1.0 / V_liq * (u[26] * (u[25] - x[25]))  # San-

    # Populate dx with odes of ion states
    dx[26] = -procA4  # Sva-
    dx[27] = -procA5  # Sbu-
    dx[28] = -procA6  # Spro-
    dx[29] = -procA7  # Sac-
    dx[30] = -procA10  # SHCO3-
    dx[31] = -procA11  # SNH3

    # Populate dx with odes for gas phase components
    dx[32] = -x[32] * q_gas / V_gas + procT8 * V_liq / V_gas # S_gas,h2
    dx[33] = -x[33] * q_gas / V_gas + procT9 * V_liq / V_gas # S_gas,ch4
    dx[34] = -x[34] * q_gas / V_gas + procT10 * V_liq / V_gas # S_gas,co2
    return dx

#%% Solve the ADM1's system of ODEs using SciPy's solve_ivp (initial value problem solver)
t_span = [0, 200] # days
y0 = 5*DIGESTERINIT # initial conditions
sol = solve_ivp(ADM1_model, t_span, y0, method = 'BDF', dense_output=True) # solve using ivp
t_out = sol.t # extract time from solution
x_out = sol.y # extract states from solution

#%% Calculate the total gas flow rate
R = 0.083145  # Universal gas constant in dm3*bar/(mol*K)
T_op = 308.15    # K
T_base = 298.15  # K
k_P = 5.0e4
P_atm = 1.013  # bar

# Partial pressure calculation using concentration
p_gas_h2 = x_out[32,:] * R * T_op / 16.0
p_gas_ch4 = x_out[33,:] * R * T_op / 64.0
p_gas_co2 = x_out[34,:] * R * T_op

K_H_h2o_base = 0.0313
p_gas_h2o = K_H_h2o_base * np.exp(5290.0 * (1.0 / T_base - 1.0 / T_op))

# Total gas pressure calculation
P_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o
q_gas_temp = k_P * (P_gas - P_atm)

# Total flow rate calculation
q_gas = q_gas_temp * P_gas/P_atm # The output gas flow is recalculated to atmospheric pressure (normalization)

#%% Compare ADM1's steady-state values from this code with ADM1's steady-state values reported in BSM2
# Steady-sate values for ADM1 from this code
x_ss_python = x_out[:, len(x_out[1,:])-1]
x_ss_python = np.append(x_ss_python, q_gas[len(q_gas)-1]) # Append with gas flow rate

# Steady-state values for ADM1 from BSM2
x_ss_BSM2 = [0.0123944521960000,0.00554315146500000,0.107407097456000,0.0123325298170000,0.0140030358520000,0.0175839179510000,0.0893147057000000,2.50550000000000e-07,0.0554902091560000,0.0951488348800000,0.0944681918740000,0.130867066076000,0.107920890509000,0.0205168226890000,0.0842203691100000,0.0436287448300000,0.312223348554000,0.931671811511000,0.338391125445000,0.335772136662000,0.101120522211000,0.677244333548000,0.284839619000000,17.2162246904400,0,0.00521009922300000,0.0122839755570000,0.0139527381860000,0.0175114392780000,0.0890351618480000,0.0856799661500000,0.00188401134000000,1.10324120000000e-05,1.65349847162100,0.0135401278050000, 2708.343137271645]

# Arrange the above two steady-state values for comparison
df_validation = pd.DataFrame({
    'x_ss_python': np.round(x_ss_python,4),
    'x_ss_BSM2': np.round(x_ss_BSM2,4) })

# Display the comparison table
pd.options.display.float_format = '{:.4f}'.format # To display 4 decimal places without scientific notation
print(df_validation)

