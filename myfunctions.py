# *new* functions for gas-fired calciner scenario
def get_dhO2(T1,T2):
    # https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Units=SI&Mask=1#Thermo-Gas
    # defined over 100-2000 K
    import numpy as np
    t = np.linspace(T1,T2,(T2-T1))/1000
    cp = np.zeros(len(t))
    for i in range(len(cp)):
        if t[i]*1000<700:
            A = 31.32234
            B = -20.23531
            C = 57.86644
            D = -36.50624
            E = -0.007374
        else:
            A = 30.03235
            B = 8.772972
            C = -3.988133
            D = 0.788313
            E = -0.741599
        cp[i] = A + B*t[i] + C*t[i]**2 + D*t[i]**3 + E/t[i]**2
    dh = np.sum(cp)
    return dh # [J/mol]

def get_dhCH4(T1,T2):
    # https://webbook.nist.gov/cgi/cbook.cgi?ID=C74828&Mask=1&Type=JANAFG&Plot=on#JANAFG
    # defined over 298-1300 K
    import numpy as np
    t = np.linspace(T1,T2,(T2-T1))
    if T1<298:
        t[np.where(t<298)]=298
    t = t/1000
    cp = np.zeros(len(t))
    A = -0.703029
    B = 108.4773
    C = -42.52157
    D = 5.862788
    E = 0.678565
    cp = A + B*t + C*t**2 + D*t**3 + E/t**2
    dh = np.sum(cp)
    return dh # [J/mol]

def get_dhH2O(T1,T2):
    # T1 == Ta [K]
    # T2 == exit temperature from calciner [K]
    
    # https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Units=SI&Mask=1#Thermo-Gas
    # https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Units=SI&Mask=2#Thermo-Condensed
    # gas phase defined over 500-1700 K
    # liquid phase defined over 298-500 K
    import numpy as np
    
    # gas-phase standard enthalpy at calciner exit T2 [kJ/mol]
    A = 30.09200 
    B = 6.832514
    C = 6.793435
    D = -2.534480
    E = 0.082139
    F = -250.8810
    H = -241.8264
    t = T2/1000 # [K]
    Hf_gas = -241.83 # [kJ/mol]
    H_gas = A*t + B*t**2/2 + C*t**3/3 + D*t**4/4 - E/t + F - H + Hf_gas
    
    # liquid-phase standard enthalpy at ambient T1 [kJ/mol]
    A = -203.6060
    B = 1523.290
    C = -3196.413
    D = 2474.455
    E = 3.855326
    F = -256.5478
    H = -285.8304
    t = T1/1000 # [K]
    Hf_liquid = -285.83 # [kJ/mol]
    H_liquid = A*t + B*t**2/2 + C*t**3/3 + D*t**4/4 - E/t + F - H + Hf_liquid
     
    dh = (H_gas-H_liquid)*1000
    return dh # [J/mol] positive value indicates enthalpy lost by the water due to cooling, available for heat recovery

def get_balanceOfPlant_gasFiredCalciner(mdot_L,mdot_G,M_OH,M_CO3,z,A,h,Z,TL,Ta,P,RH,CO2captureRate,G_inlet,G_outlet,vapor_inlet,vapor_outlet,n_hi): 
    # Description: after simulation of contactor, get power consumption (mechanical and thermal), evaporative losses, and CaCO3 losses from primary process equipment.
    # Source: cycle based on Keith (2018)
    # Output: c: array of process objects ['contactor','reactor','calciner','CaCO3','slaker','CaO','compression','CO2'] with variables:
        # m [MW]: mechanical power
        # t [MW]: thermal power
        # w [t/hr]: evaporative losses
        # c [t/hr]: CaCO3 makeup
    # Inputs:
        # mdot_L [t/h]: solvent mass flow rate (contactor)
        # mdot_G [t/h]: gas mass flow rate (contactor)
        # M_OH [mol/L solution]: inlet molar concentration of OH- in solution
        # M_CO3 [mol/L solution]: inlet molar concentration of CO3-2 in solution
        # z: stoichiometric coefficient of binding species (CO2 + 2KOH = K2CO3 + H2O)
        # A [m^2]: cross-sectional area of contactor
        # h [m]: height to pump solvent
        # Z [m]: depth of packing
        # TL [K]: liquid temperature
        # Ta [K]: air temperature
        # P [Pa]: ambient pressure
        # RH: relative humidity (fraction of saturated humidity level)
        # CO2captureRate [t/hr]: capture rate of CO2 
        # G_inlet [mol/s]: bulk molar gas flow rate at contactor inlet
        # G_outlet [mol/s]: bulk molar gas flow rate at contactor outlet
        # vapor_inlet [mol-vap/mol-gas]: mol fraction of water vapor at contactor inlet
        # vapor outlet [mol-vap/mol-gas]: mol fraction of water vapor at contactor outlet
        # n_hi: fraction recovered sensible heat (heat integration)
        # c: object instance for output
        
    import numpy as np
    from scipy.constants import g
    from myfunctions import get_MM, get_rhoAir, get_rhoWater, get_muWater, get_fanPower, get_pumpPower, get_emf, get_compressionPower
    
    captureRate = CO2captureRate # [t-CO2/hr]
    
    # get evaporation rate from change in vapor mol fraction
    V1 = G_outlet # [mol/s] bulk gas rate from contactor
    V2 = G_inlet # [mol/s] bulk gas rate to contactor
    evapLoss = vapor_outlet*V1 - vapor_inlet*V2 # [mol-H2O/s]

    # variables for power analysis
    eta_fan = 0.7 # [unitless] fan efficiency (Keith 2018)
    Fp = 10 # Wilcox (2012): 10-60 (random packing), Strigle (1987): 10-28 (structured packing), use smaller packing factor for low flow rate.
    eta_pump = 0.85 # (Wilcox 2012 (Keith 2018 gives .82))

    # reactor parameters
    bed_height = 4.5 # [m] reactor bed height (Keith 2018)
    bed_area = 6000 # [m2] approximate
    fluidizationVelocity = 0.0165 # [m/s] (Keith 2018)
    rho_CaCO3 = 2700 # [kg/m3]

    # calciner
    nu_calciner = 0.78 # fraction thermal efficiency, (Keith 2018)

    # compression
    nu_compressor = 0.85 # (Wilcox 2012)
    T_compressor = 40 + 273.15 # [K] assume isothermic compression at outlet temperature
    p_out = 15.1e6 # [Pa] outlet pressure

    # get molar mass constants # [g/mol]
    MM_OH = get_MM('OH') 
    MM_CO3 = get_MM('CO3')
    MM_H2O = get_MM('H2O')
    MM_K = get_MM('K')
    MM_CO2 = get_MM('CO2')
    MM_Ca = get_MM('Ca')
    MM_CaCO3 = get_MM('CaCO3')
    MM_CaO = get_MM('CaO')
    MM_CH4 = get_MM('CH4')
    MM_O2 = get_MM('O2')

    # --CONTACTOR--
    # define parameters for pressure drop calculations
    Vstar = mdot_G*1000/60**2/A # [kg/m^2s]
    rho_L = get_rhoWater(TL) # [kg/m^3]
    mu_L = get_muWater(TL) # [N*s/m^2] (liquid viscosity)
    rho_V = get_rhoAir(Ta,RH,P) # [kg/m^3] (density moist air )
    #Vendor specified pressure drop for Brentwood XF12560 (Holmes 2012 (Table 1))
    air_velocity = Vstar/rho_V # [m/s]
    deltaP = 7.4*air_velocity**2.14 # [Pa]
    # fan power - use vendor specified
    Pf = get_fanPower(mdot_G, deltaP, rho_V, eta_fan, Z) # [J/s]
    # solvent pumping
    Pp = get_pumpPower(mdot_L, rho_L, h, mu_L, eta_pump) # [J/s]
    contactor_mechanicalPower = (Pf+Pp)/1e6 # [MW-mechanical]
    contactor_waterLoss = evapLoss*MM_H2O*60**2/1e6 # [t-H2O/hr]
    
    # --PELLET REACTOR--
    dy = captureRate*1e6/60**2/MM_CO2 # [mol/s] molar rate of reaction, based on CO2 capture rate
    # PUMP WORK
    Q_s = dy*MM_CaCO3/1e3/rho_CaCO3 # [m3/s]
    Q_L = bed_area*fluidizationVelocity-Q_s # [m3/s]
    e = get_emf(TL) # void fraction within fluidized bed at minimum fluidization
    dp = g*(rho_CaCO3-rho_L)*(1-e)*bed_height # [Pa] pressure drop, Fogler (1997) eq. R12.3-2
    eta_i = (1-0.12*Q_L**(-0.27))*(1-mu_L**0.8) # intrinsic pump efficiency, Wilcox (2012) eq. 3.84
    Pr = dp*Q_L/(eta_i*eta_pump) # [W] wilcox (2012) eq. 3.85
    # CO3 BALANCE
    # CO3 ions from lime slurry are in balance
    # CO3 in CaCO3 fines to disposal are in balance with CaCO3 makeup
    # seed from calciner represents ~2% of mass-flow (conversion efficiency of 98% in calciner)
    # fines to disposal/makup represent ~1% of mass flow
    CaCO3_waste = dy*0.01 # [mol/s] CaCO3 waste fines, 1%
    CaCO3_makeup = CaCO3_waste # [mol/s] CaCO3 makeup 
    CaCO3_seed = dy*0.02 # [mol/s] CaCO3 seed from calciner, 2%
    S_CaCO3 = dy + CaCO3_seed + CaCO3_makeup - CaCO3_waste # [mol/s] CaCO3 solid stream from pellet reactor to calciner
    mdot_S = S_CaCO3*MM_CaCO3/1e6*60**2 # [t/hr] mass flow rate of CaCO3 from reactor to calciner
    pelletReactor_mechanicalPower = Pr/1e6 # [MW-mechanical]
    pelletReactor_calciumWaste = CaCO3_waste*MM_CaCO3/1e6*60**2 #[t-CaCO3/hr]

    # --CALCINER--
    # Calciner: CaCO3 -> CaO + CO2
    # Methane combustion: CH4 + 2O2 -> CO2 + 2H2O
    # ---------
    # (1) find molar rate of combustion reactants to match CaCO3
    # (1.1) get molar rate of calcination reaction
    S_CaO = S_CaCO3 - CaCO3_seed # [mol/s] CaO(s) hot stream to Slaker
    # (1.2) use enthalpy of calcination and combustion to get molar rate of combustion
    dH = 178.3e3 # [J/mol-CO2] reaction enthalpy of calcination (Keith, 2018)
    dHc_CH4 = 890e3 # [J/mol-CH4] enthalpy of combustion of CH4 (NIST)
    min_CH4 = dH/dHc_CH4*S_CaO # [mol-CH4/s]
    G_CH4 = min_CH4/nu_calciner # [mol-CH4/s]
    # (1.3) get t/hr methane
    mdot_CH4 = G_CH4*MM_CH4/1e6*60**2 # [t-CH4/hr]
    # ---------
    # (2) sensible heating of input streams to 900 C
    # (2.1) heating CaCO3 stream
    # <https://www.matweb.com/search/datasheet_print.aspx?matguid=bea4bfa9c8bd462093d50da5eebe78ac>
    # <https://www.engineeringtoolbox.com/standard-state-enthalpy-formation-definition-value-Gibbs-free-energy-entropy-molar-heat-capacity-d_1978.html>
    Cp_CaCO3 = 0.8343 # [J/gK]
    CaCO3_heating = S_CaCO3*MM_CaCO3*(900+273-Ta) # [J/s]
    # (2.2) heating CH4 stream
    dhCH4 = get_dhCH4(Ta,900+273) # [J/mol-CH4]
    CH4_heating = dhCH4*G_CH4 # [J/s]
    # (2.3) heating O2 stream
    dhO2 = get_dhO2(Ta,900+273) # [J/mol-O2]
    G_O2 = G_CH4*2 # [mol-O2/s]
    O2_heating = dhO2*G_O2 # [J/s]
    # (2.4) total sensible heating
    calciner_thermalSensible = (CaCO3_heating + CH4_heating + O2_heating)/1e6 # [MW] sensible heating
    # ---------
    # (3) define output streams
    # (3.1) CaO (s) hot to slaker, from (1.1)
    # (3.2) CO2 (g) hot to compressor
    G_CO2 = S_CaO.copy() + G_CH4 # [mol/s] CO2(g) hot stream to compressor
    # (3.3) H2O (g) hot to water knockout
    G_H2O = G_CH4*2 # [mol/s]
    # credit water against evap
    calciner_waterLoss = -G_H2O*MM_H2O/1e6*60**2 # [t-H2O/hr]
    # assume 50% of heat from H2O product, credit against sensible heat load 
    dhH2O = get_dhH2O(Ta,900+273) # [J/mol-H2O]
    H2O_heating = -dhH2O*G_H2O*n_hi # [J/s]
    calciner_thermalSensible = calciner_thermalSensible + H2O_heating/1e6 #[MW]
    
    # Air Separation Unit (ASU)
    # assume 238 kWh/t-O2, this is more concervative than McQueen ("typically over 200 kWh/tO2")
    mdot_O2 = G_O2*MM_O2/1e6*60**2 # [t-O2/hr]
    ASU_mechanicalPower = 238/1e3*mdot_O2 # [MW]
    
    # --SLAKER--
    eta_slaker = 0.85 # [] conversion efficiency (Keith, 2018)
    h_slaker = 63.9e3 # [J/mol-CaO] # enthalpy of slaking (Keith, 2018) 
    Cp_CaO = 53.08 # [J/molK] (at 900K) <https://webbook.nist.gov/cgi/cbook.cgi?ID=C1305788&Type=JANAFS&Table=on#JANAFS>
    Pt_slaker = h_slaker*eta_slaker*S_CaO # [J/s] thermal power produced
    Pt_CaO = Cp_CaO*(900-300)*S_CaO # [J/s] estimate of sensible heat available from hot CaO stream, cooling from 900-300 C 
    slaker_thermalLatent = -Pt_slaker/1e6*n_hi
    slaker_thermalSensible = -Pt_CaO/1e6*n_hi
    # Schorcht (2013) p.252, 5-30kWh/tonne of quicklime (CaO), assume avearage of 17.5kWh/t-CaO
    mdot_CaO = captureRate*MM_CaO/MM_CO2 # [t-CaO/hr]
    Pm_slaker = mdot_CaO*17.5/1e3 # [MW]
    slaker_mechanicalPower = Pm_slaker # [MW-mechanical]  
    
    # --COMPRESSOR--
    # (Typical pipeline pressure/temperature: 10-15 MPa/35 C)
    # COMPRESSOR DEFINITION
    # reference power usage 132 kWh/t-CO2 (Keith, 2018)
    # reference outlet pressure of 151 Bar = 15.1 MPa, 40 C (Keith, 2018)
    Pc = get_compressionPower(T_compressor,P,p_out,G_CO2) # [J/s] 
    Pc = Pc/nu_compressor/1e6 # [MW] mechanical power to compress gas  
    # cooling CO2 stream
    Cp_CO2 = 50.61 # [J/mol/K] Cp at average temperature of ~750 K, Fundamentals of Heat and Mass Transfer (7th ed., 2011), Table A.4
    Pt = G_CO2*Cp_CO2*(900+273.15-T_compressor)/1e6 # [MW] thermal energy from cooling CO2 stream
    compressor_mechanicalPower = Pc
    compressor_thermal = -Pt*n_hi
    
    # prepare output summary variables
    
    evap = contactor_waterLoss + calciner_waterLoss# t-H2O
    mechPower = (contactor_mechanicalPower +
                 compressor_mechanicalPower +
                 slaker_mechanicalPower +
                 pelletReactor_mechanicalPower+
                 ASU_mechanicalPower) # MW
    thermPower = (slaker_thermalLatent +
                  slaker_thermalSensible +
                  calciner_thermalSensible +
                  compressor_thermal) # MW
    electricalPower = mechPower + thermPower
    
    return evap, electricalPower, mdot_CH4 # [t-H2O/hr, MW, t-CH4/hr]


# function library

def get_emf(TL):
    # Description: void fraction within fluidized bed at minimum fluidization
    # Source: Broadhurst & Becker cf fluidizedBed chapter 12 (p7) (eq. R12.3-7)
    # Output: dimensionless
    # Inputs:
    #     TL [K]: liquid temperature (modeled as water)
    from scipy.constants import g
    psi = 0.6 # sphericity, between 0.5-1, 0.6 typical (same soruce)
    dp = 425.0e-6 # average pellet size 425 micrometres (Keith)
    rho_s = 2700.0 # density of CaCO3 (solid particles)
    rho_L = get_rhoWater(TL)
    mu_L = get_muWater(TL)
    eta = g*(rho_s-rho_L)
    emf = 0.586*psi**(-0.72)*(mu_L**2/(rho_L*eta*dp**3))**0.029*(rho_L/rho_s)**0.021
    return emf

def get_index(t,rh,ppm,sp,tg,TL,Ta,RH,PPM,P):
    # t, rh, ppm, sp, tg: climate data of interest
    # TL, Ta, RH, PPM, P: arrays of environmental variables where capture table is calculated
    from numpy import where, array, isclose
    i = where(abs(TL-tg) == min(abs(TL-tg)))[:]
    j = where(abs(Ta-t) == min(abs(Ta-t)))[:]
    k = where(abs(RH-rh) == min(abs(RH-rh)))[:]
    l = where(abs(PPM-ppm) == min(abs(PPM-ppm)))[:]
    m = where(abs(P-sp) == min(abs(P-sp)))[:]
    index = array([i,j,k,l,m])
    index = index.flatten()

    return index

def get_compressionPower(T,P1,P2,n):
    # isothermal per-stage compression
    # n [mol-CO2/s]
    from scipy.constants import R
    from numpy import log
    V1 = n*R*T/P1
    V2 = n*R*T/P2
    P = -n*R*T*log(V2/V1) #[J/s]
    return P

def get_groundTemp(T):
    # Description: Correlation for undisturbed ground temperature
    # Source: (Ouzzane 2014) New correlations for the prediction of the undisturbed ground temperature
    # Output: [K]
    # Inputs:
    #     T [K]: annual ambient dry bulb temperature
    from numpy import mean, shape
    if len(shape(T))==3:
        Tamb = mean(T,0)
    else:
        Tamb = mean(T) # [K] average annual temperature
    Tg = 17.898 + 0.951*Tamb # [K] undisturbed ground temperature
    return Tg

def get_MM(string):
    # Description: molar mass look-up
    # Source: wikipedia
    # Output: [g/mol]
    # Inputs:
    #     string: 'H20', 'OH', 'CO3'....
    C = 12.011 # [g/mol] carbon
    H = 1.00784 # [g/mol] hydrogen
    K = 39.0983 # [g/mol] potassium
    O = 15.999 # [g/mol] oxygen
    N = 14.0067 # [g/mol] nitrogen
    A = 39.948 # [g/mol] argon
    Ca = 40.078 # [g/mol] calcium
    H2O = 2*H + O # [g/mol] H2O
    OH = H + O # [g/mol] OH-
    CO3 = C + 3*O # [g/mol] CO3
    CO2 = C + 2*O # [g/mol] CO2
    CaOH2 = Ca + 2*OH # [g/mol] Ca(OH)2
    CaCO3 = Ca + CO3 # [g/mol] CaCO3
    CaO = Ca + O # [g/mol] CaO
    CH4 = C + 4*H # [g/mol] CH4
    O2 = 2*O # [g/mol] O2
    dryAir = 28.9635 # [g/mol] (from Tsilingiris 2008)
    MM_library = locals()
    if string in MM_library:
        return MM_library[string] # [g/mol]
    else:
        from sys import exit
        exit("error in function get_MM, no molar mass entry for "+string)

def get_rhoAir(Ta,RH,P):
    # Description: density of moist air
    # Output: [kg/m3]
    # Inputs:
    #     Ta [K]: air temperature
    #     RH [w/w_sat]: humidity ratio/saturated-humidity ratio
    #     P [Pa]: pressure
    from scipy.constants import R
    MM_dry = get_MM('dryAir') # [g-dry/mol-dry]
    MM_vap = get_MM('H2O') # [g-vap/mol-vap]
    w_sat = get_wSat(Ta,P)
    w = RH * w_sat # [kg-vap/kg-dry]
    Mw = w*MM_dry/MM_vap # [mol-vap/mol-dry]
    n = P/(R*Ta) # [mol-total/m3]
    n_dry = n / (1+Mw) # [mol-dry/m3]
    n_vap = w*n_dry # [mol-vap/m3]
    rho = (n_dry*MM_dry + n_vap*MM_vap)/1e3 # [kg/m3]

    return rho

def get_pw(T,P):
    # Description: partial pressure of water vapor at saturation
    # Source: Buck (1981) Eq. 8
    # Output: [Pa]
    # Inputs:
    #     T [K]: water temperature
    #     P [Pa]: air pressure
    from numpy import exp
    T = T-273 # [C]
    P = P/1e2 # [mb]
    Pvap = (1.0007+(3.46e-6*P))*0.61121*exp(17.502*T/(240.97+T)) # [kPa]
    Pvap = Pvap*1e3 # [Pa]
    return Pvap # [Pa]

def get_wSat(Ta,P):
    # Decription: saturated mass humidity ratio
    # Output: [kg-vap/kg-dry air]
    # Inputs:
    #     Ta [K]: air temperature
    #     P [Pa]: total air pressure
    from sys import exit
    Ma = get_MM('dryAir') # [g/mol]
    Mv = get_MM('H2O') # [g/mol]
    Psat = get_pw(Ta,P) # [Pa] saturated vapor pressure
    x = Psat/P # [mol-vap/mol-mix] mol-fraction vapor
    b = (1.0/x-1)**(-1) # [mol-vap/mol-dry air]
    w = b*Mv/Ma # [kg-vap/kg-dry air] saturated humidity ratio
    return w

def get_rhoWater(T):
    # Description: density of liquid water as a function of temperature
    #     Defined over 0<T<100 [C]
    # Source: White (2008) Fluid Mechanics
    # Output: [kg/m3]
    # Inputs:
    #     T [K]: water temperature 
    if (T<273) or (T>373):
        print('Warning: temperature exceeding range of density correlation (get_rhoWater())')
    T = T-273 # [C]
    rho = 1000-0.0178*abs(T-4)**1.7 # [kg/m^3]
    return rho # [kg/m^3]

def get_muWater(T):
    # Description: dynamic viscosity of liquid water as a function of temperature
    # Source: White (2008) Fluid Mechanics
    # Output: [kg/m/s]
    # Inputs:
    #     T [K]: water temperature
    if (T<273) or (T>373):
        print('Warning: temperature exceeding range of viscosity correlation (get_muWater())')
    from numpy import exp
    mu_ref = 1.788e-3 #[kg/m*s]
    z = 273/T
    mu = mu_ref*exp(-1.704-5.306*z+7.003*z**2)
    return mu # [kg/m/s]

def get_fanPower(mdot_G, dP, rho_G, eta, Z):
    # Description: gas blowing work, constant density (deltaP < 156kPa)
    # Source:  Wilcox (2012) eq. 3.81
    # Output: [W]
    # Inputs:
        # mdot_G: [t/h] mass flow rate
        # dP: [Pa/m] pressure drop per metre of packing
        # rho_G: [kg/m^3] gas density
        # eta: [unitless] fan efficiency
        # Z: [m] total depth of packing
    
    #convert to si untis
    mdot = mdot_G*1e3/60**2 # [kg/s]
    dP = dP*Z # [Pa] total pressure drop 
    P = (mdot*dP)/(rho_G*eta)
    return P # [w] total fan power
    
def get_pumpPower(mdot_L, rho_L, h, mu, eta):
    # Description: liquid pumping work
    # Source: Wilcox (2012) eq. 3.85
    # Output: [W]
    # Inputs:
        # mdot_L: [t/h] liquid mass flow rate
        # rho_L: [kg/m^3] liquid density
        # h: [m] elevation head
        # mu: [Pa*s] liquid dynamic visocity
    
    from scipy.constants import g
    Q = mdot_L*1e3/60**2 / rho_L # [m^3/s] volumetric flow rate
    dP = rho_L*g*h # [Pa] hydrostatic pressure distribution
    eta_i = (1-0.12*Q**(-0.27))*(1-mu**0.8) # [unitless] intrinsic efficiency "for moderate pressure and mu<0.5 Pa*s" (Eq. 3.84)
    P = (Q*dP)/(eta_i*eta)
    return P # [W]

def get_balanceOfPlant(mdot_L,mdot_G,M_OH,M_CO3,z,A,h,Z,TL,Ta,P,RH,CO2captureRate,G_inlet,G_outlet,vapor_inlet,vapor_outlet,n_hi):
    # Updated 2024-05-10: remove resource object return multiple varaibles instead
    # Description: after simulation of contactor, get power consumption (mechanical and thermal), evaporative losses, and CaCO3 losses from primary process equipment.
    # Source: cycle based on Keith (2018)
    # Output:
        # m [MW]: mechanical power
        # t [MW]: thermal power
        # w [t/hr]: evaporative losses
        # c [t/hr]: CaCO3 makeup
    # Inputs:
        # mdot_L [t/h]: solvent mass flow rate (contactor)
        # mdot_G [t/h]: gas mass flow rate (contactor)
        # M_OH [mol/L solution]: inlet molar concentration of OH- in solution
        # M_CO3 [mol/L solution]: inlet molar concentration of CO3-2 in solution
        # z: stoichiometric coefficient of binding species (CO2 + 2KOH = K2CO3 + H2O)
        # A [m^2]: cross-sectional area of contactor
        # h [m]: height to pump solvent
        # Z [m]: depth of packing
        # TL [K]: liquid temperature
        # Ta [K]: air temperature
        # P [Pa]: ambient pressure
        # RH: relative humidity (fraction of saturated humidity level)
        # CO2captureRate [t/hr]: capture rate of CO2
        # G_inlet [mol/s]: bulk molar gas flow rate at contactor inlet
        # G_outlet [mol/s]: bulk molar gas flow rate at contactor outlet
        # vapor_inlet [mol-vap/mol-gas]: mol fraction of water vapor at contactor inlet
        # vapor outlet [mol-vap/mol-gas]: mol fraction of water vapor at contactor outlet
        # n_hi: fraction recovered sensible heat (heat integration)
        
    import numpy as np
    from scipy.constants import g
    
    captureRate = CO2captureRate # [t-CO2/hr]
    
    # get evaporation rate from change in vapor mol fraction
    V1 = G_outlet # [mol/s] bulk gas rate from contactor
    V2 = G_inlet # [mol/s] bulk gas rate to contactor
    evapLoss = vapor_outlet*V1 - vapor_inlet*V2 # [mol-H2O/s]

    # variables for power analysis
    eta_fan = 0.7 # [unitless] fan efficiency (Keith 2018)
    Fp = 10 # Wilcox (2012): 10-60 (random packing), Strigle (1987): 10-28 (structured packing), use smaller packing factor for low flow rate.
    eta_pump = 0.85 # (Wilcox 2012 (Keith 2018 gives .82))

    # reactor parameters
    bed_height = 4.5 # [m] reactor bed height (Keith 2018)
    bed_area = 6000 # [m2] approximate
    fluidizationVelocity = 0.0165 # [m/s] (Keith 2018)
    rho_CaCO3 = 2700 # [kg/m3]

    # calciner
    nu_calciner = 0.78 # fraction thermal efficiency, (Keith 2018)

    # compression
    nu_compressor = 0.85 # (Wilcox 2012)
    T_compressor = 40 + 273.15 # [K] assume isothermic compression at outlet temperature
    p_out = 15.1e6 # [Pa] outlet pressure

    # get molar mass constants # [g/mol]
    MM_OH = get_MM('OH') 
    MM_CO3 = get_MM('CO3')
    MM_H2O = get_MM('H2O')
    MM_K = get_MM('K')
    MM_CO2 = get_MM('CO2')
    MM_Ca = get_MM('Ca')
    MM_CaCO3 = get_MM('CaCO3')
    MM_CaO = get_MM('CaO')

    # --CONTACTOR--
    # define parameters for pressure drop calculations
    Vstar = mdot_G*1000/60**2/A # [kg/m^2s]
    rho_L = get_rhoWater(TL) # [kg/m^3]
    mu_L = get_muWater(TL) # [N*s/m^2] (liquid viscosity)
    rho_V = get_rhoAir(Ta,RH,P) # [kg/m^3] (density moist air )
    #Vendor specified pressure drop for Brentwood XF12560 (Holmes 2012 (Table 1))
    air_velocity = Vstar/rho_V # [m/s]
    deltaP = 7.4*air_velocity**2.14 # [Pa]
    # fan power - use vendor specified
    Pf = get_fanPower(mdot_G, deltaP, rho_V, eta_fan, Z) # [J/s]
    # solvent pumping
    Pp = get_pumpPower(mdot_L, rho_L, h, mu_L, eta_pump) # [J/s]
    contactor_m = (Pf+Pp)/1e6 # [MW-mechanical]
    contactor_w = evapLoss*MM_H2O*60**2/1e6 # [t-H2O/hr]

    # --PELLET REACTOR--
    dy = captureRate*1e6/60**2/MM_CO2 # [mol/s] molar rate of reaction, based on CO2 capture rate
    # PUMP WORK
    Q_s = dy*MM_CaCO3/1e3/rho_CaCO3 # [m3/s]
    Q_L = bed_area*fluidizationVelocity-Q_s # [m3/s]
    e = get_emf(TL) # void fraction within fluidized bed at minimum fluidization
    dp = g*(rho_CaCO3-rho_L)*(1-e)*bed_height # [Pa] pressure drop, Fogler (1997) eq. R12.3-2
    eta_i = (1-0.12*Q_L**(-0.27))*(1-mu_L**0.8) # intrinsic pump efficiency, Wilcox (2012) eq. 3.84
    Pr = dp*Q_L/(eta_i*eta_pump) # [W] wilcox (2012) eq. 3.85
    # CO3 BALANCE
    # CO3 ions from lime slurry are in balance
    # CO3 in CaCO3 fines to disposal are in balance with CaCO3 makeup
    # seed from calciner represents ~2% of mass-flow (conversion efficiency of 98% in calciner)
    # fines to disposal/makup represent ~1% of mass flow
    CaCO3_waste = dy*0.01 # [mol/s] CaCO3 waste fines, 1%
    CaCO3_makeup = CaCO3_waste # [mol/s] CaCO3 makeup 
    CaCO3_seed = dy*0.02 # [mol/s] CaCO3 seed from calciner, 2%
    S = dy + CaCO3_seed + CaCO3_makeup - CaCO3_waste # [mol/s] CaCO3 solid stream from pellet reactor to calciner
    mdot_S = S*MM_CaCO3/1e6*60**2 # [t/hr] mass flow rate of CaCO3 from reactor to calciner
    reactor_m = Pr/1e6 # [MW-mechanical]
    reactor_c = CaCO3_waste*MM_CaCO3/1e6*60**2 #[t-CaCO3/hr]

    # --CALCINER--
    # Calciner: CaCO3 -> CaO + CO2
    dH = 178.3e3 # [J/mol] reaction enthalpy of calcination (Keith, 2018)
    min_power = S*dH # [J/s] minimum thermodynamic power
    Pt = min_power/nu_calciner
    # heating CaCO3 stream
    # Specific heat capacity: 0.8343 J/g-C
    # <https://www.matweb.com/search/datasheet_print.aspx?matguid=bea4bfa9c8bd462093d50da5eebe78ac>
    # <https://www.engineeringtoolbox.com/standard-state-enthalpy-formation-definition-value-Gibbs-free-energy-entropy-molar-heat-capacity-d_1978.html>
    Cp_CaCO3 = 0.8343 # [J/gK]
    CaCO3_heating = S*MM_CaCO3*(900+273-Ta)*Cp_CaCO3 # [J/s]
    # calciner conversion efficiency
    S_CaO = S-CaCO3_seed # [mol/s] CaO(s) hot stream to Slaker
    G_CO2 = S_CaO.copy() # [mol/s] CO2(g) hot stream to compressor
    calciner_t = Pt/1e6 + CaCO3_heating/1e6 # [MW-thermal] latent + sensible

    # --SLAKER--
    eta_slaker = 0.85 # [] conversion efficiency (Keith, 2018)
    h_slaker = 63.9e3 # [J/mol-CaO] # enthalpy of slaking (Keith, 2018) 
    Cp_CaO = 53.08 # [J/molK] (at 900K) <https://webbook.nist.gov/cgi/cbook.cgi?ID=C1305788&Type=JANAFS&Table=on#JANAFS>
    Pt_slaker = h_slaker*eta_slaker*S_CaO # [J/s] thermal power produced
    Pt_CaO = Cp_CaO*(900-300)*S_CaO # [J/s] estimate of sensible heat available from hot CaO stream, cooling from 900-300 C 
    slaker_t = -Pt_slaker/1e6*n_hi - Pt_CaO/1e6*n_hi
    # Schorcht (2013) p.252, 5-30kWh/tonne of quicklime (CaO), assume avearage of 17.5kWh/t-CaO
    mdot_CaO = captureRate*MM_CaO/MM_CO2 # [t-CaO/hr]
    Pm_slaker = mdot_CaO*17.5/1e3 # [MW]
    slaker_m = Pm_slaker # [MW-mechanical]
    
    # --COMPRESSOR--
    # (Typical pipeline pressure/temperature: 10-15 MPa/35 C)
    # COMPRESSOR DEFINITION
    # reference power usage 132 kWh/t-CO2 (Keith, 2018)
    # reference outlet pressure of 151 Bar = 15.1 MPa, 40 C (Keith, 2018)
    Pc = get_compressionPower(T_compressor,P,p_out,G_CO2) # [J/s] 
    Pc = Pc/nu_compressor/1e6 # [MW] mechanical power to compress gas  
    # cooling CO2 stream
    Cp_CO2 = 50.61 # [J/mol/K] Cp at average temperature of ~750 K, Fundamentals of Heat and Mass Transfer (7th ed., 2011), Table A.4
    Pt = G_CO2*Cp_CO2*(900+273.15-T_compressor)/1e6 # [MW] thermal energy from cooling CO2 stream
    compressor_m = Pc
    compressor_t = -Pt*n_hi
    
    # declare output variables
    evap = contactor_w # t-H2O/hr
    CaCO3_makeup = reactor_c # t-CaCO3/hr
    mechPower = contactor_m + reactor_m + slaker_m + compressor_m # MW
    thermPower = calciner_t + slaker_t + compressor_t  # MW
    return evap, CaCO3_makeup, mechPower, thermPower

def get_RH(dewpoint,temperature,pressure):
    # Description: calculate relative humidity from absolute meteorological measurements
    # Source: 
    # Output: ratio of humidity to saturated humidity
    # Inputs:
        # dewpoint [K]: the temperature at which the humidity is the saturated humidity
        # temperature [K]: air temperature
        # pressure [Pa]: air pressure
    w = get_wSat(dewpoint,pressure) # [kg-vap/kg-dry air] saturated mass humidity ratio at the dewpoint temp.
    wsat = get_wSat(temperature,pressure) # [kg-vap/kg-dry air] saturated mass humidity ratio at the air temp.
    RH = w/wsat
    return RH

def get_sigmaWater(T):
    # Description: Surface tension of water
    # Source: Kalova (2018) Eq. 20
    # Output: [N/m]
    # Inputs:
    #     T [K]
    Tcritical = 370 + 273.15
    tau = 1 - T/Tcritical
    sigma = 241.322*tau**1.26*(1 - 0.0589*tau**0.5 - 0.56917*tau)  # [mN/m]
    sigma = sigma*1e-3
    return sigma # [N/m]

def get_diffusivity(phi,M,T,V_A):
    # Description: Diffusion of dilute gas in solvent
    # Source: Wilke and Chang, from Wilcox (2012) Eq. 3.11
    # Output: [cm^2/s]
    # Inputs:
    #     phi: solvent association parameter
    #     M [g/mol]: solvent molecular weight
    #     T [K]: temperature
    #     V_A [cm^3/mol]: molar volume of solute at normal boiling point (34 is approximate for CO2)
    mu = get_muWater(T) # [kg/ms] solvent viscosity
    mu = mu*1e3 # [cP] conversion to cP
    D = 7.4e-8*((phi*M)**0.5*T/(mu*V_A**0.6)) # [cm2/s]
    return D # [cm^2/s]

def get_k2(T,I):
    # Description: reaction rate constant of CO2 with OH
    # Source: Pinsent (1956) eq. 5, Kucka (2002) eq. 26
    # Output: [L/mol/s]
    # Inputs:
        # T [K]: liquid temperature
        # I [mol-OH/L]: concentration of OH ions
    
    # use pinsent for temperature below 40 C
    if (T<313):
        k2 = 10**(13.635-(2895/T))
    # else Kucka for 20-50 C
    else:
        from numpy import exp
        from scipy.constants import R
        k2inf = 3.27869e13*exp(-54971/(R*T))
        B = 3.3968e-4*T**2-2.125e-1*T+33.506
        k2 = k2inf*exp(B*I)
    
    return k2 # [L/mol/s]

def get_Ho(T):
    # Description: solubility of CO2 in pure water (Van't Hoff extrapolation)
    # Source: Smith (2007) eq. 6
    # Output: [atmL/mol]
    # Inputs:
    #     T [K]
    from numpy import exp
    from scipy.constants import R
    H0 = 3.4e-2 # [mol/L*atm] constant at reference temperature
    T0 = 298.15 # [K] reference temperature
    c = 19.62*1000/R # molar gas constant (from Carroll, 1991)
    H = H0*exp(c*(1/T-1/T0)) # [mol/Latm]
    H = 1/H # return as [atmL/mol]
    return H # [atmL/mol]

def get_hG(T):
    # Description: solubility of CO2 ion species in solution
    #    (for use in Wilcox (2012) Eq. 3.5)
    # Source: Wilcox (2012) Table 3.1
    # Output: [L/mol]
    # Inputs:
    #     T [K]
    from sys import exit
    T = T - 273 # [C]
    TD = [0,15,25,40,50] #temperature Data points [degree C]
    hD = [-0.007, -0.010, -0.019, -0.026, -0.029] #hG Data points [L/mol]
    if T<TD[0]: exit('error, T < %0.0f'%TD[0])
    elif T<=TD[1]:
        hG = (T-TD[0])/(TD[1]-TD[0]) * (hD[1]-hD[0]) + hD[0]
    elif T<=TD[2]:
        hG = (T-TD[1])/(TD[2]-TD[1]) * (hD[2]-hD[1]) + hD[1]
    elif T<=TD[3]:
        hG = (T-TD[2])/(TD[3]-TD[2]) * (hD[3]-hD[2]) + hD[2]
    elif T<=(TD[4]):
        hG = (T-TD[3])/(TD[4]-TD[3]) * (hD[4]-hD[3]) + hD[3]
    else:
        # if T>53: print('warning in get_hG, T > %0.0f C, T = %0.2f C, using hG(T=50)=-0.029 L/mol)'%(TD[4],T))
        hG = hD[4]
    return hG # [L/mol]

def HinSolution(Ho, hp1, hm1, hp2, hm2, hG, cp1, cm1, cp2, cm2, zp1, zm1, zp2, zm2):
    # Description: Henry's law constant, modified for ion concentration of solution
    #     - assuming solution of 2 species + CO2 (KOH + K2CO3)
    # Source: Wilcox (2012) eq. 3.3-3.5
    # Output: [atmL/mol]
    # Inputs:
    #     Ho [atmL/mol]: solubility of CO2 in pure water
    #     hp1,...,hm2 [L/mol]: salt-effect parameters for the positive (p) and negative (m) ions of species 1 and 2
    #     hg [L/mol]:  salt-effect parameter for dissolved CO2
    #     cp1,...,cm2: [mol/L]: concentrations the positive (p) and negative (m) ions of species 1 and 2
    #     zp1,...,zm2: charge of the positive (p) and negative (m) ions of species 1 and 2
    from math import log10
    h1 = hp1+hm1+hG
    h2 = hp2+hm2+hG
    I1 = 0.5*(cp1*zp1**2+cm1*zm1**2)
    I2 = 0.5*(cp2*zp2**2+cm2*zm2**2)
    H = 10**(h1*I1 + h2*I2 + log10(Ho))
    return H # [atm*L/mol]

def interfaceConcentration(H,P):
    # Description: Dissolved CO2 at interface (by Henry's Law)
    # Source: Wilcox (2012) Eq. 3.2
    # Output: [mol/L]
    # Inputs:
        # H [atmL/mol]: solubility of CO2
        # P [atm]: partial pressure of CO2
    Ci = (1/H)*P 
    return Ci # [mol/L]

def get_aw(sigma_c,sigma,rho,Lsv,a,mu):
    # Description: correlation for wetted surface area
    # Source: Onda (1968)
    # Output: [m2/m3]
    # Inputs:
    #     sigma_c [n/m]: critical surface tension of packing material
    #     sigma [n/m]: liquid surface tension
    #     rho: [kg/m3] liquid density
    #     a: [m2/m3] packing specific area
    #     Lsv: [m/s] liquid superficial velocity

    from numpy import exp
    from scipy.constants import g
    
    L = Lsv*rho # [kg/m2s] water mass flux
    Re = L/(a*mu) # Reynolds number
    Fr = L**2*a/(rho**2*g) # Froude number
    We = L**2/(rho*sigma*a) # Weber number
    
    #Note: Wilcox (2014) gives bounds on wetted area correlation
    if (Re<0.04) or (Re>500):
        0
        print('Warning, exceeding wetted area correlation range:')
        print('0.04<Re<500')
        print('Re = %0.3e'%Re)
    if (We<1.2e-8) or (We>0.2):
        print('Warning, exceeding wetted area correlation range:')
        print('1.2e-8<We<0.27')
        print('We = %0.3e'%We)
    if (Fr<2.5e-9) or (Fr>1.8e-2):
        print('Warning, exceeding wetted area correlation range:')
        print('2.5e-9<Fr<1.8e-2')
        print('Fr = %0.3e'%Fr)
    if ((sigma_c/sigma)<0.3) or ((sigma_c/sigma)>2.0):
        print('Warning, exceeding wetted area correlation range:')
        print('0.3< sigma_c/sigma <2.0')
        print('sigma_c/sigma = %0.3e'%(sigma_c/sigma))

    aw = a*(1-exp(-1.45*(sigma_c/sigma)**0.75 * Re**0.1 * Fr**(-0.05) * We**0.2)) # [m2/m3]
    
    return aw # [m2/m3]

def get_k_L(epsilon,a,T,Lsv,sigma_c):
    # Description: liquid-phase mass-tranfer coefficient for CO2 
    # Source: Onda (1968) 'Gas absorption with chemical reaction in packed columns'
    # Output: [cm/s]
    # Input:
    #     epsilon: void fraction of packing
    #     a [m2/m3]: packing specific area
    #     T [K]: liquid temperature
    #     Lsv [m/s]: liquid superficial velocity
    #     sigma_c [n/m]: critical surface tension of packing
    from scipy.constants import g
    sigma = get_sigmaWater(T)# [N/m] liquid surface tension
    MM_H2O = get_MM('H2O') # [g/mol] molar weight of H2O 
    rho = get_rhoWater(T) # [kg/m^3]
    mu = get_muWater(T) # [kg/ms]
    aw = get_aw(sigma_c, sigma, rho, Lsv, a, mu) # [m^2/m^3] wetted surface area of packing
    dp = get_dp(a,epsilon) # [m] packing diameter
    D = get_diffusivity(2.26,MM_H2O,T,34) # [cm^2/s] CO2 diffusivity in liquid phase
    D = D*1e-4 # [m^2/s]
    kL = 0.0051*(a*dp)**0.4*(mu*g/rho)**(1/3)*(rho*Lsv/aw/mu)**(2/3)*(mu/rho/D)**(-1/2) # [m/s]
    kL = kL*100 # [cm/s]
    
    return kL # [cm/s]

def get_DB(C,T,d2):
    # Description: diffusivity of binder species (KOH)
    # Source: Bosma (1999) and See (1998)
    # Output: [cm2/s]
    # Inputs:
    #     C [mol/L]: Binder (KOH) concentration
    #     T [K]: liquid temperature
    #     d2 [m]: solute diameter, assumed spherical (OH-)
    
    if T>298:
        # Stokes-Einstein method from Bosma (1999), Eq. 4
        from scipy.constants import k, pi
        # k: [J/K] Bolzmann's constant 
        mu = get_muWater(T) # [kg/ms]
        DB = k*T/(3*pi*mu*d2) # [m2/s]
        DB = DB*1e4 # [cm2/s]
    elif T<=298:
        # Empirical correlation from See (1998)
        K1 = -7.56e-4
        K2 = 4.94e-6
        K3 = -7.77e-9
        K4 = 1.10e-5
        K5 = 4.93e-6
        K6 = -1.18e-6
        K7 = -1.07
        DB = K1 + K2*T + K3*T**2 + K4*C**0.5 + K5*C + K6*C**(3/2) + K7*C**0.5/T**2
    return DB # [cm^2/s]

def enhancementFactor(DL,KL,K2,CB,Ci,DB,z):
    # Description: Enhancement factor for absorption with reaction
    # Source: Wilcox (2012) eq. 3.41-3.46 (Film model)
    # Output: dimensionless
    # Inputs:
    #     DL [cm2/s]: Diffusion of CO2
    #     KL [cm/s]: liquid side mass transfer coefficient (CO2)
    #     K2 [L/mols]: Reaction rate constant (CO2 + OH-)
    #     CB [mol/L]: Concentration of binding species (KOH)
    #     Ci [mol/L]: Concentration of CO2 at gas-liquid interface
    #     DB [cm2/s]: Diffusion of binder species
    #     z [unitless]: molar ratio of binding species to CO2, z=2 for: CO2 + 2(KOH)->
    
    from math import sqrt, tanh
    #if diffusion of CO2 (DL) and binding species (DB) differ significantly,
    # then Ei by Higbie model (Wilcox eq. 3.47)
    if abs(1-(DB/DL))>0.2:
        Ei = sqrt(DL/DB)+CB/(z*Ci)*sqrt(DB/DL)
    else:
        Ei = 1 + (DB*CB)/(z*DL*Ci)
    
    rootM = sqrt(DL*K2*CB)/KL # unitless

    # Table 3.6 from Wilcox
    if rootM > (10*Ei):
        # instantaneous reaction
        E=Ei
    elif (rootM < (0.5*Ei)) & (rootM <= 3):
        # Pseudo-first-order reaction
        E = (rootM)/(tanh(rootM))
    elif (rootM < (0.5*Ei)) & (rootM > 3):
        # Fast pseudo-first-order reaction
        E = sqrt(DL*K2*CB)/KL
    else:
        # E by Eq.3.42
        E = 0.5*Ei
        beta = sqrt((Ei-E)/(Ei-1))
        E2 = rootM*beta/tanh(rootM*beta)
        while abs(E-E2) > 1:
            if E>E2:
                E=E-1
            if E<E2:
                E=E+0.1
            beta = sqrt((Ei-E)/(Ei-1))
            E2 = rootM*beta/tanh(rootM*beta)      
    return E

def get_dp(a,e):
    # Description: effective packing diamter
    # Source: Wilcox (2012) 
    # Output: [m]
    # Inputs:
    #     a [m2/m3]: packing specific area
    #     e: void fraction packing
    dp = 6*(1-e)/a # [m] packing diameter
    return dp

def get_CpH2O(TL):
    # Description: specific heat capacity of water (isobaric), polynomial fit over 0-100C
    # Source: https://www.engineeringtoolbox.com/specific-heat-capacity-water-d_660.html
    # Output: [J-kg/K]
    # Inputs:
    #     TL [K]: liquid temperature
    p = ([ -6.17018102e-08,   1.02757547e-04,  -6.83256242e-02,
         2.26815554e+01,  -3.76020067e+03,   2.53293002e+05]) # coefficient vector
    n = 5 # degree of polynomial
    CpH2O = (p[0]*TL**n + p[1]*TL**(n-1) + p[2]*TL**(n-2) + p[3]*TL**(n-3)
          + p[4]*TL**(n-4) + p[5]*TL**(n-5))
    return CpH2O # [J/kg-K]

def get_dwdV(w,hdAv,t_w,t,mdot_dry,P):
    # Description: change in humidity due to evaporation (per unit volume) 
    # Source: Sutherland (1983) eq. 4
    # Output: [kg-vap/kg-dry/m^3]
    # Inputs:
        # w [kg-vap/kg-dry]: humidity ratio
        # hdAv [kg/m^3/s]: mass transfer coefficient X specific area 
        # t_w [K]: water temperature
        # mdot_dry [kg-dry]: mass flow rate dry air
        # P [Pa]: air pressure
    w_sw = get_wSat(t_w,P) # [kg-vap/kg-dry-air]
    dwdV = hdAv*(w_sw-w)/mdot_dry
    return dwdV # [kg-vap/kg-dry/m^3]

def get_dhdV(dwdV,t_w,t,mdot_dry,Le,hdAv,w):
    # Description: enthalpy change of water (per unit volume)
    # Source: Sutherland (1983) eq. 6
    # Output: [J/kg-dry/m3] positive is loss of enthalpy
    # Inputs:
        # dwdV [kg-vap/kg-dry/m3]: evaporation per unit volume
        # t_w [K]: water temperature
        # t [K]: air temperature (dry bulb)
        # mdot_dry [kg-dry]: mass flow rate dry air
        # Le: Lewis number
        # hdAv [kg/m3s]: mass transfer coefficient X specific area
        # hcAv [W/m3K]: heat transfer coefficient X specific area
    #get heat of vaporization
    # Dortmund Data Bank PPDS12 Equation
    from scipy.constants import R
    MM_H2O = get_MM('H2O') # [g/mol]
    A = 5.6297
    B = 13.962
    C = -11.673
    D = 2.1784
    E = -0.3166
    Tc = 647.3
    tau = 1 - t_w/Tc
    Hvap = R*Tc*(A*tau**(1/3)+B*tau**(2/3)+C*tau+D*tau**(2)+E*tau**(6)) # [J/mol]
    Hvap = Hvap/MM_H2O*1e3 # [J/kg]

    Cma = get_Cpa(t) + get_Cpv(t)*w # [J/kg-dryK] specific heat of moist air
    hcAv = Le*hdAv*Cma # [W/m3K] convective heat transfer x specific area
    dhdV = hcAv*(t_w-t)/mdot_dry + dwdV*Hvap # [J/kg-dry/m3]
    return dhdV
  
def get_dt_w(dhdV,dV,mdot_dry,mdot_w,t_w):
    # Description: change in water temperature
    # Source:
    # Output: [K] 
    # Inputs:
        # dhdV [J/kg-dry]: water enthalpy change
        # dV [m^3]: unit volume
        # mdot_w [kg/s]: liquid mass flow rate 
        # mdot_dry [kg-dry/s]: mass flow rate dry air
        # t_w [K]: water temperature
    c_w = get_CpH2O(t_w) # [J/kg-water*K]
    dh = -dhdV*dV # [J/kg-dry] dhdV positive for energy flow from water to air
    de = dh*mdot_dry # [J/s]
    dt_w = de/mdot_w/c_w # [K]
    return dt_w

def get_dt_a(dV,t_w,t,w,dwdV,hdAv,Le,mdot_dry):
    # Description: change in air temperature
    # Source: Sutherland (1983) 
    # Output: [K]
    # Inputs:
        # dV [m^3]: unit volume
        # t_w [K]: water temperature
        # t [K]: air temperature 
        # w [kg-vap/kg-dry-air]: mass humidity ratio
        # dwdV [kg-vap/kg-dry/m3]: evaporation per unit volume
        # hdAv [kg/m3/s]: mass transfer coefficient * specific area
        # Le: Lewis Number
        # mdot_dry [kg-dry/s]: mass flow rate dry air
    Cma = get_Cpa(t) + get_Cpv(t)*(w+dwdV*dV) # [J/kg-dryK] specific heat of moist air
    hcAv = Le*hdAv*Cma # [W/m3K] convective heat transfer x specific area
    dhdV = hcAv*(t_w-t)/mdot_dry # [J/kg-dry] temperature raised by convective heat transfer
    dt = dhdV*dV/Cma # [K]
    return dt

def get_hdAv(mdot_w,mdot_dry,Vol,c,n):
    # Description: evaporation mass transfer coefficient X surface area to volume ratio
    # Source: Kuehn (1998) eq. 10.51
    # Output: [kg/m3/s]
    # Inputs:
        # mdot_w [kg/s]: liquid mass flow rate 
        # mdot_dry [kg-dry/s]: mass flow rate dry air
        # Vol [m^3]: total contactor volume
        # c: empirical constant, in absence of data c=1.3 recommended 
        # n: empirical constant, in absence of data n=0.6 recommended
    hdAv = (mdot_w/Vol)*c*(mdot_w/mdot_dry)**-n # [kg/m3/s]
    return hdAv

def get_dV(Z,A,n):
    # Description: volume of differential element
    # Source:  
    # Output: [m^3]
    # Inputs:
        # Z [m]: packing depth
        # A [m^2]: cross-sectional area
        # n: number of discreet column elements
    dV = A*Z/n
    return dV # [m^3]

def get_Cpa(T):
    # Description: specific heat capacity of dry air
    # Source: Tsilingiris (2008)  
    # Output: [J/kg-K]
    # Inputs:
        # T [K]: air temperature
    CA0 = 0.103409e1
    CA1 = -0.284887e-3
    CA2 = 0.7816818e-6
    CA3 = -0.4970786e-9
    CA4 = 0.1077024e-12
    Cpa = CA0 + CA1*T + CA2*T**2 + CA3*T**3 + CA4*T**4
    Cpa = Cpa*1e3 # [J/kg-K]
    return Cpa 

def get_Cpv(T):
    # Description: specific heat capacity of Water vapor
    # Source: Tsilingiris (2008)  
    # Output: [J/kg-K]
    # Inputs:
        # T [K]: air temperature
    t = T-273.15 # [C]
    CV0 = 1.86910989
    CV1 = -2.578421578e-4
    CV2 = 1.941058941e-5
    Cpv = CV0 + CV1*t + CV2*t**2
    Cpv = Cpv*1e3 # [J/kg-K]
    return Cpv

def liquidSideUpdate(CO2_flux,H2O_flux,L,x,b,k,w,L1,x1,b1,k1,w1,z,n):
    # Description: update solvent concentrations 
    # Source:  
    # Output:
        # CO2_flux [mol/s]: CO2 molar transfer rate gas-to-liquid
        # L [mol/s]: bulk liquid flow rate 
        # x : mol fraction CO2, liquid phase
        # b : mol fraciton OH-
        # w : mol fraction H2O
        # k : mol fraction K+
    # Inputs:
        # CO2_flux [mol/s]: CO2 molar transfer rate, gas-to-liquid
        # H2O_flux [mol/s]: water vapor molar transfer rate liquid-to-gas
        # L, L1 [mol/s]: bulk liquid flow rate, bulk liquid flow rate at inlet 
        # x : liquid phase mol fraction CO2, " at inlet
        # b, b1 : mol fraciton OH-, " at inlet
        # w, w1 : mol fraction H2O, " at inlet
        # k, k1 : mol fraction K+ " at inlet
        # z [unitless]: molar ratio of binding species to CO2 (stoichiometric constant)
        # n: number of discreet column elements
    
    # (1)CO2 + (2)KOH -> (1)K2CO3 + (1)H2O
    from sys import exit
    
    # for inlet element -1
    CO2_flux[-1] = checkBinder(CO2_flux[-1],L1,b1,z) # limit diffusion to availability of binding species
    mdotx = L1*x1 + CO2_flux[-1] # [mol/s] 
    mdotb = L1*b1 - 2*CO2_flux[-1]
    mdotw = L1*w1 - H2O_flux[-1] + CO2_flux[-1] # add 1 mol H2O for each mol K2CO3
    mdotk = L1*k1
    L[-1] = mdotx + mdotb + mdotw + mdotk
    x[-1] = mdotx/L[-1]  # mol-fraction
    b[-1] = mdotb/L[-1]
    w[-1] = mdotw/L[-1]
    k[-1] = mdotk/L[-1] 
    
    # for remaining elements
    for j in reversed(range(n-1)):
        CO2_flux[j] = checkBinder(CO2_flux[j],L[j+1],b[j+1],z) # limit diffusion to availability of binding species
        mdotx = L[j+1]*x[j+1] + CO2_flux[j] # [mol/s] 
        mdotb = L[j+1]*b[j+1] - 2*CO2_flux[j]
        mdotw = L[j+1]*w[j+1] - H2O_flux[j] + CO2_flux[j]
        mdotk = L[j+1]*k[j+1]
        L[j] = mdotx + mdotb + mdotw + mdotk
        x[j] = mdotx/L[j] # mol-fraction
        b[j] = mdotb/L[j]
        w[j] = mdotw/L[j]
        k[j] = mdotk/L[j]
    
    return CO2_flux,L,x,b,w,k

def checkBinder(CO2_flux,L,b,z):
    # Description: if binding species in solvent is exhausted, limit CO2 transfer from gas phase
    # Source:  
    # Output: [mol/s]
    # Inputs:
        # CO2_flux [mol/s]: mass transfer gas to liquid phase
        # L [mol/s]: bulk molar liquid flow rate
        # b: mol fraction limiting species (OH)
        # z: stochiometric constant of binding species (per 1 CO2)  
    max_flux = (L*b/z)
    if (CO2_flux >= max_flux):
        CO2_flux = max_flux/1.1 # avoid 0s and negatives which code doesn't like
    return CO2_flux

def gasSideUpdate(CO2_flux,H2O_flux,V,y,vap,air,V2,y2,vap2,air2,n):
    # Description: update gas concentrations
    # Source:  
    # Output:
        # CO2_flux [mol/s]: mass transfer gas to liquid phase
        # V [mol/s]: gas phase molar flow rate
        # y: mol fraction CO2 gas phase
        # vap: mol fraction H2O gas phase
        # air: mol fraciton air (less CO2)
    # Inputs:
        # CO2_flux [mol/s]: mass transfer gas to liquid phase
        # H2O_flux [mol-H2O/s]: liquid-gas mass transfer rate
        # V, V2 [mol/s]: gas phase molar flow rate, " at inlet
        # y, y2: mol fraction CO2 gas phase, " at inlet
        # vap, vap2: mol fraction H2O gas phase, " at inlet
        # air, air2: mol fraction air (less CO2), " at inlet
        # n: number of discreet column elements
    from sys import exit
    # for inlet element 0
    CO2_flux[0] = checkCO2(CO2_flux[0],V2,y2) # limit diffusion to availability of gas-phase CO2
    mdoty = V2*y2 - CO2_flux[0] # [mol/s] 
    mdotvap = V2*vap2 + H2O_flux[0]
    mdotair = V2*air2
    V[0] = mdoty + mdotvap + mdotair
    y[0] = mdoty/V[0] # mol-fraction
    vap[0] = mdotvap/V[0]
    air[0] = mdotair/V[0]   
    # for remaining elements
    for j in range(1,n):
        CO2_flux[j] = checkCO2(CO2_flux[j],V[j-1],y[j-1]) # limit diffusion to availability of gas-phase CO2
        mdoty = V[j-1]*y[j-1] - CO2_flux[j] # [mol/s] 
        mdotvap = V[j-1]*vap[j-1] + H2O_flux[j]
        mdotair = V[j-1]*air[j-1]
        V[j] = mdoty + mdotvap + mdotair
        y[j] = mdoty/V[j] # mol-fraction
        vap[j] = mdotvap/V[j]
        air[j] = mdotair/V[j]
    return CO2_flux,V,y,vap,air

def checkCO2(CO2_flux,V,y):
    # Description: if predicted CO2 transfer is greater than gas-pohase CO2, limit
    # Source:  
    # Output: [mol/s]
    # Inputs:
        # CO2_flux [mol/s]: mass transfer gas to liquid phase
        # V [mol/s]: bulk molar gas flow rate
        # y: mol fraction CO2 in gas phase
    max_flux = V*y
    if (CO2_flux >= max_flux):
        CO2_flux = max_flux/2 # avoid 0s and negatives which code doesn't like
    return CO2_flux

def get_CO2flux(CO2_flux,TL,L,b,y,w,k,x,i,P,hp1,hm1,hp2,hm2,zp1,zm1,zp2,zm2,A,epsilon,a,dvol,sigma_c,d2,z,nu_packing):
    # Description: Get CO2 molar flux gas-phase to liquid-phase
    # Output: [mol-CO2/s]
    # Inputs:
    #     TL [K]: water temperature
    #     L [mol/s]: bulk liquid molar flow rate
    #     b [kg-OH/kg-L]: mol-fraction OH liquid phase
    #     y [kg-CO2/kg-G]: mol-fraction CO2 gas phase
    #     w [kg-H2O/kg-L]: mol-fraction H2O liquid phase
    #     k [kg-K/kg-L]:  mol-fraction K (potassium) liquid phase
    #     x [kg-CO2/kg-L]: mol-fraction CO2 liquid phase **as CO3-2 aqueous**
    #     P [Pa]: process pressure
    #     hp1,...,hm2 [L/mol]: salt-effect parameters for the positive (p) and negative (m) ions of species 1 and 2
    #     hg [L/mol]:  salt-effect parameter for dissolved CO2
    #     cp1,...,cm2: [mol/L]: concentrations the positive (p) and negative (m) ions of species 1 and 2
    #     zp1,...,zm2: charge of the positive (p) and negative (m) ions of species 1 and 2
    #     A: [m^2] cross sectional area of contactor
    #     epsilon: void fraction of packing
    #     a [cm^2/cm^3]: packing surface area to volume ratio
    #     dvol [m3]: volume of differential element
    #     sigma_c [n/m]: critical surface tension of packing
    #     d2 [m]: solute diameter, assumed spherical (OH-)
    #     z: molar ratio of binding species to CO2, z=2 for: CO2 + 2(KOH)-> 
    #     CO2_flux [mol-CO2/s]: CO2 flux from previous iteration
    #     nu_packing: packing efficiency for low flow rates (incomplete wetting)

    # get process parameters based on temperature (constant)
    # -- diffusivity of CO2
    # ---- solvent association parameter phi = 2.26
    # ---- molar volume of solute at boiling point = 34 [cm^3/mol]
    MM_H2O = get_MM('H2O')
    DL = get_diffusivity(2.26,MM_H2O,TL,34) # [cm^2/s]

    # -- temp dependent Henry's law values
    Ho = get_Ho(TL) # [atm*L/mol]
    hG = get_hG(TL) # [L/mol]
    
    from sys import exit
    if (b<=0): exit('error1: liquid OH <= 0')
    if (y<=0): exit('error2: gaseous CO2 <= 0')
    if (w<=0): exit('error3: liquid H2O <= 0')
                     
    # FIND INTERFACE CONCENTRATION USING HENRY'S LAW
    # -- find concentration of binding species (CB)
    rho_H2O = get_rhoWater(TL) # [kg/m^3] density of water, empirical correlation
    M_H2O = rho_H2O/MM_H2O # [mol/L] concentration of H2O
    M_L = M_H2O/w # [mol-liquid/L] H2O concentration 
    CB = b*M_L # [mol-OH/L] binder concentration                      
    # now that we have CB get k2
    K2 = get_k2(TL,CB) # [L/mol*s] from Kucka
    # -- find Henry's law constant (H) for solution of KOH and K2CO3
    c1 = CB # [mol/L] concentraiton of K+ ions from KOH
    c2 = CB # [mol/L] concentration of OH- ions
    c3 = M_L*k - CB # [mol/L] concentration of K+ ions from K2CO3
    c4 = M_L*x # [mol/L] concentration of CO3-2 ions [K2CO3]             
    H = HinSolution(Ho,hp1,hm1,hp2,hm2,hG,c1,c2,c3,c4,zp1,zm1,zp2,zm2) # [L*atm/mol]
    # -- find interface CO2 concentration from Henry's Law (Ci)
    P_CO2 = P*y/101325 # [atm] partial pressure of CO2 **pressure is assumed constant**
    Ci = interfaceConcentration(H,P_CO2) # [mol/L]
            
    # LIQUID SUPERFICIAL VEOLCITY
    Q = (L/M_L)/1000 # [m^3/s] volumetric liquid flow rate
    lsv = Q/A # [m/s] liquid superficial velocity
    
    SA = dvol*a # [m2] effective packing surface area per element
            
    # FIND LIQUID-PHASE MASS TRANSFER COEFFICIENT
    KL = get_k_L(epsilon,a,TL,lsv,sigma_c) # [cm/s] ('a' converted from cm^-1 to m^-1)
                
    # FIND ENHANCEMENT FACTOR
    # -- get binder diffusivity (assumed KOH)
    DB = get_DB(CB,TL,d2) # [cm^2/s]
    # -- enhancement factor
    E = enhancementFactor(DL,KL,K2,CB,Ci,DB,z)
    
    # GET CO2 GAS-LIQUID FLUX
    Cinf = 0 # assume all CO2 reacts **investigate**
    
    #Get new CO2 flux
    SA = SA*1e4 # [cm2] unit conversion
    Ci = Ci/1000 # [mol/cm^3]
    J=KL*(Ci-Cinf)*E # [mol/cm^2*s]
    
    # packing penalty for low wetting rate
    SA = SA*nu_packing
    
    newflux = J*SA # [mol-CO2/s]
    if i<20:
        CO2_flux = newflux # [mol-CO2/s]
    elif i>=20: # average flux with previous finding to promote convergance
        CO2_flux = (CO2_flux + newflux)/2 # [mol-CO2/s]
            
    return CO2_flux # [mol-CO2/s]

def evaporator(A,Z,n,G_in,L_in,omega_in,Ta_in,TL_in,P,cc,nn,Le,nu_packing):
    # Description: get evaporation rates and temperature change from evaporation process
    # Outputs:
    #     TL: [K] air temperature
    #     Ta: [K] water temperature
    #     omega: [kg-vap/kg-dry air] mass humidity ratio
    #     mdot_w: [kg/s] liquid mass flow rate
    #     mdot_dry: [kg/s] dry air mass flow rate
    #     dwdV: [kg-vap/kg-dry/m3] evaporation per unit volume
    #     dhdV: [J/kg-dry/m3] enthalpy of dry air per unit volume
    #     hdAv: [kg/m3s] mass transfer coefficient * specific area
    # Inputs:
    #     A [m2]: cross-sectional area of contactor
    #     Z [m]: depth of contactor
    #     n: number of elements
    #     G_in [kg/m2s]: inlet Air mass flow rate per unit area
    #     L_in [kg/m2s]: inlet Water mass flow rate per unit area
    #     omega_in [kg-vap/kg-dry]: inlet mass humidity ratio
    #     Ta_in [K]: inlet air temp
    #     TL_in [K]: inlet water temp
    #     P [Pa]: process pressure
    #     cc: coefficient for evaportive mass transfer correlation
    #     nn: coefficient for evaportive mass transfer correlation
    #     le: Lewis number (ratio of convective heat trasfer to mass transfer)
    #     nu_packing: packing efficiency for incomplete wetting
        
    from numpy import zeros, linspace
    from sys import exit
    from math import isnan
    
    n = int(n)
    
    #constants
    dV = get_dV(Z,A,n)
    
    # packing penalty for low wetting rate
    dV = dV*nu_packing

    # initialize arrays
    Ta = zeros(n)+Ta_in # [K] air temperature
    TL = zeros(n)+TL_in # [K] water temperature
    omega = zeros(n)+omega_in # [kg-vap/kg-dry air] mass humidity ratio
    mdot_w = zeros(n)+L_in*A # [kg/s] liquid mass flow rate
    mdot_dry = zeros(n)+G_in/(1+omega_in)*A # [kg/s] dry air mass flow rate
    dwdV = zeros(n)
    dhdV = zeros(n)
    hdAv = zeros(n)
    
    # convergence checking
    threshold = 1e-4
    no_converge = 1
    wLast = 1
    iteration_count=0
    
    while (no_converge or iteration_count<10):
        
        iteration_count = iteration_count + 1
               
        for i in range(n-1): # step through gas side

            # get mass transer coefficient
            hdAv[i] = get_hdAv(mdot_w[i],mdot_dry[i],Z*A,cc,nn) # typical cc=1.3, nn=0.6
            if isnan(hdAv[i]):
                print('mdot_w[i]: %0.2e'%mdot_w[i])
                print('Z: %0.2e'%Z)
                print('A: %0.2e'%A)
                print('cc: %0.2e'%cc)
                print('mdot_dry[i]: %0.2e'%mdot_dry[i])
                print('nn: %0.2e'%nn)
                exit('hdAV = nan (ln 1016)')

            # get evaporation rates
            dwdV[i] = get_dwdV(omega[i],hdAv[i],TL[i],Ta[i],mdot_dry[i],P)

            if isnan(dwdV[i]):
                print('omega[i]: %0.2e'%omega[i])
                print('hdAv[i]: %0.2e'%hdAv[i])
                print('TL[i]: %0.2e'%TL[i])
                print('Ta[i]: %0.2e'%Ta[i])
                print('mdot_dry[i]: %0.2e'%mdot_dry[i])
                print('P: %0.2e'%P)
                exit('dwdV = nan (ln 1027)')
            dw = dwdV[i]*dV

            # update air humidity ratio
            omega[i+1] = (omega[i] + dw)*.99 + omega[i+1]*.01 # promote convergence

            # get temperature change
            dt = get_dt_a(dV,TL[i],Ta[i],omega[i],dwdV[i],hdAv[i],Le,mdot_dry[i])
            
            # update temperature 
            Ta[i+1] = (Ta[i] + dt)*0.95 + Ta[i+1]*0.05 # promote convergence (dampen)

            # check for supersaturation
            wsat = get_wSat(Ta[i+1],P)
            count = 0
            while wsat<omega[i+1]:
                # update change in humidity ratio based on saturated humidity
                if (wsat-omega[i])>0:
                    dw = (wsat-omega[i])*0.95
                elif (wsat-omega[i])<0:
                    dw = (wsat-omega[i])*1.05
                else:
                    dw = 0
                    print('here I am')
                
                # update air humidity
                omega[i+1] = omega[i] + dw
                dwdV[i] = dw/dV
                
                # get temperature change
                dt = get_dt_a(dV,TL[i],Ta[i],omega[i],dwdV[i],hdAv[i],Le,mdot_dry[i])
                
                # update temperature
                Ta[i+1] = Ta[i] + dt

                # find the saturated humidity at the new temperature
                wsat = get_wSat(Ta[i+1],P)

                count = count+1
                if count>400: exit('stuck in supersat')

        for i in reversed(range(n)[1:]): # step through liquid side
            
            # get enthalpy change
            dhdV[i] = get_dhdV(dwdV[i],TL[i],Ta[i],mdot_dry[i],Le,hdAv[i],omega[i]) # [J/kg-dryair/m3]

            # get temperature change
            dt_w = get_dt_w(dhdV[i],dV,mdot_dry[i],mdot_w[i],TL[i])
            
            # update temperature
            TL[i-1] = (TL[i] + dt_w)*0.99 + TL[i-1]*0.01
            
            if TL[i-1]<273:
                no_converge = 0
                break
            
            # update liquid mass-flow rate
            dw = dwdV[i]*dV
            mdot_w[i-1] = mdot_w[i] - dw*mdot_dry[i]
            if mdot_w[i-1]<0:
                no_converge = 0 # contactor is dry
                break
            
        # check convergence using exit air temp
        # if abs(omega[-1]-wLast)<threshold:
        
        test = abs(TL[0]-wLast)/wLast
        if test<threshold:
            no_converge=0
            
        if (iteration_count % 100) == 0:
            print('evap iteration %0.0i'%iteration_count)
            # print('humidity ratio delta = %0.5e'%(omega[-1]-wLast))
            print('TL diff ratio = %0.5e'%(test))

        # wLast = omega[-1]
        wLast = TL[0]

        if iteration_count>2000:
            exit('Line 1163 (Evaporation): no convergence after %0.0i iterations'%iteration_count)
            
    return TL, Ta, omega, mdot_w, mdot_dry, dwdV, dhdV, hdAv


def contactor(s,n,TL_in,Ta_in,M_CO3,M_OH,mdot_L,mdot_G,CO2ppm,RH,P,A,epsilon,a,sigma_c,nu_packing,d2,z,Z,hp1,hm1,hp2,hm2,zp1,zm1,zp2,zm2,cc,nn,Le,v):
    # Description: Model CO2 and H2O mass transfer, and enthalpy transfer, between the gas and liquid phases.
    #              This is a wrapper function that will call for evaporation and then call for CO2 transfer until convergence.
    # Outputs:
        # captureRate [t-CO2/hr]: CO2 capture rate in tonnes/hour
        # L, L1 [mol/s]: bulk liquid molar flow rate 
        # V, V2 [mol/s]: bulk gas molar flow rate
        # x, x1: liquid-phase mol-fraction CO2, " at inlet
        # b: binder mol-fraction
        # k: K ion mol fraction
        # w, w1: liquid H2O mol-fraction, " at inlet
        # y, y1: gas-phase mol-fraction CO2, " at inlet
        # vap, vap2: vapor mol-fraction, " at inlet
        # air: dry-air (without CO2) mol fraction
        # i: number of iterations of CO2 transfer loop for converge
        # TL [K]: liquid temperature
        # Ta [K]: air temperature
    # Inputs:
        # s: [%] sensitivity for testing CO2 flux solution convergence
        # n: number of elements in contactor arrays
        # TL_in: [K] liquid inlet temperature
        # Ta_in: [K] gas inlet tempeature
        # Patm: [Pa] process pressure
        # M_CO3: [mol/L] inlet concentration of KCO3aq
        # M_OH: [mol/L] inlet concentration of KOHaq
        # mdot_L: [t/h] inlet bulk-liquid mass flow rate
        # CO2: [CO2/dryair*1e6] ppm CO2
        # RH: [partial-pressure water vapor/partial pressure saturated water vapor] relative humidity
        # A: [m^2] cross sectional area of contactor
        # epsilon: void fraction of packing
        # a: [m^2/m^3] packing surface area to volume ratio
        # sigma_c: [N/m] critical surface tension of packing material
        # nu_packing: packing efficiency (for incomplete wetting)
        # d2: [m] OH ion diameter
        # z: stoichiometric coefficient of KOH (CO2 + 2KOH = K2CO3 + H2O)
        # Z: [m] depth of packing/height of tower
        # hp1,...,hm2 [L/mol]: salt-effect parameters for the positive (p) and negative (m) ions of species 1 and 2
        # hg [L/mol]:  salt-effect parameter for dissolved CO2
        # cp1,...,cm2: [mol/L]: concentrations the positive (p) and negative (m) ions of species 1 and 2
        # zp1,...,zm2: charge of the positive (p) and negative (m) ions of species 1 and 2
        # cc: empirical coefficient for evaporation rate correlation
        # nn: empirical coefficient for evaporation rate correlation
        # Le: Lewis number (for heat ransfer correlation)
        # v: verbose output, not implemented
    
    import numpy as np
    from numpy import array, linspace, log
    from scipy.constants import R
    from sys import exit
    from math import floor
    
    # error flags for freezing or drying
    error_flag = 0 # freezing or drying, 0 = false, -1 = true
    
    # molar mass of elements/substances of interest [g/mol]
    MM_C = get_MM('C') # [g/mol] carbon
    MM_H = get_MM('H') # [g/mol] hydrogen
    MM_K = get_MM('K') # [g/mol] potassium
    MM_O = get_MM('O') # [g/mol] oxygen
    MM_N = get_MM('N') # [g/mol] nitrogen
    MM_A = get_MM('A') # [g/mol] argon
    MM_H2O = 2*MM_H + MM_O # [g/mol] H2O
    MM_OH = MM_H + MM_O # [g/mol] OH-
    MM_CO3 = MM_C + 3*MM_O # [g/mol] CO3
    MM_CO2 = MM_C + 2*MM_O # [g/mol] CO2
    MM_air = (MM_N*2*0.78084 + MM_O*2*0.20946 + MM_A*0.00934)*1/(0.78084+0.20946+0.00934) # [g/mol] dry air, no CO2
    
    # inlet liquid conditions (1) {CO3, OH, K, H2O}
    mdot_L = mdot_L*1000/60**2 # [kg/s] unit conversion
    rho_H2O = get_rhoWater(TL_in) # [kg/m^3] density of water, empirical correlation
    M_H2O = rho_H2O/MM_H2O # [mol/L] concentration of H2O
    M_K = 2*M_CO3 + M_OH # [mol/L] concentration of K+
    M_L1 = M_K + M_H2O + M_CO3 + M_OH # [mol/L] concentration of bulk liquid at inlet
    x1 = M_CO3/M_L1 # [mol-CO3/mol-Liquid] mol-fraction inlet CO3
    b1 = M_OH/M_L1 # [mol-OH/mol-liquid] mol-fraction inlet OH-
    k1 = M_K/M_L1 # [mol-K/mol-liquid] mol-fraction inlet K+
    w1 = M_H2O/M_L1 # [mol-H2O/mol-liquid] mol fraction inlet H2O
    MM_L1 = (MM_CO3*x1 + MM_OH*b1 + MM_K*k1 + MM_H2O*w1)/1000 # [kg/mol] molar-mass of bulk liquid at inlet
    L1 = mdot_L/MM_L1 # [mol/s] bulk liquid molar flow rate
    
    # inlet gas conditions (2) {air, CO2, H2O}
    mdot_G = mdot_G*1000.0/60**2 # [kg/s] unit conversion
    M_G = (P/(R*Ta_in))/1000 # [mol/L] bulk gas concentration, constant for constant P,T, ideal gas law
    pw = get_pw(Ta_in,P) # [Pa] partial pressure of water vapor at saturation
    p_vap = RH*pw # [Pa] partial pressure of water vapor
    M_vap = (p_vap/(R*Ta_in))/1000 # [mol/L] vapor concentration
    vap2 = M_vap/M_G # [mol-vapor/mol-gas] mol-fraction inlet water vapor
    air2 = (1-vap2)*(1e6-CO2ppm)/1e6 # [mol-air/mol-gas] mol-fraction air less CO2
    y2 = 1-vap2-air2 # [mol-CO2/mol-gas] mol-fraction CO2
    MM_G2 = (MM_air*air2 + MM_CO2*y2 + MM_H2O*vap2)/1000 # [kg/mol] molar mass of bulk gas at inlet
    V2 = mdot_G/MM_G2 # [mol/s] bulk gas flow rate
       
    # contactor geometry
    dvol = A*Z/n # [m3] volume of element
        
    # initialize arrays
    # the 0th element is the bottom of the column (gas inlet, 2) and the n-1 element is the top (liquid-inlet, 1).
    n = int(n) # cast float as integer
    f = [1.0]*n
    f = array(f)

    # set initial conditions to inlet conditions
    # -- liquid phase
    TL = f[:]*TL_in # [K] liquid temperature 
    L = f[:]*L1 # [mol/s] bulk molar flow rate liquid phase
    x = f[:]*x1 # mol-fraction CO2 liquid phase **as CO3-2 aqueous**
    b = f[:]*b1 # mol-fraction OH liquid phase
    k = f[:]*k1 # mol-fraction K liquid phase
    w = f[:]*w1 # mol-fraction H2O liquid phase
    # -- gas phase
    Ta = f[:]*Ta_in # [K] gas temperature
    V = f[:]*V2 # [mol/s] bulk molar flow rate gas phase
    y = f[:]*y2 # mol-fraction CO2 gas phase
    air = f[:]*air2 # mol-fraction air (less CO2) gas phase
    vap = f[:]*vap2 # mol-fraction water vapor gas phase 
    # -- mass-transfer
    CO2_flux = f[:]*0.0 # [mol/s] molar flux CO2 gas-to-liquid
    H2O_flux = f[:]*0.0 # [mol/s] molar flux H2O liquid-to-gas
    
    # set up evaporation process arrays
    omega = f[:]*0.0 # [kg-vap/kg-dry air] mass humidity ratio
    mdot_w = f[:]*0.0 # [kg/s] liquid mass flow rate
    mdot_dry = f[:]*0.0 # [kg/s] dry air mass flow rate
    dwdV = f[:]*0.0 # [kg-vap/kg-dry/m3] evaporation per unit volume
    dhdV = f[:]*0.0 # [J/kg-dry/m3] enthalpy of dry air per unit volume
    hdAv = f[:]*0.0 # [kg/m3s] mass transfer coefficient * specific area
    
    # iterative code below here
    dJ = 1 # variable to track convergence
    i = 0 # track number of iterations
    
    # get temp/humidity changes from evaporation
    MM_dryAir = get_MM('dryAir')
    omega_in = vap2/(1-vap2) * MM_H2O/MM_dryAir
    G_in = mdot_G/A
    L_in = mdot_L/A
    TL,Ta,omega,mdot_w,mdot_dry,dwdV,dhdV,hdAv = evaporator(A,Z,n,G_in,L_in,omega_in,Ta_in,TL_in,P,cc,nn,Le,nu_packing)
    
    if np.any(TL<273):
        print('warning: liquid temp below freezing')
        error_flag=-1
    
    # update mol-fractions and molar flow rate - gas phase
    H2O_flux = dwdV*dvol*MM_dryAir/MM_H2O * (V-V*vap) # mol-H2O/s
    Vnew = V + H2O_flux # gas bulk molar flow rate
    vap = (vap*V+H2O_flux)/Vnew # vapor mol fraction
    y = y*V/Vnew
    air = air*V/Vnew
    V = Vnew
    
    # update mol-fractions and molar flow rate - Liquid phase
    mdot_H2O = mdot_w-L*(x*MM_CO3+b*MM_OH+k*MM_K)*1e-3
    Lnew = mdot_H2O/(MM_H2O*1e-3) +L*(x+b+k)
        
    if np.any(Lnew<=0):
        print('warning: contactor is dry')
        error_flag=-1
    
    x = x*L/Lnew
    b = b*L/Lnew
    k = k*L/Lnew
    w = mdot_H2O/(MM_H2O*1e-3)/Lnew
    L = Lnew
    
    if error_flag==-1: # freezing or drying
        return error_flag, L, V, x, b, k, w, y, vap, air, i, x1, y2, w1, vap2, L1, V2, TL, Ta
    
    while (abs(dJ) > (s/100)): # run until gas-to-liquid flux converges to s percent
        Jsum = sum(CO2_flux) # store previous solution
        
        for j in range(n):            
            # find CO2 flux
            CO2_flux[j] = (
                get_CO2flux(
                CO2_flux[j],TL[j],L[j],b[j],y[j],w[j],k[j],x[j],i,
                P,hp1,hm1,hp2,hm2,zp1,zm1,zp2,zm2,A,epsilon,a,dvol,sigma_c,d2,z,nu_packing))
            
        # flux arrays are populated, update liquid and gas flows
        
        # check flux against binder (b) and CO2 (y) availability
        # -- limit CO2_flux to binder [OH] availability 
        CO2_flux = liquidSideUpdate(CO2_flux,H2O_flux,L,x,b,k,w,L1,x1,b1,k1,w1,z,n)[0]
        # -- limit CO2flux to gas-phase CO2 availability
        CO2_flux = gasSideUpdate(CO2_flux,H2O_flux,V,y,vap,air,V2,y2,vap2,air2,n)[0]    
                       
        # UPDATE LIQUID AND GAS ARRAYS
        # -- update liquid side
        CO2_flux,L,x,b,w,k = liquidSideUpdate(CO2_flux,H2O_flux,L,x,b,k,w,L1,x1,b1,k1,w1,z,n)
        # -- update gas side
        CO2_flux,V,y,vap,air = gasSideUpdate(CO2_flux,H2O_flux,V,y,vap,air,V2,y2,vap2,air2,n)                       
        
        # Cleanup
        dJ = (sum(CO2_flux)-Jsum)/sum(CO2_flux)
        i = i+1
        if i==(50): exit('error: CO2 flux not converging after %0i iterations'%i)
    
    # output variables
    captureRate = (L[0]*x[0]-L1*x1)*MM_CO2*60**2/1e6 #[t-CO2/hr]
    
    return captureRate, L, V, x, b, k, w, y, vap, air, i, x1, y2, w1, vap2, L1, V2, TL, Ta
