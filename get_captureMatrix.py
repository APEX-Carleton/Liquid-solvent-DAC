from sys import argv
from datetime import datetime
from myfunctions import contactor, get_balanceOfPlant_gasFiredCalciner, get_balanceOfPlant
import numpy as np
import copy

rangesInput_fn = argv[1]
matrixOutput_fn = argv[2]
logfile = matrixOutput_fn+'_log.txt'
output_logfile = open(logfile,'w')

#climate variables
TL = 21+273.0 # [K] Liquid inlet temperature (effects K2, Ho, hG, rho_air, rho_water) defined over 0-50
Ta = 21+273.0 # [K] Gas inlet temperature
CO2ppm = 400.0 # [moles-CO2/moles-dry-air*1e6]  ppm dry air
P = 101325.0 # [Pa]
RH = 0.63 # relative humidity fraction [0-1]

# inlet flow conditions
mdot_L = 35000.0 # [t/h] mass flow rate of liquid
mdot_G = 251000.0 # [t/h] mass flow rate of gas
M_OH = 1.1 # [mol/L solution] molar concentration of OH- in solution
M_CO3 = 0.45 # [mol/L solution] molar concentration of CO3-2 in solution

# Contactor and packing dimensions
A = 20*200*10.0 # [m^2] cross-sectional area
Z = 7.0 # [m] total height/depth (of packing)
epsilon = 0.90 # void fraction of packing
a = 210.0 # [m2/m3] Surface area to volume ratio for packed unit volume (210 from Keith)
sigma_c = 0.0307 # [N/m] critical surface tension of packing material, (~0.3 for polyethylene, polypropelene)
nu_packing = 0.8 # packing efficiency for low wetting rate (Keith), coefficient applied to surface area for mass transfer
h = 20 # [m] height of contactor for calculating solvent pump work

# evaporation process
cc = 1.3
nn = 0.6
Le = 0.9

#fraction process heat recovery (balance of plant)
n_hi = 0.5

# modeling variables
n = 1000 # number of column elements
s = 0.01 # [%] percent agreement for convergence of net CO2 flux

# stoichiometric coefficient KOH
z = 2 # (CO2 + 2KOH = K2CO3 + H2O)

# Henry's Law for solutions
# Ho and hG function of T
# for KOH (from Wilcox table 3.1)
hp1 = 0.074 # [L/mol] for K+
hm1 = 0.066 # [L/mol] for OH-
zp1 = 1 # charge on K+ ion
zm1 = -1 # charge on OH ion
# for K2CO3
hp2 = 0.074 # [L/mol] for K+
hm2 = 0.021 # [l/mol] for CO3
zp2 = 1 # charge on K+ ion
zm2 = -2 # charge on CO3-2 ion

# OH ion diameter for calculating DB
d2 = 2.2e-10 # [m] (https://water.lsbu.ac.uk/water/ionisoh.html)

# verbose output
v = 0

# set variable array
contactorInput = ([s,n,TL,Ta,M_CO3,M_OH,mdot_L,mdot_G,CO2ppm,RH,P,A,epsilon,a,sigma_c,nu_packing,d2,z,Z,
                   hp1,hm1,hp2,hm2,zp1,zm1,zp2,zm2,cc,nn,Le,v])

# single run, for testing and getting conditions throughout the contactor
if 0:
    npzfile = np.load(rangesInput_fn) # ranges of climate variables
    Ta = npzfile['airTemperatureRange'] # [K] air temperature array
    RH = npzfile['relativeHumidityRange'] # Relative humidity
    CO2ppm = npzfile['CO2ppmRange'] # CO2ppm dry air
    P = npzfile['pressureRange'] # [Pa] atmospheric pressure
    
    contactorInput[2] = Ta[0] #TL
    contactorInput[3] = Ta[0]
    contactorInput[9] = 0.69
    contactorInput[8] = 400.0
    contactorInput[10] = 100000.0
    
    captureRate, L, V, x, b, k, w, y, vap, air, i, x1, y2, w1, vap2, L1, V2, TL, Ta = contactor(*contactorInput)
    print('capture rate: %0.2f t/hr'%captureRate)
    np.save('output_single_run_arrays.npy',[L, V, x, b, k, w, y, vap, air, TL, Ta])

# process an array of climate conditions, and get balance of plant
if 1: # get temp, humidity, CO2, pressure

    npzfile = np.load(rangesInput_fn) # ranges of climate variables
    Ta = npzfile['airTemperatureRange'] # [K] air temperature array
    #TL = npzfile['waterTemperatureRange'] # [K] liquid temperature array [0: TL=Ta, 1:TL=Tg]
    RH = npzfile['relativeHumidityRange'] # Relative humidity
    CO2ppm = npzfile['CO2ppmRange'] # CO2ppm dry air
    P = npzfile['pressureRange'] # [Pa] atmospheric pressure
    
    template = np.zeros([len(Ta),len(RH),len(CO2ppm),len(P)])
    
    YY = np.zeros(np.shape(template)) # CO2 capture rate
    V2 = np.zeros(np.shape(template)) # inlet gas bulk molar flow rate
    V1 = np.zeros(np.shape(template)) # outlet gas bulk molar flow rate
    vap2 = np.zeros(np.shape(template)) # inlet vapor mol fraction
    vap1 = np.zeros(np.shape(template)) # outlet vapor mol fraction
    L1 = np.zeros(np.shape(template)) # inlet liquid bulk molar flow rate
    L2 = np.zeros(np.shape(template)) # outlet liquid bulk molar flow rate
    evap_A = np.zeros(np.shape(template))
    CaCO3_makeup = np.zeros(np.shape(template))
    mechPower_A = np.zeros(np.shape(template))
    thermPower_A = np.zeros(np.shape(template))
    evap_B = np.zeros(np.shape(template))
    electricalPower_B = np.zeros(np.shape(template))
    mdot_CH4_B = np.zeros(np.shape(template))

    #for i in range(len(TL)):
    for i in range(len(Ta)):
        for j in range(len(RH)):
            for k in range(len(CO2ppm)):
                for l in range(len(P)):
                    contactorInput[2] = Ta[i] # assuming liquid inlet is at air temperature
                    contactorInput[3] = Ta[i]
                    contactorInput[9] = RH[j]
                    contactorInput[8] = CO2ppm[k]
                    contactorInput[10] = P[l]
                    # simulate contactor
                    output = contactor(*contactorInput)
                    YY[i,j,k,l] = output[0]
                    vap1[i,j,k,l] = output[8][-1]
                    vap2[i,j,k,l] = output[14]
                    V2[i,j,k,l] = output[16]
                    V1[i,j,k,l] = output[2][-1]
                    L1[i,j,k,l] = output[15]
                    L2[i,j,k,l] = output[1][0]

                    # get balance of plant electrical
                    evap_A[i,j,k,l], CaCO3_makeup[i,j,k,l], mechPower_A[i,j,k,l], thermPower_A[i,j,k,l] = get_balanceOfPlant(mdot_L,mdot_G,M_OH,M_CO3,z,A,h,Z,Ta[i],Ta[i],P[l],RH[j],YY[i,j,k,l],V2[i,j,k,l],V1[i,j,k,l],vap2[i,j,k,l],vap1[i,j,k,l],n_hi)
                    # get gas-fired balance with updated heat integration
                    evap_B[i,j,k,l], electricalPower_B[i,j,k,l], mdot_CH4_B[i,j,k,l] = get_balanceOfPlant_gasFiredCalciner(mdot_L,mdot_G,M_OH,M_CO3,z,A,h,Z,Ta[i],Ta[i],P[l],RH[j],YY[i,j,k,l],V2[i,j,k,l],V1[i,j,k,l],vap2[i,j,k,l],vap1[i,j,k,l],n_hi)

                # print output
                now = datetime.now()
                current_time= now.strftime("%H:%M:%S")
                print(current_time,'[%0.0i,%0.0i,%0.0i,%0.0i]: Ta = %0.1f C, RH = %0.2f, CO2 = %0.0i ppm, P = %0.0f kPa, capture rate = %0.2f t-CO2/hr'%(i,j,k,l,Ta[i]-273,RH[j],CO2ppm[k],P[l]/1e3,YY[i,j,k,l]),file=output_logfile)
    output_logfile.close()

    np.savez(matrixOutput_fn,
             Ta=Ta,
             RH=RH,
             CO2ppm=CO2ppm,
             P=P,
             CO2captureRate=YY,
             G_inlet=V2,
             G_outlet=V1,
             L_inlet=L1,
             L_outlet=L2,
             vapor_inlet=vap2,
             vapor_outlet=vap1,
             evap_B=evap_B,
             electricalPower_B=electricalPower_B,
             mdot_CH4_B=mdot_CH4_B,
             evap_A=evap_A,
             CaCO3_makeup=CaCO3_makeup,
             mechPower_A=mechPower_A,
             thermPower_A=thermPower_A)
