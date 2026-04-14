#EvaporativeFunctions.py

import numpy as np
import scipy.constants as constants
from scipy.constants import pi, speed_of_light as cLight, h, atomic_mass, Boltzmann, epsilon_0, hbar
a_0 = constants.physical_constants['Bohr radius'][0] #Bohr radius [m]
a = 98*a_0 #s-wave scattering length [m] from https://arxiv.org/pdf/1204.1591
m = 86.909180520 * atomic_mass #Rb-87 atomic mass [kg]
taulife = 26.2348E-9 #excited-state lifetime [s]
naturallinewidth = 1/(2*pi*taulife) #natural linewidth [Hz]
wavelength = 1064e-9 #laser wavelength [m]
Delta = cLight/wavelength - cLight/780e-9   #detuning ν_L - ν_0 [Hz]
Delta_omega = 2*np.pi*Delta                 #angular detuning Δω [rad/s]
Gamma_omega = 2*np.pi*naturallinewidth      #natural linewidth Γ [rad/s]
E_r = (h**2/wavelength**2)/(2*m) #recoil energy [J]
k_Boltz = Boltzmann #Boltzmann constant [J/K]
c = cLight #speed of light [m/s]
alpha_ground_si = 7.94e-6*h #ground-state static polarizability α_0 [J/(V/m)^2]
omega_res_THz = 2*pi*377.1074635 #D1 line angular frequency ω_0 [rad/s * 10^12]
omega_res_Hz = omega_res_THz*1e12 #D1 line angular frequency ω_0 [rad/s]
omega_laser = 2*pi*c/wavelength #laser angular frequency ω_L [rad/s]
alpha_detuned = (omega_res_Hz**2 * alpha_ground_si)/(omega_res_Hz**2 - omega_laser**2) #dynamic polarizability α(ω_L) [J/(V/m)^2]
alpha_natural = alpha_detuned/(c*epsilon_0) #alternative form, currently not used

def eta_ev(T, trap_depth, k_Boltz = Boltzmann):
    eta = trap_depth/(k_Boltz*T)
    return(eta)

def phase_space_density(N, T, geometric, k_Boltz = Boltzmann, h=h):
    '''
    Phase space density in the harmonic limit as in pg. 258 of O'Hara's thesis.
    https://jet.physics.ncsu.edu/theses/pdf/OHara.pdf
    '''
    traposcfreq = geometric #/(2*np.pi) #Converting to Hz from rads/s if the frequency is not already in rads/s, currently commented out
    insideterm = (h*traposcfreq)/(k_Boltz*T)
    psd = N*np.power(insideterm, 3)
    return(psd)

def thermal_db(T, mass = m, k_Boltz = Boltzmann, hbar = hbar):
    '''
    The thermal deBroglie wavelength.
    '''
    num = 2*np.pi*np.power(hbar, 2)
    den = mass*k_Boltz*T
    db = np.sqrt(num/den)
    return(db)
    
def peak_density(N, T, geometric): 
    '''
    Under the approx the peak density is 
    Phase Space Density divided by Thermal DeBroglie Wavelength Cubed
    As in page 258
    '''
    psd = phase_space_density(N, T, geometric)
    db = thermal_db(T)
    n0 = psd/np.power(db, 3)
    return(n0)

def scattering_cross_section(scatteringlength=a):
    return(8*pi*a**2)
        
def mean_speed(T, mass=m, k_Boltz=Boltzmann):
    '''
    As in page 258
    '''
    v_bar = np.sqrt(8*k_Boltz*T/(pi*mass))
    return(v_bar)
        
def Gamma_el(N, T, geometric_frequency):
    n_0 = peak_density(N, T, geometric_frequency)
    sigma = scattering_cross_section()
    v_bar = mean_speed(T)
    return(n_0*sigma*v_bar)
    
def Gamma_ev(N, T, trap_depth, geometric_frequency):
    '''
    Valid for large eta, (eta>4), it could be eta>8, I'm reviewing the literature
    '''
    eta = eta_ev(T, trap_depth)
    evaporationrate = Gamma_el(N, T, geometric_frequency) * (eta - 4) * np.exp(-eta)
    return(evaporationrate)
        
def Gamma_3b(N, T, geometric, dparam = 1.5, L_3 = 4.3e-41, mass = m):
    '''
    Returns per‑particle three‑body loss rate.
    The treatment for three-body loss is from the 2018 paper: https://arxiv.org/pdf/quant-ph/0602010
    The dparam = 1.5 for a harmonic trap, for anharmonic traps this must be edited
    L_3 is the value given in most experimental papers, Roy et. al made the choice of naming it K_3.
    
    '''
    density = peak_density(N, T, geometric)
    threebodyrate = np.power(3, -dparam)*(L_3)*density**2
    return(threebodyrate)

def Gamma_sc(trapdepth, detuning = Delta_omega, Gamma = Gamma_omega): 
    """
    Spontaneous scattering rate at trap center for far-detuned light.
    Two-level far-detuned formula:
        Γ_sc = (Γ / |Δ|) * |U| / ħ
    with Δ in angular frequency and U = trapdepth in Joules.
    """
    return (Gamma / np.abs(detuning)) * np.abs(trapdepth) / hbar

def Gamma_bg(rate=0.1):
    '''
    The background lifetime must be independently measured, this rate is 1/backgroundlifetime.
    There is a clever way to numerically estimate this, see pages 106-107 of O'Hara's Thesis
    '''
    return(rate)

def N_dot_only_ev(N, T, trap_depth, geometric_frequency):
    '''
    To compare with true evaporation.
    '''
    dNdt = -(Gamma_ev(N, T, trap_depth, geometric_frequency))*N
    return (dNdt)

def T_dot_only_ev(N, T, ModulationTerm, trap_depth, geometric_frequency, RecoilEnergy=E_r):
    #Modulation term is omega_bar_dot_over_omega_bar
    gammaev = Gamma_ev(N, T, trap_depth, geometric_frequency)
    eta = eta_ev(T, trap_depth)
    Efficiency = (eta + (eta-5)/(eta-4) - 3)
    dTdt = -((gammaev/3)*Efficiency - ModulationTerm)*T
    return(dTdt)
        
def N_dot(N, T, trap_depth, geometric_frequency):
    '''
    From Roy et. al
    '''
    dNdt = -(Gamma_ev(N, T, trap_depth, geometric_frequency) + Gamma_3b(N, T, geometric_frequency) + Gamma_bg())*N
    return(dNdt)

def T_dot(N, T, ModulationTerm, trap_depth, geometric_frequency, RecoilEnergy=E_r):
    '''
    From Roy et. al
    '''
    #Modulation term is omega_bar_dot_over_omega_bar
    gammaev = Gamma_ev(N, T, trap_depth, geometric_frequency)
    gamma3b = Gamma_3b(N, T, geometric_frequency)
    gammasc = Gamma_sc(trap_depth)
    eta = eta_ev(T, trap_depth)
    Efficiency = (eta + (eta-5)/(eta-4) - 3)
    dTdt = -((gammaev/3)*Efficiency - gamma3b/3 - ModulationTerm)*T + (gammasc*RecoilEnergy)/k_Boltz
    return(dTdt)
