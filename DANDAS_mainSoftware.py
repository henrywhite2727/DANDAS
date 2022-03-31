# # Dark matter ANnihilation and Decay Application Software (DANDAS)
# ##### Code written by: Henry White, 20109413
# ##### With lots of help from Aaron Vincent and Diyaselis Delgado 



import numpy as np
get_ipython().run_line_magic('pip', 'install vegas #(this line was used to install vegas in jupyter notebooks)')
import vegas
import pandas as pd
from astropy import units
import matplotlib.pyplot as plt
from scipy import interpolate
from astropy.coordinates import SkyCoord,EarthLocation, Galactic, AltAz, ICRS, get_icrs_coordinates
from astropy.time import Time
import scipy.optimize
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.stats import poisson
import scipy.optimize
from IPython.display import Image
from scipy.interpolate import UnivariateSpline
from scipy.special import gammaln
from scipy.optimize import fsolve
import timeit



def rescaling(mass,sigmaV,J,D):
    '''
    This function calculates DM decay lifetime to per nu + nubar flavor by rescaling annihilation cross section data
    (to all neutrino flavors). Because the cross section data is to per flavor, no experiment flavor detection factor 
    needs to be added.

    mass: array in units of GeV, 2 times mass range used for annihilation should be input (due to decay assumptions)
    sigmaV: annihilation cross section array in [cm^3/s]
    J: The J factor for the given experiment in [Gev^2 cm^-5 sr]
    D: The D factor for a given experiment in [GeV cm^-2 sr]
    tau: the lifetimes in s
    '''

    flux=np.zeros(len(mass))
    tau=np.zeros(len(mass))

    for i in range(0,len(mass)):
        flux[i]=(1/(4*np.pi))*(sigmaV[i]/(2*(mass[i]/2)**2))*(1/3)*2*J   #mass is divided by 2 due to assumption in annihilation paper (2DM -> 2 nu)
        tau[i]=(1/(4*np.pi))*(1/flux[i])*(1/(3*mass[i]))*2*D      

    return tau

def inv_rescaling(mass,tau,J,D):
    '''
    This function calculates DM cross section per nu + nubar flavor by rescaling lifetime limit data
    (to all neutrino flavors). Because the cross section data is per flavor, no experiment flavor detection factor 
    needs to be added.
    '''
    flux=np.zeros(len(mass))
    sigmaV=np.zeros(len(mass))

    for i in range(0,len(mass)):
        flux[i]=(1/(4*np.pi))*(1/((mass[i])*tau[i]))*(1/3)*2*D   #mass is divided by 2 due to assumption in annihilation paper (2DM -> 2 nu)
        sigmaV[i]=((4*np.pi))*(flux[i])*(2*mass[i]**2)*(3/2)/J      
                                                                     
    return sigmaV

def bin_flux2Lifetime(alpha,delta,mass,flux,D):
    '''
    This function calculates the lifetime limit based on a binned neutrino flux. The mathematical 
    formulation is derived in 'Dark Matter Decay to Neutrinos'
    
    '''
    tau=((2*D*(alpha-1))/((3*mass**2)*16*np.pi**2))*((10**(delta/2)-10**(-delta/2))*flux)**-1
    return tau

def bin_f2l_variable_delta(alpha,upper_delta,lower_delta,mass,flux,D):
    '''
    This function calculates the lifetime limit based on a binned neutrino flux with non constant bin
    width. The mathematical formulation is derived in 'Dark Matter Decay to Neutrinos'
    
    '''
    tau=((2*D*(alpha-1))/((3*mass**2)*16*np.pi**2))*((10**(upper_delta/2)-10**(-lower_delta/2))*flux)**-1
    return tau

def bin_f2l_alpha1(delta,mass,flux,D):
    '''
    This function calculates the lifetime limit based on a binned neutrino flux with a flat
    power spectrum. The mathematical formulation is derived in 'Dark Matter Decay to Neutrinos'
    
    '''
    
    tau=((2*D)/((3*mass**2)*16*np.pi**2))*(delta*np.log(10)*flux)**-1
    return tau

def diff_flux2Lifetime(mass,flux,D):
    '''
    This function calculates the lifetime limit based on a differential neutrino flux. The mathematical 
    formulation is derived in 'Dark Matter Decay to Neutrinos'
    
    '''
    tau=(1/(4*np.pi))*(1/flux)*(1/(3*mass))*2*D
    return tau


def binning(mass,flux):
    '''
    #This function is going to take unbinned flux data and associated DM mass as inputs and output binned flux data along with
    #a (essentially) continuous range of mass values
    '''
    flux_logbin_func=interp1d(np.log10(mass),np.log10(flux),kind='nearest')
    new_log_mass=np.linspace(np.log10(mass[0]),np.log10(mass[-1]),1000)
    flux_log_binned=flux_logbin_func(new_log_mass)
    binned_mass=10**new_log_mass
    binned_flux=10**flux_log_binned

    return binned_mass, binned_flux


def log_centre_bin(low,high):
    '''#This function takes the edges of a bin and returns their logarithmic centre

    '''
    log_centre=(np.log10(low)+np.log10(high))/2
    width=(np.log10(high)-np.log10(low))
    centre_val=10**log_centre
    return centre_val

def log_width_bin(low,high):
    width=(np.log10(high)-np.log10(low))
    return width   

def auger_factors(E):
    D1_aug=0.11e23 #[90,95]
    D2_aug=0.35e23 #[75,90]
    D3_aug=0.33e23 #[60,75]

    J1_aug = 1.001651628542065e+22
    J2_aug = 2.83874979745116e+22
    J3_aug = 2.677412870123818e+22

    #importing Auger Exposure Arrays
    LogExp1=np.loadtxt('Data_Files/LogExp1.txt',delimiter=',')
    LogExp2=np.loadtxt('Data_Files/LogExp2.txt',delimiter=',')
    LogExp3=np.loadtxt('Data_Files/LogExp3.txt',delimiter=',')

    #Interpolating functions for each exposure range
    exp1=interp1d(LogExp1[:,0],LogExp1[:,1])
    exp2=interp1d(LogExp2[:,0],LogExp2[:,1])
    exp3=interp1d(LogExp3[:,0],LogExp3[:,1])

    logexp1=np.zeros(len(E))
    logexp2=np.zeros(len(E))
    logexp3=np.zeros(len(E))

    #Checking if elements of E are outside of the interpolation range
    for i in range(0,len(E)):
        if np.log10(E[i])<=LogExp1[0,0]:
            continue
        else:
            logexp1[i]=exp1(np.log10(E[i]))

        if np.log10(E[i])<=LogExp2[0,0]:
            continue
        else:
            logexp2[i]=exp2(np.log10(E[i]))

        if np.log10(E[i])<=LogExp3[0,0]:
            continue
        else:
            logexp3[i]=exp2(np.log10(E[i]))        

    #"Un-log10 ing" the arrays
    Exp1=10**logexp1
    Exp2=10**logexp2
    Exp3=10**logexp3


    exptot=Exp1+Exp2+Exp3

    J_array=(J1_aug*Exp1/exptot + J2_aug*Exp2/exptot + J3_aug*Exp3/exptot)
    D_array=(D1_aug*Exp1/exptot + D2_aug*Exp2/exptot + D3_aug*Exp3/exptot)

    return J_array,D_array



def DANDAS(Halo_Profile, Antiparticle_Nature,plot_preference,reduce_runtime, data,**kwargs):
    start1 = timeit.default_timer()
    
    if Halo_Profile=='NFW':
        scaling_factor_J=1
        scaling_factor=1
        
        J_allsky=2.3082231779640816e23*scaling_factor_J
        D_allsky=2.6470090649443034e23*scaling_factor

        D_tambo=0.001e23*scaling_factor
        J_tambo=0.0009e23 *scaling_factor_J

        J_cta=0.074e23*scaling_factor_J
        D_cta=0.003e23*scaling_factor 

        J_anita=0.018e23*scaling_factor_J 
        D_anita=0.052e23*scaling_factor 

        J_grand=0.28e23*scaling_factor_J
        D_grand=0.298e23*scaling_factor
        
        J_auger1=0.10e23*scaling_factor_J       #for [90,95] degree range
        J_auger2=0.28e23*scaling_factor_J          #for [75,90] degree range
        J_auger3=0.27e23*scaling_factor_J          #for [60,75] degree range
        D_auger1=0.11e23*scaling_factor          #for [90,95] degree range
        D_auger2=0.35e23*scaling_factor          #for [75,90] degree range
        D_auger3=0.33e23*scaling_factor          #for [60,75] degree range
        
        J_pone1=0.87e23*scaling_factor_J           #for cos(zenith)=[-1,-0.5]
        J_pone2=1.2e23*scaling_factor_J            #for cos(zenith)=[-0.5,0.5]
        J_pone3=0.13e23*scaling_factor_J           #for cos(zenith)=[0.5,1]
        D_pone1=0.83e23*scaling_factor           #for cos(zenith)=[-1,-0.5]
        D_pone2=1.35e23*scaling_factor           #for cos(zenith)=[-0.5,0.5]
        D_pone3=0.47e23*scaling_factor           #for cos(zenith)=[0.5,1]
        

        
        
    elif Halo_Profile=='Einasto':
            
        D_allsky=2.7236e+23
        J_allsky=2.22e23
        
        D_anita=5.431e+21
        J_anita=2.0084e21
        
        D_grand=3.011e+22
        J_grand=2.591e+22
        
        D_cta=3.145e20
        J_cta=9.732e+21
        
        D_tambo=1.0284e+20
        J_tambo=8.63e+19
        
        J_auger1=9.51e+21            #for [90,95] degree range
        J_auger2=2.665e+22         #for [75,90] degree range
        J_auger3=2.462e+22           #for [60,75] degree range
        D_auger1=1.1518e+22          #for [90,95] degree range
        D_auger2=3.504e+22           #for [75,90] degree range
        D_auger3=3.435e+22          #for [60,75] degree range   
        
        J_pone1=7.56e+22    #for cos(zenith)=[-1,-0.5]
        J_pone2=1.169e+23   #for cos(zenith)=[-0.5,0.5]
        J_pone3=1.2659e+22   #for cos(zenith)=[0.5,1]
        D_pone1=8.328e+22    #for cos(zenith)=[-1,-0.5]
        D_pone2=1.383e+23    #for cos(zenith)=[-0.5,0.5]
        D_pone3=4.818e+22   #for cos(zenith)=[0.5,1]
        
        
            
    elif Halo_Profile=='NFW Custom' or Halo_Profile=='Einasto Custom' or 'Custom Density Function':        
        #If the user specified different parameters, these will be applied by DANDAS
        r0= kwargs.get('r0', None) 
        rs=kwargs.get('rs', None) 
        rho_0= kwargs.get('rho_0', None)
        r_halo=kwargs.get('r_halo', None)
        gamma=kwargs.get('gamma', None)
        alpha=kwargs.get('alpha', None)
        
        #if user leaves one parameter blank, DANDAS will apply the best-fit value
        if r_halo is None:
            r_halo = 200 * 3.0857e21 # maximum Halo radius [kpc]; convert kpc to cm;
        if r0 is None:
            r0 =  8.127 * 3.0857e21 # distance from the Sun to GC [kpc]; convert kpc to cm
        if rs is None:
            rs = 20 * 3.0857e21     # scale radius [kpc]; convert kpc to cm
        if rho_0 is None:
            rho_0 = 0.4             # local densitity [GeV/cm^-3]
        if gamma is None:
            gamma=1.2              # slope parameter (for NFW)
        if alpha is None:
            alpha=0.155             #slope parameter (Einasto) 
        
        '''The following section (lines 166-561) calculates the J and D factors using Monte Carlo Integration methods, specifically the
        vegas+ package. This code was written by Diyaselis Delgado López and modified by Henry White to suit DANDAS. '''
        
        # galactic center definition
        gc = get_icrs_coordinates('Galactic Center')

        # calculating upper limit of line of sight (los)
        psi = 0                 # angle between GC and los [rad]
        
        decay_xmax = np.sqrt(r_halo**2 - (r0**2)*(np.sin(psi)**2)) + r0*np.cos(psi) # upper limit of los integration

        # exposure selection for TAMBO and GRAND
        global exp # TAMBO: 0 = [95º,90º], 1 = [90º,75º], 2 = [75º,60º]
           # P-ONE: 0 = [-1,-0.5], 1 = [-0.5,0.5], 2 = [0.5,1]

        # ==============================================================================
        # ---------------------------- class definition --------------------------------
        # ==============================================================================

        class Dfactor:

            def __init__(self,experiment):
                self.experiment = experiment

                if experiment == 'allsky':

                    frame = 'icrs'
                    location = None

                    # determines whether to do 4D integral
                    integ_4D = False

                    # limits of solid angle for All-Sky
                    ra_equiv_lower = 0
                    ra_equiv_upper = 2 * np.pi

                    dec_equiv_lower = -np.pi/2
                    dec_equiv_upper = np.pi/2

                elif experiment == 'grand':

                    frame = 'altaz'
                    # define coordinates specific to GRAND
                    location = EarthLocation(lat=40.14 * units.deg, lon=94.66 * units.deg, height=1500 * units.m)

                    # determines whether to do 4D integral
                    integ_4D = True

                    # limits of solid angle for GRAND
                    # limits are based on the bin edges of Fig. 24 of arVix: 1810.09994
                    ra_equiv_lower = 0 # az
                    ra_equiv_upper = 2 * np.pi # az

                    dec_equiv_lower = np.deg2rad(-4.616) # alt
                    dec_equiv_upper = np.deg2rad(4.602) # alt

                    # GRAND data files for fractional acceptance
                    alt_filename = 'Data_Files/alt_GRAND.txt'

                elif experiment == 'anita':

                    frame = 'icrs'
                    # define coordinates specific to ANITA
                    location = EarthLocation(lat=-77.85 * units.deg, lon=166.67 * units.deg, height=3500 * units.m)

                    # determines whether to do 4D integral
                    integ_4D = False

                    # limits of solid angle for ANITA
                    ra_equiv_lower = 0
                    ra_equiv_upper = 2 * np.pi

                    dec_equiv_lower = np.deg2rad(1.5)
                    dec_equiv_upper = np.deg2rad(4)

                elif experiment == 'cta':

                    frame = 'galactic'
            # define coordinates specific to CTA
                    location = EarthLocation(lat=-24.68 * units.deg, lon=-70.32 * units.deg, height=2408 * units.m)

            # determines whether to do 4D integral
                    integ_4D = False

            # limits of solid angle for CTA
            # limits are based on the Fig. 1 of arVix: 1408.4131

            # constants definition
                    r1 = 0.55 # deg
                    delta_cut = 1.36 # deg
                    b_offset = 1.42 # deg
                    horizontal_cut  = 0.3

                elif experiment == 'tambo':

                    frame = 'altaz'
            # define coordinates specific to TAMBO
                    location = EarthLocation(lat=-15.66 * units.deg, lon=-72.18 * units.deg, height=3658 * units.m)

            # determines whether to do 4D integral
                    integ_4D = True

            # limits of solid angle for TAMBO
            # limits are based on the bin edges of Fig. 4 of arVix: 2002.06475
                    ra_equiv_lower = np.deg2rad(90) # az
                    ra_equiv_upper = np.deg2rad(270) # az

                    dec_equiv_lower = np.deg2rad(-15) # alt
                    dec_equiv_upper = np.deg2rad(35) # alt

            # TAMBO data files for fractional acceptance
                    az_filename = 'Data_Files/tambo_azimuth.dat'
                    alt_filename = 'Data_Files/tambo_elevation.dat'

                elif experiment == 'auger':

                    frame = 'altaz'
            # define coordinates specific to Auger
                    location = EarthLocation(lat=-35.4634 * units.deg, lon=-69.5848 * units.deg, height=1400 * units.m)

            # determines whether to do 4D integral
                    integ_4D = True

            # limits of solid angle for Auger
                    ra_equiv_lower = 0 # az
                    ra_equiv_upper = 2*np.pi # az

            # select region of exposure limits
                    alt_range1 = np.deg2rad(90 - 95) # altitude = 90º - zenith angle
                    alt_range2 = np.deg2rad(90 - 90) # altitude = 90º - zenith angle
                    alt_range3 = np.deg2rad(90 - 75) # altitude = 90º - zenith angle
                    alt_range4 = np.deg2rad(90 - 60) # altitude = 90º - zenith angle

                    alt_lower = [alt_range1,alt_range2,alt_range3]
                    alt_upper = [alt_range2,alt_range3,alt_range4]

                    dec_equiv_lower = alt_lower[exp] # alt
                    dec_equiv_upper = alt_upper[exp] # alt

                elif experiment == 'pone':

                    frame = 'altaz'
            # define coordinates specific to P-ONE
                    location = EarthLocation(lat=47.46 * units.deg, lon=127.46 * units.deg, height=-2660 * units.m)

            # determines whether to do 4D integral
                    integ_4D = True

            # limits of solid angle for P-ONE
                    ra_equiv_lower = 0 # az
                    ra_equiv_upper = 2*np.pi # az

            # select region of exposure limits
                    alt_range1 = np.arcsin(-1)   # cos(zen) = sin(alt)
                    alt_range2 = np.arcsin(-0.5) # cos(zen) = sin(alt)
                    alt_range3 = np.arcsin(0.5)  # cos(zen) = sin(alt)
                    alt_range4 = np.arcsin(1)    # cos(zen) = sin(alt)

                    alt_lower = [alt_range1,alt_range2,alt_range3]
                    alt_upper = [alt_range2,alt_range3,alt_range4]

                    dec_equiv_lower = alt_lower[exp] # alt
                    dec_equiv_upper = alt_upper[exp] # alt

                else:
                    end

                # check if input is correct
                list = ['allsky','grand','anita','cta', 'tambo', 'auger', 'pone']
                if (experiment in list) == False:
                    print('Experiment not defined. \n Please specify an experiment from the list: \n'+', '.join(list))

        # ======================================================================
        # ------------------------ helper functions ----------------------------
        # ======================================================================

                def exp_factor(ra_equiv,dec_equiv):
                    '''
                    factor that maps the exposure space of an specific experiment

                    Parameters
                    ----------
                    ra_equiv : right ascension equivalent [rad]
                    dec_equiv: declination equivalent [rad]

                    Returns
                    -------
                    factor: either 1 or 0 depending whether the integration is inside
                            the experimental exposure range
                            depending on the experiment the factor would be waited
                    '''

                    # define weight
                    weight = 1

                    if experiment == 'cta':
                        # convert to degrees
                        l = np.rad2deg(ra_equiv)
                        b = np.rad2deg(dec_equiv)

                        # visual limits from Fig.1 in degrees
                        if np.abs(l) > 1.5 and np.abs(b) > 1.5:
                            factor = 0
                        else: # exclude outer circle regions
                            if delta_cut < np.sqrt(l**2 + b**2):
                                factor = 0
                            elif r1 > np.sqrt(l**2+(b-b_offset)**2): # offset FoV circle
                                factor = 0
                            elif horizontal_cut > np.abs(l):  # horizontal band
                                factor = 0
                            else:
                                factor = 1
                    else:
                        if ra_equiv >= ra_equiv_lower and ra_equiv <= ra_equiv_upper and dec_equiv >= dec_equiv_lower and dec_equiv <= dec_equiv_upper:
                            factor = 1
                            if experiment == 'grand':
                                weight = get_alt_weight(dec_equiv) * (4*np.pi) # 4*pi factor for solid angle unit corrections
                            elif experiment == 'tambo':
                                weight = (get_az_weight(ra_equiv) * get_alt_weight(dec_equiv)) * (4*np.pi) # 4*pi factor for solid angle unit corrections
                        else:
                            factor = 0

                    return factor * weight


                def get_az_weight(az):
                    '''
                    gets the azimuth weight based on pdf distribution

                    Parameters
                    ----------
                    az : azimuth [rad]

                    Returns
                    -------
                    spline : spline interpolation at a given azimuth
                    '''
                    # create pandas DataFrame object from az data files
                    az_data = pd.read_csv(az_filename, header=None, names=["Azimuth", "Fractional Acceptance"], engine='python') # azimuth [deg]
                    az_integ = np.trapz(az_data['Fractional Acceptance'],az_data['Azimuth'])
                    norm_az_data = az_data['Fractional Acceptance'] / az_integ
                    az_data['Azimuth'] = np.deg2rad(az_data['Azimuth']) + np.pi # South facing for TAMBO experiment
                    az_spline = interpolate.UnivariateSpline(az_data['Azimuth'],norm_az_data, k=2, s=1e-10)
                    return az_spline(az)


                def get_alt_weight(alt):
                    '''
                    gets the altitude weight based on pdf distribution

                    Parameters
                    ----------
                    alt : altitude [rad]

                    Returns
                    -------
                    spline : spline interpolation at a given altitude
                    '''
                    # create pandas DataFrame object from alt data file
                    alt_data = pd.read_csv(alt_filename, header=None, names=["Altitude", "Fractional Acceptance"], engine='python') # altitude [deg]
                    alt_integ = np.trapz(alt_data['Fractional Acceptance'], alt_data['Altitude'])
                    norm_alt_data = alt_data['Fractional Acceptance'] / alt_integ
                    alt_data['Altitude'] = np.deg2rad(alt_data['Altitude'])
                    alt_spline = interpolate.UnivariateSpline(alt_data['Altitude'], norm_alt_data,  k=2, s=1e-10)
                    return alt_spline(alt)


                def coords_conversion(ra_equiv, dec_equiv, obstime, location, frame, rad=True):
                    '''
                    converts Horizontal coordinates to Equatorial coordinates

                    Parameters
                    ----------
                    ra_equiv: right ascension equivalent [rad]
                    dec_equiv: declination equivalent [rad]
                    obstime: observation time based on experiment's exposure
                    location: experiment location
                    frame: coordinate frame (e.g. 'altaz', 'galactic')
                    rad: indicates if coordinates are in degrees or radians (rad by default)

                    Returns
                    -------
                    ra,dec: equatorial coordinates
                    '''

                    if rad == True:
                        coords = SkyCoord(ra_equiv * units.rad, dec_equiv * units.rad, frame=frame,location=location,obstime=obstime)
                    else:
                        coords = SkyCoord(ra_equiv * units.deg, dec_equiv * units.deg, frame=frame,location=location,obstime=obstime)

                    # galactic coordinates conversion
                    ra = coords.transform_to(ICRS()).ra.radian
                    dec = coords.transform_to(ICRS()).dec.radian
                    return ra,dec


                def reduce_eval(factor):
                    '''
                    reduces number of integrand evaluations computed
                    '''
                    if factor == 0:
                        return 0.0


                def rho_DM(ra,dec,x):
                    '''
                    Parameters
                    ----------
                    ra : right ascension [rad]
                    dec: declination [rad]
                    x: distance of line of sight [cm]

                    Returns
                    -------
                    rho_DM: DM densitity in equatorial coordinates
                    '''
                    # calculating galactocentric distance based on eq. 3
                    los = SkyCoord(ra * units.rad, dec * units.rad, frame='icrs')
                    psi = gc.separation(los).radian
                    r = np.sqrt(r0**2 - 2*x*r0*np.cos(psi) + x**2)

                    if Halo_Profile=='NFW Custom':
                        # calculating scale density rho_s [GeV/cm^-3] for NFW profile
                        rho_s = rho_0 / ((2**(3 - gamma))/(((r0/rs)**gamma)*(1 + (r0/rs))**(3 - gamma))) # [GeV/cm^-3]

                        # calculating DM density rho_DM [GeV/cm^-3] based on NFW profile
                        rho_DM = rho_s * (2**(3 - gamma))/(((r/rs)**gamma)*(1 + (r/rs))**(3 - gamma)) # [GeV/cm^-3]

                    elif Halo_Profile=='Einasto Custom':
                        #calculating scale density for Einasto profile
                        rho_s_ein = rho_0 / np.exp((-2/alpha)*(((r0/rs)**alpha)-1))

                        #Calculating DM density for Einasto profile
                        rho_DM=rho_s_ein * np.exp((-2/alpha)*(((r/rs)**alpha)-1))
                        
                    elif 'Custom Density Function':
                        rho_func=kwargs.get('rho_function', None)
                        
                        rho_DM=rho_func(r)


                    return rho_DM
                

                def decay_integrand(vars):
                    '''
                    Parameters
                    ----------
                    vars: array of integration variables

                    Returns
                    -------
                    integrand: densitity squared times cos(dec)
                    '''
                    # define variables
                    ra_equiv = vars[0]
                    dec_equiv = vars[1]
                    x = vars[2]

                    if integ_4D == True:
                        time = vars[3]
                        time_factor = 1/24
                        # define observed time with UTC offset for 1 day exposure
                        obstime = Time('2021-6-10 13:39:30') - time * units.hour
                    else:
                        time_factor = 1
                        obstime = Time('2021-6-10 13:39:30')

                    if frame == 'galactic':
                        # shift l by 180º
                        ra_equiv -= np.pi

                    # define factor based on experiment exposure
                    factor = exp_factor(ra_equiv,dec_equiv)
                    reduce_eval(factor)

                    # convert to equatorial coordinates
                    ra,dec = coords_conversion(ra_equiv, dec_equiv, obstime, location, frame)

                    integrand = rho_DM(ra,dec,x) * np.cos(dec_equiv)

                    return np.multiply(integrand,factor) * time_factor
            
                def annihilation_integrand(vars):
                    '''
                    Parameters
                    ----------
                    vars: array of integration variables

                    Returns
                    -------
                    integrand: densitity squared times cos(dec)
                    '''
                    # define variables
                    ra_equiv = vars[0]
                    dec_equiv = vars[1]
                    x = vars[2]

                    if integ_4D == True:
                        time = vars[3]
                        time_factor = 1/24
                        # define observed time with UTC offset for 1 day exposure
                        obstime = Time('2021-6-10 13:39:30') - time * units.hour
                    else:
                        time_factor = 1
                        obstime = Time('2021-6-10 13:39:30')

                    if frame == 'galactic':
                        # shift l by 180º
                        ra_equiv -= np.pi

                    # define factor based on experiment exposure
                    factor = exp_factor(ra_equiv,dec_equiv)
                    reduce_eval(factor)

                    # convert to equatorial coordinates
                    ra,dec = coords_conversion(ra_equiv, dec_equiv, obstime, location, frame)

                    integrand = (rho_DM(ra,dec,x)**2) * np.cos(dec_equiv)

                    return np.multiply(integrand,factor) * time_factor


                # ======================================================================

                if integ_4D == True:
                    # execute 4D integral for decay
                    integrate = vegas.Integrator([[0, 2*np.pi], [-np.pi/2, np.pi/2], [0, decay_xmax], [0,24]])
                else:
                    # execute triple integral for decay
                    integrate = vegas.Integrator([[0, 2*np.pi], [-np.pi/2, np.pi/2], [0, decay_xmax]])

                # train the integrator and discard results
                integrate(decay_integrand, nitn=10, neval=1000)

                if experiment == 'cta':
                    decay_result = integrate(decay_integrand, nitn=30, neval=3000)
                    annihilation_result=integrate(annihilation_integrand, nitn=30, neval=3000)
                    
                else:
                    decay_result = integrate(decay_integrand, nitn=10, neval=1000)
                    annihilation_result = integrate(annihilation_integrand, nitn=10, neval=1000)

                self.dfactor = decay_result
                self.jfactor = annihilation_result
                self.meanD = decay_result.mean
                self.meanJ = annihilation_result.mean
                self.sdev = decay_result.sdev
                #self.error = decay_result.sdev/decay_result.mean

        
        # ==============================================================================
        # --------------------------- Calculate J and D factors ------------------------------
        # ==============================================================================  
        
        allsky_dfactor = Dfactor('allsky')
        D_allsky=allsky_dfactor.meanD
        J_allsky=allsky_dfactor.meanJ
      
        if reduce_runtime==False:

            grand_dfactor = Dfactor('grand')
            D_grand=grand_dfactor.meanD
            J_grand=grand_dfactor.meanJ
            
            tambo_dfactor = Dfactor('tambo')
            D_tambo=tambo_dfactor.meanD
            J_tambo=tambo_dfactor.meanJ

            k=0
            D_pone=np.zeros(3)
            J_pone=np.zeros(3)
            for exp in range(3):
                pone_dfactor = Dfactor('pone')
                D_pone[k]=pone_dfactor.meanD
                J_pone[k]=pone_dfactor.meanJ
                k=k+1

            #Converting arrays to floats to ensure consistency of J and D factor format across methods
            D_pone1=D_pone[0]
            D_pone2=D_pone[1]
            D_pone3=D_pone[2]
            J_pone1=J_pone[0]
            J_pone2=J_pone[1]
            J_pone3=J_pone[2]

        '''
        The following section of DANDAS computes all of the annihilation and decay limits based on the users assumption.
        Recalculating the annihilation cross section data used in the "Dark Matter Annihilation to Neutrinos" paper was deemed
        to be out of the scope of this project so data is imported and rescaled based on the users assumptions (about Halo profile, antiparticle nature, and flavor).
        
        Lifetime limits are calculated within DANDAS due to their minimal run time. Further,
        by including the process by which each limit was calculated, there is greater transparency with other researchers about the process
        applied to calculate their 'customized' DM annihilation/decay parameters.
        
        Annihilation cross section data is imported and scaled according to assumptions in lines ___ to ___.
        
        Decay lifetime data is calculated in lines __ to ___.
        '''
    
    
    #Importing Annihilation Cross Section Data and accounting for antiparticle nature assumption
    
    if Antiparticle_Nature=='Majorana':
        ann_factor=1
    elif Antiparticle_Nature=='Dirac':
        ann_factor=2    #If DM is distinct from its antiparticle, then a 2x higher annihilation cross section is required to match the neutrino fluxes measured
    
    #Borexino Experiment (calculated from data from Fig. 4 in arxiv: 1909.02422v1)
    bor_ann=np.loadtxt('Data_Files/BorexinoAnnihilationLimits.txt')
    bor_mass=bor_ann[:,0]/1000   #converting from MeV -> GeV
    bor_sigmaV=(bor_ann[:,1]/2)*ann_factor*(J_allsky/2.3e23)    #Due to factor of 2 issue in previously digitized data
    
    #Kamlamd Experiment (Calculated from data in arxiv: 1909.02422v1)
    kamData=np.loadtxt('Data_Files/kamland.txt')
    kam_mass=(kamData[:,0])/1000  #dividing by 1000 to go MeV->GeV
    kam_sigmaV=kamData[:,1]*ann_factor*(J_allsky/2.3e23)
    
    #Superkamiokande data (Calculated from Wan Linyan's PhD thesis: Experimental Studies on Low Energy Electron Antineutrinos and Related Physics)
    Sk_nuebar_Data=np.loadtxt('Data_Files/SK4_nuebar.csv', delimiter=',')
    Sk_nuebar_mass=(Sk_nuebar_Data[:,0])   
    Sk_nuebar_sigmaV=Sk_nuebar_Data[:,1]*ann_factor 
    
    #JUNO Experiment (data from https://arxiv.org/abs/1507.05613)
    juno_data=np.loadtxt('Data_Files/juno.txt')
    junoMass=juno_data[:,0]   
    junosigmaV=(juno_data[:,1]/2)*ann_factor*(2.3e23/J_allsky) #Due to factor of 2 issue in previously digitized data
    
    #Cross section limit calculated from SuperK atmospheric neutrino data, IceCube atmospheric neutrino data, and Icecube-HE (neutrinos from astrophysical sources)
    congl=np.loadtxt('Data_Files/SK_IC_conglomerate.csv',delimiter=',')  #digitizing data in fig. 2 of https://arxiv.org/abs/1912.09486
    congl_mass=congl[:,0]
    congl_sigmaV=congl[:,1]*ann_factor*(2.3e23/J_allsky)
    
    #SuperK analysis by (Olivares et al.) found at from http://etheses.dur.ac.uk/13142/1/PhD_thesis_Olivares_FINAL.pdf?DDD25+
    sk_ol1=np.loadtxt('Data_Files/SK_Oliv_sigmaV.csv',delimiter=',')
    sk_ol_mass=sk_ol1[:,0]
    sk_ol_sigmaV=sk_ol1[:,1]*ann_factor*(2.3e23/J_allsky)
    
    #SuperK analysis conducted in Katarzyna Frankiewicz PhD thesis at https://arxiv.org/abs/1510.07999
    SK_data1=np.loadtxt('Data_Files/SK_katarzyna.csv',delimiter=',')
    SK_mass1=SK_data1[:,0]
    SK_sigmaV1=SK_data1[:,1]*ann_factor*(2.3e23/J_allsky)
    
    #HyperK analysis conducted by (Bell et al.) found at https://arxiv.org/pdf/2005.01950.pdf
    hk=np.loadtxt('Data_Files/HK_sigmaV2.csv',delimiter=',')
    hk_mass=hk[:,0]
    hk_sigmaV=hk[:,1]*4*ann_factor*(2.3e23/J_allsky)    #cross section is multiplied by 4 due to 20 yr exposure in their paper (all cross sections calculated within DANDAS are for a 5 yr exposure time)

    #IceCube DeepCore analysis in https://arxiv.org/abs/2107.11224
    IC_deep=np.loadtxt('Data_Files/IC_Antares_best_sigmaV_lims.csv',delimiter=',')  #this is an erroneously titled data file, the data is not associated with the ANTARES neutrino experiment
    IC_deep_mass=IC_deep[:,0]
    IC_deep_sigmaV=IC_deep[:,1]*ann_factor*(2.3e23/J_allsky)
    
    #ANTARES neutrino telescope analysis in https://arxiv.org/abs/1612.04595
    ant_data_alb=np.loadtxt('Data_Files/Antares_alb.csv',delimiter=',')
    ant_mass2=ant_data_alb[:,0]
    ant_sigmaV=ant_data_alb[:,1]*ann_factor*(2.3e23/J_allsky)
    
    #IceCube neutrino observatory combined analysis from https://arxiv.org/pdf/1606.00209.pdf and https://arxiv.org/pdf/1705.08103.pdf
    IceCube=np.loadtxt("Data_Files/IceCube_sigmaV_fig2.csv",delimiter=',')
    IC_mass=IceCube[:,0]
    IC_sigmaV=IceCube[:,1]*ann_factor*(2.3e23/J_allsky)
    
    
    #KM3NET experiment data from https://pos.sissa.it/358/552/pdf
    km3_data=np.loadtxt('Data_Files/KM3NET_sigmaV.csv',delimiter=',')
    km3_mass=km3_data[:,0]  
    km3_sigmaV=(km3_data[:,1]/np.sqrt(5))*ann_factor*(2.3e23/J_allsky)        #sqrt(5) is so cross section corresponds to a 5 yr exposure
    
    #PONE experiment data from https://arxiv.org/abs/1912.09486
    P1_data=np.loadtxt('Data_Files/PONE_sigmaV.csv',delimiter=',')
    P1_mass=P1_data[:,0]    #Indexing is based on length of sigmaV dataset
    P1_sigmaV=P1_data[:,1]*ann_factor*(2.3e23/J_allsky)
    
    #Icecube neutrino experiment  analysis of annihilation to electron neutrinos in https://arxiv.org/abs/1903.12623
    IC_bhat_nue=np.loadtxt('Data_Files/IC_bhat_nue_sigmaV.csv',delimiter=',')
    IC_bhat_nue_mass=IC_bhat_nue[:,0]*1000000 #converting from PeV -> GeV
    IC_bhat_nue_sigmaV=(IC_bhat_nue[:,1]*2/3)*ann_factor*(2.3e23/J_allsky)    #factor of 1/3 is due to their assumption of single channel decay (I assume equal branching ratio all flavors)
    
    if reduce_runtime==False:
        #GRAND experiment analysis from https://arxiv.org/abs/1912.09486
        grand_data=np.loadtxt('Data_Files/NewGRAND.txt',delimiter=',')
        grand_mass=grand_data[:,0]
        grand_sigmaV=grand_data[:,1]*ann_factor*(0.28e23/J_grand)

    #RNO-G data from https://arxiv.org/abs/1912.09486
    rnogdata=np.loadtxt('Data_Files/newRNOG.txt',delimiter=',')
    rnog_mass=rnogdata[:,0]
    rnog_sigmaV=(rnogdata[:,1]/2)*ann_factor*(2.3e23/J_allsky)     #Due to factor of 2 issue in previously digitized data
    
    #IceCube Extra High Energy (EHE) analysis in https://arxiv.org/abs/1912.09486
    icehe_data=np.loadtxt("Data_Files/NewICEHE.txt",delimiter=',')
    icehe_mass=icehe_data[:,0]
    icehe_sigmaV=icehe_data[:,1]*ann_factor*(2.3e23/J_allsky)
    
    if reduce_runtime==False:

        #Importing TAMBO annihilation cross section data from https://arxiv.org/abs/1912.09486 
        tamboData=np.loadtxt('Data_Files/stambo.txt',delimiter=',')
        tambo_Mass_ann=tamboData[:,0] 
        tambo_sigmaV=(tamboData[:,1]/2)*ann_factor*(0.0009e23/J_tambo)  #due to factor of 2 issue in previously digitized data


    
    
    
    


    ''' Calculating Lifetime limits based on data from different experiments '''
    
    #importing Borexino flux data (Fig. 4 in arxiv: 1909.02422v1)
    bor_data2=np.loadtxt("Data_Files/borexino_data.dat")
    BorData=(np.transpose(bor_data2))
    bor_mass2=BorData[0]*2/1000                    #to convert MeV to GeV, factor of 2 because 1 DM -> 2 nu
    bor_flux2=BorData[1]*2     #Adding factor of 2 so decay limit is per flavor (not just nue_bar)


    bor_tau2=diff_flux2Lifetime(bor_mass2, bor_flux2, D_allsky)


    #KamLand is a bit more of a process as it involves the combination two separate analyses
    #Importing Kamland flux data (also in arxiv: 1909.02422v1)
    KamData2=np.loadtxt('Data_Files/kamland_dphidE.dat')   
    Kamland=np.transpose(KamData2)
    kam_mass2=Kamland[0]*2/1000                      #to convert MeV to GeV                 
    kam_flux2=Kamland[1]*2                          #same flavor detection factor as for Borexino  

    #Importing updated Kamland Flux limits
    KamData3=np.loadtxt('Data_Files/Kamland_updated_flux.csv',delimiter=',')
    kam_mass3=KamData3[:,0]*2/1000 
    kam_flux3=KamData3[:,1]*2 

    #Calculating lifetimes using flux2Lifetime equation
    kam_tau2=(1/(4*np.pi))*(1/kam_flux2)*(1/(3*kam_mass2))*2*D_allsky
    kam_tau3=(1/(4*np.pi))*(1/kam_flux3)*(1/(3*kam_mass3))*2*D_allsky


    #Binning non-updated kamland lifetime data
    kam_bin2=interp1d(kam_mass2,kam_tau2, kind='nearest')
    kam_mass_interp2=np.linspace(kam_mass2[0],kam_mass2[-1],100)
    kam_tau_older=kam_bin2(kam_mass_interp2)

    #Binning updated kamland lifetime data
    kam_bin3=interp1d(kam_mass3,kam_tau3, kind='nearest')
    kam_mass_interp3=np.linspace(kam_mass3[0],kam_mass3[-1],100)
    kam_tau_updated=kam_bin3(kam_mass_interp3)


    #Determining strongest limits from mix of old and updated data
    kam_mass_mix=np.linspace(kam_mass2[0],kam_mass2[-1],10000)
    kam_best_tau_lims=np.zeros(10000)

    for i in range(0,10000):
        #Accouting for the fact that non-updated data covers a wider mass range
        if kam_mass_mix[i]<=kam_mass3[0] or kam_mass_mix[i]>=kam_mass3[-1]:
            kam_best_tau_lims[i]=kam_bin2(kam_mass_mix[i])

        else:

            if kam_bin2(kam_mass_mix[i]) > kam_bin3(kam_mass_mix[i]):
                kam_best_tau_lims[i]=kam_bin2(kam_mass_mix[i])
            else:
                kam_best_tau_lims[i]=kam_bin3(kam_mass_mix[i])

    '''
    #Calculating lifetime limit directly from data in https://arxiv.org/pdf/2002.06475.pdf
    '''
    #Importing acceptance and N_background data from https://arxiv.org/pdf/2002.06475.pdf
    tambo_accept=np.loadtxt('Data_Files/tambo_acceptance_u.dat',skiprows=1,delimiter=',')
    tambo_acc=tambo_accept[:,1]

    tambo_data=np.loadtxt('Data_Files/Tambo_Nbk.csv',delimiter=',')

    Tambo_binCentre_log=tambo_data[:,0] #bin centres in units of log10(E/eV), bins have a logarithmic width of 0.5
    Tambo_binCentres=(10**Tambo_binCentre_log)*1e-9     #Converting from log10(E/eV) units to GeV
    Tambo_Nbk=tambo_data[:,1]*(5/3)     #Expected number of tau neutrino events over 3 yrs within each bin (converted to be over 5 yrs)


    #Finding the number of DM events that are 2 sigma above null hypothesis (no DM) using poisson cdf

    tambo_mvec=Tambo_binCentres 

    Ndm_tambo=np.zeros(len(tambo_mvec))    #number of DM events  
    phi_tambo=np.zeros(len(tambo_mvec))    #Flux of neutrinos due to DM

    tambo_exposure=60*60*24*365.25*3   #Their background is given for a 3 year exposure
    
    import warnings   #This command is used so that warnings related to precision of optiization are ignored (as error is negligible compared to confidence level of projected sensitivity)
    warnings.filterwarnings("ignore")

    #Finding flux of neutrinos due to DM at each mass given for background model
    for i in range(0,len(tambo_mvec)):
        func1=lambda y: (poisson.cdf(Tambo_Nbk[i],Tambo_Nbk[i]+y)-.05)**2
        yopt_t = scipy.optimize.fmin(func=func1, x0=[0],disp=False)
        Ndm_tambo[i]=yopt_t[0] 
        #calculating the expected flux 
        phi_tambo[i]=Ndm_tambo[i]/(tambo_acc[i]*tambo_exposure*4*np.pi)   #has units of cm^-2 s^-1 sr^-1 (maybe has a GeV^-1 due to binning?)



    tambo_mass=tambo_mvec*2 #due to decay assumptions 
    if reduce_runtime==False:
        tambo_tau_d=diff_flux2Lifetime(tambo_mass,phi_tambo,D_tambo)  

    if reduce_runtime==False:
        #Importing P-ONE Data
        P1_data=np.loadtxt('Data_Files/PONE.txt')

        P1_mass2=P1_data[1:49]   
        #P1_sigmaV=P1_data[51:99]


        #importing P-ONE effective area data
        P1_aeffs=np.loadtxt('Data_Files/PONEAeffs.dat')

        #Saving effective area for each angular range. First and last data points are omitted as they are zero
        aeff1=P1_aeffs[1:-1,1]
        aeff2=P1_aeffs[1:-1,2]
        aeff3=P1_aeffs[1:-1,3]
        aefftot=aeff1+aeff2+aeff3

        J_P1=(J_pone1*aeff1/aefftot + J_pone2*aeff2/aefftot + J_pone3*aeff3/aefftot)*aefftot
        D_P1=(D_pone1*aeff1/aefftot + D_pone2*aeff2/aefftot + D_pone3*aeff3/aefftot)*aefftot

        P1_mass_ann=P1_mass2     #labelling this "annihilation mass" as it is not doubled (2 nu -> 2 DM assumption for annihilation)
        P1_mass_decay=P1_mass2*2 #due to decay process (1 DM -> 2 nu)

        #Calculating lifetimes using method based off "PONEBackgroundBit.mat" code

        def icresinterp(E):

            a=1.259
            b=0.7172  #currently unknown where these values are from
            c=1.198
            logsig=a-np.tanh(b*np.log10(E)-c)
            return logsig


        gam = 2.28     #alpha value for power law of flux as a function of energy
        phinormastro=1.44e-18  
        #P1_mvec=np.logspace(P1_mass[0],P1_mass[-1],48) #Goes from approximately 10^3 to 10^7 over 48 points (to match D array)

        P1_mvec=P1_mass_ann  #calculating lifetimes using decay range (doubled M_x from annihilation paper)

        gam_atm=3.39  #value from Aaron's code
        phinormAtm=10**(-13.93) # from https://icecube.wisc.edu/news/view/317

        #Creating function that calculates effective area at a specific energy by weighing contribution of each line
        getaeff=interp1d(P1_mass_ann,np.pi*aeff1+2*np.pi*aeff2+np.pi*aeff3)


        #Defining energy resolution function 
        def Eresfunc(E,Ei):
            '''
            #Uses resolution to smooth over data in each bin using a Gaussian method
            '''
            term1=np.log10(E/Ei)**2
            term2=2*icresinterp(Ei)**2
            return np.exp(-(term1/term2))

        #Initializing necessary arrays
        FluxinBin_astro=np.zeros(len(P1_mvec))
        FluxinBin_atm=np.zeros(len(P1_mvec))
        P1_tau_sanitycheck=np.zeros(len(P1_mvec))

        Emin=P1_mass_ann[0]
        Emax=P1_mass_ann[-1]

        #Astrophysical flux equations
        dphiAstrodE = lambda E: phinormastro*(E/1e5)**-gam
        dphiAtmdE= lambda E: phinormAtm*(E/1e4)**-gam_atm

        exposure = 5*525600*60 #5 years



        #Calculating expected astrophysical and atmospheric neutrino background signal
        for i in range(0,len(P1_mvec)):
            Ei=P1_mvec[i]
            energy_resolution= lambda E: Eresfunc(E,Ei)
            integrand=lambda E: dphiAstrodE(E)*energy_resolution(E)*getaeff(E)
            integrand_atm=lambda E: dphiAtmdE(E)*energy_resolution(E)*getaeff(E)
            FluxinBin_raw1=integrate.quad(integrand, Emin, Emax)
            FluxinBin_raw2=integrate.quad(integrand_atm, Emin, Emax)
            FluxinBin_astro[i]=FluxinBin_raw1[0]                               #omitting error term output by scipy quad
            FluxinBin_atm[i]=FluxinBin_raw2[0]  
        #P1_tau_sanitycheck[i]=(1/(4*np.pi))*(1/FluxinBin[i])*(1/(3*P1_mvec[i]))*2*D_P1[i]

        #Calculating number of neutrino events based on flux and exposure
        NinBin_atm=FluxinBin_atm*exposure
        NinBin_astro=FluxinBin_astro*exposure


        #Finding the number of DM events that are 2 sigma above null hypothesis (no DM)
        N_DM=np.zeros(len(P1_mvec))    

        for i in range(0,len(P1_mvec)):
            func1=lambda y: (poisson.cdf(NinBin_astro[i]+NinBin_atm[i],NinBin_astro[i]+NinBin_atm[i]+y)-.05)**2
            yopt = scipy.optimize.fmin(func=func1, x0=[0],disp=False)
            N_DM[i]=yopt[0]


        #Calculating lifetime limit using equation 1 in 'DM Decay to Neutrinos'
        P1_flux_direct=N_DM/exposure  

        tau_P1_direct=(1/(4*np.pi*P1_flux_direct*3*(P1_mass_decay)))*2*(D_pone1*aeff1 + D_pone2*aeff2 + D_pone3*aeff3)
        svlim = (4*np.pi*2*3*(P1_flux_direct)*(P1_mass_ann)**2)/(2*(J_pone1*aeff1 + J_pone2*aeff2 + J_pone3*aeff3))

    ant_mass=ant_mass2*2  #factor of 2 for decay

    #The previously published data was rescaled to a lifetime limit
    tau_ant=rescaling(ant_mass,ant_sigmaV,J_allsky,D_allsky)


    #JUNO target number calculation
    D_allsky=2.6470090649443034*10**23

    avogadro_number = 6.022140857e23
    ton = 1.0e6 # grams
    kT = 1.0e3*ton
    mass = 17.*kT
    O_atoms = 1.2e33
    years = 365.*24.*60.*60.
    time_of_observation = 10.*years

    juno = np.loadtxt('Data_Files/juno_background_events.csv', delimiter = ',')

    juno_bgspl = UnivariateSpline(juno[:,0], np.log10(juno[:,1]*juno[:,0]), k = 5, s = 0.001)

    def get_background_expectation(energy):
        energy*=1e3
        if(energy < juno[:,0].min()):
            return 0
        elif(energy > juno[:,0].max()):
            return 0
        else:
            return(10**juno_bgspl(energy))

    juno_xs = np.loadtxt('Data_Files/ibd_xs_juno.txt', delimiter = ' ')
    energy = juno_xs[:,0]
    xs = juno_xs[:,1]
    
    
    
    xs_spline = UnivariateSpline(np.log10(energy), np.log10(xs), k=5, s=0.1)
    splmin = energy.min()
    splmax = energy.max()


    def get_signal_expectation(D_allsky, tau, E_nu):
        kappa = 2

        mx=E_nu*2

        constant=(1./(4.*np.pi)) * (1/(3.*mx*tau))*2*D_allsky/(2)
        
        e_depo = mx/2 #- shift*1e-3 (HW: this line is not used)

        signal_spectrum = time_of_observation * O_atoms * constant * 10**xs_spline(np.log10(E_nu*1e3))

        return signal_spectrum

    def log_poisson(k, mu):

        ''''''
        ln_poisson = -mu + k * np.log(mu) - gammaln(k+1)
        return ln_poisson.sum()    


    def fit_d_2llh(tau, E_nu):
        #mx=mx*2
        signal = get_signal_expectation(D_allsky, tau, E_nu)
        expectation = get_background_expectation(E_nu)
        d_2llh = -2. * ( log_poisson(signal + expectation, expectation) - log_poisson(expectation, expectation))

        return d_2llh - 2.71 #Could you check if this is the right change in -2 ln(L)for 90% CL 


    jmxs = np.linspace(9, 38, 100)
    jmxs/=1e3
    J_allSky = 2.23e3 # GeV^2 cm^-5 # s-channel

    jsigmav_list = []
    #ds_list = []
    for E_nu in jmxs:
        jsigmav_list.append(fsolve(fit_d_2llh, 1e22, args = (E_nu))[0])

    #Changing variable names for clarity
    juno_mass=jmxs*2  #factor of 2 is for decay
    juno_tau=jsigmav_list




    #Importing GRAND data from annihilation paper
    grand_data=np.loadtxt('Data_Files/NewGRAND.txt',delimiter=',')
    grand_mass=grand_data[:,0]*2
    grand_sigmaV=grand_data[:,1]

    #Importing flux data from fig.4 in https://arxiv.org/pdf/1810.09994.pdf
    grand=np.loadtxt('Data_Files/GRAND_flux.csv',delimiter=',')
    E_grand=grand[:,0]
    grand_flux=(grand[:,1]/(E_grand**2))/3  #converting all flavor flux to per flavor flux
    grand_DM_mass=E_grand*2

    #Calculating lifetime limit based on binned flux data (with alpha=1 as the assumed power spectrum)
    if reduce_runtime==False:
        grand_tau_d=bin_f2l_alpha1(1,grand_DM_mass,grand_flux,D_grand)


    #Importing annihilation cross section limits from https://arxiv.org/abs/1912.09486
    dune_dat=np.loadtxt('Data_Files/DUNE_sigmaV.csv',delimiter=',')
    dune_mass=dune_dat[:,0]*2
    dune_sigmaV=dune_dat[:,1]*ann_factor

    #Rescaling annihilation limits from https://arxiv.org/abs/1912.09486
    dune_tau=rescaling(dune_mass,dune_sigmaV,J_allsky,D_allsky)


    #Importing cross section from https://arxiv.org/pdf/1606.00209.pdf (Aartsen 2016a)
    IC_2016a=np.loadtxt("Data_Files/IceCube_sigmaV_2016a.csv",delimiter=',')
    IC_2016a_mass=IC_2016a[:,0]
    IC_2016a_sigmaV=IC_2016a[:,1]

    #Importing cross section from Table 2 in https://arxiv.org/pdf/1705.08103.pdf (Aartsen 2017b)
    IC_2017b=np.loadtxt("Data_Files/IC_2017b_table2_sigmaV.csv",delimiter=',')
    IC_2017b_mass=IC_2017b[:,0]
    IC_2017b_sigmaV=IC_2017b[:,1]

    #appending both data sets so that the analyses can be combined into a signle line
    IC_best_sigmaV_lims=np.append(IC_2017b_sigmaV[0:7],IC_2016a_sigmaV[3:])
    IC_best_mass=np.append(IC_2017b_mass[0:7],IC_2016a_mass[3:])

    #rescaling annihilation limits to decay limits
    IC_decay_mass=IC_best_mass*2
    tau_IC=rescaling(IC_decay_mass,IC_best_sigmaV_lims,J_allsky,D_allsky)


    #Importing annihilation cross section data from conference proceedings ( https://arxiv.org/abs/2107.11224 )
    IC_ant=np.loadtxt('Data_Files/IC_Antares_best_sigmaV_lims.csv',delimiter=',')
    IC_ant_mass=IC_ant[:,0]*2
    IC_ant_sigmaV=IC_ant[:,1]

    IC_ant_tau=rescaling(IC_ant_mass,IC_ant_sigmaV,J_allsky,D_allsky)


    #Importing flux data from https://arxiv.org/pdf/1807.01820.pdf
    ic_ehe=np.loadtxt('Data_Files/IC_EHE_energy_flux.csv',delimiter=',')
    ic_ehe_Enu=ic_ehe[:,0]
    ic_ehe_flux=(ic_ehe[:,1]/3)/ic_ehe_Enu**2  #Their flux is divided by 3 as they show the ALL FLAVOR flux (and we want flux per flavor so we can get lifetime per flavor)
    ic_ehe_mass=ic_ehe_Enu*2

    ic_ehe_tau=bin_f2l_alpha1(1,ic_ehe_mass,ic_ehe_flux,D_allsky)


    #importing data and removing points that are 0
    final_data=np.loadtxt('Data_Files/final_final_final_s_wave_Mmin_-3.0_halo_Sergio_combined_limits_glaactic.txt')
    final_massFull=final_data[0] #mass in GeV
    final_mass=final_massFull[13:-7]*2
    final_sigmaVFull=final_data[1]
    final_sigmaV=final_sigmaVFull[13:-7]/4  #look in experimentcomparisons file for proof this is necessary


    #Importing IC Bhattachrya's DM -> nue decay lifetime line
    IC_bhat_nue=np.loadtxt('Data_Files/IC_bhat_nue_tau.csv',delimiter=',')
    IC_bhat_nue_mass2=(IC_bhat_nue[:,0]*1000000)*2 #converting from PeV -> GeV
    IC_bhat_nue_tau=IC_bhat_nue[:,1]/3    #factor of 1/3 is due to their assumption of single channel decay

    #Calculating rescaled lifetime limits from annihilation limits
    alpha=2    #power law associated with these neutrino fluxes
    tau_final=rescaling(final_mass,final_sigmaV,J_allsky,D_allsky)

    #from table G.1 in HESE 7.5 year data paper
    IC_astro_delta=np.array([log_width_bin(4.2e4,8.83e4),log_width_bin(8.83e4,1.86e5),log_width_bin(1.86e5,3.91e5),log_width_bin(3.91e5,8.23e5),log_width_bin(8.23e5,1.73e6),log_width_bin(1.73e6,3.64e6),log_width_bin(3.64e6,7.67e6)])

    #Calculating SK lifetime limits directly from flux data in Fig 1 of annihilation paper
    #Note: I'm using both eqn. 14 in overleaf and classic lifetime equation
    SK_nue2=np.loadtxt('Data_Files/Fig1_SK_nue.csv',delimiter=',')
    SK_nue_mass2=SK_nue2[:,0]
    SK_nue_flux2=SK_nue2[:,1]/(SK_nue_mass2**2)  #as data is given in E^2 times flux
    SK_nue_mass=SK_nue_mass2*2          #factor of 2 for decay

    #importing bin edges shown in Fig. 1 of annihilation paper (horizontal error bars)
    SK_nue_bin_data = np.loadtxt('Data_Files/SK_nue_bin_edges.csv', delimiter=',')
    SK_nue_bin_edges = np.log10(SK_nue_bin_data[:,0])
    SK_nue_delta=np.zeros(len(SK_nue_bin_edges)-1)
    for i in range(0,len(SK_nue_bin_edges)-1):
        SK_nue_delta[i]=SK_nue_bin_edges[i+1]-SK_nue_bin_edges[i]


    #Calculating IceCube limits (using fig. 1 data) using overleaf method
    IceCube_Astro=np.loadtxt('Data_Files/IceCube_AstrophysicalFlux.csv',delimiter=',')
    IC_astro_mass=IceCube_Astro[:,0]

    IC_astro_uppersigma_data=np.loadtxt('Data_Files/IC_astro_uppersigma.csv',delimiter=',')
    IC_a_sigma_data=IC_astro_uppersigma_data[:,1]
    IC_astro_sigma=np.zeros(7)
    k=0
    for i in range(0,len(IC_astro_sigma),2):
        IC_astro_sigma[k]=IC_a_sigma_data[i+1]-IC_a_sigma_data[i]
        k=k+1

    IC_astro_flux=(IceCube_Astro[:,1]+1.64*IC_astro_sigma)/(IC_astro_mass**2)
    IC_astro_mass=IceCube_Astro[:,0]*2

    IC_astro_tau3=bin_flux2Lifetime(alpha,IC_astro_delta,IC_astro_mass,IC_astro_flux,D_allsky)

    #Calculating IC astro lifetime using table G.1. (Bayesian Analysis) in HESE 7.5 year data paper 
    IC_astro_E=np.array([log_centre_bin(4.2e4,8.83e4),log_centre_bin(8.83e4,1.86e5),log_centre_bin(1.86e5,3.91e5),log_centre_bin(3.91e5,8.23e5),log_centre_bin(8.23e5,1.73e6),log_centre_bin(1.73e6,3.64e6),log_centre_bin(3.64e6,7.67e6)])
    IC_astro_DM_mass=IC_astro_E*2
    IC_astro_flux_G1=np.array([1.3e-17,3.9e-18,8.6e-20,1.9e-21,7.9e-21,7.2e-22,5.4e-24])  
    IC_astro_flux_uppersigma=np.array([1.2e-17,1.5e-18,20e-20,47e-21,12e-21,28e-22,280e-24])
    IC_astro_flux_upperlim=(IC_astro_flux_G1+1.64*IC_astro_flux_uppersigma)/3 #factor of 1/3 as it gives flux to ALL flavors 
    IC_astro_delta=np.array([log_width_bin(4.2e4,8.83e4),log_width_bin(8.83e4,1.86e5),log_width_bin(1.86e5,3.91e5),log_width_bin(3.91e5,8.23e5),log_width_bin(8.23e5,1.73e6),log_width_bin(1.73e6,3.64e6),log_width_bin(3.64e6,7.67e6)]) 
    IC_astro_tau_G1=bin_flux2Lifetime(alpha,IC_astro_delta,IC_astro_DM_mass,IC_astro_flux_upperlim,D_allsky)

    #Calculating IC_nue delta array (assuming each point is in the direct centre of the bin)
    IC_nue=np.loadtxt('Data_Files/IceCube_nue.csv',delimiter=',')
    IC_nue_mass2=IC_nue[:,0]
    IC_nue_flux2=IC_nue[:,1]/(IC_nue_mass2**2)
    IC_nue_mass=IC_nue_mass2*2

    IC_nue_delta=np.zeros(len(IC_nue_flux2))
    for i in range(0,len(IC_nue_flux2)-1):
        IC_nue_delta[i]=log_width_bin(IC_nue_mass2[i],IC_nue_mass2[i+1])


    #Calculating IC_nue delta values (recognizing that points are not at centre of bin)
    IC_nue_bin_edges=np.loadtxt('Data_Files/IC_nue_bin_edges.csv', delimiter=',')  
    IC_nue_bin_edges=np.log10(IC_nue_bin_edges[:,0])  #ignoring flux component of array
    IC_nue_upper_delta=np.zeros(4)  #array that will contain distance from most likely value to upper edge of bin
    IC_nue_lower_delta=np.zeros(4)  #array that will contain distance from most likely value to lower edge of bin

    k=0
    for i in range(0,len(IC_nue_bin_edges),2):
        IC_nue_upper_delta[k]=IC_nue_bin_edges[i+1]-np.log10(IC_nue_mass2[k])  #Not sure if I should be using E_nue (called IC_nue_mass2) or m_DM (which is equal to 2*E_nue)
        IC_nue_lower_delta[k]=np.log10(IC_nue_mass2[k])-IC_nue_bin_edges[i]
        k=k+1 #used so mass array only increments by 1

    #Calculating upper limit of IC_nue flux data shown in figure 1
    IC_nue_upper_sigma=np.loadtxt('Data_Files/IC_nue_upper_sigma.csv',delimiter=',')
    IC_nue_upper_sigma=IC_nue_upper_sigma[:,1]
    IC_nue_sigma=np.zeros(4)

    j=0
    for i in range(0,len(IC_nue_upper_sigma),2):
        IC_nue_sigma[j]=(IC_nue_upper_sigma[i+1]-IC_nue_upper_sigma[i])/(IC_nue_mass2[j]**2)
        j=j+1

    IC_nue_flux_upperlim=IC_nue_flux2+1.64*IC_nue_sigma

    #calculating lifetimes using variable delta method
    IC_nue_tau_vardelta=bin_f2l_variable_delta(2,IC_nue_upper_delta,IC_nue_lower_delta,IC_nue_mass,IC_nue_flux_upperlim,D_allsky)


    #Applying  eqn. 14 in overleaf
    alpha=2
    delta=0.2 #logarithmic bin width of about 0.2 log(E/GeV)
    SK_nue_tau2=bin_flux2Lifetime(alpha,SK_nue_delta,SK_nue_mass,SK_nue_flux2,D_allsky)
    IC_nue_tau2=bin_flux2Lifetime(alpha,delta,IC_nue_mass2,IC_nue_flux2,D_allsky)

    #Appending IC nue, SK nue, and IC astro to make one solid line

    combo_mass1=np.append(SK_nue_mass,IC_nue_mass)
    combo_mass_tot=np.append(combo_mass1,IC_astro_DM_mass)
    combo_tau1=np.append(SK_nue_tau2,IC_nue_tau_vardelta)
    combo_tau_tot=np.append(combo_tau1,IC_astro_tau_G1)
    combo_mass_binned,combo_tau_binned=binning(combo_mass_tot,combo_tau_tot)

    #Binning flux data (for IC nue, astro, and SK atm)
    IC_nue_mass_bin,IC_nue_flux_bin=binning(IC_nue_mass2,IC_nue_flux2)
    #IC_astro_mass_bin,IC_astro_flux_bin=binning(IC_astro_mass,IC_astro_flux)
    SK_nue_mass_bin,SK_nue_flux_bin=binning(SK_nue_mass,SK_nue_flux2)

    IC_nue_tau_interp=bin_flux2Lifetime(alpha,delta,IC_nue_mass_bin,IC_nue_flux_bin,D_allsky)
    #IC_astro_tau_interp=bin_flux2Lifetime(alpha,delta,IC_astro_mass_bin,IC_astro_flux_bin,D_allsky)
    SK_nue_tau_interp=bin_flux2Lifetime(alpha,delta,SK_nue_mass_bin,SK_nue_flux_bin,D_allsky)




    #This involves a lot of code as it is also a combination of old (labelled 4) and new (labelled 5) limits
    #Importing strongest (lowest) flux limits from fig. 6.6 in Linyan thesis
    Sk_nuebarData4=np.loadtxt('Data_Files/Linyan_strongest_flux_lims.csv', delimiter=',')   
    Sk_nuebarData4=np.transpose(Sk_nuebarData4)
    Sk_nuebar_mass4=Sk_nuebarData4[0]*2/1000                    #decay factor of 2, MeV conversion                
    Sk_nuebar_flux4=Sk_nuebarData4[1]*2                    #same flavor detection factor as for Borexino 

    #Importing updated SK flux limits from fig. 25 in https://arxiv.org/pdf/2109.11174.pdf
    Sk_nuebarData5=np.loadtxt('Data_Files/SK_2021_obs_flux.csv',delimiter=',')   
    Sk_nuebarData5=np.transpose(Sk_nuebarData5)
    Sk_nuebar_mass5=Sk_nuebarData5[0]*2/1000                    #decay factor of 2, MeV conversion                
    Sk_nuebar_flux5=Sk_nuebarData5[1]*2  


    SK_nuebar_tau4=diff_flux2Lifetime(Sk_nuebar_mass4,Sk_nuebar_flux4,D_allsky)
    SK_nuebar_tau5=diff_flux2Lifetime(Sk_nuebar_mass5,Sk_nuebar_flux5,D_allsky)

    #Binning SK data
    binning_func=interp1d(Sk_nuebar_mass4,SK_nuebar_tau4, kind='nearest')
    binning_func5=interp1d(Sk_nuebar_mass5,SK_nuebar_tau5, kind='nearest')

    sk_mass_interp=np.linspace(Sk_nuebar_mass4[0],Sk_nuebar_mass4[-1],100)
    sk_mass_interp5=np.linspace(Sk_nuebar_mass5[0],Sk_nuebar_mass5[-1],100)

    #Determining strongest limits from mix of old and updated data
    sk_mass_mix=np.linspace(sk_mass_interp5[0],sk_mass_interp5[-1],1000)
    sk_best_tau_lims=np.zeros(1000)

    for i in range(0,1000):
        if binning_func(sk_mass_mix[i])>binning_func5(sk_mass_mix[i]):
            sk_best_tau_lims[i]=binning_func(sk_mass_mix[i])
        else:
            sk_best_tau_lims[i]=binning_func5(sk_mass_mix[i])


    #Importing flux data from pg 35 of updated paper: https://arxiv.org/pdf/2010.12279.pdf
    RNOG_a=np.loadtxt('Data_Files/RNOG_Aaron.csv',delimiter=',')
    rnog_a_mass=RNOG_a[:,0]*2  #for decay
    rnog_a_flux=(RNOG_a[:,1]/(RNOG_a[:,0]**2))/3 #1/3 is to convert an all flavor flux to a per flavor flux

    rno_delta=1  #due to one decade energy binning
    #Using flux data from updated paper: https://arxiv.org/pdf/2010.12279.pdf
    rno_tau_a=bin_f2l_alpha1(rno_delta,rnog_a_mass,rnog_a_flux,D_allsky)


    #Using data from https://arxiv.org/pdf/2005.01950.pdf
    hk_mass_d=hk[:,0]*2
    hk_tau_d=rescaling(hk_mass_d,hk_sigmaV,J_allsky,D_allsky)


    #Using annihilation cross section data from https://arxiv.org/abs/1510.07999
    SK_mass2=SK_mass1*2

    SK_tau1=rescaling(SK_mass2,SK_sigmaV1,2.3e23,D_allsky)


    #Importing annihilation cross section data from https://pos.sissa.it/358/552/pdf
    km3_mass2=km3_data[:,0]*2  #for decay of 1 DM to 2 neutrinos

    km3_tau=rescaling(km3_mass2,km3_sigmaV,J_allsky,D_allsky)


    #Using annihilation cross section data from http://etheses.dur.ac.uk/13142/1/PhD_thesis_Olivares_FINAL.pdf?DDD25+
    sk_ol_mass2=sk_ol_mass*2

    #Rescaling annihilation limit to lifetime limit
    sk_ol_tau=rescaling(sk_ol_mass2,sk_ol_sigmaV,J_allsky,D_allsky)


    #Importing data from Figure 5 in https://arxiv.org/pdf/1911.02561.pdf
    gen2=np.loadtxt('Data_Files/IC_gen2_E2flux.csv',delimiter=',')
    gen2_E=gen2[:,0]   
    gen2_flux=gen2[:,1]/(3*gen2_E**2)   #factor of 3 because they show an all flavour flux
    gen2_mass=gen2_E*2

    gen2_tau=bin_f2l_alpha1(1,gen2_mass,gen2_flux,D_allsky)

    #importing data from fig 6
    gen2_6=np.loadtxt('Data_Files/IC_Gen2_Fig6.csv',delimiter=',')
    gen2_E_6=gen2_6[:,0]   
    gen2_flux_6=(gen2_6[:,1]/(gen2_E_6**2))*624.151   #factor of 624 to convert erg -> GeV
    gen2_mass_6=gen2_E_6*2

    gen2_tau_6=bin_flux2Lifetime(2, 1,gen2_mass_6,gen2_flux_6,D_allsky)

    #Importing annihilation cross section from DM annihilation paper
    gen2_r=np.loadtxt('Data_Files/IC_gen2_sigmaV.csv',delimiter=',')
    gen2_r_mass=gen2_r[:,0]*2
    gen2_sigmaV=gen2_r[:,1]

    #rescaling annihilation
    gen2_tau_r=rescaling(gen2_r_mass,gen2_sigmaV,J_allsky,D_allsky)


    #Plotting lifetime limits if desired by the user
    if plot_preference!='No_plot':
        plt.rcParams.update({'font.size': 12})
        plt.rcParams["font.family"] = "serif"
        fsize=16
         
        #Graphics for Lifetime Limits plot
        plt.figure(1,figsize=(19,11))
        Bor_col='#1f77b4'
        plt.plot(bor_mass2,bor_tau2, label='Borexino',linewidth=1.5,color=Bor_col,alpha=0.7, ls='-')
        plt.fill_between(bor_mass2,bor_tau2, y2=1e-10,color=Bor_col,alpha=0.25,zorder=2)
        plt.text(5e-3, 1e19, 'Borexino', fontsize=fsize,color=Bor_col,clip_on=True)

        kam_col='#ff7f0e'
        plt.plot(kam_mass_mix, kam_best_tau_lims, label='KamLand',linewidth=1.5,color=kam_col,alpha=0.7, ls='-')
        plt.fill_between(kam_mass_mix, kam_best_tau_lims, y2=1e-10,color=kam_col,alpha=0.25,zorder=2)
        plt.text(7e-4, 1e22, 'KamLAND', fontsize=fsize,color=kam_col,clip_on=True)

        jun_col='#2ca02c'
        plt.plot(juno_mass,juno_tau,label='JUNO',linewidth=1.5,color=jun_col,alpha=0.7, ls='--')
        plt.text(5e-3, 6e23, 'JUNO', fontsize=fsize,color=jun_col,clip_on=True)

        sk_ol_col='#731fb4'
        plt.plot(sk_ol_mass2,sk_ol_tau,label='SK (Olivares et al.)',linewidth=1.5,color=sk_ol_col,alpha=0.7, ls='-')
        plt.fill_between(sk_ol_mass2,sk_ol_tau, y2=1e-10,color=sk_ol_col,alpha=0.25,zorder=2)
        plt.text(4e-2, 2e21, 'SK (Olivares et al.)', fontsize=fsize,color=sk_ol_col,clip_on=True)

        sknu_col='#d62728'
        plt.plot(sk_mass_mix,sk_best_tau_lims, label='$SK -\overline {v_e}\ $',linewidth=1.5,color=sknu_col,alpha=0.7, ls='-')
        plt.fill_between(sk_mass_mix,sk_best_tau_lims, y2=1e-10,color=sknu_col,alpha=0.25,zorder=2)
        plt.text(3e-2, 3e24, '$SK -\overline {v_e}\ $', fontsize=fsize,color=sknu_col,clip_on=True)

        hk_col='#ed26b5'
        plt.plot(hk_mass_d,hk_tau_d,linewidth=1.5,color=hk_col,alpha=0.7, ls='--')
        plt.text(9e-2, 1e24, 'HK (Bell et al.) ', fontsize=fsize,color=hk_col,clip_on=True)

        SK_col='#20B2AA'
        plt.plot(SK_mass2,SK_tau1, label='SuperK', linewidth=1.5,color=SK_col,alpha=0.7, ls='-')
        plt.fill_between(SK_mass2,SK_tau1, y2=1e-10,color=SK_col,alpha=0.25,zorder=2)
        plt.text(10, 2e24, 'SK', fontsize=fsize,color=SK_col,clip_on=True)

        IC_col='#9467bd'
        plt.plot(IC_decay_mass,tau_IC,label='IceCube', linewidth=1.5,color=IC_col,alpha=0.7, ls='-')
        plt.fill_between(IC_decay_mass,tau_IC, y2=1e-10,color=IC_col,alpha=0.25,zorder=2)
        plt.text(100, 1.5e25, 'IceCube', fontsize=fsize,color=IC_col,clip_on=True)


        ant_col='#e377c2'
        plt.plot(ant_mass,tau_ant, label='ANTARES',linewidth=1.5,color=ant_col,alpha=0.7, ls='-')
        plt.fill_between(ant_mass,tau_ant, y2=1e-10,color=ant_col,alpha=0.25,zorder=2)
        plt.text(2000, 6.5e27, 'ANTARES', fontsize=fsize,color=ant_col,clip_on=True)

        icant_col='#7f7f7f'
        plt.plot(IC_ant_mass,IC_ant_tau, label='IC-DeepCore',linewidth=1.5,color=icant_col,alpha=0.7, ls='-')
        plt.fill_between(IC_ant_mass,IC_ant_tau, y2=1e-10,color=icant_col,alpha=0.25,zorder=2)
        plt.text(10, 6.5e26, 'IC-DeepCore', fontsize=fsize,color=icant_col,clip_on=True)

        bhat_col='#d62728'
        plt.plot(IC_bhat_nue_mass2,IC_bhat_nue_tau,label='IceCube (Bhattacharya, 2019)',linewidth=1.5,color=bhat_col,alpha=0.7, ls='-')
        plt.fill_between(IC_bhat_nue_mass2,IC_bhat_nue_tau, y2=1e-10,color=bhat_col,alpha=0.25,zorder=2)
        plt.text(6e5, 3e27, 'IceCube (Bhattacharya et al.)', fontsize=fsize,color=bhat_col,clip_on=True)

        cong_col='#8c564b'
        plt.plot(combo_mass_binned, combo_tau_binned, label= 'SK atm. and IceCube-HE',linewidth=1.5,color=cong_col,alpha=0.7, ls='-')
        plt.fill_between(combo_mass_binned, combo_tau_binned, y2=1e-10,color=cong_col,alpha=0.25,zorder=2)
        plt.text(6e-1, 1e22, 'SK atm.', fontsize=fsize,color=cong_col,clip_on=True)
        plt.text(4e6, 9e28, 'IceCube-HE', fontsize=fsize,color=cong_col,clip_on=True)

        ehe_col='#17becf'
        plt.plot(ic_ehe_mass,ic_ehe_tau,label='IceCube-EHE',linewidth=1.5,color=ehe_col,alpha=0.7, ls='-')
        plt.fill_between(ic_ehe_mass,ic_ehe_tau, y2=1e-10,color=ehe_col,alpha=0.25,zorder=2)
        plt.text(3e10, 6e27, 'IceCube-EHE', fontsize=fsize,color=ehe_col,clip_on=True)

        if reduce_runtime==False:
            P1_col='#2ca02c'
            plt.plot(P1_mass_decay,tau_P1_direct,label='P-ONE',linewidth=1.5,color=P1_col,alpha=0.7, ls='--')
            plt.text(1e5, 1e29, 'P-ONE', fontsize=fsize,color=P1_col,clip_on=True)
            
            grand_col='#e377c2'
            plt.plot(grand_DM_mass,grand_tau_d,label='GRAND200k',linewidth=1.5,color=grand_col,alpha=0.7, ls='--')
            plt.text(4e8, 9e28, 'GRAND', fontsize=fsize,color=grand_col,clip_on=True)
                    
            tam_col='#731fb4'
            plt.plot(tambo_mass,tambo_tau_d,label='TAMBO',linewidth=1.5,color=tam_col,alpha=0.7, ls='--')
            plt.text(6e5, 8e26, 'TAMBO', fontsize=fsize,color=tam_col,clip_on=True)

        km3_col='#ff7f0e'
        plt.plot(km3_mass2,km3_tau,label='KM3NET',linewidth=1.5,color=km3_col,alpha=0.7, ls='--')
        plt.text(500, 5e28, 'KM3NET', fontsize=fsize,color=km3_col,clip_on=True)

        rno_col='#1f77b4'
        plt.plot(rnog_a_mass,rno_tau_a,label='RNO-G',linewidth=1.5,color=rno_col,alpha=0.7, ls='--')
        plt.text(8e9, 4e28, 'RNO-G', fontsize=fsize,color=rno_col,clip_on=True)

        dune_col='#8a8107'
        plt.plot(dune_mass,dune_tau,label='DUNE',linewidth=1.5,color=dune_col,alpha=0.7, ls='--')
        plt.text(3e-1, 2e23, 'DUNE', fontsize=fsize,color=dune_col,clip_on=True)


        if plot_preference=='Full_Plot':
            plt.xlim(1e-3,1e11)
            plt.ylim(1e18,1e30)
            
        elif plot_preference=='Custom_plot':
            plot_axes = kwargs.get('plot_axes_dec', None)
            plt.xlim(plot_axes[0],plot_axes[1])
            plt.ylim(plot_axes[2],plot_axes[3])
                
        plt.yscale("log")
        plt.xscale("log")
        plt.ylabel(r'$\tau_\chi$  [s]')
        plt.xlabel(r'$m_\chi$ [GeV]')
        
        plt.savefig("Dark Matter Lifetime Limits.pdf",dpi=200)
        
        
        #Graphics for Annihilation Cross Section Limits plot
        plt.figure(2,figsize=(19,11))
        fsize=16
        
        

        Bor_col='#1f77b4'
        plt.plot(bor_mass,bor_sigmaV, label='Borexino',linewidth=1.5,color=Bor_col,alpha=0.7, ls='-')
        plt.fill_between(bor_mass,bor_sigmaV, y2=1e-15,color=Bor_col,alpha=0.25,zorder=2)
        plt.text(bor_mass[0],bor_sigmaV[0]*5, 'Borexino', fontsize=fsize,color=Bor_col,clip_on=True)

        kam_col='#ff7f0e'
        plt.plot(kam_mass, kam_sigmaV, label='KamLand',linewidth=1.5,color=kam_col,alpha=0.7, ls='-')
        plt.fill_between(kam_mass, kam_sigmaV, y2=1e-15,color=kam_col,alpha=0.25,zorder=2)
        plt.text(kam_mass[0]/8,kam_sigmaV[0], 'KamLAND', fontsize=fsize,color=kam_col,clip_on=True)

        jun_col='#2ca02c'
        plt.plot(junoMass,junosigmaV,label='JUNO',linewidth=1.5,color=jun_col,alpha=0.7, ls='--')
        plt.text(junoMass[0]/4, junosigmaV[0]/3, 'JUNO', fontsize=fsize,color=jun_col,clip_on=True)

        sk_ol_col='#731fb4'
        plt.plot(sk_ol_mass,sk_ol_sigmaV,label='SK (Olivares et al.)',linewidth=1.5,color=sk_ol_col,alpha=0.7, ls='-')
        plt.fill_between(sk_ol_mass,sk_ol_sigmaV, y2=1e-10,color=sk_ol_col,alpha=0.25,zorder=2)
        plt.text(sk_ol_mass[-10], sk_ol_sigmaV[-10]*10, 'SK (Olivares et al.)', fontsize=fsize,color=sk_ol_col,rotation=90,clip_on=True)

        sknu_col='#d62728'
        plt.plot(Sk_nuebar_mass,Sk_nuebar_sigmaV, label='$SK -\overline {v_e}\ $',linewidth=1.5,color=sknu_col,alpha=0.7, ls='-')
        plt.fill_between(Sk_nuebar_mass,Sk_nuebar_sigmaV, y2=1e-10,color=sknu_col,alpha=0.25,zorder=2)
        plt.text(Sk_nuebar_mass[0]/4, Sk_nuebar_sigmaV[0], '$ SK -\overline {v_e}\ $', fontsize=fsize,color=sknu_col,clip_on=True)

        hk_col='#ed26b5'
        plt.plot(hk_mass,hk_sigmaV,linewidth=1.5,color=hk_col,alpha=0.7, ls='--')
        plt.text(8e-2, 2e-25, 'HK (Bell et al.) ', fontsize=fsize,color=hk_col,clip_on=True)

        SK_col='#20B2AA'
        plt.plot(SK_mass1,SK_sigmaV1, label='SuperK', linewidth=1.5,color=SK_col,alpha=0.7, ls='-')
        plt.fill_between(SK_mass1,SK_sigmaV1, y2=1e-10,color=SK_col,alpha=0.25,zorder=2)
        plt.text(SK_mass1[0], SK_sigmaV1[0]*3, 'SuperK', fontsize=fsize,color=SK_col,clip_on=True)

        IC_col='#9467bd'
        plt.plot(IC_mass,IC_sigmaV,label='IceCube', linewidth=1.5,color=IC_col,alpha=0.7, ls='-')
        plt.fill_between(IC_mass,IC_sigmaV, y2=1e-10,color=IC_col,alpha=0.25,zorder=2)
        plt.text(IC_mass[0], IC_sigmaV[0], 'IceCube', fontsize=fsize,color=IC_col,clip_on=True)


        ant_col='#e377c2'
        plt.plot(ant_mass2,ant_sigmaV, label='ANTARES',linewidth=1.5,color=ant_col,alpha=0.7, ls='-')
        plt.fill_between(ant_mass2,ant_sigmaV, y2=1e-10,color=ant_col,alpha=0.25,zorder=2)
        plt.text(600, 1e-24, 'ANTARES', fontsize=fsize,color=ant_col,clip_on=True)

        icant_col='#7f7f7f'
        plt.plot(IC_deep_mass,IC_deep_sigmaV, label='IC-DeepCore',linewidth=1.5,color=icant_col,alpha=0.7, ls='-')
        plt.fill_between(IC_deep_mass,IC_deep_sigmaV, y2=1e-10,color=icant_col,alpha=0.25,zorder=2)
        plt.text(IC_deep_mass[0], 2e-25, 'IC-DeepCore', fontsize=fsize,color=icant_col,clip_on=True)

        bhat_col='#d62728'
        plt.plot(IC_bhat_nue_mass/2,IC_bhat_nue_sigmaV,label='IceCube (Bhattacharya, 2019)',linewidth=1.5,color=bhat_col,alpha=0.7, ls='-')
        plt.fill_between(IC_bhat_nue_mass/2,IC_bhat_nue_sigmaV, y2=1e-10,color=bhat_col,alpha=0.25,zorder=2)
        plt.text(IC_bhat_nue_mass[0], IC_bhat_nue_sigmaV[0]*2, 'IceCube (Bhattacharya et al.)', fontsize=fsize,color=bhat_col,rotation=90, clip_on=True)

        cong_col='#8c564b'
        plt.plot(congl_mass, congl_sigmaV, label= 'SK atm. and IceCube-HE',linewidth=1.5,color=cong_col,alpha=0.7, ls='-')
        plt.fill_between(congl_mass,congl_sigmaV, y2=1e-10,color=cong_col,alpha=0.25,zorder=2)
        plt.text(congl_mass[1], congl_sigmaV[1]*7, 'SK atm.', fontsize=fsize,color=cong_col,clip_on=True)
        plt.text(congl_mass[-1]/6, congl_sigmaV[-1]*1.5, 'IceCube-HE', fontsize=fsize,color=cong_col,clip_on=True)

        ehe_col='#17becf'
        plt.plot(icehe_mass,icehe_sigmaV,label='IceCube-EHE',linewidth=1.5,color=ehe_col,alpha=0.7, ls='-')
        plt.fill_between(icehe_mass,icehe_sigmaV, y2=1e-10,color=ehe_col,alpha=0.25,zorder=2)
        plt.text(icehe_mass[0]*1.3,icehe_sigmaV[0], 'IceCube-EHE', fontsize=fsize,color=ehe_col,clip_on=True)

        if reduce_runtime==False:

            P1_col='#2ca02c'
            plt.plot(P1_mass,P1_sigmaV,label='P-ONE',linewidth=1.5,color=P1_col,alpha=0.7, ls='--')
            plt.text(P1_mass[0], 1.9e-25, 'P-ONE', fontsize=fsize,color=P1_col,clip_on=True)
            
            grand_col='#e377c2'
            plt.plot(grand_mass,grand_sigmaV,label='GRAND200k',linewidth=1.5,color=grand_col,alpha=0.7, ls='--')
            plt.text(rnog_mass[4]*12, rnog_sigmaV[4]*2, 'GRAND', fontsize=fsize,color=grand_col,clip_on=True)
            
            tam_col='#731fb4'
            plt.plot(tambo_Mass_ann,tambo_sigmaV,label='TAMBO',linewidth=1.5,color=tam_col,alpha=0.7, ls='--')
            plt.text(congl_mass[-1]/6, 4e-22, 'TAMBO', fontsize=fsize,color=tam_col,clip_on=True)


        km3_col='#ff7f0e'
        plt.plot(km3_mass/2,km3_sigmaV,label='KM3NET',linewidth=1.5,color=km3_col,alpha=0.7, ls='--')
        plt.text(P1_mass[0], 2e-26, 'KM3NET', fontsize=fsize,color=km3_col,clip_on=True)
        
        rno_col='#1f77b4'
        plt.plot(rnog_mass,rnog_sigmaV,label='RNO-G',linewidth=1.5,color=rno_col,alpha=0.7, ls='--')
        plt.text(rnog_mass[4]*3, rnog_sigmaV[4], 'RNO-G', fontsize=fsize,color=rno_col,clip_on=True)

        dune_col='#8a8107'
        plt.plot(dune_mass/2,dune_sigmaV,label='DUNE',linewidth=1.5,color=dune_col,alpha=0.7, ls='--')
        plt.text(dune_mass[4]/2,dune_sigmaV[4], 'DUNE', fontsize=fsize,color=dune_col,clip_on=True)

        plt.yscale("log")
        plt.xscale("log")
        plt.ylabel(r'$ \langle \sigma\nu \rangle $ $ [cm^3/s] $')
        plt.xlabel(r'$m_\chi$ [GeV]')
        
        
        
        if plot_preference=='Full_Plot':
            plt.ylim(1e-26,1e-19)
            plt.xlim(1e-3,1e10)
            
            
        elif plot_preference=='Custom_plot':
            plot_axes = kwargs.get('plot_axes_ann', None)
            plt.xlim(plot_axes[0],plot_axes[1])
            plt.ylim(plot_axes[2],plot_axes[3])
        
        plt.savefig("Dark Matter Annihilation Cross Section Limits.pdf",dpi=200)
        '''
        The following section allows the user to input any mass and receive the upper bound on the
        annihilation cross section limit or the lower bound on lifetime if desired
        
        '''
        if data==True:
             #Appending all of the strongest annihilation limits        
            sigmaV_full=np.append(bor_sigmaV[0:14],kam_sigmaV[:3])
            sigmaV_full=np.append(sigmaV_full,Sk_nuebar_sigmaV)
            sigmaV_full=np.append(sigmaV_full,sk_ol_sigmaV[17:])
            sigmaV_full=np.append(sigmaV_full,congl_sigmaV[:8])
            sigmaV_full=np.append(sigmaV_full,SK_sigmaV1[:2])
            sigmaV_full=np.append(sigmaV_full,IC_deep_sigmaV)
            sigmaV_full=np.append(sigmaV_full,ant_sigmaV[3:12])
            sigmaV_full=np.append(sigmaV_full,congl_sigmaV[28:])
            sigmaV_full=np.append(sigmaV_full,icehe_sigmaV[1:])


            #Appending all of the corresponding masses for the annihilation limits  
            mass1_full=np.append(bor_mass[0:14],kam_mass[:3])
            mass1_full=np.append(mass1_full,Sk_nuebar_mass)
            mass1_full=np.append(mass1_full,sk_ol_mass[17:])
            mass1_full=np.append(mass1_full,congl_mass[:8])
            mass1_full=np.append(mass1_full,SK_mass1[:2])
            mass1_full=np.append(mass1_full,IC_deep_mass)
            mass1_full=np.append(mass1_full,ant_mass2[3:12])
            mass1_full=np.append(mass1_full,congl_mass[28:])
            mass1_full=np.append(mass1_full,icehe_mass[1:])

            strongest_sigmaV=interp1d(mass1_full,sigmaV_full)
            m_x = kwargs.get('m_x', None)
            #print('the upper bound on annihilation cross section at for a DM particle with mass',m_x,' GeV is sigmaV=',strongest_sigmaV(m_x))

            #Appending all of the strongest lifetime limits        
            tau_full=np.append(bor_tau2[0:14],kam_best_tau_lims[:4680])
            tau_full=np.append(tau_full,sk_best_tau_lims[136:])
            tau_full=np.append(tau_full,sk_ol_tau[17:])
            tau_full=np.append(tau_full,combo_tau_binned[:97])
            tau_full=np.append(tau_full,SK_tau1[:2])
            tau_full=np.append(tau_full,IC_ant_tau)
            tau_full=np.append(tau_full,tau_ant[20:68])
            tau_full=np.append(tau_full,combo_tau_binned[707:])
            tau_full=np.append(tau_full,ic_ehe_tau[1:])

            #Appending all of the corresponding masses        
            mass2_full=np.append(bor_mass2[0:14],kam_mass_mix[:4680])
            mass2_full=np.append(mass2_full,sk_mass_mix[136:])
            mass2_full=np.append(mass2_full,sk_ol_mass[17:])
            mass2_full=np.append(mass2_full,combo_mass_binned[:97])
            mass2_full=np.append(mass2_full,SK_mass1[:2])
            mass2_full=np.append(mass2_full,IC_ant_mass)
            mass2_full=np.append(mass2_full,ant_mass2[20:68])
            mass2_full=np.append(mass2_full,combo_mass_binned[707:])
            mass2_full=np.append(mass2_full,ic_ehe_mass[1:])

            strongest_tau=interp1d(mass2_full,tau_full)
            m_x = kwargs.get('m_x', None)
            #print('the lower bound on DM lifetime at for a DM particle with mass',m_x,' GeV is t=',strongest_tau(m_x))



    output=1    
    if data==True:
        return np.array([[strongest_sigmaV(m_x)],[strongest_tau(m_x)]])
    else:
        return 1


