# DANDAS

The Dark mAtter ANnihilation and DecAy Software (DANDAS) aims to make research on dark matter easier. This is done by outputting the strongest constraints on dark matter (DM) annihilation or decay to neutrinos based on a researchers assumptions about DM halo profile and antiparticle nature. DANDAS is capable of outputting data showing the upper bound on annihilation cross section (cm^3/s) or lower bound on DM lifetime (s) based on data from neutrino experiments around the globe. If desired, DANDAS can also return the strongest constraints for both DM annihilation and decay to neutrinos as a data file as well. 

Examples of how to properly use DANDAS are included in the notebook: 'User-Open this notebook to use DANDAS.ipynb'.

A detailed description of all of the inputs to DANDAS are included below:

INPUTS                




    Halo_Profile: Describes how dark matter density is distributed within a galaxy.
                  i) NFW halo profile where they can...
                      a) 'NFW' : use rho_0=0.4 GeV cm^-3, rs=20 kpc, gamma=1.2, and R_0=8.127 kpc (Best-Fit Values) for fastest run time (<<5s)
                      b) 'NFW Custom' : user inputs their desired values for all parameters to calculate J and D factors 

                    
                  ii) Einasto halo profile
                      a) 'Einasto' : use rho_0=0.4 GeV cm^-3, rs=20 kpc, alpha=0.155 and R_0=8.127 kpc and alpha=0.155 for fastest run time 
                      c) 'Einasto Custom' : input their desired values for all parameters and go through full MC integration 
                      
                  iii) 'Custom Density Function'
                      -This allows a user to input their desired density function
                     
                  
    Antiparticle Nature: Whether a DM particle is assumed to be distinct from its antiparticle nature or not. Note that this
                         parameter will only affect the annihilation cross section parameter. Inputs can be
                         i) 'Majorana' : Assumes DM is its own antiparticle
                         ii) 'Dirac'   : Assumes DM is distinct from its antiparticle counterpart
                         
    plot_preference: Depending on what the user wants to see, they will be able to customize the plot as follows:
                 i) 'Full_Plot' : This option will show the user annihilation cross section and lifetime over a DM mass range of 10^-2-10^11 GeV
                 ii) 'Custom_plot' : This option allows the user to 'zoom in' and only look at the region of the plot they care about
                         a) If 'Custom_plot' is chosen, the user MUST specify their desired limits for the x and y axis as a **kwarg (see below)
                         
                 iii) 'No_plot' : If no plot is desired, this is the input for you!
                 
    reduce_runtime:  This input can reduce the run-time (in the case of a custom halo profile) by omitting calculations based on data from PONE, GRAND, and TAMBO   experiments.
                     i) True: If the user wants to omit lifetime and annihilation cross section calculations based on on data from PONE, TAMBO, and GRAND to significantly reduce run time
                     
                     ii) False: If user wants to see limits calculated from PONE, TAMBO, and GRAND data. Results in a significant (>30 minute) increase in           run time if a custom DM halo profile is input.
                 
    data:  This option allows the user to save the strongest annihilation cross section and decay limits as an array. Inputs are
                 i) True : This will output the bounds on annihilation cross section and lifetime as an array over the mass range specified in the m_x input. This array will be saved within the variable used to call DANDAS. Note that these bounds are only based on published neutrino data, NOT projected sensitivies to neutrinos (shown on plots as dotted lines) 
                 ii) False: DANDAS will not output any data
                
    Optional inputs (**kwargs)
    
    i)     m_x: The mass value or range (in GeV) that Annihilation Cross Section (cm^3/s) and Lifetime (s) are calculated over. This input can be given as a float or an array of floats
    
    ii)    rho_0: The known DM density used to calibrate the DM density function (GeV/cm^3). Note that this value is only input if 'NFW Custom' or 'Einasto Custom' is desired by the user
    
    iii)   r0: The distance (in cm) to the point where the scale DM density of the milky way is true. Note that this value is only
                  input if 'NFW Custom' or 'Einasto Custom' is desired by the user.
                  
    iv)    rho_function: If a 'Custom Density function' is desired for the DM halo profile, it will be passed in with rho_function. This function must fit the following specifications to work within DANDAS:
                           a) It's only input can be distance from the Milky Way's galactic centre. This input must be given in cm. All other parameters must be defined within the function.
                           b) It must output the DM density in GeV/cm^3.
                           c) rho_func cannot be a lambda function.
                           
    v)  plot_axes_ann: This controls the axes of the annihilation cross section plot output by DANDAS. It should be input as a numpy
                    array of the following form ([x_low, x_high, y_low, y_high]).
    
    vi) plot_axes_dec:  This controls the axes of the lifetime limit plot output by DANDAS. It should be input as a numpy
                    array of the following form ([x_low, x_high, y_low, y_high]).
    
