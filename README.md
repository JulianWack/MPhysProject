# MPhysProject
Code produced during my MPhys project investigating the Fourier Acceleration of the N=2 principal chiral model in 2 dimensions.

The main code and data is contained in the SU2xSU2 directory. A pervious version, not suitable for long runs is contained in SU2xSU2_preproduction. 
I also conducted a study of the quantum harmonic oscillator in the QHO directory.
To run the code, the following set up is required:
- numpy, scipy, matplotlib libraries
- alive\_progress, astropy libraries
- plotting style sheet: add content of folder plotting\_style to the matplotlib configuration directory which can be located running \path{matplotlib.get_configdir()}
