# XFEL-numerical-simulation-related
Some Python scripts for X-ray free-electron laser simulations
They are mostly used for my daliy analysis and simulation. Wil be updated from time to time. Comments are welcomed.

# Related code
Elegant: https://ops.aps.anl.gov/elegant.html
Genesis 1.4: https://github.com/svenreiche/Genesis-1.3-Version4

# desciption of the libraries
SDDS_class.py : io related tasks for Elegant outputs (eg. convert SDDS file to h5)

Genesis_analysis.py : io / plotting related tasks for Genesis outputs

EEHG_model.py : calculate analytical results for Echo-enabled harmonic generation (EEHG)

dynamic_diffraction.py : calculate numerical results on dynamic diffraction theory (can be used for self-seeding). Currently, only Diamond is supported.

BeamOptics_class.py : deal with linear propagation of electrons in lattice

# GUIs
plot_beamline.py : based on the SDDS_class.py library, interpret the lattice layout (.magn file) and depict it interactively. Requires PyQt6.
TODO: add the utility to show the simulated optic parameters (i.e. alpha, beta, gamma, etc.)
