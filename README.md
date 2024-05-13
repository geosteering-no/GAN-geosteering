# GAN-geosteering
The code to reproduce the results of the paper "DISTINGUISH workflow: a new paradigm of dynamic well placement using generative machine learning" presented at ECMOR 2024.

**This repository is a work in progress, and some of the dependencies are not open yet due to contractual obligations.** We are working on opening all the components or replacing them with open analogs. Please create an issue to stay up-to-date with the developments. 

# Setup

USE a virtual environment and update pip and setup-tools

Install the following:
1. PET: https://github.com/Python-Ensemble-Toolbox/PET
2. MCWD tool: https://gitlab.com/mshahriarish/deep-borehole-inverse-problem.git
  - Use branch "workingAdaptiveFull"
4. GAN tool: https://git.openlab.iris.no/seal/gan-geosteering.git

Note: Downgrade scikit-learn to v0.22 !

# Initiallize
1. Initialize the simulator to fit with PET by instructions in GAN/README.md
   1.1 Note: Fix all paths in the GAN/GAN.py
2. Generate the data by running python data/write_data_var.py
  2.1 Note: remember to fix all paths in write_data_var.py

# Run code
Run the EnKF sequential assimilation by

`
python run_script.py
`
