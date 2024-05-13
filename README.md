# GAN-geosteering
The code to reproduce the results of the paper "DISTINGUISH workflow: a new paradigm of dynamic well placement using generative machine learning" presented at ECMOR 2024.

# Setup

USE a virtual environment and update pip and setup-tools

Install the following:
1. PET: https://github.com/Python-Ensemble-Toolbox/PET
2. MCWD tool: https://gitlab.com/mshahriarish/deep-borehole-inverse-problem.git
3. GAN tool: https://git.openlab.iris.no/seal/gan-geosteering.git

# Initiallize
1. Initialize the simulator to fit with PET by instructions in GAN/README.md
2. Generate the data by running python data/write_data_var.py

# Run code
Run the EnKF sequential assimilation by

`
python run_script.py
`
