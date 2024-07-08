# GAN-geosteering
The code to reproduce the results of the paper "DISTINGUISH workflow: a new paradigm of dynamic well placement using generative machine learning" presented at ECMOR 2024.

**This repository is a work in progress, and some of the dependencies are not open yet due to contractual obligations.** We are working on opening all the components or replacing them with open analogs. Please create an issue to stay up-to-date with the developments. 

# Setup

USE a virtual environment and update pip and setup-tools

Install the following:
1. PET: https://github.com/Python-Ensemble-Toolbox/PET
* Including the setup of PET in your **virtual environment**, which can be done by running `python3 -m pip install -e .` in the PET folder.
2. MCWD tool: https://gitlab.com/mshahriarish/deep-borehole-inverse-problem.git
  - Use branch "workingAdaptiveFull"
3. GAN tool: https://github.com/alin256/gan-geosteering-prestudy-internal

Note: Downgrade scikit-learn to v0.22 !

# Initiallize
1. Initialize the simulator to fit with PET by instructions in [GAN/README.md](GAN/README.md)

   1.1 Note: Fix all paths in the [GAN/GAN.py](GAN/GAN.py)
3. Generate the data by running python [data/write_data_var.py](data/write_data_var.py)

   2.1 Note: remember to fix all paths in [write_data_var.py](write_data_var.py)

# Run code
Run the EnKF sequential assimilation by

`
python run_script.py
`

# Run the Web Application
The Web Application is by Streamlit. First, ensure that Streamlit is installed by

'
pip install streamlit
'

Then start the application by the following command

'
streamlit run Run_WF_streamlit_Dash.py
'
