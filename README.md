# GAN-geosteering
The code to reproduce the results of the paper ["DISTINGUISH workflow: a new paradigm of dynamic well placement using generative machine learning"](#cite-as) presented at ECMOR 2024.

## Updates towards fully open-source dependences

**This repository is a work in progress, and some of the dependencies are not open yet due to contractual obligations.** We are working on opening all the components or replacing them with open analogs. Please create an issue to stay up-to-date with the developments. 

Curently, the updated latent-vector-to-logs fully machine-learning simulator is grouped in the sub-folder [gan_update](https://github.com/geosteering-no/GAN-geosteering/tree/main/gan_update). 

To run full modeling sequence, use [vector_to_log.py](https://github.com/geosteering-no/GAN-geosteering/blob/main/gan_update/vector_to_log.py)

We will likely move it to a separate repository when the setup is further verified. 

## Cite as:

Alyaev, S., Fossum, K., Djecta, H. E., Tveranger, J., & Elsheikh, A. (2024). **DISTINGUISH Workflow: a New Paradigm of Dynamic Well Placement Using Generative Machine Learning**. In *ECMOR 2024*. (Vol. 2024, No. 1, pp. 1-16). European Association of Geoscientists & Engineers. DOI: https://doi.org/10.3997/2214-4609.202437018

### Latex

```
@article{fossum2024ensemble,
  title={Ensemble history-matching workflow using interpretable SPADE-GAN geomodel},
  author={Fossum, Kristian and Alyaev, Sergey and Elsheikh, Ahmed H},
  journal={First Break},
  volume={42},
  number={2},
  pages={57--63},
  year={2024},
  publisher={European Association of Geoscientists \& Engineers},
  doi={https://doi.org/10.3997/2214-4609.202437018}
}
```

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

#
`
