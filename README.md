Welcome to our github! Here you can find code for all GOSDT and XGBoost models associated with the paper:

> Integrated single-cell multiomic analysis of HIV latency reversal reveals novel regulators of viral reactivation.

**REPRODUCING RESULTS**

The first step is to clone this repository locally:

> `git clone https://github.com/CalebKornfein/latency.git`

**ENVIRONMENT SETUP**

All code was run in a conda environment. Conda can be downloaded [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#). We use conda version 4.12.0 and Python version 3.9.12, which can be installed [here](https://www.python.org/downloads/). Once conda is successfully installed, you can clone a version of our environment by opening terminal, navigating to the base of the latency github folder, and running the command:

>`conda env create -f environment.yml python=3.9.12`

We have also attached a requirements.txt containing all packages in the environment. If any error occurs in installing a package using environment.yml, try pip-installing the package in the conda environment.

**MODELING**

Useful scripts, in order of sequence to run:

- 1: **spearman.py**
    - Merges RNA and Motif data for donor 1 and donor 2
    - Calculates Spearman correlation coefficients
    - Generates train and test data splits
- 2a: **gosdt_sweep.py**
    - Runs all GOSDT models, including the threshold guess stages
- 2b: **xgboost_model.py**
    - Runs all XGBoost models
- 3a: **gosdt_visualize.py**
    - Generates GOSDT-related paper graphics
- 3b: **xgboost_visualize.py**
    - Generates XGBoosdt-related paper graphics

All scripts should be run from the root directory path (to maintain pathing consistency). Each script can be run in the terminal using be run using the command

> `python SCRIPT_NAME.py`

Beware of long script runtimes. Analysis was done using a base 2021 Macbook Pro (16GB RAM, M1 processor), and estimated runtimes for the scripts are:

- spearman: ~3 hours
- gosdt_sweep: ~8 hours
- xgboost_model: ~6 hours
- gosdt_visualize: < 10 minutes
- xgboost_visualize: < 10 minutes

**CONCLUSION**

For questions, you can write to the github owner at caleb.kornfein@gmail.com.