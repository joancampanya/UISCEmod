# -*- coding: utf-8 -*-
"""
UISCEmod version 072023
Created on 1st July 2023

# SUMMARY
UISCEmod: Software for modelling and characterizing water level time series focused
on ephemeral karstic wetlands from Ireland. UISCEmod includes two hydrological models, 
empirical (EM) and lumped (LM), that requires minimal input information 
(meteorological, water levels, and volume-stage-area curves). The calibration process for both 
modelling approaches is automated following a Bayesian approach. 

UISCEmod is divided in Input,Calibration and Outputs modules.
The Inputs module combines meteorological datasets, R and ETo, to generate 
ER time series, which are then considered for the hydrological models. 
The calibration module is used to calibrate the hydrological models and 
can be ignored if model parameters are already defined. The third module, 
Outputs, generates the products including: forward solution for stage and 
volume time series, probability density function (pdf) of the calibrated model 
parameters, and a set of products to help evaluating convergence of the 
calibration process as well as the fit of the data. 

######
General code structure
######
The main functions used by UISCEmod_v072023 are in the UISCEmod_library_v072023
UISCEmod is divided in three modules:
    Inputs: UISCEmod_module_inputs.py
    Calibration: UISCEmod_module_calibration.py
    Outputs: UISCEmod_module_outputs.py

#######
LICENCE
#######
UISCEmod is under the license: Attribution 4.0 International (CC BY 4.0)

#########
REFERENCE
#########
If using UISCEmod make reference to:
Campanyà,J.,McCormack,T.,Gill,L.W.,Johnston,P.M.,Licciardi,A.,Naughton,O.,2023. 
UISCEmod: Open-source software for modelling water level time series in 
ephemeral karstic wetlands. Environmental Modelling & Software 167, 105761.
https://doi.org/10.1016/j.envsoft.2023.105761

#######
FUNDING
#######
UISCEmod was developed within a project funded by Geological Survey Ireland 
(Ref: 2019-TRP-GW-001). 

#######
DISCLAIMER
#######
Although every effort has been made to ensure that UISCEmod works correctly,
we cannot guarantee that the script is excempt of issues. Neither 
South East Technological University, Geological Survey Ireland, 
nor the authors accept any responsibility whatsoever for loss or damage 
occasioned, or claimed to have been occasioned, in part or in full as a 
result of using of UISCEmod.

@author: Joan Campanya i Llovet 
South East Technological University (SETU), Ireland
"""

# %%
# Import libraries
import click
import warnings
import logging

from . import uisce
from . import inputs
from . import calibration
from . import outputs

logger = logging.getLogger(__name__)

@click.command()
@click.option("-j", "--job-extension", default="", help="Define job extension if you are planning to send multiple jobs at the same time")
@click.option("-v", "--verbose", count=True)
def cli(job_extension, verbose):
    if not verbose:
        warnings.filterwarnings("ignore")
    elif verbose == 1:
        logger.setLevel(logging.INFO)
    elif verbose >= 2:
        logger.setLevel(logging.DEBUG)

    # Get job information
    (df_sites, main_path) = uisce.inputs.get_paths_and_initial_information(job_ext=job_extension)

    for s_len in df_sites.index:
        # # Generate input variables
        click.echo("\n#####\nINPUT MODULE:")
        inputs.job_in(main_path, s_len, job_extension)

        # Perform calibration
        if df_sites.loc[s_len]["mode"] == "calibration":
            print("\n#####\nCALIBRARION MODULE:")

            calibration.job_cal(main_path, s_len, job_extension)

        # Generate outputs
        print("\n#####\nOUTPUT MODULE:")
        outputs.job_out(main_path, s_len, job_extension)
