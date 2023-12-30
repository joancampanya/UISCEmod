# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:53:56 2023

UISCEmod Input module.
Get main variables and information from input files and combines 
meteorological datasets, R and ETo, to generate ER time series, which are 
then considered for the hydrological models. The outputs from this module 
are used in the UISCEmod calibration and outputs modules.

@author: Joan Campanya i Llovet 
South East Technological University (SETU), Ireland
"""
import UISCEmod_library_v072023 as uisce
import pickle


######################################################################
def job_in(main_path, s_len, job_ext):
    # Get main paths and details about the jobs from
    # job_parameters.csv
    (
        main_path,
        path_out,
        path_parameters,
        path_spill,
        path_coord,
        path_sites_input_ts,
        path_sites_info,
        df_sites,
        step_2_tun,
    ) = pickle.load(open(main_path + "/job_parameters" + job_ext + ".p", "rb"))

    # Get information about the job to perform
    # and generate the appropiate directiories
    (
        site_name,
        ER_update,
        model_approach,
        version,
        iterations,
        burn_in,
        tunning,
        path_sav,
        path_results,
    ) = uisce.inputs.get_job_details(path_sites_info, path_out, df_sites, s_len)

    # Inform about the job being performed
    print("\n# " + str(site_name) + "  :  " + str(version))

    # Get details for inversion and forward processes
    (
        em_model_par,
        lm_model_par,
        run_var,
        check_var_samp,
    ) = uisce.inputs.get_model_parameters(
        path_parameters + site_name + "_UISCEmod_info.csv",
        df_sites.iloc[s_len],
        path_results + "/copy_input_file.csv",
    )
    # ####
    # Get meteorological data
    print("    # getting meteorological data...")
    (
        df_precip_sel,
        df_et_sel,
        df_precip_evap,
        df_in,
    ) = uisce.inputs.get_meteorological_datasets(
        path_sites_input_ts, run_var, em_model_par, lm_model_par, site_name, ER_update
    )

    # Get groundwater level data
    print("    # getting gwl data...")
    # Get available gwl data for site of interest
    df_wl = uisce.inputs.get_water_level_data(
        path_sites_input_ts + site_name + "_UISCEmod_input_ts.csv"
    )

    # Convert to volume and divide between calibration, validation
    # and forward datasets.
    (
        df_cal,
        df_val,
        df_for,
        df_all,
        min_values_vol,
        spillpoint_vol,
        df_in_cal,
        df_in_val,
        df_in_for,
        initial_vol,
    ) = uisce.preprocessing.initial_conditions(
        path_sav, df_wl.copy(), run_var, lm_model_par, em_model_par, df_in.copy()
    )

    # re-format the parameters to be ready for calibration/forward
    (
        forward,
        variables,
        v_range,
        v_init,
        mask_range,
        initial_vol,
        f_vsa,
        param_steps_ini,
        hsm_step_ini,
    ) = uisce.preprocessing.model_parameters(
        lm_model_par, em_model_par, run_var, df_all, path_sav, site_name, initial_vol
    )

    # Store relevant variables
    pickle.dump(
        [
            run_var,
            v_init,
            param_steps_ini,
            hsm_step_ini,
            mask_range,
            iterations,
            check_var_samp,
            v_range,
            df_in_cal,
            variables,
            step_2_tun,
            spillpoint_vol,
            lm_model_par,
            em_model_par,
            min_values_vol,
            f_vsa,
            initial_vol,
            df_cal,
            forward,
            path_results,
            df_precip_evap,
            path_out,
            site_name,
            version,
            df_precip_sel,
            df_et_sel,
            df_val,
            df_for,
            path_sav,
            path_sites_input_ts,
        ],
        open(path_out + site_name + "_" + version + "/input_var.p", "wb"),
    )

    return ()


if __name__ == "__main__":
    job_in(main_path, s_len, job_ext)
