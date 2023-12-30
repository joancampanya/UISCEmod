# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:54:37 2023


UISCEmod Output module.
The output module generates the UISCEmod products including: forward solution
time series, probability density function (pdf) of the calibrated model 
parameters, and a set of products to help evaluating convergence of the 
calibration process as well as the fit of the data. It uses outputs from
the UISCEmod input and the calibration modules.

@author: Joan Campanya i Llovet 
South East Technological University (SETU), Ireland
"""
from . import uisce
import pickle


######################################################################
def job_out(main_path, s_len, job_ext):
    (_, _, _, _, _, _, _, df_sites, _) = pickle.load(
        open(main_path + "/job_parameters" + job_ext + ".p", "rb")
    )

    # Get input parameters
    (
        run_var,
        _,
        _,
        _,
        mask_range,
        _,
        check_var_samp,
        v_range,
        _,
        variables,
        _,
        spillpoint_vol,
        lm_model_par,
        em_model_par,
        min_values_vol,
        f_vsa,
        initial_vol,
        df_cal,
        _,
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
    ) = pickle.load(
        open(
            main_path
            + "/out/"
            + df_sites.loc[s_len]["site_name"]
            + "_"
            + df_sites.loc[s_len]["version"]
            + "/input_var.p",
            "rb",
        )
    )

    # Get calibration outputs
    (
        accepted,
        accepted_lik,
        accepted_tot,
        accepted_met,
        df_accepted_ts,
        df_accepted_ts_tmp,
        main_path,
    ) = pickle.load(
        open(
            main_path + "/out/" + site_name + "_" + version + "/calibration_var.p", "rb"
        )
    )

    ################################################################
    # OUTPUTS (module 3)
    ################################################################
    if run_var["uisce_mode"] in ["calibration", "sample_the_prior"]:
        if len(accepted) > 0:
            uisce.outputs.store_likelihood_values(
                accepted_lik, path_out, site_name, version
            )

            # plot dependency between variables
            uisce.visualization.variables_dependency(
                accepted,
                accepted_tot,
                mask_range,
                variables,
                path_results,
                run_var,
                site_name,
                check_var_samp,
            )

            # reformat accepted parameters
            (accepted, accepted_met) = uisce.outputs.reformat_accepted_parameters(
                accepted, accepted_met
            )

            # Update and store accepted time series
            (df_accepted_ts) = uisce.outputs.store_accepted_ts(
                df_accepted_ts,
                df_accepted_ts_tmp,
                df_cal,
                (
                    path_out
                    + str(site_name)
                    + "_"
                    + str(version)
                    + "/calibration/accepted_ts_calibration_process.csv"
                ),
            )

            # Store accepted model parameters
            uisce.outputs.store_accepted_parameters(
                accepted,
                accepted_met,
                variables,
                (path_out + str(site_name) + "_" + str(version) + "/calibration/"),
                site_name,
                run_var,
            )

            # Assess distribution of the errors
            uisce.outputs.errors_analysis(
                df_accepted_ts.copy(),
                df_cal.copy(),
                min_values_vol[0],
                accepted.copy(),
                (path_out + str(site_name) + "_" + str(version) + "/calibration"),
                export_errors=True,
            )

    if run_var["uisce_mode"] in ["forward", "calibration", "sample_the_prior"]:
        # visualize results (calibration, validation, sample de prior, forward)
        uisce.outputs.results_and_products(
            df_precip_sel.copy(),
            df_et_sel.copy(),
            df_precip_evap.copy(),
            df_cal.copy(),
            df_val.copy(),
            df_for.copy(),
            main_path,
            path_sav,
            path_results,
            path_sites_input_ts,
            run_var,
            lm_model_par,
            em_model_par,
            f_vsa,
            v_range,
            mask_range,
            min_values_vol,
            spillpoint_vol,
            initial_vol,
            check_var_samp,
            text_inv="",
            text_sdp="sample_de_prior",
            text_for="forward",
        )

    return ()


if __name__ == "__main__":
    job_out(main_path, s_len, job_ext)
