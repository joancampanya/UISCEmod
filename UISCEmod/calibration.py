# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:54:17 2023

UISCEmod Calibration module.
This module is used by UISCEmod to calibrate the hydrological models.
It reads variables and products from the input module and generate 
products to be used for the output module.

@author: Joan Campanya i Llovet 
South East Technological University (SETU), Ireland
"""
import UISCEmod_library_v072023 as uisce
import pickle
import datetime


######################################################################
def job_cal(main_path, s_len, job_ext):
    (_, _, _, _, _, _, _, df_sites, _) = pickle.load(
        open(main_path + "/job_parameters" + job_ext + ".p", "rb")
    )

    (
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
        _,
        _,
        _,
        _,
        _,
        _,
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

    ###############################################################
    # CALIBRATION (Module 2)
    ###############################################################
    # Calibration process
    if run_var["uisce_mode"] in ["calibration", "sample_the_prior"]:
        counts_conv = 0

        # Define initial variables prior to calibration
        (
            x,
            x_lik,
            precip_var,
            accepted,
            accepted_tot,
            accepted_lik,
            accepted_tot_lik,
            accepted_met,
            accepted_tot_met,
            rejected,
            rejected_met,
            df_accepted_ts,
            df_accepted_ts_tmp,
            cws,
            st_range,
            hsm_step,
            last_change,
            var_2_tun,
        ) = uisce.mcmc.define_initial_variables(
            v_init, run_var, param_steps_ini, hsm_step_ini, mask_range
        )

        # MCMC process
        stable_process = 0
        while stable_process == 0:
            s0 = datetime.datetime.now()
            for i in range(0, iterations):
                if i + 1 == iterations:
                    stable_process = 10
                if i in [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]:
                    s1 = datetime.datetime.now()
                    print("iteration: " + str(i) + "\n    " + str(s1 - s0))

                # Apply tunning to change the MCMC step size when appropiate
                if (
                    (i == int(iterations * run_var["th_acceptance"]))
                    and (check_var_samp == False)
                    and (run_var["tunning"] > 0)
                ):
                    (st_range, hsm_step) = uisce.mcmc.tunning(
                        x,
                        x_lik,
                        v_range,
                        mask_range,
                        df_in_cal.copy(),
                        precip_var,
                        st_range,
                        variables,
                        step_2_tun,
                        var_2_tun,
                        run_var,
                        spillpoint_vol,
                        lm_model_par,
                        em_model_par,
                        min_values_vol,
                        f_vsa,
                        initial_vol,
                        df_cal.copy(),
                        forward,
                        path_results,
                        param_steps_ini,
                        hsm_step,
                        df_precip_evap.copy(),
                        tunning_iterations=run_var["tunning"],
                    )

                (
                    x_new,
                    df_in_sel,
                    precip_var_new,
                    last_change,
                ) = uisce.mcmc.transition_model_parameters(
                    x.copy(),
                    v_range.copy(),
                    mask_range.copy(),
                    df_in_cal.copy(),
                    precip_var,
                    st_range.copy(),
                    last_change,
                    hsm_step,
                    tunning=[],
                    iteration=i,
                )

                ####
                # Rough check if the calibration process is converging or not.
                # if not converging allow for wider steps (full range) every 100 iteration
                # if this does not work then re-start the calibration process
                # with new random initial parameters.
                check_conv = uisce.mcmc.check_convergence(
                    df_cal, x, i, iterations, run_var, check_var_samp
                )

                if check_conv == False:
                    (x_new, precip_var_new) = uisce.mcmc.get_random_uniform_variables(
                        x_new.copy(), v_range.copy(), df_in_cal.copy()
                    )

                    counts_conv = counts_conv + 1
                    if counts_conv == 1:
                        print(
                            "    WARNING: It looks that the inversion is not converging. Trying wider steps..."
                        )

                    if counts_conv > 500:
                        print(
                            "\n    # ALERT: The process is not converging.\n         The inversion is re-starting with random initial values"
                        )
                        print("\n\n\n")
                        print("# NEW ITERATION PROCESS:")
                        x = x_new.copy()
                        x_lik = -1e99
                        precip_var = precip_var_new
                        counts_conv = 0
                        break

                #####
                # Check that the new parameters fit within the prior
                # If they do evaluate if they get accepted or not
                if uisce.mcmc.prior(x_new, v_range, variables) == 0:
                    print("ISSUES at generating the prior!")
                    break

                else:
                    if check_var_samp == True:
                        # Accept all the proposed parameters
                        x = x_new
                        precip_var = precip_var_new
                        accepted_tot.append(x_new)
                        accepted.append(x_new)
                        accepted_met.append(precip_var)
                        accepted_tot_met.append(precip_var)

                    if check_var_samp == False:
                        # Update Calibration inputs parameters
                        if run_var["model_approach"] == "LM":
                            cal_inputs = [
                                x_new,
                                df_in_sel.copy(),
                                variables,
                                spillpoint_vol,
                                lm_model_par["window_lm"],
                                min_values_vol[0],
                                initial_vol,
                                f_vsa,
                                lm_model_par["ref_date"],
                                df_precip_evap.copy(),
                            ]

                        if run_var["model_approach"] == "EM":
                            cal_inputs = [
                                x_new.copy(),
                                df_in_sel.copy(),
                                variables,
                                em_model_par["window_em"],
                                min_values_vol[0],
                                mask_range,
                            ]

                        ####
                        # Compute forward solution and likelihood
                        (x_new_lik, ts_tmp) = uisce.mcmc.lik_normal(
                            df_cal.copy(), forward, cal_inputs.copy(), run_var
                        )

                        ####
                        # If there is no accepted value, get the first one
                        if i == 0:
                            x = x_new.copy()
                            x_lik = x_new_lik
                            precip_var = precip_var_new
                            accepted_tot.append(x_new)
                            accepted_tot_lik.append(x_lik)

                        # Otherwise base it on the acceptance criteria
                        elif uisce.mcmc.acceptance(x_lik, x_new_lik) == True:
                            # update model parameters and reference likelihood
                            x = x_new.copy()
                            x_lik = x_new_lik
                            precip_var = precip_var_new

                            accepted_tot.append(x_new)
                            accepted_tot_met.append(precip_var_new)

                            # Accept every "# variables" accepted iterations
                            if i > iterations * run_var["th_acceptance"]:
                                accepted.append(x_new)
                                accepted_lik.append(x_lik)
                                accepted_met.append(precip_var_new)

                                if str(i).endswith("00"):
                                    print(
                                        "accepted at iteration: "
                                        + str(i)
                                        + " | likelihood: "
                                        + str(round(x_lik, 2))
                                    )

                                    ###################################################################
                                    # store and visualize relevant outputs from calibration process
                                    if len(accepted) > 0:
                                        uisce.outputs.store_accepted_parameters(
                                            accepted.copy(),
                                            accepted_met.copy(),
                                            variables,
                                            (
                                                path_out
                                                + str(site_name)
                                                + "_"
                                                + str(version)
                                                + "/calibration/"
                                            ),
                                            site_name,
                                            run_var,
                                        )

                                # Select time series to keep. Not all of them are kept
                                # as may take too much memory and slow the calibration
                                # process
                                (
                                    df_accepted_ts,
                                    df_accepted_ts_tmp,
                                ) = uisce.mcmc.keep_time_series(
                                    df_accepted_ts_tmp,
                                    df_accepted_ts,
                                    ts_tmp,
                                    i,
                                )

                            # Plot corner plot to show evolution of the
                            # inversion process and how variables are
                            # evolving. May help to spot issues in the
                            # calibration process
                            if str(i).endswith("0000"):
                                uisce.visualization.corner_plot(
                                    accepted_tot.copy(),
                                    "corner_plot_accepted_all",
                                    mask_range.copy(),
                                    variables.copy(),
                                    path_results,
                                    run_var["model_approach"],
                                    site_name,
                                    check_var_samp,
                                )

                        else:
                            # If parameters are not accepted store the rejected values
                            rejected.append(x_new)
                            rejected_met.append(precip_var)

        # # ####
        # # Check proportion of accepted models
        if check_var_samp == False:
            uisce.mcmc.proportion_accepted_models(
                accepted_tot, accepted, rejected, run_var["th_acceptance"]
            )

    print(" storing calibration variables...")
    pickle.dump(
        [
            accepted,
            accepted_lik,
            accepted_tot,
            accepted_met,
            df_accepted_ts,
            df_accepted_ts_tmp,
            main_path,
        ],
        open(
            main_path + "/out/" + site_name + "_" + version + "/calibration_var.p", "wb"
        ),
    )

    return ()


if __name__ == "__main__":
    job_cal(main_path, s_len, job_ext)
