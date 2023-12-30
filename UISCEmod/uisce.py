# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:55:43 2022

Functions considered for UISCEmod

NOTE: Originally the EM approach was named TF. Any reference within here
to the TF is to the EM approach of the current version of UISCEmod.

@author: Joan Campanya i Llovet 
South East Technological University (SETU), Ireland
"""
import pandas as pd
import numpy as np
import datetime
import random
import pickle

import corner
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
import shutil

from os import walk
from scipy import signal
from scipy import interpolate

"""
Uncomment the random seed to test and check updates in UISCEmod code
"""
# random.seed(2023)

# General formats for storing the data for stage and volume time series
format_num_stg = "%.2f"
format_num_vol = "%.0f"

# random state for specific functions
random_state = random.randint(0, 1e6)


#######################################################################
#######################################################################
class visualization:
    ###################################################################
    def variables_dependency(
        accepted,
        accepted_tot,
        mask_range,
        variables,
        path_results,
        run_var,
        site_name,
        check_var_samp,
    ):
        """
        Generate corner plots for all accepted model parameters
        and only considering parameters accepted after burn in

        Parameters
        ----------
        accepted : list
            Accepted model parameters after burn in
        accepted_tot : list
            Accepted model parameters from beginning of
            the inversion
        mask_range : list
            True and False deffining which variables are
            included in the inversion process
        variables : list
            DESCRIPTION.
        path_results : str
            Path where to store the results
        run_var : dictionary
            Contains information about site and the job being
            performed
        site_name : str
            Name of the site
        check_var_samp : bool
            Check if is we are sampling the prior

        Returns
        -------
        None.

        """

        for i_corn in range(0, 2):
            try:
                if i_corn == 0:
                    var_accepted = accepted.copy()
                    txt = "corner_plot_accepted_after_burn_in"
                if i_corn == 1:
                    var_accepted = accepted_tot.copy()
                    txt = "corner_plot_accepted_all"

                # Generate corner plots
                visualization.corner_plot(
                    var_accepted,
                    txt,
                    mask_range,
                    variables,
                    path_results,
                    run_var["model_approach"],
                    site_name,
                    check_var_samp,
                )
            except:
                print("    ~ Issues generating corner plots")

        return ()

    ###################################################################
    def stage_discharge_curves(
        obs, df_y_s, df_y_all, df_y_all_s, df_outflow, path_results, mode
    ):
        """
        Plot stage flow plots. For TF plots netflow vs stage
        for LM plots netflow vs stage and discharge vs stage.

        Parameters
        ----------
        obs : dataframe
            Observed volume time series
        df_y_s : dataframe
            Observed stage time series
        df_y_all : dataframe
            Model volume time series
        df_y_all_s : dataframe
            Model stage time series
        df_outflow : dataframe
            Model discharge time series
        path_results : str
            path where to store the outputs
        mode : str
            Specify if using EM or LM approaches

        Returns
        -------
        None.

        """

        # simplify the plot considering the last 100
        # time series
        if len(df_y_all.columns > 100):
            df_y_all = df_y_all.iloc[:, -100::]
            df_y_all_s = df_y_all_s.iloc[:, -100::]
            df_outflow = df_outflow.iloc[:, -100::]

        # Shift the outflow one day as the discharge
        # is based on stage value of the previous day
        # and we want to assess disch compare to stage
        df_outflow = df_outflow.shift(periods=-1)

        # Get netflow from observed volume time series
        net_flow = obs.squeeze().diff().to_frame().copy()
        net_flow.columns = ["net flow"]
        net_flow_plot = net_flow.copy()[net_flow.index.isin(df_y_s.index)]

        # Get stage values from measured data
        stage_plot_meas = df_y_s.copy()[df_y_s.index.isin(net_flow.index)]

        # Get netflow estimates from modelled time series
        mod_net_flow_plot = df_y_all.diff(axis=0).copy()[
            df_y_all.index.isin(df_y_s.index)
        ]
        stage_plot = df_y_all_s.copy()[df_y_all_s.index.isin(net_flow.index)]

        # Generate the plot
        fig, ax = plt.subplots(figsize=(20, 10))

        ax.plot(
            np.array(stage_plot_meas.shift(0)),
            np.array(net_flow_plot),
            "r*",
            alpha=0.5,
            label="measured Net Flow",
        )

        for i_num, i in enumerate(df_y_all.columns):
            if i_num == 0:
                ax.plot(
                    np.array(stage_plot.iloc[:, i_num]),
                    np.array(mod_net_flow_plot.iloc[:, i_num]),
                    "bD",
                    markerfacecolor="none",
                    alpha=1 / len(df_y_all.columns),
                    label="Estimated Net Flow",
                )
            else:
                ax.plot(
                    np.array(stage_plot.iloc[:, i_num]),
                    np.array(mod_net_flow_plot.iloc[:, i_num]),
                    "bD",
                    markerfacecolor="none",
                    alpha=1 / len(df_y_all.columns),
                    label=None,
                )

        min_val_p = np.nanmin(net_flow_plot)
        min_val_p = np.min([min_val_p, np.nanmin(mod_net_flow_plot)])

        try:
            outflow_plot = df_outflow.copy()[df_outflow.index.isin(df_y_s.index)].copy()

            # Avoid multiple legends
            if i_num == 0:
                ax.plot(
                    np.array(stage_plot),
                    np.array(outflow_plot),
                    "k+",
                    alpha=1 / len(df_y_all.columns),
                    label="estimated Discharge",
                )
            else:
                ax.plot(
                    np.array(stage_plot),
                    np.array(outflow_plot),
                    "k+",
                    alpha=1 / len(df_y_all.columns),
                    label=None,
                )

            min_val_p = np.min([min_val_p, np.nanmin(outflow_plot)])
        except:
            pass

        # Add labels and store the plot
        ax.legend()
        ax.hlines(
            y=0,
            xmin=stage_plot.min().min(),
            xmax=stage_plot.max().max(),
            color="gray",
            lw=2,
            ls="--",
        )

        ax.set_xlabel("Stage [m]")
        ax.set_ylabel("Flow [m${^3}$/day]")

        plt.savefig(path_results + "/" + str(mode) + "/" + "stage_vs_net_flow_plot.png")

        plt.close()

        return ()

    ###################################################################
    def net_in_out_flows(
        obs, mod_ts, outflow, inflow, path_out, df_in, df_precip, mode
    ):
        """
        Plot netflow for measured data and compared with
        Netflow, recharge and discharge from modelled data

        Parameters
        ----------
        obs : DataFrame
            Measured volume time series
        mod_ts : DataFrame
            Modelled volume time series
        outflow : DataFrame
            Modelled outflow time series
        inflow : DataFrame
            Modelled inflow time series
        path_out : str
            path where to store the data
        df_in : DataFrame
            Effective rainfall time series
        df_precip : DataFrame
            Precipitation time series
        mode : str
            Specify if using EM or LM approach

        Returns
        -------
        None.

        """

        fig, ax = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(10, 8),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )

        # add netflow
        net_flow = obs.squeeze().diff().to_frame().copy()
        net_flow.columns = ["Net Flow (measured)"]
        net_flow.plot(ax=ax[0], color="mediumorchid", style="-", lw=1)

        # add outflow
        outflow = outflow[outflow.index >= obs.index[0]]
        outflow = outflow[outflow.index <= obs.index[-1]]

        inflow = inflow[inflow.index >= obs.index[0]]
        inflow = inflow[inflow.index <= obs.index[-1]]

        estimated_netflow = mod_ts.squeeze().diff(axis=0).copy()

        for i_num, i in enumerate(estimated_netflow.columns):
            if i_num == 0:
                outflow_tmp = outflow[[i]]
                inflow_tmp = inflow[[i]]
                estimated_netflow_tmp = estimated_netflow[[i]]

                outflow_tmp.columns = ["outflow"]
                inflow_tmp.columns = ["inflow"]
                estimated_netflow_tmp.columns = ["Net Flow (model)"]
                legend = ["outflow", "inflow", "Net Flow (model)"]
            else:
                outflow_tmp = outflow[i]
                inflow_tmp = inflow[i]
                estimated_netflow_tmp = estimated_netflow[i]

                legend = [None, None, None]

            outflow_tmp.plot(ax=ax[0], color="r", style="-", lw=0.1, legend=legend[0])

            inflow_tmp.plot(ax=ax[0], color="b", style="-", lw=0.1, legend=legend[1])

            estimated_netflow_tmp.plot(
                ax=ax[0], color="k", style="--", lw=0.1, legend=legend[2]
            )

        df_in_plot = df_in[df_in.index >= obs.index[0]]
        df_in_plot = df_in_plot[df_in_plot.index <= obs.index[-1]]

        df_precip_plot = df_precip[df_precip.index >= obs.index[0]]
        df_precip_plot = df_precip_plot[df_precip_plot.index <= obs.index[-1]]
        df_precip_plot.columns = ["Rainfall"]
        df_in_plot = df_in_plot.mean(axis=1).to_frame()
        df_in_plot.columns = ["Effective Rainfall"]

        ax[1].bar(df_precip_plot.index, df_precip_plot.mean(axis=1), facecolor="salmon")

        ax[1].bar(
            df_in_plot.index,
            df_in_plot.mean(axis=1),
            facecolor="k",
        )

        try:
            ax[0].set_ylabel("Volume [$m^3$]")
            ax[1].set_ylabel("[mm]")

            plt.suptitle("Netflow Inflow and Outflow TS")
            plt.tight_layout()
        except:
            pass

        plt.savefig(path_out + "/" + str(mode) + "/" + "net_in_out_flows.png")

        plt.close()

        return ()

    ###################################################################
    def ts_imaging_empirical_errors(
        run_var,
        df_y_all,
        df_y,
        path_sav,
        min_stage,
        q_up,
        q_2up,
        q_down,
        q_2down,
        df_in_plot,
        df_precip_plot,
        min_val,
        path_results,
        sel_time_series,
    ):
        """
        Plot comparing model and mesured time series for
        volume and stage. It considers the errors empirically
        defined

        Parameters
        ----------
        run_var : dictionary
            Contains information about site and the job being
            performed
        df_y_all : DataFrame
            Model volume time series
        df_y : DataFrame
            Measured volume time series
        path_sav : str
            path to stage volume curves
        min_stage : float
            minimum value for stage plots
        q_up : float
            value for quantile 0.86.
        q_2up : float
            value for quantile 0.975
        q_down : float
            value for quantile 0.16
        q_2down : float
            value for quantile 0.025
        df_in_plot : DataFrame
            Effective rainfall time series
        df_precip_plot : DataFrame
            Precipitation time series
        min_val : float
            minimum volume
        path_results : str
            path where to store the plot
        sel_time_series : str
            Description of the process being run
            calibration, validation, forward...

        Returns
        -------
        None.

        """

        range_plot = 1
        if run_var["model_volume"] == True:
            range_plot = 2
            df_y_mod = df_y_all.median(axis=1).to_frame()

            df_mod_s = preprocessing.convert_hydrol_param(
                path_sav,
                df_y_mod.copy(),
                hydro_out="stage",
                hydro_in="volume",
                min_stage=min_stage,
                return_f=False,
                simplify_f=-999,
            )

            df_mod_s_up = preprocessing.convert_hydrol_param(
                path_sav,
                df_y_mod.copy() + q_up,
                hydro_out="stage",
                hydro_in="volume",
                min_stage=min_stage,
                return_f=False,
                simplify_f=-999,
            )

            df_mod_s_down = preprocessing.convert_hydrol_param(
                path_sav,
                df_y_mod.copy() + q_down,
                hydro_out="stage",
                hydro_in="volume",
                min_stage=min_stage,
                return_f=False,
                simplify_f=-999,
            )

            df_mod_s_2up = preprocessing.convert_hydrol_param(
                path_sav,
                df_y_mod.copy() + q_2up,
                hydro_out="stage",
                hydro_in="volume",
                min_stage=min_stage,
                return_f=False,
                simplify_f=-999,
            )

            df_mod_s_2down = preprocessing.convert_hydrol_param(
                path_sav,
                df_y_mod.copy() + q_2down,
                hydro_out="stage",
                hydro_in="volume",
                min_stage=min_stage,
                return_f=False,
                simplify_f=-999,
            )

            df_y_s = preprocessing.convert_hydrol_param(
                path_sav,
                df_y.copy(),
                hydro_out="stage",
                hydro_in="volume",
                min_stage=min_stage,
                return_f=False,
                simplify_f=-999,
            )

        for i in range(0, range_plot):
            if i == 0:
                df_mod_plot = df_y_mod.copy()
                df_up = df_mod_plot + q_up
                df_2up = df_mod_plot + q_2up
                df_down = df_mod_plot + q_down
                df_2down = df_mod_plot + q_2down
                df_meas_plot = df_y.copy()

                ext = "stage"
                if range_plot == 2:
                    ext = "volume"

            if i == 1:
                df_mod_plot = df_mod_s.copy()
                df_up = df_mod_s_up
                df_2up = df_mod_s_2up
                df_down = df_mod_s_down
                df_2down = df_mod_s_2down
                df_meas_plot = df_y_s.copy()
                ext = "stage"
                if sel_time_series == "calibration":
                    min_val = run_var["min_stage"][0]
                if sel_time_series == "validation":
                    min_val = run_var["min_stage"][1]
                if sel_time_series == "forward":
                    min_val = run_var["min_stage"][2]

            fig, ax = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(10, 8),
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=True,
            )

            ax[1].bar(
                df_precip_plot.index,
                df_precip_plot.mean(axis=1),
                width=1,
                facecolor="pink",
            )

            ax[1].bar(
                df_in_plot.index,
                df_in_plot.mean(axis=1),
                width=1,
                facecolor="k",
            )

            ax[1].set_ylim(0, 1.05 * df_precip_plot.mean(axis=1).max())
            ax[1].set_ylabel("precipitation [mm]")

            if ext == "volume":
                ax[0].set_ylabel("volume [m$^3$]")
            if ext == "stage":
                ax[0].set_ylabel("stage [m]")
            fig.tight_layout()

            # plot measured time series
            ax[0].fill_between(
                df_mod_plot.index,
                df_up.squeeze(),
                df_down.squeeze(),
                color="blue",
                alpha=0.1,
                lw=0,
            )

            ax[0].fill_between(
                df_mod_plot.index,
                df_2up.squeeze(),
                df_2down.squeeze(),
                color="blue",
                alpha=0.1,
                lw=0,
            )

            df_meas_plot.plot(ax=ax[0], color="orange", legend=False)

            df_mod_plot.median(axis=1).plot(ax=ax[0], color="blue", lw=0.5, zorder=10)

            # Add horizontal lines associated to quantiles (flood duration)
            for iq in np.array([0.05, 0.5, 0.95]):
                ax[0].hlines(
                    y=df_mod_plot.quantile(
                        iq,
                        axis=0,
                        interpolation="nearest",
                    ),
                    xmin=df_y_all.index[0],
                    xmax=df_y_all.index[-1],
                    color="blue",
                    alpha=0.25,
                    ls="--",
                )

                ax[0].hlines(
                    y=df_meas_plot.quantile(
                        iq,
                        axis=0,
                        interpolation="nearest",
                    ),
                    xmin=df_y.index[0],
                    xmax=df_y.index[-1],
                    color="orange",
                    alpha=0.25,
                    ls="--",
                )

            # Compute evaluation metrix
            try:
                NSE = outputs.NSE(df_mod_plot.copy(), df_meas_plot.copy())

                (KGE, kge_1, kge_2, kge_3) = outputs.KGE(
                    df_mod_plot.copy(), df_meas_plot.copy()
                )

                BIAS = outputs.BIAS(df_mod_plot.copy(), df_meas_plot.copy())
            except:
                NSE = "N/A"
                KGE = "N/A"
                BIAS = "N/A"

            # # ####
            # percentatge of points within errors
            # make sure we only compare where there is data to compare
            df_mod_plot = df_mod_plot[df_mod_plot.index.isin(df_meas_plot.index)]
            df_meas_plot = df_meas_plot[df_meas_plot.index.isin(df_mod_plot.index)]
            df_up = df_up[df_up.index.isin(df_mod_plot.index)]
            df_2up = df_2up[df_2up.index.isin(df_mod_plot.index)]
            df_down = df_down[df_down.index.isin(df_mod_plot.index)]
            df_2down = df_2down[df_2down.index.isin(df_mod_plot.index)]

            ppwe_68_a = [df_meas_plot.squeeze() <= df_up.squeeze()]
            ppwe_68_b = [df_meas_plot.squeeze() >= df_down.squeeze()]
            ppwe_68 = np.array(ppwe_68_a)[0] * np.array(ppwe_68_b)[0]
            ppwe_68 = ppwe_68[np.array(df_meas_plot.squeeze().to_list()) > min_val]
            prop_68 = np.round(np.nansum(ppwe_68) / len(ppwe_68), 2)

            ppwe_95_a = [df_meas_plot.squeeze() <= df_2up.squeeze()]
            ppwe_95_b = [df_meas_plot.squeeze() >= df_2down.squeeze()]
            ppwe_95 = np.array(ppwe_95_a)[0] * np.array(ppwe_95_b)[0]
            ppwe_95 = ppwe_95[np.array(df_meas_plot.squeeze().to_list()) > min_val]
            prop_95 = np.round(np.nansum(ppwe_95) / len(ppwe_95), 2)

            plt.xlabel("Date")

            plt.suptitle(
                "Empirical errors  "
                + str(run_var["site_name"])
                + " (NSE: "
                + str(NSE)
                + ", KGE: "
                + str(KGE)
                + ", BIAS[%]: "
                + str(BIAS)
                + ") \n"
                + "prop. of fittet points: "
                + "68% int:"
                + str(prop_68)
                + ", "
                + "96% int:"
                + str(prop_95)
            )

            plt.tight_layout()
            plt.savefig(
                path_results
                + "/"
                + str(sel_time_series)
                + "/"
                + "/"
                + "modelled_TS_empirical errors"
                + "_NSE_"
                + str(NSE)
                + "_KGE_"
                + str(KGE)
                + "_BIAS_"
                + str(int(BIAS))
                + "_"
                + str(np.round(prop_68, 2))
                + "_"
                + str(np.round(prop_95, 2))
                + "_"
                + str(ext)
                + ".png"
            )

            plt.close()
        return ()

    ###################################################################

    def ts_imaging_model_parameters_propagation(
        run_var,
        df_y_all,
        df_y,
        df_y_all_s,
        df_y_s,
        df_in_plot,
        df_precip_plot,
        path_results,
        sel_time_series,
        min_val,
    ):
        """
        Plot comparing modelled and measured time series for
        volume and stage. It only consider errors derived from
        pdf of the model parameters

        Parameters
        ----------
        run_var : Directory
            Contains information about site and the job being
            performed
        df_y_all : DataFrame
            Model volume time series
        df_y : DataFrame
            Measured volume time series
        df_y_all_s : DataFrame
            Model stage time series
        df_y_s : DataFrame
            Measured stage time series
        df_in_plot : DataFrame
            Effective rainfall time series
        df_precip_plot : DataFrame
            Precipitation time series
        path_results : str
            path where to store the results
        sel_time_series : str
            job being perfoemd (calibration, validation, forward...)
        min_val : float
            minimum volume

        Returns
        -------
        None.

        """
        range_plot = 1
        if run_var["model_volume"] == True:
            range_plot = 2

        for i in range(0, range_plot):
            if i == 0:
                df_mod_plot = df_y_all.copy()
                df_meas_plot = df_y.copy()
                ext = "stage"
                if range_plot == 2:
                    ext = "volume"

            if i == 1:
                df_mod_plot = df_y_all_s.copy()
                df_meas_plot = df_y_s.copy()
                ext = "stage"

            fig, ax = plt.subplots(figsize=(10, 6))
            barax = ax.twinx()

            # plot input precipitation and effective rainfall
            barax.bar(
                df_in_plot.index,
                df_in_plot.mean(axis=1),
                facecolor="steelblue",
                alpha=0.3,
            )

            barax.bar(
                df_precip_plot.index,
                df_precip_plot.mean(axis=1),
                facecolor="rosybrown",
                alpha=0.3,
            )

            barax.set_ylim(0, 100)
            barax.set_ylabel("precipitation [mm]")
            if ext == "volume":
                ax.set_ylabel("volume [m$^3$]")
            if ext == "stage":
                ax.set_ylabel("stage [m]")
            ax.set_xlabel("Date")
            fig.tight_layout()

            # plot measured time series
            ax.fill_between(
                df_mod_plot.index,
                df_mod_plot.quantile(0.84, axis=1),
                df_mod_plot.quantile(0.16, axis=1),
                color="blue",
                alpha=0.1,
                lw=0,
            )

            ax.fill_between(
                df_mod_plot.index,
                df_mod_plot.quantile(0.975, axis=1),
                df_mod_plot.quantile(0.025, axis=1),
                color="blue",
                alpha=0.1,
                lw=0,
            )

            df_meas_plot.plot(ax=ax, color="orange", legend=False)

            df_mod_plot.median(axis=1).plot(ax=ax, color="blue", lw=0.5, zorder=10)

            # Add horizontal lines associated to quantiles (flood duration)
            for iq in np.array([0.05, 0.5, 0.95]):
                ax.hlines(
                    y=df_mod_plot.median(axis=1).quantile(
                        iq,
                        interpolation="nearest",
                    ),
                    xmin=df_y_all.index[0],
                    xmax=df_y_all.index[-1],
                    color="blue",
                    alpha=0.25,
                    ls="--",
                )

                ax.hlines(
                    y=df_meas_plot.quantile(
                        iq,
                        axis=0,
                        interpolation="nearest",
                    ),
                    xmin=df_y.index[0],
                    xmax=df_y.index[-1],
                    color="orange",
                    alpha=0.25,
                    ls="--",
                )

            # Compute evaluation metrix
            try:
                NSE = outputs.NSE(df_mod_plot.median(axis=1).to_frame(), df_meas_plot)

                (KGE, kge_1, kge_2, kge_3) = outputs.KGE(
                    df_mod_plot.median(axis=1).to_frame(), df_meas_plot
                )

                BIAS = outputs.BIAS(df_mod_plot.median(axis=1).to_frame(), df_meas_plot)
            except:
                NSE = "N/A"
                KGE = "N/A"
                BIAS = "N/A"

            plt.xlabel("Date")

            plt.suptitle(
                "Model param propagation "
                + str(run_var["site_name"])
                + " NSE: "
                + str(NSE)
                + ", KGE: "
                + str(KGE)
                + " BIAS[%]: "
                + str(int(BIAS))
            )

            plt.tight_layout()
            plt.savefig(
                path_results
                + "/"
                + str(sel_time_series)
                + "/"
                + "modelled_TS_model_param_propagation"
                + "_NSE_"
                + str(NSE)
                + "_KGE_"
                + str(KGE)
                + "_BIAS_"
                + str(BIAS)
                + str(ext)
                + ".png"
            )

            plt.close()

        return ()

    ###################################################################

    def autocorrelation_analysis_model_parameters(
        accepted, variables, mask_range, path_results, m_ap, site_name
    ):
        """
        Assess autocorrelation of the values of the accepted model parameters
        from the McMC of the calibration process. In theory there should not
        be correlation between one value and the previous one. If there is it
        can mean that the step size is too small or too large


        Parameters
        ----------
        accepted : array
            Contains the accepted model parameters from the calibration process
        variables : array
            Name of the variables
        mask_range : array
            Specify which variables where calibrated
        path_results : str
            path where to store the results
        m_ap : str
            mode used for the calibration process (EM or LM)
        site_name : str
            name of the site

        Returns
        -------
        None.

        """

        try:
            accepted_var = np.array(
                [list(x[np.array(mask_range) == True]) for x in accepted]
            )
            variables_var = np.array(variables)[np.array(mask_range) == True]

            df_var = pd.DataFrame(accepted_var, columns=list(variables_var))

            # resamplig applied as only one variable was modified at a time
            re_sampling = len(df_var.columns)
            df_var = df_var.iloc[::re_sampling, :]

            lag = df_var.shape[0]
            if lag > 200:
                lag = 200

            auto_corr_list = []
            for l in range(0, lag):
                auto_corr = []
                for j in range(0, len(df_var.columns)):
                    auto_corr_0 = df_var.iloc[:, j].squeeze().autocorr(lag=l)

                    auto_corr.append(auto_corr_0)
                auto_corr_list.append(auto_corr)

            df_auto = pd.DataFrame(
                np.array(auto_corr_list), index=np.arange(lag), columns=df_var.columns
            )

            fig, ax = plt.subplots()
            df_auto.plot(ax=ax)

            ax.legend(loc=0)
            ax.set_title("Auto-correlation for all variables")
            ax.set(xlabel="lag", ylabel="autocorrelation", ylim=(-1, 1))
            plt.tight_layout()
            plt.savefig(path_results + "/model_parameters_autocorrelation_analysis.png")

            plt.close()

        except:
            print("      ~ ISSUES at performing correlation analysis between vairables")

        return ()

    ###################################################################
    def accepted_gamma_distributions(
        window, accepted, variables, path_results, m_ap, site_name
    ):
        """

        Parameters
        ----------
        window : int
            length of the window for computing gamma distribution
        accepted : array
            Contains the accepted model parameters from the calibration process
        variables : array
            Name of the variables
        path_results : str
            path where to store the results
        m_ap : str
            mode used for the calibration process (EM or LM)
        site_name : str
            name of the site

        Returns
        -------
        None.

        """

        print("    computing accepted transfer fucntions...")
        r_win = np.linspace(0, window, window)

        fig, ax = plt.subplots(nrows=1, ncols=1)

        df_g1 = pd.DataFrame()

        x_int = int(len(accepted) / 1000)
        if x_int < 1:
            x_int = 1

        gamma_dist = r_win * 0

        indx_a = [
            x for x in range(0, len(variables)) if variables[x].startswith("a_gamma")
        ][0]

        indx_b = [
            x for x in range(0, len(variables)) if variables[x].startswith("b_gamma")
        ][0]

        for i_num, i in enumerate(accepted[0:-1:x_int]):
            gamma_dist = stats.gamma.pdf(
                r_win, a=accepted[i_num][indx_a], scale=accepted[i_num][indx_b]
            )

            gamma_dist = gamma_dist / np.sum(gamma_dist)
            df_g1 = pd.concat(
                [df_g1, pd.DataFrame(gamma_dist, columns=[str(i_num)])], axis=1
            )

        df_g1_plot = df_g1[df_g1 > np.max(np.max(df_g1)) * 1e-3]

        print("    imaging accepted transfer functions...")
        for i_num, i in enumerate([df_g1_plot]):
            ax.fill_between(
                i.index,
                i.quantile(q=0.84, axis=1).values,
                i.quantile(q=0.16, axis=1).values,
                color="k",
                alpha=0.2,
                lw=0,
            )

            ax.fill_between(
                i.index,
                i.quantile(q=0.98, axis=1).values,
                i.quantile(q=0.02, axis=1).values,
                color="k",
                alpha=0.2,
                lw=0,
            )

            ax.plot(i.median(axis=1), color="k")

        ax.set_title("Accepted Gamma Parameters")
        ax.set_xlabel("Antecedent days")
        ax.set_ylabel("Proportion of Recharge")

        ax.invert_xaxis()

        plt.tight_layout()
        plt.savefig(path_results + "/" + "gamma_distribution.png")

        plt.close()
        return ()

    ###################################################################
    def plot_distribution_model_parameters(
        model_par,
        accepted,
        variables,
        mask_range,
        path_results,
        m_ap,
        s_name,
        check_var=False,
    ):
        """
        Generates trace plots and histograms for the accepted model
        parameters from the calibration process

        Parameters
        ----------
        model_par : dictionary
            contains information about the input file used for calibration
            process
        accepted : array
            Contains the accepted model parameters from the calibration process
        variables : array
            Name of the variables
        mask_range : array
            Specify which variables where calibrated
        path_results : str
            path where to store the results
        m_ap : str
            mode used for the calibration process (EM or LM)
        s_name : str
            name of the site
        check_var : bool, optional
            Specify if we are sampling the prior or not. The default is False.

        Returns
        -------
        None.

        """

        # visualize accepted values for each variable independently
        accepted_plot = []
        range_plot = []
        v_range = model_par["v_range"]
        mask_range = np.array(mask_range)

        for i in range(0, accepted.shape[0]):
            accepted_plot.append(list(accepted[i, :][mask_range == True]))

        for i in range(0, len(mask_range)):
            if mask_range[i] in [True, "True"]:
                range_plot.append(v_range[i])

        accepted_plot = np.array(accepted_plot)
        variables_plot = np.array(variables)[mask_range == True]

        fig = plt.figure(figsize=(18, 10))
        for i in range(0, accepted_plot.shape[1]):
            ax = fig.add_subplot(2, accepted_plot.shape[1], i + 1)

            ax.plot(accepted_plot[:, i], lw=0.1)

            ax.set_title("Trace for $" + variables_plot[i] + "$")

            ax.set_xlabel("Iteration")
            ax.set_ylabel(variables_plot[i])
            ax = fig.add_subplot(
                2, accepted_plot.shape[1], accepted_plot.shape[1] + i + 1
            )

            a_level = ax.hist(accepted_plot[:, i], bins=20)

            try:
                h_line = np.sum(a_level[0]) / (
                    (range_plot[i][1] - range_plot[i][0])
                    / (a_level[1][1] - a_level[1][0])
                )

                ax.axhline(y=h_line, color="r", linestyle="--")
            except:
                print(" ~ issues with h line")

            ax.set_ylabel("Frequency (normed)")
            ax.set_xlabel(variables_plot[i])
            ax.set_title("Histogram of $" + variables_plot[i] + "$")

        fig.tight_layout()

        if check_var == False:
            plt.savefig(path_results + "/" + "trace_and_distribution_plots.png")
        else:
            plt.savefig(
                path_results + "/" + "trace_and_distribution_plots_prior_sampling.png"
            )
        plt.close()

        return ()

    ###################################################################
    def corner_plot(
        var_accepted,
        txt,
        mask_range,
        variables,
        path_results,
        m_ap,
        site_name,
        check_var_samp,
        color_sel="k",
    ):
        """
        Generates corner plor of accepted model parameters. Can help to assess
        if the calibration process is converging, has converged, or is having
        problems

        Parameters
        ----------
        var_accepted : array
            accepted model parameters
        txt : str
            extension to add when storing the plots
        mask_range : array
            Specify which variables where calibrated
        variables : array
            Name of the variables/model parameters
        path_results : str
            path where to store the results
        m_ap : str
            mode used for the calibration process (EM or LM)
        site_name : str
            name of the site
        check_var_samp : bool
            Specify if we are sampling the prior or not. The default is False.
        color_sel : str, optional
            Color for the plot. The default is 'k'.

        Returns
        -------
        None.

        """

        # avoid plotting variables without dynamic range
        acc_range = np.max(var_accepted, axis=0) - np.min(var_accepted, axis=0)
        for i_num, i in enumerate(acc_range):
            if i <= 0:
                mask_range[i_num] = False

        # plot dependency between variables
        try:
            accepted_plot = [
                list(x[np.array(mask_range) == True]) for x in var_accepted
            ]
            variable_plot = np.array(variables)[np.array(mask_range) == True]

            corner.corner(
                np.array(accepted_plot),
                labels=list(variable_plot),
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 12},
                color=color_sel,
            )

            if check_var_samp == False:
                plt.savefig(path_results + "/calibration/" + txt + ".png")
            else:
                plt.savefig(
                    path_results
                    + "/calibration/"
                    + "sampling_the_prior_"
                    + txt
                    + ".png"
                )
            plt.close()
        except:
            print("    ~ Issues at generating the corner plots")

        return ()


#######################################################################
#######################################################################
class outputs:
    ###################################################################
    def store_likelihood_values(accepted_lik, path_out, site_name, version):
        """
        Store likelihood values of accepted model parameters

        Parameters
        ----------
        accepted_lik : array
            array with likelihood values of accepted model parameters
        path_out : str
            path where to store the data
        site_name : str
            name of the site
        version : str
            version of the job

        Returns
        -------
        None.

        """

        df_lik = pd.DataFrame(accepted_lik)
        df_lik.to_csv(
            path_out
            + str(site_name)
            + "_"
            + str(version)
            + "/calibration/accepted_lik_cal_process.csv"
        )

        return ()

    ###################################################################
    def results_and_products(
        df_precip_sel,
        df_et_sel,
        df_precip_evap,
        df_cal,
        df_val,
        df_for,
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
        text_sdp="",
        text_for="",
    ):
        """
        export results and products from calibration/validation and forward calculations


        Parameters
        ----------
        df_precip_sel:DataFrame
            precipitation time series
        df_et_sel: DataFrame
            evapotranspiration time series
        df_precip_evap:DataFrame
            precipitation minus evaporation time series
        df_cal: DataFrame
            water level time series calibration
        df_val: DataFrame
            water level time series validations
        df_for: DataFrame
            water level time series forward
        main_path:str
            path to UISCEmod general directory
        path_sav:str
            path to stage area volume curves
        path_results:str
            path where to store the results
        path_sites_input_ts:str
            path to input time series
        run_var : dictionary
            Contains information about site and the job being
            performed
        lm_model_par : dictionary
            contains information for calibration of LM approach
        em_model_par : dictionary
            contains information for calibration of EM approach
        f_vsa:funtion
            fountions to convert from volume to stage and area
        v_range: array
            range of possible values for each model parameters
        mask_range: array
            specify which parameters are being calibrated
        min_values_vol:array
            minimum volumes to be considered for calibration,validation, forward
        spillpoint_vol: float
            elevation of the spillpoint/overtop
        initial_vol:float
            initial volume to use for the LM approach
        check_var_samp:bool
            specify if we are sampling the prior
        text_inv: str
            Text to be added to inversion products. default: ''
        text_sdp: str
            Text to be added to products related to sampling the prior. default: ''
        text_for: str
            Text to be added to products related to forward calculations. default: ''

        Returns
        -------
        None.

        """

        if run_var["model_approach"] == "EM":
            additional_param = []

        if run_var["model_approach"] == "LM":
            additional_param = [
                pd.concat(
                    [
                        df_precip_sel.copy(),
                        df_et_sel.copy(),
                    ],
                    axis=1,
                ),
                spillpoint_vol,
                f_vsa,
                initial_vol,
            ]

        if run_var["model_approach"] == "EM":
            model_par = em_model_par
        if run_var["model_approach"] == "LM":
            model_par = {
                "v_range": v_range,
                "ref_date": lm_model_par["ref_date"],
                "window_lm": lm_model_par["window_lm"],
            }

            for ij in range(0, len(mask_range)):
                if mask_range[ij] in ["True", True]:
                    mask_range[ij] = True
                else:
                    mask_range[ij] = False

        if run_var["uisce_mode"] == "calibration":
            sel_modes = ["calibration", "validation"]
            text = text_inv

        if run_var["uisce_mode"] == "sample_the_prior":
            sel_modes = ["calibration"]
            text = text_sdp

        if run_var["uisce_mode"] == "forward":
            sel_modes = ["forward"]
            text = text_for

        for mode in sel_modes:
            try:
                if mode == "calibration":
                    print("   visualizing calibration...")
                    df = df_cal.copy()
                    min_value = min_values_vol[0]
                    sel_time_series = run_var["pot_time_series"][0]
                    impose_overlap = True

                if mode == "validation":
                    print("   visualizing validation...")
                    df = df_val.copy()
                    min_value = min_values_vol[1]
                    sel_time_series = run_var["pot_time_series"][1]
                    impose_overlap = True

                if mode == "forward":
                    print("   visualizing forward...")
                    df = df_for.copy()
                    min_value = min_values_vol[2]
                    sel_time_series = run_var["pot_time_series"][2]
                    impose_overlap = False

                outputs.outputs_analysis(
                    df.copy(),
                    check_var_samp,
                    df_precip_sel.copy(),
                    df_et_sel.copy(),
                    main_path,
                    run_var,
                    path_sav,
                    min_value,
                    additional_param,
                    mask_range,
                    path_results,
                    model_par,
                    sel_time_series,
                    df_precip_evap.copy(),
                    path_sites_input_ts,
                    impose_overlap=impose_overlap,
                    text=text,
                )

            except:
                print(" ~ Something went wrong when generating the output files...")

        return ()

    ###################################################################
    def compute_forward_solution(
        obs,
        m_ap,
        path_results,
        df_in,
        window,
        min_val,
        mask_range,
        additional_param,
        model_par,
        sel_time_series,
        run_var,
        df_precip_evap,
        text="",
        export_additional_ts=True,
        impose_overlap=False,
    ):
        """


        Parameters
        ----------
        obs : DataFrame
            measured gwl data
        m_ap : str
            selected model approach (EM or LM)
        path_results : str
            path where to store results
        df_in : DataFrame
            Contains ER time series
        window : Int
            Lentgh of the window for which to calculate the gamma distribution
        min_val : float
            smallest values to model
        mask_range : array
            specifies which variables need to be calibrated or not
        additional_param : list
            Contains information necessary for the forward calculations
        model_par : dictionary
            Information about the calibration process
        sel_time_series : str
            mode used in this job (calibration, validation, forward...)
        run_var : dictionary
            Contains information about site and the job being
            performed
        df_precip_evap : DataFRame
            precipitation minus evaporation time series
        text : str, optional
            extension to be used when storing the results. The default is ''.
        export_additional_ts :bool, optional
            Export discharge and recharge time series. The default is True.
        impose_overlap : bool, optional
            Impose overlap between obs and model time series.
            The default is False.

        Returns
        -------
        Forward calculations including water levels/volume time series,
        recharge and discharge time series (for LM approach), and confidence
        intervals based on quantiles
        """

        df_y_all = pd.DataFrame()
        df_disch = pd.DataFrame()
        df_rech = pd.DataFrame()
        df_spill = pd.DataFrame()

        # propagation
        print("    Computing model propagation...")
        # Select forward and related variables
        if m_ap == "LM":
            forward = forwards.lm_reservoir_model

        if m_ap == "EM":
            forward = forwards.gamma_transfer_function

        # Read accepted model parameters
        f = []
        for dirpath, dirnames, filenames in walk(path_results + "/calibration/"):
            f.extend(filenames)
            break

        file = [x for x in f if x.startswith("accepted_model_parameters.csv")]

        df_models = pd.read_csv(
            path_results + "/calibration/" + file[0], index_col=0, comment="#"
        )

        # select random models if there are too many
        # to speed up the process of plotting
        max_forward_calculations = np.min(np.array([500, int(run_var["iterations"])]))
        if df_models.shape[0] > max_forward_calculations:
            df_models_sel = df_models.sample(
                n=max_forward_calculations, random_state=random_state
            )
        else:
            df_models_sel = df_models.copy()

        # ####
        # Compute time series and rech/disch for LM model
        variables_all = list(df_models_sel.columns)
        for i in range(0, df_models_sel.shape[0]):
            i_var = list(df_models_sel.iloc[i])
            x_var = df_in[i_var[-1]]

            # define input parameters and compute the forward
            if m_ap == "EM":
                inputs = [
                    i_var[0 : len(variables_all) - 1],
                    x_var,
                    variables_all,
                    window,
                    min_val,
                    mask_range,
                ]

                wl_ts = forward(inputs)

            if m_ap == "LM":
                y_var_f = np.array(i_var[0 : len(variables_all) - 1])
                inputs = [
                    y_var_f,
                    x_var.copy(),
                    variables_all[0:-1],
                    additional_param[1],
                    window,
                    min_val,
                    float(additional_param[3]),
                    additional_param[2],
                    model_par["ref_date"],
                    df_precip_evap.copy(),
                ]

                (wl_ts, disch, rech, spill) = forward(inputs)

                wl_ts.columns = [str(i)]
                disch.columns = [str(i)]
                rech.columns = [str(i)]
                spill.columns = [str(i)]

            df_y_all = pd.concat([df_y_all, wl_ts], axis=1)

            if m_ap == "LM":
                df_disch = pd.concat([df_disch, disch], axis=1)

                df_rech = pd.concat([df_rech, rech], axis=1)

                df_spill = pd.concat([df_spill, spill], axis=1)

        df_y_all.columns = list(range(len(df_y_all.columns)))
        df_disch.columns = list(range(len(df_disch.columns)))
        df_rech.columns = list(range(len(df_rech.columns)))
        df_spill.columns = list(range(len(df_spill.columns)))

        # Compare with measured data
        if (sel_time_series in ["calibration", "validation"]) or (
            impose_overlap == True
        ):
            df_y_all = df_y_all[df_y_all.index.isin(obs.index)]
            df_disch = df_disch[df_disch.index >= obs.index[0]]
            df_rech = df_rech[df_rech.index >= obs.index[0]]
            df_spill = df_spill[df_spill.index >= obs.index[0]]

        # Store time series
        if run_var["model_volume"] == True:
            ext = "volume"
            format_num = format_num_vol
        else:
            ext = "_stage_"
            format_num = format_num_stg

        df_y_all.to_csv(
            path_results
            + "/"
            + str(sel_time_series)
            + "/"
            + "model_"
            + str(ext)
            + "_ts"
            + text
            + ".csv",
            float_format=format_num,
        )

        if (
            (m_ap == "LM")
            and (run_var["model_volume"] == True)
            and (export_additional_ts == True)
        ):
            df_disch.to_csv(
                path_results
                + "/"
                + str(sel_time_series)
                + "/"
                + "model_"
                + "outflow"
                + "_ts"
                + text
                + ".csv",
                float_format=format_num,
            )

            df_rech.to_csv(
                path_results
                + "/"
                + str(sel_time_series)
                + "/"
                + "model_"
                + "inflow"
                + "_ts"
                + text
                + ".csv",
                float_format=format_num,
            )

        # export spill volumes of water
        if df_spill.max().max() > 1:
            df_spill.to_csv(
                path_results
                + "/"
                + str(sel_time_series)
                + "/"
                + "model_"
                + "overtop"
                + "_ts"
                + text
                + ".csv",
                float_format=format_num,
            )

        try:
            df_err = pd.read_csv(
                path_results + "/calibration/errors_quantiles.csv",
                index_col=0,
                comment="#",
            )

            q_up = df_err.loc[0.840][0]
            q_2up = df_err.loc[0.975][0]
            q_down = df_err.loc[0.160][0]
            q_2down = df_err.loc[0.025][0]

        except:
            q_up = np.array(i_var[-2])
            q_2up = 2 * np.array(i_var[-2])
            q_down = -np.array(i_var[-2])
            q_2down = -2 * np.array(i_var[-2])

        list_cols = list(range(0, len(df_y_all.columns)))
        df_y_all.columns = list_cols
        if m_ap == "LM":
            df_disch.columns = list_cols
            df_rech.columns = list_cols
            df_spill.columns = list_cols

        return (df_y_all, df_disch, df_rech, q_up, q_2up, q_down, q_2down)

    #######################################################################

    def NSE(df_y_mod, df_y_meas):
        """
        Compute NSE for data fit assessment

        Parameters
        ----------
        df_y_mod : DataFrame
            Modelled times series
        df_y_meas : DataFrame
            Measured time series

        Returns
        -------
        NSE value

        """

        df_y_mod = df_y_mod[df_y_mod.index.isin(df_y_meas.index)]
        df_y_meas = df_y_meas[df_y_meas.index.isin(df_y_mod.index)]

        NSE_t = []
        NSE_b = []
        for i in range(0, len(df_y_mod)):
            if np.isnan(df_y_meas.iloc[i][0] * df_y_mod.iloc[i][0]) == False:
                NSE_t.append((df_y_meas.iloc[i][0] - df_y_mod.iloc[i][0]) ** 2)
                NSE_b.append((df_y_meas.iloc[i][0] - np.nanmean(df_y_meas)) ** 2)
        NSE = 1 - (np.nansum(NSE_t) / np.nansum(NSE_b))
        return np.round(NSE, 2)

    #######################################################################

    def KGE(df_y_mod, df_y_meas):
        """
        Compute KGE for data fit assessment

        Parameters
        ----------
        df_y_mod : DataFrame
            Modelled times series
        df_y_meas : DataFrame
            Measured time series

        Returns
        -------
        KGE values and the value for each of the three
        components deffining the KGE

        """

        df_y_mod = df_y_mod[df_y_mod.index.isin(df_y_meas.index)]
        df_y_meas = df_y_meas[df_y_meas.index.isin(df_y_mod.index)]
        # df_y_mod[np.isnan(df_y_meas) == True] = np.nan

        try:
            cor = df_y_mod.corrwith(df_y_meas.squeeze(), axis=0)[0]
            # cor,_ = pearsonr(df_y_mod.squeeze(),df_y_meas.squeeze())
            kge_1 = (cor - 1) ** 2
            # kge_2 = ((df_y.rep_val.std()/np.array(df_y.measured).std()) - 1)**2
            kge_2 = ((np.nanstd(df_y_mod) / np.nanstd(df_y_meas)) - 1) ** 2

            kge_3 = ((np.nanmean(df_y_mod) / np.nanmean(df_y_meas)) - 1) ** 2
            KGE = 1 - np.sqrt(kge_1 + kge_2 + kge_3)
        except:
            KGE = -9999.0
            kge_1 = -9999.0
            kge_2 = -9999.0
            kge_3 = -9999.0
            cor = -9999.0

        return (
            np.round(KGE, 2),
            cor,
            np.nanstd(df_y_mod) / np.nanstd(df_y_meas),
            np.nanmean(df_y_mod) / np.nanmean(df_y_meas),
        )

    ###################################################################
    def BIAS(df_y_mod, df_y):
        """
        Compute BIAS for data fit assessment

        Parameters
        ----------
        df_y_mod : DataFrame
            Modelled times series
        df_y : DataFrame
            Measured time series

        Returns
        -------
        BIAS value

        """

        max_norm = df_y.max()[0]
        min_norm = df_y.min()[0]

        df_y_n = (df_y - min_norm) / (max_norm - min_norm)
        df_y_mod_n = (df_y_mod - min_norm) / (max_norm - min_norm)
        bias = np.round(
            100 * np.nanmean((df_y_n.squeeze() - df_y_mod_n.mean(axis=1))), 2
        )

        return bias

    ###################################################################

    def store_accepted_ts(df_accepted_ts, df_accepted_ts_tmp, df_cal, path_out):
        """
        store accepted time series

        Parameters
        ----------
        df_accepted_ts : DataFrame
            Accepted time series
        df_accepted_ts_tmp : DataFrame
            Accepted time series that where not yet included
        df_cal : DataFrame
            Measured time series
        path_out : str
            Path where to store the accepted time series

        Returns
        -------
        DataFrame with all accepted time series including the measured time
        series to facilitate comparison

        """

        df_accepted_ts = pd.concat([df_accepted_ts, df_accepted_ts_tmp], axis=1)

        df_accepted_ts_store = pd.concat([df_cal, df_accepted_ts], axis=1)

        df_accepted_ts_store.to_csv(path_out, float_format=format_num_vol)

        return df_accepted_ts

    ###################################################################

    def outputs_analysis(
        obs,
        check_var_samp,
        df_precip,
        df_et,
        main_path,
        run_var,
        path_sav,
        min_val,
        additional_param,
        mask_range,
        path_results,
        model_par,
        sel_time_series,
        df_precip_evap,
        path_sites_input_ts,
        std_solution=False,
        export_additional_ts=True,
        text="",
        impose_overlap=False,
    ):
        """
        Compute forward and assess quality of the calibration process
        and data fit

        Parameters
        ----------
        obs : DataFrame
            measured gwl data
        check_var_samp : bool
            Specify if we are sampling the prior
        df_precip : DataFrame
            precipitation time series
        df_et : DataFrame
            Evapotranspiration time series
        main_path : str
            main path from where UISCEmod is running
        run_var : dictionary
            Contains information about site and the job being
            performed
        path_sav : str
            path to stage area volume curves
        min_val : float
            minimum value to calculate
        additional_param : list
            Contains information necessary for the forward calculations
        mask_range : array
            specifies which variables need to be calibrated or not
        path_results : str
            path where to store results
        model_par : dictionary
            Information about the calibration process
        sel_time_series : str
            mode used in this job (calibration, validation, forward...)
        df_precip_evap : DataFrame
            Dataframe with precipitation minus evapotranspiration times series
        path_sites_input_ts : str
            path where the input time series are
        std_solution : bool, optional
            Specify if you want to use the constrained std when deffining the
            errors instead of quantiles. The default is False.
        export_additional_ts : bool, optional
            export discharge and recharge time series for LM approach. The default is True.
        text : str, optional
            extension to add when storing the results. The default is ''.
        impose_overlap : bool, optional
            impose overlap between obs and model time series. The default is False.

        Returns
        -------
        None.

        """

        site_name = run_var["site_name"]
        m_ap = run_var["model_approach"]

        if run_var["model_approach"] == "EM":
            window = model_par["window_em"]

        if run_var["model_approach"] == "LM":
            window = model_par["window_lm"]

        # ####
        # Read accepted model parameters
        f = []
        for dirpath, dirnames, filenames in walk(path_results + "/calibration/"):
            f.extend(filenames)
            break

        file = [x for x in f if x.startswith("accepted_model_parameters.csv")]

        df_models = pd.read_csv(
            path_results + "/calibration/" + file[0], index_col=0, comment="#"
        )

        accepted = np.array(df_models.iloc[:, 0:-1])
        accepted_met = np.array(df_models.iloc[:, -1])
        variables = list(df_models.columns[0:-1])

        sel_hsmax = np.unique(df_models["met_input"])
        df_models["met_input"] = [int(x.split("_")[1]) for x in df_models["met_input"]]
        accepted_all = np.array(df_models)
        variables_all = list(df_models.columns)
        model_par_all = model_par.copy()
        met_data_range = np.array(
            [
                float(run_var["met_data"][1][0].split("_")[1]),
                float(run_var["met_data"][1][-1].split("_")[1]),
            ]
        )
        model_par_all["v_range"] = np.concatenate(
            (model_par_all["v_range"], [met_data_range])
        )
        if len(set(df_models["met_input"])) > 1:
            mask_range_all = np.append(mask_range, np.array([True]))
        else:
            mask_range_all = np.append(mask_range, np.array([False]))

        # Generate recharge based on values from accepted model parameters
        run_var["met_data"][1] == list(set(accepted_met))

        if run_var["uisce_mode"] == "climate_change_SL":
            df_in = pd.read_csv(
                path_sites_input_ts + site_name + "_climate_change_ER_ts.csv",
                index_col=0,
                parse_dates=[0],
            )
        else:
            df_in = pd.read_csv(
                path_sites_input_ts + site_name + "_ER_ts.csv",
                index_col=0,
                parse_dates=[0],
            )

        if sel_time_series == "calibration":
            st_in = run_var["st_time_series"][0]
            end_in = run_var["end_time_series"][0]

        elif sel_time_series == "validation":
            st_in = run_var["st_time_series"][1]
            end_in = run_var["end_time_series"][1]

        else:
            st_in = run_var["st_time_series"][2]
            end_in = run_var["end_time_series"][2]

        # Make sure we keep dates for spinup period
        if run_var["model_approach"] == "LM":
            if st_in > model_par["ref_date"]:
                st_in = model_par["ref_date"]

        # Keep additional dates to run the gamma distribution
        df_in = df_in[df_in.index >= st_in - datetime.timedelta(days=window + 10)]
        df_in = df_in[df_in.index <= end_in]

        df_precip_evap = df_precip_evap[df_precip_evap.index.isin(df_in.index)]

        if sel_time_series == "calibration":
            print("    imaging model parameters...")
            visualization.plot_distribution_model_parameters(
                model_par_all,
                accepted_all,
                variables_all,
                mask_range_all,
                path_results + "/" + str(sel_time_series) + "/",
                m_ap,
                site_name,
                check_var=check_var_samp,
            )

            if run_var["uisce_mode"] != "sample_the_prior":
                visualization.accepted_gamma_distributions(
                    window,
                    accepted,
                    variables,
                    path_results + "/" + str(sel_time_series) + "/",
                    m_ap,
                    site_name,
                )

        ###########################################################
        if check_var_samp != True:
            # # ##########
            # # get errors
            df_err = pd.read_csv(
                path_results + "/calibration/errors_quantiles.csv",
                index_col=0,
                comment="#",
            )

            # Define confidence intervals
            q_up = df_err.loc[0.840][0]
            q_2up = df_err.loc[0.975][0]
            q_down = df_err.loc[0.160][0]
            q_2down = df_err.loc[0.025][0]

            # # Measured
            df_y = pd.DataFrame(obs.values, index=obs.index, columns=["measured"])

            ##########################################################
            (
                df_y_all,
                df_disch,
                df_rech,
                q_up,
                q_2up,
                q_down,
                q_2down,
            ) = outputs.compute_forward_solution(
                obs,
                m_ap,
                path_results,
                df_in.copy(),
                window,
                min_val,
                mask_range,
                additional_param,
                model_par,
                sel_time_series,
                run_var,
                df_precip_evap.copy(),
                text=text,
                export_additional_ts=export_additional_ts,
                impose_overlap=impose_overlap,
            )

            if (run_var["uisce_mode"] in ["calibration", "validation"]) or (
                impose_overlap == True
            ):
                df_y_all = df_y_all[df_y_all.index.isin(obs.index)]
                obs = obs[obs.index.isin(df_y_all.index)]

                # Show error distribution for segments of the time series that
                # overlap with measured data
                outputs.errors_analysis(
                    df_y_all.copy(),
                    obs.copy(),
                    min_val,
                    accepted.copy(),
                    (path_results + "/" + sel_time_series + "/"),
                    std_solution=std_solution,
                )

                for i in obs.index:
                    if np.isnan(obs.loc[i])[0] == True:
                        df_y_all.loc[[i]] = np.nan * df_y_all.loc[[i]]

            # get stage time series
            if sel_time_series == "calibration":
                min_i = 0
            elif sel_time_series == "validation":
                min_i = 1
            else:
                min_i = 2

            min_stage = run_var["min_stage"][min_i]
            site_name = run_var["site_name"]

            if run_var["model_volume"] == True:
                df_y_all_s = preprocessing.convert_hydrol_param(
                    path_sav,
                    df_y_all.copy(),
                    hydro_out="stage",
                    hydro_in="volume",
                    min_stage=min_stage,
                    return_f=False,
                    simplify_f=-999,
                )

                df_y_s = preprocessing.convert_hydrol_param(
                    path_sav,
                    df_y.copy(),
                    hydro_out="stage",
                    hydro_in="volume",
                    min_stage=min_stage,
                    return_f=False,
                    simplify_f=-999,
                )

                df_y_all_s_store = pd.concat([df_y_s, df_y_all_s], axis=1)

                df_y_all_s_store.to_csv(
                    path_results + "/" + sel_time_series + "/"
                    # + str(run_var['site_name'])
                    + "model_stage_ts.csv",
                    float_format=format_num_stg,
                )

                # Prepare meteorological data
                df_in_plot = df_in[df_in.index >= df_y_all.index[0]]
                df_in_plot = df_in_plot[df_in_plot.index <= df_y_all.index[-1]]
                df_in_plot = df_in_plot * 1000

                # Get only the selected hs_max
                df_in_plot = df_in_plot[sel_hsmax]

                df_precip_plot = df_precip[df_precip.index >= df_y_all.index[0]]
                df_precip_plot = df_precip_plot[
                    df_precip_plot.index <= df_y_all.index[-1]
                ]

                # ####
                # plot time series using model parameters propagation
                print("    imaging time series from model parameters propagation...")
                visualization.ts_imaging_model_parameters_propagation(
                    run_var,
                    df_y_all.copy(),
                    df_y.copy(),
                    df_y_all_s.copy(),
                    df_y_s.copy(),
                    df_in_plot.copy(),
                    df_precip_plot.copy(),
                    path_results,
                    sel_time_series,
                    min_val,
                )

                # ####
                # plot time series using empirical errors
                print("    imaging time series with empirical errors...")
                visualization.ts_imaging_empirical_errors(
                    run_var,
                    df_y_all.copy(),
                    df_y.copy(),
                    path_sav,
                    min_stage,
                    q_up,
                    q_2up,
                    q_down,
                    q_2down,
                    df_in_plot.copy(),
                    df_precip_plot.copy(),
                    min_val,
                    path_results,
                    sel_time_series,
                )

                # # ####
                # # Plot stage discharge plots using net flow and modelled discharge
                print("    Imaging stage discharge curves...")
                visualization.stage_discharge_curves(
                    obs,
                    df_y_s,
                    df_y_all,
                    df_y_all_s,
                    df_disch,
                    path_results,
                    sel_time_series,
                )

                # # plot recharge discharge netflow comparisons
                if run_var["model_approach"] == "LM":
                    visualization.net_in_out_flows(
                        obs,
                        df_y_all,
                        df_disch,
                        df_rech,
                        path_results,
                        df_in_plot,
                        df_precip_plot,
                        sel_time_series,
                    )

            # ####
            if sel_time_series == "calibration":
                visualization.autocorrelation_analysis_model_parameters(
                    accepted,
                    variables,
                    mask_range,
                    path_results + "/" + sel_time_series + "/",
                    m_ap,
                    site_name,
                )

        return ()

    ###################################################################
    def store_accepted_parameters(
        accepted, accepted_met, variables, path_out, site_name, run_var, txt=""
    ):
        """
        Store accepted model parameters from the calibration process

        Parameters
        ----------
        accepted : array
            accepted model parameters
        accepted_met : array
            accepted Hsmax model parameters (SMD model)
        variables : array
            Name of the model parameters
        path_out : str
            path where to store the model parameters
        site_name : str
            name of the site
        run_var : dictionary
            Contains information about site and the job being
            performed
        txt : str, optional
            text to add at the name of the file with model parameters.
            The default is ''.

        Returns
        -------
        DataFrame with all accepted model parameters, including Hsmax

        """

        (accepted_r, accepted_met_r) = outputs.reformat_accepted_parameters(
            accepted.copy(), accepted_met.copy()
        )

        # Define dataframes
        df_accepted = pd.DataFrame(accepted_r, columns=variables)

        df_met = pd.DataFrame(accepted_met_r, columns=["met_input"])

        df_all = pd.concat([df_accepted, df_met], axis=1)

        # store the accepted models
        df_all.to_csv(
            path_out + "accepted_model_parameters.csv" + ".csv", float_format="%.3f"
        )

        return df_all

    ###################################################################

    def errors_analysis(
        df_accepted_ts,
        df_gwl,
        min_val,
        accepted,
        path_out,
        export_errors=False,
        std_solution=False,
    ):
        """
        Assess distribution of the errors between model and
        measured time series

        Parameters
        ----------
        df_accepted_ts : DataFrame
            Accepted model time series
        df_gwl : DataFrame
            Measured time series
        min_val : float
            minimum value to consider.
        accepted : array
            contains the accepted model parameters
        path_out : str
            path where to store the errors analysis
        export_errors : bool, optional
            Export errors in quantiles. The default is False.
        std_solution : bool, optional
            Plot the std of the solution. Usefull when running synthetic
            studies to check that we can recover the solutions.
            The default is False.

        Returns
        -------
        None.

        """

        ####
        # Calculate and store the errors
        df_errors_accepted_ts = df_accepted_ts.copy()
        for i_err in df_accepted_ts.columns:
            df_errors_accepted_ts[i_err] = df_accepted_ts[i_err] - df_gwl.squeeze()

            df_errors_accepted_ts[i_err] = df_errors_accepted_ts[i_err][
                df_gwl.squeeze() > min_val
            ]

        df_errors_accepted_ts.to_csv(
            path_out + "/error_ts_calibration_process.csv", float_format=format_num_vol
        )

        ####
        # Image the errors and compare with the assumption that they are
        # gaussian distributed

        # histogram errors
        x = np.array(df_errors_accepted_ts.melt().value.dropna())
        num_bins = 30
        plt.figure()
        plt.hist(x, bins=num_bins, facecolor="b", alpha=0.5, density=True)

        # guassian distriobution based on calibrated std
        range_val = np.max(np.abs(x))
        y = np.linspace(-1 * range_val, range_val, 10000)

        plt.plot(y, stats.norm.pdf(y, 0, np.quantile(accepted[:, -1], 0.5)), color="k")

        plt.fill_between(
            y,
            stats.norm.pdf(y, 0, np.quantile(accepted[:, -1], 0.84)),
            stats.norm.pdf(y, 0, np.quantile(accepted[:, -1], 0.16)),
            color="k",
            lw=0,
            alpha=0.25,
        )

        plt.fill_between(
            y,
            stats.norm.pdf(y, 0, np.quantile(accepted[:, -1], 0.975)),
            stats.norm.pdf(y, 0, np.quantile(accepted[:, -1], 0.025)),
            color="k",
            lw=0,
            alpha=0.25,
        )

        if std_solution != False:
            plt.plot(y, stats.norm.pdf(y, 0, std_solution), color="orange", ls="--")

        plt.savefig(path_out + "/error_distribution_analysis.png")

        plt.close()

        if export_errors == True:
            #####
            # Get quantiles to define the errors (in case they don't follow a
            # gaussian distribution) - They will be used later for modelling/
            # forward/climate change anlysis
            q_errors = []
            for i_q in np.arange(0, 1.005, 0.005):
                q_errors.append(df_errors_accepted_ts.melt().value.quantile(q=i_q))

            df_err_q = pd.DataFrame(q_errors, np.arange(0, 1.005, 0.005))

            # store the errors in each subfolder
            df_err_q.to_csv(
                path_out.rsplit("/", 1)[0] + "/calibration/errors_quantiles.csv",
                float_format="%.3f",
            )
            df_err_q.to_csv(
                path_out.rsplit("/", 1)[0] + "/validation/errors_quantiles.csv",
                float_format="%.3f",
            )
            df_err_q.to_csv(
                path_out.rsplit("/", 1)[0] + "/forward/errors_quantiles.csv",
                float_format="%.3f",
            )

        return ()

    ###################################################################

    def reformat_accepted_parameters(accepted, accepted_met):
        """
        Reformat the accepted model parameters to facilitate output
        analysis

        Parameters
        ----------
        accepted : array
            accapted model parameters
        accepted_met : array
            accepted SMD model parameters (Hsmax)

        Returns
        -------
        the same as inputs but with new format

        """

        # Create a list with accepted parameters
        accepted = np.array(accepted)
        accepted_met = np.array(accepted_met)

        # ####
        # Keep only unique solutions
        # This should not be necessary if the process of selecting
        # variables was working properly, but helps when also
        # selecting best meteorological parameters
        accepted_mix = np.concatenate((accepted, np.array([accepted_met]).T), axis=1)

        _, indexes = np.unique(accepted_mix, axis=0, return_index=True)

        accepted_2 = [accepted_mix[index] for index in sorted(indexes)]
        accepted_2 = np.array(accepted_2)
        accepted = accepted_2[:, 0:-1]
        accepted = accepted.astype(np.float)
        accepted_met = accepted_2[:, -1]

        accepted = np.round(accepted, 4)

        return (accepted, accepted_met)


#######################################################################
#######################################################################
class mcmc:
    ###################################################################
    def get_random_uniform_variables(x, v_range, df_in):
        """
        get random value from a range of values assuming all values
        within the range have the same probability

        Parameters
        ----------
        x : array
            current values for each model parameter
        v_range : array
            range of possible values for each model parameter
        df_in_cal : DataFrame
            DataFrame with effective rainfal time series
            for potential Hsmax values

        Returns
        -------
        new vector x with new values for model parameters and new Hsmax value.

        """

        x_new = x.copy()
        for ix in range(0, len(x)):
            x_new[ix] = random.uniform(v_range[ix][0], v_range[ix][1])
            precip_var_new = random.choice(df_in.columns)

        return (x_new, precip_var_new)

    ###################################################################
    def check_convergence(df_cal, x, i, iterations, run_var, check_var_samp):
        """
        Check if the calibration process is converging. If it is doing better
        than assuming a flat line with value equal to the mean of the measured
        time series.

        Parameters
        ----------
        df_cal : DataFrame
            measured time series
        x : array
            model parameters (last value of the accepted model parameters from
                              the McMC)
        i : int
            iteration
        iterations : int
            max number of iterations
        run_var : dictionary
            Contains information about site and the job being
            performed
        check_var_samp : bool
            Specify if we are sampling the prior

        Returns
        -------
        True or False

        """

        if (
            (x[-1] >= 0.5 * df_cal.std()[0])
            and (i > 50000)
            and (i < int(iterations * run_var["th_acceptance"]))
            and (check_var_samp == False)
            and (str(i).endswith("00"))
        ):
            check_conv = False
        else:
            check_conv = True
        return check_conv

    ###################################################################
    def keep_time_series(df_accepted_ts_tmp, df_accepted_ts, ts_tmp, i):
        """
        Decide which of the accepted time series to keep. This is performed
        to avoid memory issues. It resamples from the tmp accepted time series
        keeping 10 every 500.

        Parameters
        ----------
        df_accepted_ts_tmp : DataFrame
            Accepted time series that will be resampled before added into
            the accepted time series
        df_accepted_ts : DataFrame
            Accepted time series
        ts_tmp : Series
            subset of accepted time series
        i : int
            iteration

        Returns
        -------
        Updated dataframes with accepted time series

        """

        # Keep accepted time series
        ts_tmp = ts_tmp.to_frame()
        ts_tmp.columns = [str(i)]
        ts_tmp = np.round(ts_tmp, 2)
        df_accepted_ts_tmp = pd.concat([df_accepted_ts_tmp, ts_tmp], axis=1)

        # keep only representative time
        # series to avoid memory issues
        # This is only for later plotting the
        # results of calibration
        if len(df_accepted_ts_tmp.columns) > 500:
            print("resampling stores time series...")
            df_accepted_ts_tmp = df_accepted_ts_tmp.sample(
                10, axis=1, random_state=random_state
            )

            df_accepted_ts = pd.concat([df_accepted_ts, df_accepted_ts_tmp], axis=1)

            df_accepted_ts_tmp = pd.DataFrame()

        return (df_accepted_ts, df_accepted_ts_tmp)

    ###################################################################
    def define_initial_variables(
        v_init, run_var, param_steps_ini, hsm_step_ini, mask_range
    ):
        """
        Generate parameters for mcmc process

        Parameters
        ----------
        v_init : array
            initial model parameters
        run_var : dictionary
            Contains information about site and the job being
            performed
        param_steps_ini : array
            range of variation for each model parameter when deffining
            new potential values for the model parameters within the
            mcmc process
        hsm_step_ini : float
            step size for varying variables in % of the range
        mask_range : array
            define which variables will be calibrated

        Returns
        -------
        list of initial parameters for mcmc process

        """

        # Additional parameters for MCMC analysis
        x = v_init
        x_lik = -99999
        precip_var = run_var["met_data"][0][0]

        accepted = []
        accepted_tot = []
        rejected = []
        accepted_met = []
        accepted_tot_met = []
        rejected_met = []
        accepted_lik = []
        accepted_tot_lik = []
        df_accepted_ts = pd.DataFrame()
        df_accepted_ts_tmp = pd.DataFrame()

        cws = 0
        st_range = param_steps_ini.copy()
        hsm_step = hsm_step_ini
        last_change = 0

        var_2_tun = [xk for xk in range(0, len(mask_range)) if mask_range[xk] == True]

        return (
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
        )

    ###################################################################
    def tunning(
        x,
        x_lik,
        v_range,
        mask_range,
        df_in_cal,
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
        df_cal,
        forward,
        path_results,
        param_steps_ini,
        hsm_step,
        df_precip_evap,
        tunning_iterations=5000,
    ):
        """
        perform tunning wtihin the mcmc process to improve sampling of the
        prior. This part of the code should be simplified.

        Parameters
        ----------
        x: array
            current model parameters from mcmc
        x_lik: float
            likelihood for the current model parameters
        v_range: array
            range of possible values for model parameters
        mask_range:array
            defines which variables are calibrated
        df_in_cal: DataFrame
            potential effective rainfall time series considered during
            calibration
        precip_var:str
            current Hsmax
        st_range:array
            range of variability for each variable when deffining new model
            parameters
        variables: array
            name of the model parameters
        step_2_tun:array
            suggested new lengths of variability for each variable when deffining
            the new model parameters
        var_2_tun: array
            define variables that need to be tunned
        run_var : dictionary
            Contains information about site and the job being
            performed
        spillpoint_vol:float
            value of the spillpoint/overtop in volume
        lm_model_par:dictionary
            information related to calibration of the lm approach
        em_model_par:dictionary
            information related to calibration of the em approach
        min_values_vol:array
            minumum volumes to be modelled
        f_vsa:function
            converts from volume to stage/area
        initial_vol:float
            value of the initial volume for the lm approach
        df_cal:DataFrame
            measured time series
        forward:funtion
            related to the forward process (em or lm)
        path_results:str
            path where to store the results
        param_steps_ini : array
            range of variation for each model parameter when deffining
            new potential values for the model parameters within the
            mcmc process
        hsm_step_ini : float
            step size for varying variables in % of the range
        df_precip_evap:DataFrame
            precipitation minus evaporation times series
        tunning_iterations:int
            number of iterations to be considered for each test during
            the tunning.Default value:5000

        Returns
        -------
        new step size for model parameters and smd model parameter (hsmax)

        """

        print("\n    tunning steps length... \n")

        # store the latest values before starting the tunning
        x_orig = x.copy()
        x_lik_orig = x_lik.copy()

        # Deffine initial values before starting the tunning
        # tunning matrix with large autocorrelation
        tunning_eval_array = 1e99 * np.ones([len(var_2_tun) + 1, len(step_2_tun)])

        # Autocorrelation dataframe
        df_autocorrelation = pd.DataFrame(
            tunning_eval_array.copy(),
            index=list(var_2_tun) + ["hsm"],
            columns=list(step_2_tun),
        )

        # acceptance ratio dataframe
        df_acceptance = pd.DataFrame(
            tunning_eval_array.copy(),
            index=list(var_2_tun) + ["hsm"],
            columns=list(step_2_tun),
        )

        # Optimal acceptance ratio based on Gelman et al. (1996)
        p_jump = [0.441, 0.352, 0.316, 0.279, 0.275, 0.266, 0.261, 0.255, 0.261, 0.267]

        # Apply tunning for each variable (i) for selected ranges (j)
        for i in range(0, len(var_2_tun) + 1):
            print("# Variable: " + str(i))
            count_200 = 0
            for j in range(0, len(step_2_tun)):
                avoid_hsm_tunning = False
                if count_200 > 0:
                    break

                print("    # Step to tun:" + str(step_2_tun[j]))

                if i == len(var_2_tun):
                    print("    Hs_max")
                    var_tun = ["hsm", step_2_tun[j]]
                    hsm_step = step_2_tun[j]

                else:
                    # define variable to tun
                    var_tun = [
                        var_2_tun[i],
                        (v_range[var_2_tun[i], 1] - v_range[var_2_tun[i], 0])
                        * step_2_tun[j],
                    ]

                # initial conditions before tunning
                x_tun = x_orig.copy()
                x_lik = x_lik_orig.copy()
                accepted = []
                rejected = []

                # Start tunning for the number of selected iterations
                for k in range(0, tunning_iterations):
                    if avoid_hsm_tunning == True:
                        break

                    (
                        x_new,
                        df_in_sel,
                        precip_var_new,
                        last_change,
                    ) = mcmc.transition_model_parameters(
                        x_tun.copy(),
                        v_range.copy(),
                        mask_range.copy(),
                        df_in_cal.copy(),
                        precip_var,
                        st_range.copy(),
                        1e9,
                        hsm_step,
                        tunning=var_tun,
                        iteration=k,
                    )

                    #####
                    # Check that the new parameters fit within the prior
                    # If they do evaluate if they get accepted or not
                    if mcmc.prior(x_new, v_range, variables) == 0:
                        print("ISSUES at generating the prior!")
                        break

                    if mcmc.prior(x_new, v_range, variables) == 1:
                        # Get input parameters used to compute the forward
                        # and evaluate the likelihood
                        if run_var["model_approach"] == "LM":
                            cal_inputs = [
                                x_new.copy(),
                                df_in_sel.copy(),
                                variables,
                                spillpoint_vol,
                                lm_model_par["window_lm"],
                                min_values_vol[0],
                                initial_vol,
                                f_vsa,
                                lm_model_par["ref_date"],
                                df_precip_evap,
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
                        (x_new_lik, ts_tmp) = mcmc.lik_normal(
                            df_cal.copy(), forward, cal_inputs.copy(), run_var
                        )

                        ####
                        # If there is no accepted value, get the first one
                        vals_to_add = np.append(
                            x_new, float(precip_var_new.split("_")[1])
                        )

                        # Otherwise base it on the acceptance criteria
                        if mcmc.acceptance(x_lik, x_new_lik):
                            # update model parameters and reference likelihood
                            x = x_new
                            x_lik = x_new_lik
                            precip_var = precip_var_new

                            accepted.append(vals_to_add)
                        else:
                            rejected.append(vals_to_add)

                # evaluate condition
                try:
                    (ratio_val, autocorrelation_val) = mcmc.update_tunning_evaluation(
                        np.array(accepted)[:, i].copy(),
                        np.array(rejected)[:, i].copy(),
                        tunning_iterations,
                    )
                except:
                    ratio_val = 0
                    autocorrelation_val = 200

                if autocorrelation_val < 5:
                    autocorrelation_val = 1
                df_acceptance.iloc[i, j] = ratio_val
                df_autocorrelation.iloc[i, j] = autocorrelation_val

                print("        # ratio: " + str(np.round(ratio_val, 2)))
                print("        # autocorrelation: " + str(autocorrelation_val))

                if (
                    (var_tun[0] != "hsm")
                    and (ratio_val < 0.1)
                    and (autocorrelation_val == 200)
                ):
                    count_200 = count_200 + 1

        print("\n    # Summary correlation:")
        print(df_autocorrelation)

        print("\n    # Summary accepted ratio:")
        print(df_acceptance)

        (new_step_size) = mcmc.select_steps_sizes_from_tunning(
            df_autocorrelation.iloc[0:-1, :],
            df_acceptance.iloc[0:-1, :],
            p_jump,
            var_2_tun,
            st_range,
            param_steps_ini,
            v_range,
            variables,
            path_results,
        )

        min_auto = df_autocorrelation.loc["hsm"].min()
        hsm_ratio_tmp = df_acceptance.loc["hsm"][
            df_autocorrelation.loc["hsm"] == min_auto
        ]

        hsm_ratio_tmp = np.abs(hsm_ratio_tmp - p_jump[len(var_2_tun)])
        new_hsm_step_size = hsm_ratio_tmp.idxmin()

        return (new_step_size, new_hsm_step_size)

    ###################################################################
    def select_steps_sizes_from_tunning(
        df_autocorrelation,
        df_acceptance,
        p_jump,
        var_2_tun,
        st_range,
        std_ranges_ini,
        v_range,
        variables,
        path_results,
    ):
        """
        Select the step sizes affer all tunning options have been testes

        Parameters
        ----------
        df_autocorrelation : DataFrame
            autocorrelation values from tunning
        df_acceptance : DataFrame
            ratio of acceptance from tunning
        p_jump : array
            optimal acceptance ratio depedning on the number of variables
        var_2_tun : array
            model parameters from which tunning was performed
        st_range : array
            ranges of variability for when deffining the new model parameters
        std_ranges_ini : array
            initial step size
        v_range : array
            list of potential values for each model parameter
        variables : array
            name model parameters
        path_results : str
            path to where store the results

        Returns
        -------
        new step size to be considered when deffining the potential new variables

        """

        tunned_ranges = df_autocorrelation.iloc[:, 0].copy() * 0
        for i_l in range(0, len(tunned_ranges)):
            current_lag = 1e9
            current_accept_ratio = 0
            for j_l in range(0, len(df_autocorrelation.columns)):
                if df_autocorrelation.iloc[i_l, j_l] < current_lag:
                    tunned_ranges.iloc[i_l] = df_autocorrelation.columns[j_l]
                    current_lag = df_autocorrelation.iloc[i_l, j_l]
                    current_accept_ratio = df_acceptance.iloc[i_l, j_l]
                if df_autocorrelation.iloc[i_l, j_l] == current_lag:
                    if np.abs(
                        df_acceptance.iloc[i_l, j_l] - p_jump[len(var_2_tun) - 1]
                    ) < np.abs(current_accept_ratio - p_jump[len(var_2_tun) - 1]):
                        tunned_ranges.iloc[i_l] = df_autocorrelation.columns[j_l]
                        current_lag = df_autocorrelation.iloc[i_l, j_l]
                        current_accept_ratio = df_acceptance.iloc[i_l, j_l]

        print("####################### TUNNED RANGES: \n" + str(tunned_ranges))
        for k_tun in range(len(st_range)):
            if k_tun in tunned_ranges.index:
                st_range[k_tun] = tunned_ranges.loc[k_tun] * (
                    v_range[k_tun, 1] - v_range[k_tun, 0]
                )

                df_steps_1 = pd.DataFrame(
                    std_ranges_ini, columns=["original"], index=variables
                )

                df_steps_2 = pd.DataFrame(st_range, columns=["tunned"], index=variables)

                df_steps = pd.concat([df_steps_1, df_steps_2], axis=1)

                df_steps.to_csv(
                    path_results + "/calibration/tunned_step_size.csv",
                    float_format="%.4f",
                )
        return st_range

    ###################################################################
    def update_tunning_evaluation(accept_tunning, reject_tunning, tunning_iterations):
        """
        calculate autocorrelation and acceptance ratio of model parameters

        Parameters
        ----------
        accept_tunning : array
            accepted values
        reject_tunning : array
            rejected values
        tunning_iterations : int
            total number of iterations

        Returns
        -------
        acceptance ratio and length (number of iterations) for the
        autocorrelation to go below 0.2

        """

        # Check autocorrelation and acceptance ratio
        lag_time = 200
        df_accept_tun = pd.Series(accept_tunning)

        try:
            first_autocorr = 0
            auto_corr = []

            if len(df_accept_tun) > 200:
                for l in range(0, 200):
                    auto_corr_0 = df_accept_tun.autocorr(lag=l)
                    auto_corr.append(auto_corr_0)

                    if (first_autocorr == 0) and (np.abs(auto_corr_0) <= 0.2):
                        lag_time = l
                        first_autocorr = 1
                    if (first_autocorr == 1) and (np.abs(auto_corr_0) > 0.2):
                        lag_time = l
                        first_autocorr = 0

        except:
            pass

        accept_ratio = len(accept_tunning) / (len(accept_tunning) + len(reject_tunning))

        return (accept_ratio, lag_time)

    ###################################################################
    def proportion_accepted_models(accepted_tot, accepted, rejected, th_acceptance):
        """
        caluclate proportion of accepted model parameters for the total
        mcmc process and after burn in process.

        Parameters
        ----------
        accepted_tot : array
            accepted model parameters since begining of the mcmc proces
        accepted : array
            accepted model parameters after burn in
        rejected : array
            rejected model parameters since begining of mcmc process
        th_acceptance : int
            proportion of iterations before performin burn in

        Returns
        -------
        None.

        """

        print(
            "####\n    Total proportion of accepted models): "
            + str(np.round(len(accepted_tot) / (len(accepted_tot) + len(rejected)), 4))
        )

        print(
            "####\n    Proportion of accepted models after burn in: "
            + str(
                np.round(
                    len(accepted)
                    / ((1 - th_acceptance) * (len(accepted_tot) + len(rejected))),
                    4,
                )
            )
        )

        return ()

    ###################################################################
    def acceptance(x_acc, x_new_acc):
        """
        Decide if the new model parameters (candidate) are accepted or not

        Parameters
        ----------
        x : array
            current model parameters (last ones added in the mcmc)
        x_new : array
            new candidate model parameters

        Returns
        -------
        True or False

        """
        x_new_acc = np.array(x_new_acc)
        x_acc = np.array(x_acc)
        if x_new_acc > x_acc:
            return True

        else:
            accept = random.uniform(0, 1)
            # Since we did a log likelihood, we need to exponentiate
            # in order to compare to the random number
            # less likely x_new are less likely to be accepted
            return np.log(accept) <= (x_new_acc - x_acc)

    ######################################################################
    def lik_normal(obs, forward, inputs, run_var):
        """
        Parameters
        ----------
        x : list
            variables to be used to compute the forward plus the standard
            deviation. Standard deviation should be the last variable.
        x_var : array
            sampling points where forward will be calculated
        obs : array
            response of "measured" data used to constrain the
            variables
        forward : function
            function that computes the forward problem using
            variables in x

        Returns
        -------
        Likelihood
        modelled times series

        """
        # Assume that the distribution of the errors is normally distributed
        # around mean or median
        if run_var["model_approach"] == "EM":
            f_inputs = forward(inputs)
            min_val = 1.1 * inputs[4]

        if run_var["model_approach"] == "LM":
            f_inputs, _, _, _ = forward(inputs)
            min_val = 1.1 * inputs[5]

        # Select only periods of time for the ones there is data
        if obs.shape[0] != f_inputs.shape[0]:
            f_inputs = f_inputs[f_inputs.index.isin(obs.index)]
            obs = obs[obs.index.isin(f_inputs.index)]

        # Convert to series
        obs_all = obs.squeeze()
        f_inputs_all = f_inputs.squeeze()

        # Avoid flat lines when assessing the fit of the data
        if min_val != False:
            mask_orig = np.array([obs_all <= min_val])
        else:
            mask_orig = np.array([obs_all <= -9999])

        mask = mask_orig.copy()
        for i in range(-5, 5):
            tmp_mask = np.roll(mask_orig, i)
            mask = mask + tmp_mask

        obs = obs_all.copy()[mask[0, :] == False]
        f_inputs = f_inputs_all.copy()[mask[0, :] == False]

        # Compute likelihood
        a1 = -0.5 * np.log((inputs[0][-1] ** 2) * 2 * np.pi)
        a2t = (obs - f_inputs) ** 2
        a2b = 2 * (inputs[0][-1] ** 2)
        lik = np.nansum(a1 - (a2t / a2b))

        if np.sum(obs[obs.index.isin(f_inputs.index)]) == 0:
            print("    ##### ISSUE: MEASURED AND MODEL TIME SERIES DO NOT OVERLAP ####")

        return (lik, f_inputs_all)

    #######################################################################
    def prior(x, x_range, variables):
        """
        check that the model parameters agree with the potential values
        Parameters
        ----------
        x : list
            values model parameters
        x_range : list
            range of values for model parameters

        Returns
        -------
        int
            Returns 1 if the variables are within the valid range of numbers.
            Otherwise it returns 0: is absolutely unileky that this
            solution is valid

        """

        for i in range(0, len(x)):
            if x[i] < x_range[i][0] or x[i] > x_range[i][1]:
                print("ISSUES with " + str(variables[i]))
                return 0
        else:
            return 1

    ###################################################################
    ###################################################################
    def transition_model_parameters(
        x,
        v_range,
        mask_range,
        df_in,
        precip_var,
        st_range,
        last_change,
        range_hsm_step,
        tunning=[],
        iteration=-9999,
    ):
        """
        Generate new candidates of model parameters.
        Only one parameter is modified each time

        Parameters
        ----------
        x : array
            current model parameters (last from mcmc)
        v_range : array
            potential values for model parameters
        mask_range : array
            define which model parameters are included in the calibration
        df_in : DataFrame
            potential effective rainfall times series
        precip_var : str
            current Hs max value
        st_range : array
            range of variability when defining new model parameters
        last_change : int
            last variable that was modified
        range_hsm_step : float
            range of variability for hsmax
        tunning : list, optional
            variables to apply tunning. The default is [].
        iteration : int, optional
            current iteration within the calibration process. The default is -9999.

        Returns
        -------

        updated model parameters, effective rainfall time series, hsmax, and
        the variable that was modified.

        """

        new_x = x.copy()
        df_in_sel = df_in[precip_var].copy()

        if iteration != 0:
            # Define values that can be modified in this iteration
            pot_vals = np.array(list(range(len(x))))
            pot_vals = pot_vals[np.array(mask_range) == True]
            total_var = len(pot_vals) + 1
            x_rand = random.random()

            if "hsm" in tunning:
                x_rand = 1

            # #########################################################
            # Transition for SMD model (Hs_max variable)
            if (
                (x_rand > 1 - (1.0 / total_var))
                and (len(df_in.columns) > 1)
                and (last_change != -999)
                and ((len(tunning) == 0) or (tunning[0] == "hsm"))
            ):
                # ####
                # precipitation index
                precip_var_tmp = precip_var

                # index for current value within the array of solutions
                pot_values = sorted(df_in.columns)
                c_ind = pot_values.index(precip_var)

                while precip_var_tmp == precip_var:
                    range_hsm = int(range_hsm_step * len(df_in.columns))
                    if range_hsm < 1:
                        range_hsm = 1

                    c_list = np.arange(-1 * range_hsm, range_hsm) + c_ind

                    pot_cols = []
                    for i in c_list:
                        j = i
                        if j >= len(df_in.columns):
                            j = j - len(df_in.columns)
                        if j < 0:
                            j = j + len(df_in.columns)
                        pot_cols.append(j)

                    # select new parameter
                    precip_var_tmp = pot_values[random.choice(pot_cols)]
                precip_var = precip_var_tmp
                df_in_sel = df_in[[precip_var]].copy()

                # avoid changing this variable the next itreation
                if tunning == "hsm":
                    last_change = 1e9
                else:
                    last_change = -999

            else:
                # #########################################################
                # Transition for hydrological model parameters
                new_x_tmp = new_x.copy()

                # Force to change a different parameter in each transition
                pot_index = np.array([xk for xk in pot_vals if xk != last_change])

                # Apply the transition considering if
                # tunning is being performed or not
                if len(tunning) > 0:
                    i = tunning[0]
                    step = tunning[1]
                    last_change = 1e9
                else:
                    i = random.choice(pot_index)
                    step = st_range[i]
                    last_change = i

                s_min = new_x_tmp[i] - step
                s_max = new_x_tmp[i] + step

                while new_x_tmp[i] == new_x[i]:
                    # define new parameter and make it fit within the prior
                    new_x_tmp[i] = random.uniform(s_min, s_max)
                    if new_x_tmp[i] > v_range[i][1]:
                        new_x_tmp[i] = v_range[i][0] + (new_x_tmp[i] - v_range[i][1])
                    if new_x_tmp[i] < v_range[i][0]:
                        new_x_tmp[i] = v_range[i][1] + (new_x_tmp[i] - v_range[i][0])

                new_x[i] = new_x_tmp[i]

        return (new_x, df_in_sel, precip_var, last_change)


#######################################################################
#######################################################################
class forwards:
    ###################################################################
    def gamma_transfer_function(inputs):
        """
        Forward for the EM approach

        Parameters
        ----------
        inputs : List
            list of parameters necessary to compute the forward following
            the EM approach

        Returns
        -------
        model time series

        """

        x = inputs[0]
        x_var = inputs[1]
        window = inputs[3]
        try:
            min_value = inputs[4]
        except:
            min_value = 0

        r_win = np.linspace(0, window, window)

        # Define transfer functions with all components
        gamma_dist = stats.gamma.pdf(r_win, a=x[2], scale=x[3])

        gamma_dist = gamma_dist / np.sum(gamma_dist)

        gamma_dist_r = np.flipud(gamma_dist)
        gamma_dist_pad = np.pad(gamma_dist_r, (0, len(gamma_dist_r)), "constant")

        # Create padding
        try:
            x_mod = signal.convolve(x_var, np.flipud(gamma_dist_pad)[1::], mode="same")
        except:
            x_mod = signal.convolve(
                x_var.squeeze(), np.flipud(gamma_dist_pad)[1::], mode="same"
            )

        # Reformat as pandas series
        df_x_mod = pd.Series(x_mod, index=x_var.index)

        # Add linear regression components
        y = x[0] * df_x_mod + x[1]

        # force volumes to be equal or larger than zero
        if min_value > 0:
            y = y.where(y > min_value, min_value)
        else:
            y = y.where(y > 0, 0)

        return y.iloc[window::]

    ###################################################################
    def lm_reservoir_model(inputs):
        """
        Forward for the LM approach

        Parameters
        ----------
        inputs : List
            list of parameters necessary to compute the forward following
            the LM approach

        Returns
        -------
        model volume time series, recharge and discharge time series, and
        time series with volume discharged from spillpoint/overtop

        """
        model_params = inputs[0]
        try:
            infiltration = inputs[1].squeeze()
        except:
            infiltration = inputs[1]

        params_name = np.array(inputs[2])
        spillpoint = inputs[3]
        window = inputs[4]
        min_val = inputs[5]
        initial_volume = inputs[6]
        [f_vs, f_va] = inputs[7]
        ref_time = inputs[8]
        precip_evap = inputs[9] / 1000

        spillpoint = np.min([spillpoint, f_vs.x.max()])

        # met_data = inputs[]
        # Get model variables
        c1 = model_params[params_name == "C1_lm"]
        a_gamma = model_params[params_name == "a_gamma_lm"]
        b_gamma = model_params[params_name == "b_gamma_lm"]
        coef_flow = model_params[params_name == "coef_flow"]
        beta = model_params[params_name == "beta"]
        const_flow = model_params[params_name == "const_flow"]
        stage_flow = model_params[params_name == "stage_flow"]

        # ####
        # Define recharge convolving gamma distribution
        # over effective rainfall
        r_win = np.linspace(0, window, window)

        # Define transfer functions with all components
        gamma_dist = stats.gamma.pdf(r_win, a=a_gamma[0], scale=b_gamma[0])

        # normalize area of gamma function to 1
        gamma_dist = gamma_dist / np.sum(gamma_dist)

        gamma_dist_r = np.flipud(gamma_dist)
        gamma_dist_pad = np.pad(gamma_dist_r, (0, len(gamma_dist_r)), "constant")

        # Create padding
        recharge = signal.convolve(
            infiltration, np.flipud(gamma_dist_pad)[1::], mode="same"
        )

        # Reformat as pandas series
        df_recharge = pd.Series(recharge, index=infiltration.index)

        # # force recharge to be positive by ignoring negative values
        df_recharge = df_recharge.where(df_recharge > 0, 0)

        df_recharge = df_recharge.loc[ref_time::]

        # ####
        # Define volume, discharge arrays
        vol = np.zeros(df_recharge.shape[0])
        vol[0] = initial_volume
        discharge_f = np.zeros(vol.shape)
        spill_f = np.zeros(vol.shape)

        recharge_input = np.zeros(df_recharge.shape[0])
        rech_index = df_recharge.index

        spill_point_f = f_vs(spillpoint)

        ####
        precip_evap = precip_evap[precip_evap.index.isin(df_recharge.index)]
        if len(precip_evap.index) != len(df_recharge.index):
            print("Issues with precip_evap time series")
        # Calculate volume and discharge
        for i_r, rech_in_o in enumerate(df_recharge.iloc[0::]):
            if i_r == 0:
                continue

            i = i_r
            # ####
            # Get direct rain and consider evaporation
            # (Rain - Evaporation) * area of the flood
            # get area of the flood from volume using tables

            try:
                area = f_va(vol[i - 1])
            except:
                area = f_va.y.max()

            rain_evap = precip_evap.iat[i] * area

            rech_in = rech_in_o * (c1[0] - area)

            ####
            # Compute Discharge
            # select active discharges
            coef_flow_tmp = []
            const_flow_tmp = []
            beta_tmp = []

            try:
                if vol[i - 1] < f_vs.x.min():
                    vol_tmp = f_vs.x.min()
                elif vol[i - 1] > f_vs.x.max():
                    vol_tmp = f_vs.x.max()
                else:
                    vol_tmp = vol[i - 1]
                stg = f_vs(vol_tmp)

            except:
                stg = spill_point_f

            # limit discharge when water level is above
            # discharge point

            if stage_flow <= stg:
                coef_flow_tmp = coef_flow
                const_flow_tmp = const_flow
                beta_tmp = beta
            else:
                coef_flow_tmp = 0
                const_flow_tmp = 0
                beta_tmp = 1

            # compute discharge
            # discharge = 0
            discharge = (
                coef_flow_tmp * ((stg - stage_flow) ** beta_tmp)
            ) + const_flow_tmp

            # Compute volume
            vol[i] = rech_in + discharge + rain_evap + vol[i - 1]

            recharge_input[i] = rech_in

            # Not allowance for negative volumes
            if vol[i] < 0:
                vol[i] = 0

            if vol[i] > spillpoint:
                spill_f[i] = vol[i] - spillpoint
                vol[i] = spillpoint

            discharge_f[i] = discharge

        # return a dataframe
        df_vol = pd.Series(vol, index=rech_index)

        df_recharge_in = pd.DataFrame(recharge_input, index=rech_index)

        df_discharge = pd.DataFrame(discharge_f, index=rech_index)

        df_spill = pd.DataFrame(spill_f, index=rech_index)

        # force evaluated volumes to be equal or larger than zero
        # or the minimum volume
        if min_val > 0:
            df_vol = df_vol.where(df_vol > min_val, min_val)
        else:
            df_vol = df_vol.where(df_vol > 0, 0)

        return (df_vol, df_discharge, df_recharge_in, df_spill)


#######################################################################
#######################################################################
class preprocessing:
    ###################################################################
    def model_parameters(
        lm_model_par, em_model_par, run_var, df_all, path_sav, site_name, initial_vol
    ):
        """
        re-format the parameters to be ready for calibration/forward

        Parameters
        ----------
        lm_model_par : dictionary
            contains information for calibration of LM approach
        em_model_par : dictionary
            contains information for calibration of EM approach
        run_var : dictionary
            Contains information about site and the job being
            performed
        df_all : DataFrame
            water level time series
        path_sav : str
            path to stage area volume curves
        site_name : str
            name of the site
        initial_vol : float
            initial volume for the lm approach

        Returns
        -------
        re-formated model parameters

        """

        # Pre-process for inversion - forward
        if run_var["model_approach"] == "LM":
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
            ) = preprocessing.parameters_lm(
                lm_model_par, initial_vol, df_all, path_sav, site_name
            )

        # TF model
        if run_var["model_approach"] == "EM":
            (
                forward,
                variables,
                v_range,
                v_init,
                mask_range,
                param_steps_ini,
                hsm_step_ini,
            ) = preprocessing.parameters_em(em_model_par)
            f_vsa = []

        return (
            forward,
            variables,
            v_range,
            v_init,
            mask_range,
            initial_vol,
            f_vsa,
            param_steps_ini,
            hsm_step_ini,
        )

    ###################################################################
    def initial_conditions(path_sav, df_wl, run_var, lm_model_par, em_model_par, df_in):
        """
        Divide the time series into calibration, validation and forward datasets.
        It also converts from stage to volume time series as modelling is
        performed with volume time series.

        Parameters
        ----------
        path_sav : str
            path to stage area volume conversion curves
        df_wl : DataFrame
            measured water level time series
        run_var : dictionary
            Contains information about site and the job being
            performed
        em_model_par : dictionary
            contains information for calibration of EM approach
        lm_model_par : dictionary
            contains information for calibration of LM approach
        df_in : DataFrame
            effective rainfall time series

        Returns
        -------
        time series for calibration, validation and forward calculations
        as well as relevant information such as initial volume, spillpoint,
        and minumum volumes to model.
        """

        # Get initial condition for water level data
        (
            df_cal,
            df_val,
            df_for,
            df_all,
            min_values_vol,
            spillpoint_vol,
        ) = preprocessing.wl_data(path_sav, df_wl.copy(), run_var)

        # Get initial conditions for effective rainfall
        (
            df_in_cal,
            df_in_val,
            df_in_for,
            initial_vol,
        ) = preprocessing.initial_er_conditions(
            run_var, lm_model_par, em_model_par, df_cal, df_val, df_for, df_in
        )

        return (
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
        )

    ###################################################################
    def wl_data(path_sav, df, run_var):
        """
        Divide water level data into calibration, validation and forward time
        series. It also converts it from stage to volume and apply any filters
        such as only focusing on part of the time series (e.g. top values).

        Parameters
        ----------
        path_sav : str
            path to stage area volume conversion curves
        df : DataFrame
            measured water level time series
        run_var : dictionary
            Contains information about site and the job being
            performed

        Returns
        -------
        Water level time series for calibration, validation and forward calculations
        as well as relevant information such as spillpoint,
        and minumum volumes to model.
        """

        # Generate Calibration, validation and forward datasets
        # calibration
        df_cal = df[df.index >= run_var["st_time_series"][0]]
        df_cal = df_cal[df_cal.index <= run_var["end_time_series"][0]]

        # validation
        df_val = df[df.index >= run_var["st_time_series"][1]]
        df_val = df_val[df_val.index <= run_var["end_time_series"][1]]

        # forward
        df_for = df[df.index >= run_var["st_time_series"][2]]
        df_for = df_for[df_for.index <= run_var["end_time_series"][2]]

        df_all = df.copy()

        # Get min stage values to be considered
        min_cal = run_var["min_stage"][0]
        min_val = run_var["min_stage"][1]
        min_for = run_var["min_stage"][2]

        # Generate volume datasets if approapiate
        if run_var["model_volume"] == True:
            df_cal = preprocessing.convert_hydrol_param(
                path_sav,
                df_cal.copy(),
                hydro_out="volume",
                hydro_in="stage",
                min_stage=-1e2,
                return_f=False,
                simplify_f=-999,
            )

            df_val = preprocessing.convert_hydrol_param(
                path_sav,
                df_val.copy(),
                hydro_out="volume",
                hydro_in="stage",
                min_stage=-1e2,
                return_f=False,
                simplify_f=-999,
            )

            df_for = preprocessing.convert_hydrol_param(
                path_sav,
                df_for.copy(),
                hydro_out="volume",
                hydro_in="stage",
                min_stage=-1e2,
                return_f=False,
                simplify_f=-999,
            )

            df_all = preprocessing.convert_hydrol_param(
                path_sav,
                df_all.copy(),
                hydro_out="volume",
                hydro_in="stage",
                min_stage=-1e2,
                return_f=False,
                simplify_f=-999,
            )

            spillpoint = preprocessing.convert_hydrol_param(
                path_sav,
                float(run_var["spillpoint"]),
                hydro_out="volume",
                hydro_in="stage",
                min_stage=-1e2,
                return_f=False,
                simplify_f=-999,
            )

            min_cal = preprocessing.convert_hydrol_param(
                path_sav,
                float(min_cal),
                hydro_out="volume",
                hydro_in="stage",
                min_stage=-1e2,
                return_f=False,
                simplify_f=-999,
            )

            min_val = preprocessing.convert_hydrol_param(
                path_sav,
                float(min_val),
                hydro_out="volume",
                hydro_in="stage",
                min_stage=-1e2,
                return_f=False,
                simplify_f=-999,
            )

            min_for = preprocessing.convert_hydrol_param(
                path_sav,
                float(min_for),
                hydro_out="volume",
                hydro_in="stage",
                min_stage=-1e2,
                return_f=False,
                simplify_f=-999,
            )

        if min_val > spillpoint:
            min_val = df_val.min()
            min_cal = df_cal.min()
            min_for = df_for.min()

        # ####
        # Apply filters to calibration and validation datasets to focus
        # the inversion on particular parts of the time series
        q_low_cal = df_cal.quantile(run_var["q_low"][0])[0]
        q_low_val = df_cal.quantile(run_var["q_low"][1])[0]
        q_low_for = df_cal.quantile(run_var["q_low"][2])[0]

        if run_var["q_high"][0] < 1:
            q_high_cal = df_cal.quantile(run_var["q_high"][0])[0]
        else:
            q_high_cal = 1e99

        if run_var["q_high"][1] < 1:
            q_high_val = df_cal.quantile(run_var["q_high"][1])[0]
        else:
            q_high_val = 1e99

        if run_var["q_high"][2] < 1:
            q_high_for = df_cal.quantile(run_var["q_high"][2])[0]
        else:
            q_high_for = 1e99

        # cut_off = job_var['calibration']['cut_off'] * df_cal.max()[0]
        cut_off_cal = df_cal.max()[0] * (run_var["cut_off"][0])
        cut_off_val = df_cal.max()[0] * (run_var["cut_off"][1])
        cut_off_for = df_cal.max()[0] * (run_var["cut_off"][2])

        sel_low_cal = np.max([q_low_cal, cut_off_cal, min_cal])

        sel_low_val = np.max([q_low_val, cut_off_val, min_val])

        sel_low_for = np.max([q_low_for, cut_off_for, min_for])

        # apply filters to calibration dataset
        df_cal_nan = np.isnan(df_cal)
        df_cal = df_cal.where(df_cal >= sel_low_cal, sel_low_cal)

        df_cal = df_cal.where(df_cal_nan == False, np.nan)

        df_cal = df_cal.where(df_cal <= q_high_cal, np.nan)

        # Apply filters to validation dataset
        df_val_nan = np.isnan(df_val)
        df_val = df_val.where(df_val >= sel_low_val, sel_low_val)

        df_val = df_val.where(df_val_nan == False, np.nan)

        df_val = df_val.where(df_val <= q_high_val, np.nan)

        # Apply filters to validation dataset
        df_for_nan = np.isnan(df_for)
        df_for = df_for.where(df_for >= sel_low_for, sel_low_for)

        df_for = df_for.where(df_for_nan == False, np.nan)

        df_for = df_for.where(df_for <= q_high_for, np.nan)

        min_values = [min_cal, min_val, min_for]
        return (df_cal, df_val, df_for, df_all, min_values, spillpoint)

    ###################################################################
    def parameters_em(em_model_par):
        """
        Get information for calibration/forward process of the EM approach

        Parameters
        ----------
        em_model_par : dictionary
            contains information for calibration of EM approach

        Returns
        -------
        Information relevant for calibration/forward of EM approach

        """

        variables = em_model_par["variables"]
        v_range = em_model_par["v_range"]
        mask_range = em_model_par["mask_range"]
        v_init = em_model_par["v_init"]
        forward = forwards.gamma_transfer_function
        param_steps_ini = em_model_par["step_size_em"] * (v_range[:, 1] - v_range[:, 0])
        mask_range = list(map(lambda x: x == "True", mask_range))
        hsm_step_ini = em_model_par["step_size_em"]

        return (
            forward,
            variables,
            v_range,
            v_init,
            mask_range,
            param_steps_ini,
            hsm_step_ini,
        )

    ###################################################################
    def parameters_lm(lm_model_par, initial_vol, df_all, path_sav, site_name):
        """
        Get information for calibration/forward process of the LM approach

        Parameters
        ----------
        lm_model_par : dictionary
            contains information for calibration of LM approach
        initial_vol : float
            initial volume lm approach
        df_all : DataFrame
            measured volume time series
        path_sav : str
            path to stage area volume curves
        site_name : str
            name of the site

        Returns
        -------
        Information relevant for calibration/forward of LM approach

        """

        forward = forwards.lm_reservoir_model

        v_range = []
        v_init = []
        variables = []
        mask_range = []
        for i in lm_model_par.keys():
            if (i in ["ini_vol", "ref_date", "window_lm", "step_size_lm"]) == False:
                variables.append(i)
                v_init.append(lm_model_par[i][2])
                v_range.append(lm_model_par[i][0:2])
                mask_range.append(lm_model_par[i][3])

        v_init = np.array([float(x) for x in v_init])

        v_range = np.array([(float(x), float(y)) for x, y in v_range])

        # Automatically detect the initial volume
        if initial_vol == -9999:
            try:
                ref_time = lm_model_par["ref_date"]
                initial_vol = df_all.loc[ref_time][0]

                if np.isnan(initial_vol) == True:
                    initial_vol = 0
            except:
                print(
                    "#### NOTE_1: Could not select initial volume automatically\n"
                    + "          please check that there is groundwater level"
                    + " data for the initial data you specified.\n"
                    + "      By default we will be using 0"
                )

                initial_vol = 0

        (f_vs, _) = preprocessing.convert_hydrol_param(
            path_sav,
            df_all.copy(),
            hydro_out="stage",
            hydro_in="volume",
            min_stage=-1e2,
            return_f=True,
            simplify_f=-6,
        )

        (f_va, _) = preprocessing.convert_hydrol_param(
            path_sav,
            df_all.copy(),
            hydro_out="area",
            hydro_in="volume",
            min_stage=-1e2,
            return_f=True,
            simplify_f=-6,
        )

        # Define steps to explore the prior
        param_steps_ini = lm_model_par["step_size_lm"] * (v_range[:, 1] - v_range[:, 0])

        mask_range = list(map(lambda x: x == "True", mask_range))
        hsm_step_ini = lm_model_par["step_size_lm"]

        return (
            forward,
            variables,
            v_range,
            v_init,
            mask_range,
            initial_vol,
            [f_vs, f_va],
            param_steps_ini,
            hsm_step_ini,
        )

    ###################################################################
    def initial_er_conditions(
        run_var, lm_model_par, em_model_par, df_cal, df_val, df_for, df_in
    ):
        """
        Extract effective rainfall time series for calibration, validation,
        and forward calculations

        Parameters
        ----------
        run_var : dictionary
            Contains information about site and the job being
            performed
        lm_model_par : dictionary
            contains information for calibration of LM approach
        em_model_par : dictionary
            contains information for calibration of EM approach
        df_cal : DataFrame
            water level time series for calibration
        df_val : DataFrame
            water level time series for validation
        df_for : DataFrame
            water level time series for forward calculations
        df_in : DataFrame
            effective rainfall time series

        Returns
        -------
        effective rainfall times eries for calibration, validation and forward
        calculations as well as initial volume for lm approach.

        """

        # Define initial parameters. When the input (meteorological) time series
        # shoud start
        if run_var["model_approach"] == "EM":
            ini_cal = run_var["st_time_series"][0] - datetime.timedelta(
                days=em_model_par["window_em"]
            )
            ini_val = run_var["st_time_series"][1] - datetime.timedelta(
                days=em_model_par["window_em"]
            )
            ini_for = run_var["st_time_series"][2] - datetime.timedelta(
                days=em_model_par["window_em"]
            )
            initial_vol = -9999.0
        if run_var["model_approach"] == "LM":
            initial_vol = float(lm_model_par["ini_vol"][0])
            ini_cal = lm_model_par["ref_date"] - datetime.timedelta(
                days=lm_model_par["window_lm"]
            )
            ini_val = lm_model_par["ref_date"] - datetime.timedelta(
                days=lm_model_par["window_lm"]
            )
            ini_for = lm_model_par["ref_date"] - datetime.timedelta(
                days=lm_model_par["window_lm"]
            )

        end_cal = np.max(df_cal.index + datetime.timedelta(days=1))
        df_in_cal = df_in.loc[ini_cal:end_cal]

        end_val = np.max(df_val.index + datetime.timedelta(days=1))
        df_in_val = df_in.loc[ini_val:end_val]

        end_for = run_var["end_time_series"][2] + datetime.timedelta(days=1)
        df_in_for = df_in.loc[ini_for:end_for]

        return (df_in_cal, df_in_val, df_in_for, initial_vol)

    ###################################################################
    def convert_hydrol_param(
        path_sav,
        df_y,
        hydro_out="stage",
        hydro_in="volume",
        min_stage=-1e2,
        return_f=False,
        simplify_f=-999,
    ):
        """
        Parameters
        ----------
        path_sav : str
            path to stage area volume conversion curves
        df_y : pandas dataframe
            Dataframe with data that needs to be converted to a different
            hydrological parameter
        SAR_ref : str
            SAR reference code for the site
        hydro_out: str, optional
            Specify the output hydrological parameter ('stage','area','volume').
            The default is 'stage'.
        hydro_in: str, optional
            Specify the input hydrological parameter ('stage','area','volume').
            The default is 'volume'.
        min_stage : float, optional
            Minimum stage value to be considered. Any potential stage value below
            this number will be replaced by this number. The default is -1e2.

        Returns: df_y_converted
        -------
        Converts dataframe time series between hydrological parameters. It assumes
        that all time series within the dataframe are for the same site and are
        related to the same hydrological parameter.

        """
        if hydro_out == "stage":
            indx = "Stage"
        if hydro_out == "area":
            indx = "Area"
            min_stage = 0
        if hydro_out == "volume":
            indx = "Volume"
            min_stage = 0

        if hydro_in == "stage":
            indx_r = "Stage"
        if hydro_in == "area":
            indx_r = "Area"
            min_stage = 0
        if hydro_in == "volume":
            indx_r = "Volume"
            min_stage = 0

        # prepare for conversion between hydrological parameters
        df_v2s = pd.read_csv(path_sav, comment="#")

        df_v2s = df_v2s.dropna()

        df_v2s = df_v2s.sort_values(by=df_v2s.columns[0])
        df_v2s.index = np.arange(0, df_v2s.shape[0])

        f = interpolate.interp1d(df_v2s[indx_r], df_v2s[indx])

        if type(df_y) != float:
            df_nan_mask = df_y.isna()

            # Avoid negative volume/area values and impose minimum stage
            df_y = df_y.where(df_y > min_stage, min_stage)

            df_y = df_y.where(df_nan_mask == False, np.nan)

            if simplify_f > 0:
                interval = int(len(f.x) / simplify_f)
                sel_vals = np.arange(0, len(f.x), interval)

                if sel_vals[-1] != len(f.x) - 1:
                    sel_vals = np.append(sel_vals, [len(f.x) - 1])

                if (len(sel_vals) > 0) and (len(sel_vals) < len(f.x)):
                    f.x = f.x[sel_vals]
                    f.y = f.y[sel_vals]

            min_val = np.nanmin(df_v2s[indx_r])
            max_val = np.nanmax(df_v2s[indx_r])

            mask = df_y < min_stage
            df_y.where(mask == False, np.nan, inplace=True)

            mask = df_y < min_val
            df_y.where(mask == False, min_val, inplace=True)

            mask = df_y > max_val
            df_y.where(mask == False, max_val, inplace=True)

            # Apply conversion
            df_y_converted = df_y.copy()
            for i in df_y_converted.columns:
                df_y_converted[i] = f(df_y[i])

            if len(df_y_converted.columns) == 1:
                if indx == "Stage":
                    df_y_converted.columns = ["stage[m]"]
                if indx == "Area":
                    df_y_converted.columns = ["area[m2]"]
                if indx == "Volume":
                    df_y_converted.columns = ["volume[m2]"]
        else:
            try:
                df_y_converted = float(f(df_y))
            except:
                if df_y >= f.x.max():
                    df_y_converted = f.y.max()
                elif df_y <= f.x.min():
                    df_y_converted = f.y.min()

        if return_f == False:
            return df_y_converted

        if return_f == True:
            return (f, df_y_converted)


#######################################################################
#######################################################################
class inputs:
    ###################################################################
    def get_meteorological_datasets(
        path_sites_input_ts, run_var, em_model_par, lm_model_par, site_name, ER_update
    ):
        """
        Get meteorological data for pre-processing for UISCEmod modelling

        Parameters
        ----------
        path_sites_input_ts : str
            path to meteorological time series
        run_var : dictionary
            Contains information about site and the job being
            performed
        em_model_par : dictionary
            contains information for calibration of EM approach
        lm_model_par : dictionary
            contains information for calibration of LM approach
        site_name : str
            Name of the site
        ER_update : bool
            specify if effective rianfall time series needs to be calculated
            (True) or if you will be using pre-deffined time series (False)

        Returns
        -------
        return meteorological time series: precipitation, evapotranspiration,
        precipitation minus evaporation, and effective rainfall time series

        """
        (
            df_precip_sel,
            df_et_sel,
            df_precip_evap,
        ) = inputs.get_meteorological_data_per_site(
            path_sites_input_ts, run_var, em_model_par, lm_model_par, site_name
        )

        # ####
        # Generate time series for recharge (input for the hydrological model) in [m]
        if ER_update == True:
            inputs.generate_potential_effective_rainfall_datasets(
                run_var.copy(),
                df_precip_sel.copy(),
                df_et_sel.copy(),
                path_sites_input_ts,
                site_name,
            )

        # ####
        # get potential effective rainfall time series
        df_in = pd.read_csv(
            path_sites_input_ts + site_name + "_ER_ts.csv", index_col=0, parse_dates=[0]
        )

        return (df_precip_sel, df_et_sel, df_precip_evap, df_in)

    ###################################################################
    def get_job_details(path_sites_info, path_out, df_sites, s_len):
        """
        Get information about the jobs that need to be executed

        Parameters
        ----------
        path_sites_info : str
            path to input directory where info about sites is stored
        path_out : str
            path where to store the results
        df_sites : DataFrame
            Contains information about the jobs to be executed
        s_len : int
            specify which of all the programmed jobs is currently running

        Returns
        -------
        Information about the job that needs to be executed

        """

        (
            site_name,
            ER_update,
            model_approach,
            version,
            iterations,
            burn_in,
            tunning,
        ) = df_sites[
            [
                "site_name",
                "gen_ER",
                "model_approach",
                "version",
                "iterations",
                "burn_in",
                "tunning",
            ]
        ].loc[
            s_len
        ]

        # create folder to store the results, if it does not exist
        path_results = path_out + str(site_name) + "_" + str(version)
        path_sav = path_sites_info + site_name + "_UISCEmod_SAV.csv"
        inputs.create_folder(path_results)

        return (
            site_name,
            ER_update,
            model_approach,
            version,
            int(iterations),
            burn_in,
            tunning,
            path_sav,
            path_results,
        )

    ###################################################################
    def get_paths_and_initial_information(job_ext=""):
        """
        Access to paths to run UISCEmod and information about the jobs to run.
        If you copyed the UISCEmod folder and kept the internal structure you
        should not change anything here, independently to where the folder is
        in your computer. If this is modified UISCEmod may not fins the paths

        Parameters
        ----------
        job_ext : txt, optional
            extension to be added to the jobs. The default is ''.
            this is only relevant if running multiple jobs at the same
            time as they may share temporary files.

        Returns
        -------
        relevant paths and information about the jobs to run

        """

        ####
        # paths (static datasets - relatively stable over time)
        main_path = os.getcwd().rsplit("\\", 1)[0] + "/"
        path_out = main_path + r"/out/"

        # Path to precipitation dataset for modelling
        path_parameters = main_path + "in/sites_info/site_parameters/"
        path_sites = main_path + r"in/sites_info/job_parameters.csv"
        path_spill = main_path + r"in/sites_info/spillpoint_stg.csv"
        path_coord = main_path + r"/datasets/Site_xy_all2708.csv"
        path_sites_info = main_path + "/in/sites_info/"
        path_sites_input_ts = main_path + "/in/sites_input_ts/"

        #######################################################################
        # Get sites information
        df_sites = pd.read_csv(path_sites, comment="#")

        #######################################################################
        # Define tunning steps as a proportion of the range
        # of potential values for each variable
        step_2_tun = np.array(
            [  # 1e-5,
                5e-5,
                1e-4,
                2.5e-4,
                5e-4,
                7.5e-4,
                1e-3,
                2.5e-3,
                5e-3,
                7.5e-3,
                1e-2,
                2.5e-2,
                5e-2,
                1e-1,
                5e-1,
            ]
        )

        pickle.dump(
            [
                main_path,
                path_out,
                path_parameters,
                path_spill,
                path_coord,
                path_sites_input_ts,
                path_sites_info,
                df_sites,
                step_2_tun,
            ],
            open(main_path + "/job_parameters" + job_ext + ".p", "wb"),
        )

        return (df_sites, main_path)

    ###################################################################
    def create_folder(path_results):
        """
        Automatically generates internal directories for the jobs that are
        executed.

        Parameters
        ----------
        path_results : str
            path where to store the results from UISCEmod

        Returns
        -------
        None.

        """

        if not os.path.exists(path_results + "/calibration/"):
            os.makedirs(path_results + "/calibration/")
        if not os.path.exists(path_results + "/validation/"):
            os.makedirs(path_results + "/validation/")
        if not os.path.exists(path_results + "/forward/"):
            os.makedirs(path_results + "/forward/")
        return ()

    ###################################################################
    def get_water_level_data(path):
        """
        get water level time series for site of interest following
        UISCEmod formats

        Parameters
        ----------
        path : str
            path where to get water level time series

        Returns
        -------
        water level time series

        """

        # read the water levels
        df = pd.read_csv(path, index_col=0, usecols=["Date", "gwl[m]"], comment="#")

        df.index = pd.to_datetime(df.index)

        # Remove potential duplicates
        df = df[~df.index.duplicated(keep="first")]

        # Sort the data in ascending order
        df.sort_index(ascending=True, inplace=True)

        # Ensure there is data for every day, even if it
        # is a NaN
        all_dates = pd.date_range(df.index[0], df.index[-1], freq="D")

        df = df.reindex(all_dates, fill_value=np.nan)

        return df

    ###################################################################
    def generate_potential_effective_rainfall_datasets(
        run_var, df_precip_sel, df_et_sel, path_sites_input_ts, site_name
    ):
        """
        Generate potential effective rainfall time series based on potential
        Hsmax values

        Parameters
        ----------
        run_var : dictionary
            Contains information about site and the job being
            performed.
        df_precip_sel : DataFrame
            precipitation time series
        df_et_sel : DataFrame
            evapotransiration times eries
        path_sites_input_ts : str
            path where to input time series are stored. potential effective
            rainfall time series will be stored there.
        site_name : str
            name of the site

        Returns
        -------
        None.

        """

        df_precip_sel = df_precip_sel.squeeze()
        df_et_sel = df_et_sel.squeeze()
        # t0 = datetime.datetime.now()
        df_in_all = pd.DataFrame()

        for i_num, i in enumerate(run_var["met_data"][1]):
            df_tmp = pd.Series()

            if "Rch" in i:
                smd = np.zeros(df_precip_sel.shape[0])
                aet = np.zeros(df_precip_sel.shape[0])
                rec_lin = np.zeros(df_precip_sel.shape[0])
                smd_max = float(i.split("_")[1])

                if smd_max != 0:
                    for j in range(1, df_precip_sel.shape[0]):
                        const = (smd_max - smd[j - 1]) / smd_max
                        aet[j] = df_et_sel.iat[j] * const
                        smd[j] = smd[j - 1] - df_precip_sel.iat[j] + aet[j]

                        if smd[j] < 0:
                            rec_lin[j] = -1 * smd[j]
                            smd[j] = 0

                        if smd[j] > smd_max:
                            smd[j] = smd_max

                    df_tmp = pd.Series(rec_lin, index=df_precip_sel.index)
                else:
                    df_tmp = df_precip_sel.copy().squeeze() - df_et_sel.squeeze()
                    df_tmp = df_tmp.where(df_tmp > 0, 0)

            df_tmp = df_tmp.to_frame()
            df_tmp.columns = [str(i)]
            df_in_all = pd.concat([df_in_all, df_tmp], axis=1)

        # Make sure there are no issues with the time series data later
        df_in_all = df_in_all[~df_in_all.index.duplicated(keep="first")]
        df_in_all.sort_index(ascending=True, inplace=True)

        df_in_all.index = pd.to_datetime(df_in_all.index)
        df_in_all = df_in_all / 1000.0

        df_in_all.to_csv(path_sites_input_ts + site_name + "_ER_ts.csv")
        return ()

    ###################################################################
    def get_meteorological_data_per_site(
        path_sites_input_ts, run_var, em_model_par, lm_model_par, site_name
    ):
        """
        Get meteorological time series for specific site following UISCEmod
        format

        Parameters
        ----------
        path_sites_input_ts : str
            path to input time series
        run_var : dictionary
            Contains information about site and the job being
            performed.
        em_model_par : dictionary
            contains information for calibration of EM approach
        lm_model_par : dictionary
            contains information for calibration of LM approach
        site_name : str
            name of the site

        Returns
        -------
        meteorological time series. precipitation, evapotranspiration
        and precipitation minus evaporation.

        """

        # select times of interest considering window size
        early_time = np.min(
            [np.min(run_var["st_time_series"]), lm_model_par["ref_date"]]
        )

        if run_var["model_approach"] == "EM":
            st_time = early_time - datetime.timedelta(
                days=em_model_par["window_em"] + 10
            )

        if run_var["model_approach"] == "LM":
            st_time = early_time - datetime.timedelta(
                days=lm_model_par["window_lm"] + 10
            )

        end_time = np.max(run_var["end_time_series"])

        # Select data for site of interest
        # Select precipitation values
        df_met = pd.read_csv(
            path_sites_input_ts + site_name + "_UISCEmod_input_ts.csv",
            index_col=0,
            usecols=["Date", "R[mm]", "ET[mm]", "EVAP[mm]"],
            comment="#",
        )

        df_met.index = pd.to_datetime(df_met.index)
        df_precip_sel = df_met[["R[mm]"]]
        df_et_sel = df_met[["ET[mm]"]]
        df_evap_sel = df_met[["EVAP[mm]"]]

        # Get only times of interest
        df_precip_sel = df_precip_sel[df_precip_sel.index >= st_time]
        df_precip_sel = df_precip_sel[df_precip_sel.index <= end_time]
        df_et_sel = df_et_sel[df_et_sel.index >= st_time]
        df_et_sel = df_et_sel[df_et_sel.index <= end_time]
        df_evap_sel = df_evap_sel[df_evap_sel.index >= st_time]
        df_evap_sel = df_evap_sel[df_evap_sel.index <= end_time]

        df_precip_evap = df_precip_sel.squeeze() - df_evap_sel.squeeze()
        df_precip_evap.columns = ["precip_evap"]

        return (df_precip_sel, df_et_sel, df_precip_evap)

    ###################################################################
    def get_model_parameters(path, df_sites, path_out):
        """
        Get modelling information for specific site
        from "XXXX_UISCEmod_info.csv" file.

        Parameters
        ----------
        path : str
         path to file with modelling parameters.
        df_sites : DataFrame
            information about the jobs to be performed
        path_out : str
            path where to store the results. A copy of the information file is
            added to each output to know what where the conditions for
            calibration/forward of the site

        Returns
        -------
        Information about modelling parameters for EM and LM approach as well
        as information about the site ans jobs that need to be executed, and
        specifies if we are sampling the prior or not

        """

        shutil.copyfile(path, path_out)
        # Read input parameters file
        # considering # as comments, and ignoring white spaces
        df = pd.read_csv(
            path, index_col=0, header=None, low_memory=False, comment="#", sep=":"
        )

        df.columns = ["values"]
        df["values"] = df["values"].map(str.strip)

        # ####
        # Data constrains

        # Edges of considered time series for calibration,
        # validation, and forwatd calculations
        pot_time_series = df.at["time_series", "values"][1:-1].split(",")

        st_time_series = [
            pd.to_datetime(x)
            for x in (df.at["st_time_series", "values"][1:-1].split(","))
        ]

        end_time_series = [
            pd.to_datetime(x)
            for x in (df.at["end_time_series", "values"][1:-1].split(","))
        ]

        # ####
        # Focus only on subsections of the times series
        # cut of - ignore values smaller than cut off
        # relative to maximum value
        cut_off = [float(x) for x in (df.at["cut_off", "values"][1:-1].split(","))]

        q_low = [float(x) for x in (df.at["q_low", "values"][1:-1].split(","))]

        q_high = [float(x) for x in (df.at["q_high", "values"][1:-1].split(","))]

        # minimum stage to be considered (i.e. elevation of the datalogger)
        min_stage = [float(x) for x in (df.at["min_stage", "values"][1:-1].split(","))]

        # specify if modelling volume or not.
        # If False it will model stage time series
        model_volume = eval(df.at["model_volume", "values"])

        # spill point information
        spillpoint = eval(df.at["spillpoint", "values"])

        # ####
        # Meteorological data
        in_met_data = df.at["met_data", "values"][1:-1].split(",")
        if in_met_data[-1] == "False":
            met_data = ["Rch_" + "%03d" % int(in_met_data[3])]
            or_met_data = met_data
        else:
            hs_max_values = np.arange(
                int(in_met_data[0]),
                int(in_met_data[1]) + int(in_met_data[2]),
                int(in_met_data[2]),
            )

            # Get potential datasets
            met_data = ["Rch_" + "%03d" % x for x in hs_max_values]

            if in_met_data[3] == "*":
                or_met_data = [random.choice(met_data)]
            else:
                or_met_data = ["Rch_" + "%03d" % int(in_met_data[3])]

        met_data = [or_met_data, met_data]

        # ####
        # TF parameters
        window_em = int(df.at["window_em", "values"])

        step_size_em = 0.01 * float(df.at["step_size_em", "values"])

        variables_em = ["C1_em", "C0_em", "a_gamma_em", "b_gamma_em", "std_em"]

        for i_num, i in enumerate(variables_em):
            if df.at[i, "values"][1:-1].split(",")[2] == "*":
                new_val = str(
                    random.uniform(
                        float(df.at[i, "values"][1:-1].split(",")[0]),
                        float(df.at[i, "values"][1:-1].split(",")[1]),
                    )
                )

                df.loc[i, "values"] = (
                    "["
                    + df.at[i, "values"][1:-1].split(",")[0]
                    + ","
                    + df.at[i, "values"][1:-1].split(",")[1]
                    + ","
                    + new_val
                    + ","
                    + df.at[i, "values"][1:-1].split(",")[3]
                    + "]"
                )

        # Get range of potential values
        v_range_em = np.float_(
            [
                df.at["C1_em", "values"][1:-1].split(",")[0:2],
                df.at["C0_em", "values"][1:-1].split(",")[0:2],
                df.at["a_gamma_em", "values"][1:-1].split(",")[0:2],
                df.at["b_gamma_em", "values"][1:-1].split(",")[0:2],
                df.at["std_em", "values"][1:-1].split(",")[0:2],
            ]
        )

        # Get initial values
        v_init_em = np.float_(
            [
                df.at["C1_em", "values"][1:-1].split(",")[2],
                df.at["C0_em", "values"][1:-1].split(",")[2],
                df.at["a_gamma_em", "values"][1:-1].split(",")[2],
                df.at["b_gamma_em", "values"][1:-1].split(",")[2],
                df.at["std_em", "values"][1:-1].split(",")[2],
            ]
        )

        # Get mask for unblocked/blocked variables
        mask_range_em = [
            df.at["C1_em", "values"][1:-1].split(",")[3],
            df.at["C0_em", "values"][1:-1].split(",")[3],
            df.at["a_gamma_em", "values"][1:-1].split(",")[3],
            df.at["b_gamma_em", "values"][1:-1].split(",")[3],
            df.at["std_em", "values"][1:-1].split(",")[3],
        ]

        # ####
        # LM parameters
        window_lm = int(df.at["window_lm", "values"])
        step_size_lm = 0.01 * float(df.at["step_size_lm", "values"])
        ref_date_lm = pd.to_datetime(df.at["ref_date", "values"][1:-1])
        C1_lm = df.at["C1_lm", "values"][1:-1].split(",")
        a_gamma_lm = df.at["a_gamma_lm", "values"][1:-1].split(",")
        b_gamma_lm = df.at["b_gamma_lm", "values"][1:-1].split(",")
        coef_flow = df.at["coef_flow", "values"][1:-1].split(",")
        beta = df.at["beta", "values"][1:-1].split(",")
        const_flow = df.at["const_flow", "values"][1:-1].split(",")
        ini_vol = df.at["initial_volume", "values"][1:-1].split(",")
        stage_flow = df.at["stage_flow", "values"][1:-1].split(",")
        std_lm = df.at["std_lm", "values"][1:-1].split(",")

        # ####
        # Define dictionaries with relevant information
        em_model_parameters = {
            "variables": variables_em,
            "v_range": v_range_em,
            "v_init": v_init_em,
            "mask_range": mask_range_em,
            "window_em": window_em,
            "step_size_em": step_size_em,
        }

        lm_model_parameters = {
            "C1_lm": C1_lm,
            "a_gamma_lm": a_gamma_lm,
            "b_gamma_lm": b_gamma_lm,
            "coef_flow": coef_flow,
            "beta": beta,
            "const_flow": const_flow,
            "ini_vol": ini_vol,
            "stage_flow": stage_flow,
            "std_lm": std_lm,
            "ref_date": ref_date_lm,
            "window_lm": window_lm,
            "step_size_lm": step_size_lm,
        }

        for i in lm_model_parameters.keys():
            if i in [
                "C1_lm",
                "a_gamma_lm",
                "b_gamma_lm",
                "coef_flow",
                "beta",
                "const_flow",
                "stage_flow",
                "std_lm",
            ]:
                if lm_model_parameters[i][2] == "*":
                    lm_model_parameters[i][2] = str(
                        random.uniform(
                            float(lm_model_parameters[i][0]),
                            float(lm_model_parameters[i][1]),
                        )
                    )

        (
            site_name,
            u_mode,
            model_approach,
            version,
            iterations,
            burn_in,
            tunning,
        ) = df_sites[
            [
                "site_name",
                "mode",
                "model_approach",
                "version",
                "iterations",
                "burn_in",
                "tunning",
            ]
        ]

        run_variables = {
            "model_approach": model_approach,
            "iterations": iterations,
            "th_acceptance": burn_in,
            "pot_time_series": pot_time_series,
            "st_time_series": st_time_series,
            "end_time_series": end_time_series,
            "met_data": met_data,
            "spillpoint": spillpoint,
            "uisce_mode": u_mode,
            "model_volume": model_volume,
            "min_stage": min_stage,
            "cut_off": cut_off,
            "q_low": q_low,
            "q_high": q_high,
            "site_name": site_name,
            "version": version,
            "tunning": tunning,
        }

        if u_mode == "sample_the_prior":
            check_var_samp = True
        else:
            check_var_samp = False

        return (em_model_parameters, lm_model_parameters, run_variables, check_var_samp)

    ###################################################################
    def get_spillpoints(path_spill, df_sites):
        """

        Parameters
        ----------
        path_spill : str
            path to file with spillpoint information
        df_sites : DataFrame
            include names of sites for which we want to know the spillpoint

        Returns
        -------
        Elevation of the spill point [m]. If not spillpoint
        information is found for some of the sites it returns 1e9 for
        these sites.

        """
        df_sp = pd.read_csv(path_spill, comment="#")

        sites_info_sp = {}
        for i in df_sites["site_name"]:
            try:
                sp = df_sp[i].iloc[0]
            except:
                sp = 1e9

            sites_info_tmp = {i: {"spill_point": sp}}

            sites_info_sp.update(sites_info_tmp)

        return sites_info_sp
