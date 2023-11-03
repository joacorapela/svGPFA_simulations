import sys
import os
import traceback
import random
import cProfile, pstats
import pickle
import argparse
import configparser
import torch
import plotly.graph_objects as go

import gcnu_common.utils.config_dict
import svGPFA.stats.svGPFAModelFactory
import svGPFA.stats.svEM
import svGPFA.utils.configUtils
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_res_number", help="simuluation result number",
                        type=int, default=91450833)
#                         type=int, default=68760574)
#                         type=int, default=95195429)
#                         type=int, default=32451751)
    parser.add_argument("--est_init_number", help="estimation init number",
                        type=int, default=551)
    parser.add_argument("--n_repeats", help="number of repeats of the estimation of models in each device",
                        # type=int, default=10)
                        type=int, default=5)
    parser.add_argument("--cuda_device_index", help="cuda device index where to run the model",
                        type=int, default=0)
    parser.add_argument("--est_init_config_filename_pattern",
                        help="estimation initialization configuration "
                             "filename pattern",
                        type=str,
                        default="../data/{:08d}_estimation_metaData.ini")
    parser.add_argument("--sim_res_config_filename_pattern",
                        help="simulation result configuration filename "
                             "pattern",
                        type=str,
                        default="../results/{:08d}_simulation_metaData.ini")
    parser.add_argument("--results_metadata_filename_pattern",
                        help="results metadata filename pattern",
                        type=str,
                        default="../results/{:08d}_gpuBugReport_metadata.ini")
    parser.add_argument("--results_filename_pattern",
                        help="results filename pattern",
                        type=str,
                        default="../results/{:08d}_gpuBugReport.pickle")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../figures/{:08d}_gpuBugReport.{:s}")
    args = parser.parse_args()

    sim_res_number = args.sim_res_number
    est_init_number = args.est_init_number
    n_repeats = args.n_repeats
    cuda_device_index = args.cuda_device_index
    est_init_config_filename_pattern = args.est_init_config_filename_pattern
    sim_res_config_filename_pattern = args.sim_res_config_filename_pattern
    results_metadata_filename_pattern = args.results_metadata_filename_pattern
    results_filename_pattern = args.results_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    # load data
    sim_res_config_filename = sim_res_config_filename_pattern.format(
        sim_res_number)
    sim_res_config = configparser.ConfigParser()
    sim_res_config.read(sim_res_config_filename)
    sim_res_filename = sim_res_config["simulation_results"]["sim_res_filename"]
    with open(sim_res_filename, "rb") as f:
        sim_res = pickle.load(f)
    spikes_times = sim_res["spikes_times"]
    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])

    spikes_times = [[spikes_times[r][n].tolist() for n in range(n_neurons)]
                    for r in range(n_trials)]

    sim_init_config_filename = sim_res_config["simulation_params"]["sim_init_config_filename"]
    sim_init_config = configparser.ConfigParser()
    sim_init_config.read(sim_init_config_filename)
    trials_start_times = [float(str) for str in sim_init_config["data_structure_params"]["trials_start_times"][1:-1].split(",")]
    trials_end_times = [float(str) for str in sim_init_config["data_structure_params"]["trials_end_times"][1:-1].split(",")]

    est_init_config_filename = est_init_config_filename_pattern.format(
        est_init_number)
    est_init_config = configparser.ConfigParser()
    est_init_config.read(est_init_config_filename)
    n_latents = int(est_init_config["model_structure_params"]["n_latents"])

    # get results_number
    results_prefix_used = True
    while results_prefix_used:
        results_number = random.randint(0, 10**8)
        results_metadata_filename = results_metadata_filename_pattern.format(results_number)
        if not os.path.exists(results_metadata_filename):
           results_prefix_used = False
    results_filename = results_filename_pattern.format(results_number)

    # save metadata
    results_metadata = configparser.ConfigParser()
    results_metadata["params"] = {"sim_res_number": sim_res_number,
                                  "est_init_number": est_init_number,
                                  "n_repeats": n_repeats,
                                  "cuda_device_index": cuda_device_index}
    with open(results_metadata_filename, "w") as f: results_metadata.write(f)

    #    get args_info
    args_info = svGPFA.utils.initUtils.getArgsInfo()

    #    build dynamic parameter specifications
    dynamic_params = svGPFA.utils.initUtils.getParamsDictFromArgs(
        n_latents=n_latents, n_trials=n_trials, args=vars(args),
        args_info=args_info)
    #    build configuration file parameter specifications
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=est_init_config).get_dict()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    #    build default parameter specificiations
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_latents=n_latents)

    devices = ["cpu", f"cuda:{cuda_device_index}"]
    elapsed_times = torch.empty((len(devices), n_repeats), dtype=torch.double)
    lower_bounds = torch.empty((len(devices), n_repeats), dtype=torch.double)
    num_fun_eval = torch.empty((len(devices), n_repeats), dtype=torch.double)
    num_iter = torch.empty((len(devices), n_repeats), dtype=torch.double)

    for d, device in enumerate(devices):
        r = 0
        while r < n_repeats:
            print(f"Processing device {device}, repeat {r}")
            #    finally, get the parameters from the dynamic,
            #    configuration file and default parameter specifications
            params, kernels_types = svGPFA.utils.initUtils.getParamsAndKernelsTypes(
                n_trials=n_trials, n_neurons=n_neurons, n_latents=n_latents,
                trials_start_times=trials_start_times,
                trials_end_times=trials_end_times,
                dynamic_params_spec=dynamic_params,
                config_file_params_spec=config_file_params,
                default_params_spec=default_params)

            # build kernels
            kernels_params0 = params["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["kernels_params0"]
            kernels = svGPFA.utils.miscUtils.buildKernels(
                kernels_types=kernels_types, kernels_params=kernels_params0)

            # create model
            kernelMatrixInvMethod = svGPFA.stats.svGPFAModelFactory.kernelMatrixInvChol
            indPointsCovRep = svGPFA.stats.svGPFAModelFactory.indPointsCovChol
            model = svGPFA.stats.svGPFAModelFactory.SVGPFAModelFactory.buildModelPyTorch(
                conditionalDist=svGPFA.stats.svGPFAModelFactory.PointProcess,
                linkFunction=svGPFA.stats.svGPFAModelFactory.ExponentialLink,
                embeddingType=svGPFA.stats.svGPFAModelFactory.LinearEmbedding,
                kernels=kernels, kernelMatrixInvMethod=kernelMatrixInvMethod,
                indPointsCovRep=indPointsCovRep)

            model.setParamsAndData(
                measurements=spikes_times,
                initial_params=params["initial_params"],
                eLLCalculationParams=params["ell_calculation_params"],
                priorCovRegParam=params["optim_params"]["prior_cov_reg_param"])

            model.to(device=device)

            # maximize lower bound
            svEM = svGPFA.stats.svEM.SVEM_PyTorch()
            try:
                lowerBoundHist, elapsedTimeHist, terminationInfo,\
                iterationsModelParams, num_fun_eval[d, r], num_iter[d, r] = \
                    svEM.maximizeSimultaneously(model=model,
                                                optim_params=params["optim_params"])
                elapsed_times[d, r] = elapsedTimeHist[-1]
                lower_bounds[d, r] = lowerBoundHist[-1]
            except Exception as e:
                print("Exception detected. Retrying")
                stack_trace = traceback.format_exc()
                print(e)
                print(stack_trace)
                breakpoint()
            print("Device {:s}, Repeat {:d}: elapsed time {:f}, "
                  "elapsed time per function call {:f}, "
                  "lower bound {:f}".format(device, r, elapsed_times[d, r],
                                            elapsed_times[d, r]/num_fun_eval[d, r],
                                            lower_bounds[d, r]))
            r += 1

    results_to_save = {"elapsed_times": elapsed_times,
                       "lower_bounds": lower_bounds,
                       "num_fun_eval": num_fun_eval,
                       "num_iter": num_iter,
                      }
    with open(results_filename, "wb") as f: pickle.dump(results_to_save, f)

    hoovertext = [[],[]]
    for d, device in enumerate(devices):
        for r in range(n_repeats):
            hoovertext[d].append(
                (f"num_fun_eval: {num_fun_eval[d,r]}<br>"
                 f"num_iter: {num_iter[d,r]}<br>"
                 f"elapsed_time_per_funcall: "
                 f"{elapsed_times[d,r]/num_fun_eval[d,r]}"))

    fig = go.Figure()
    for d in range(len(devices)):
        trace = go.Scatter(x=elapsed_times[d,:], y=lower_bounds[d, :],
                           mode="markers", name=devices[d],
                           hovertext=hoovertext[d], hoverinfo="text")
        fig.add_trace(trace)
    fig.update_layout(xaxis_title="Elapsed Time (sec)")
    fig.update_layout(yaxis_title="Lower Bound")

    png_fig_filename = fig_filename_pattern.format(results_number, "png")
    fig.write_image(png_fig_filename)
    html_fig_filename = fig_filename_pattern.format(results_number, "html")
    fig.write_html(html_fig_filename)

    # fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
