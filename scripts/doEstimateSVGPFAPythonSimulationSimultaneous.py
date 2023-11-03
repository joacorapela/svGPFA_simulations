import sys
import os
import pdb
import random
import cProfile, pstats
import pickle
import argparse
import configparser
import torch

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
    parser.add_argument("--device", help="device where to run the model (e.g., cpu or cuda:0)",
                        type=str, default="cpu")
    parser.add_argument("--profile", help="perform profiling", action="store_true")
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
    parser.add_argument("--profiler_filename_pattern",
                        help="profiler filename pattern", type=str,
                        default="../results/{:08d}_estimatedModel.prof")
    parser.add_argument("--model_save_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default= "../results/{:08d}_estimatedModel.pickle")
    args = parser.parse_args()

    sim_res_number = args.sim_res_number
    est_init_number = args.est_init_number
    device = torch.device(args.device)
    profile = args.profile
    est_init_config_filename_pattern = args.est_init_config_filename_pattern
    sim_res_config_filename_pattern = args.sim_res_config_filename_pattern
    profiler_filename_pattern = args.profiler_filename_pattern
    model_save_filename_pattern = args.model_save_filename_pattern

    # load data
    sim_res_config_filename = sim_res_config_filename_pattern.format(
        sim_res_number)
    sim_res_config = configparser.ConfigParser()
    sim_res_config.read(sim_res_config_filename)
    # sim_init_config_filename = \
    #     sim_res_config["simulation_params"]["sim_init_config_filename"]
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

    #    build dynamic parameter specifications
    args_info = svGPFA.utils.initUtils.getArgsInfo()
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
    #    finally, get the parameters from the dynamic,
    #    configuration file and default parameter specifications
    params, kernels_types = svGPFA.utils.initUtils.getParamsAndKernelsTypes(
        n_trials=n_trials, n_neurons=n_neurons, n_latents=n_latents,
        trials_start_times=trials_start_times,
        trials_end_times=trials_end_times,
        dynamic_params_spec=dynamic_params,
        config_file_params_spec=config_file_params,
        default_params_spec=default_params)

    # build model_save_filename
    estPrefixUsed = True
    while estPrefixUsed:
        est_res_number = random.randint(0, 10**8)
        estim_res_metadata_filename = "../results/{:08d}_estimation_metaData.ini".format(est_res_number)
        if not os.path.exists(estim_res_metadata_filename):
            estPrefixUsed = False
    model_save_filename = model_save_filename_pattern.format(est_res_number)

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
    def getKernelParams(model):
        kernelParams = model.getKernelsParams()[0]
        return kernelParams

    svEM = svGPFA.stats.svEM.SVEM_PyTorch()
    if profile:
        pr = cProfile.Profile()
        pr.enable()
    lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = \
        svEM.maximizeSimultaneously(model=model,
                                    optim_params=params["optim_params"])
    if profile:
        pr.disable()
        profiler_filename = profiler_filename_pattern.format(est_res_number)
        pr.dump_stats(profiler_filename)

    print("Elapsed time {:f}".format(elapsedTimeHist[-1]))

#     # save estimated values
#     estimResConfig = configparser.ConfigParser()
#     estimResConfig["simulation_params"] = {"sim_res_number": sim_res_number}
#     estimResConfig["optim_params"] = params["optim_params"]
#     estimResConfig["estimation_params"] = {
#         "est_init_number": est_init_number,
#     }
#     with open(estim_res_metadata_filename, "w") as f:
#         estimResConfig.write(f)
# 
#     resultsToSave = {"lowerBoundHist": lowerBoundHist,
#                      "elapsedTimeHist": elapsedTimeHist,
#                      "terminationInfo": terminationInfo,
#                      "iterationModelParams": iterationsModelParams,
#                      "model": model}
#     with open(model_save_filename, "wb") as f:
#         pickle.dump(resultsToSave, f)
#     print("Saved results to {:s}".format(model_save_filename))

    pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
