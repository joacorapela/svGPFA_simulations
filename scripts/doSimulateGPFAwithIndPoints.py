
import sys
import os
import random
import torch
import pickle
import argparse
import configparser
import matplotlib.pyplot as plt

import gcnu_common.utils.config_dict
import gcnu_common.stats.pointProcesses.sampling
import svGPFA.simulations.simulations
import svGPFA.utils.configUtils
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils
import svGPFA.stats.svPosteriorOnLatents
import svGPFA.stats.svPosteriorOnIndPoints
import svGPFA.stats.kernelsMatricesStore

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_init_config_number", help="Simulation initialization configuration number", type=int)
    parser.add_argument("--trial_to_plot", help="Trial to plot", type=int, default=0)
    parser.add_argument("--sim_init_config_filename_pattern",
                        help="Simulation initialization configuration filename pattern",
                        type=str,
                        default= "../data/{:08d}_simulation_metaData.ini")
    parser.add_argument("--sim_res_config_filename_pattern",
                        help="Simulation result configuration filename pattern",
                        type=str,
                        default="../results/{:08d}_simulation_metaData.ini")
    parser.add_argument("--sim_res_filename_pattern",
                        help="Simulation result filename pattern",
                        type=str,
                        default="../results/{:08d}_simRes.pickle")
    args = parser.parse_args()

    sim_init_config_number = args.sim_init_config_number
    trial_to_plot = args.trial_to_plot
    trial_to_plot = args.trial_to_plot
    sim_init_config_filename_pattern = args.sim_init_config_filename_pattern
    sim_res_config_filename_pattern = args.sim_res_config_filename_pattern
    sim_res_filename_pattern = args.sim_res_filename_pattern

    # load data and initial values
    sim_init_config_filename = sim_init_config_filename_pattern.format(
        sim_init_config_number)
    sim_init_config = configparser.ConfigParser()
    sim_init_config.read(sim_init_config_filename)
    n_latents = int(sim_init_config["control_variables"]["n_latents"])
    n_neurons = int(sim_init_config["control_variables"]["n_neurons"])
    n_time_steps_CIF = int(sim_init_config["control_variables"]["n_time_steps_CIF"])
    trials_start_times = svGPFA.utils.initUtils.strTo1DDoubleTensor(sim_init_config["data_structure_params"]["trials_start_times"])
    trials_end_times = svGPFA.utils.initUtils.strTo1DDoubleTensor(sim_init_config["data_structure_params"]["trials_end_times"])
    n_trials = len(trials_start_times)
    prior_cov_reg_param = float(sim_init_config["control_variables"]["prior_cov_reg_param"])
    latents_cov_reg_param = float(sim_init_config["control_variables"]["latents_cov_reg_param"])

    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=sim_init_config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)

    kernels_params0, kernels_types = \
        svGPFA.utils.initUtils.getKernelsParams0AndTypes(
            n_latents=n_latents, config_file_params=config_file_params)
    kernels = svGPFA.utils.miscUtils.buildKernels(
        kernels_types=kernels_types, kernels_params=kernels_params0)

    # begin horrible patch
    # for k in range(n_latents):
    #     kernels[k]._scale = 0.1
    # end horrible patch

    n_ind_points = svGPFA.utils.initUtils.getParam(
        section_name="ind_points_locs_params0", param_name="n_ind_points",
        config_file_params=config_file_params)
    ind_points_locs = svGPFA.utils.initUtils.getIndPointsLocs0(
        n_latents=n_latents, n_trials=n_trials,
        config_file_params=config_file_params,
        n_ind_points=n_ind_points,
        trials_start_times=trials_start_times,
        trials_end_times=trials_end_times,
    )

    trials_times = svGPFA.utils.miscUtils.getTrialsTimes(
        start_times=trials_start_times,
        end_times=trials_end_times,
        n_steps=n_time_steps_CIF)

    var_mean = svGPFA.utils.initUtils.getVariationalMean0(
        n_latents=n_latents, n_trials=n_trials, n_ind_points=n_ind_points,
        config_file_params=config_file_params)
    var_mean_new_format = [[var_mean[k][r, :, :] for k in range(n_latents)]
                           for r in range(n_trials)]

    var_cov = svGPFA.utils.initUtils.getVariationalCov0(
        n_latents=n_latents, n_trials=n_trials, n_ind_points=n_ind_points,
        config_file_params=config_file_params)
    var_cov_new_format = [[var_cov[k][r, :, :] for k in range(n_latents)]
                           for r in range(n_trials)]

    # C, d = svGPFA.utils.configUtils.getLinearEmbeddingParams(
    #     CFilename=sim_init_config["embedding_params"]["C_filename"],
    #     dFilename=sim_init_config["embedding_params"]["d_filename"])
    C, d = svGPFA.utils.initUtils.getLinearEmbeddingParams0(
        n_neurons=n_neurons, n_latents=n_latents,
        config_file_params=config_file_params)

    # indPointsMeans = svGPFA.utils.configUtils.getIndPointsMeans(n_trials=n_trials, n_latents=n_latents, config=sim_init_config)
    simulator = svGPFA.simulations.simulations.GPFAwithIndPointsSimulator()
    print("Computing latents samples")
    latents_samples, latents_means, latents_STDs, Kzz = \
        simulator.getLatentsSamplesMeansAndSTDs(
            var_mean=var_mean_new_format,
            var_cov=var_cov_new_format,
            kernels=kernels,
            ind_points_locs=ind_points_locs,
            # trialsTimes=latentsTrialsTimesMatrix,
            trials_times=trials_times,
            prior_cov_reg_param=prior_cov_reg_param,
            latents_cov_reg_param=latents_cov_reg_param,
            dtype=C.dtype,
        )
#     exit = False
#     attemptNumber = 1
#     while not exit:
#         latents_samples, latents_means, latents_STDs, KzzChol = simulator.getLatentsSamplesMeansAndSTDs(
#             var_mean=var_mean,
#             kernels=kernels,
#             ind_points_locs=ind_points_locs,
#             trialsTimes=latentsTrialsTimesMatrix,
#             prior_cov_reg_param=prior_cov_reg_param,
#             latents_cov_reg_param=latents_cov_reg_param,
#             dtype=C.dtype)
#         maxLatentsSamples = torch.tensor([torch.max(latents_samples[k]) for k in range(n_latents)]).max()
#         print("Attempt number: {:d}, max lLatents: {:.02f}".format(attemptNumber, maxLatentsSamples))
# 
#         if(maxLatentsSamples>1.0):
#             exit = True
#         else:
#             attemptNumber += 1
    cif_values = simulator.getCIF(n_trials=n_trials,
                                  latents_samples=latents_samples,
                                  C=C, d=d, link_function=torch.exp)
    plt.figure()

    for k in range(n_latents):
        plt.subplot(n_latents, 1, k+1)
        plt.plot(trials_times[trial_to_plot, :, 0],
                 latents_samples[trial_to_plot][k, :])
        plt.xlabel("Time (sec)")
        plt.ylabel("Latent")
        if k == 0:
            plt.title("Trial: {:d}".format(trial_to_plot))

    plt.figure()
    plt.show(block=False)

    for n in range(n_neurons):
        plt.plot(trials_times[trial_to_plot, :, 0],
                 cif_values[trial_to_plot][n])
    plt.xlabel("Time (sec)")
    plt.ylabel("CIF")
    plt.title("Trial: {:d}".format(trial_to_plot))

    plt.show()
    import pdb; pdb.set_trace()
    print("Getting spikes times")
    sampling_func = gcnu_common.stats.pointProcesses.sampling.sampleInhomogeneousPP_thinning
    spikes_times = simulator.simulate(cif_trials_times=trials_times,
                                      cif_values=cif_values,
                                      sampling_func=sampling_func)

    sim_res = {"trials_times": trials_times,
               "latents_samples": latents_samples,
               "latents_means": latents_means,
               "latents_STDs": latents_STDs,
               "var_mean": var_mean,
               "var_cov": var_cov,
               "ind_points_locs": ind_points_locs,
               "C": C, "d": d,
               "cif_values": cif_values,
               "spikes_times": spikes_times,
               "Kzz": Kzz}

    random_prefix_used = True
    while random_prefix_used:
        sim_number = random.randint(0, 10**8)
        sim_res_config_filename = sim_res_config_filename_pattern.format(
            sim_number)
        if not os.path.exists(sim_res_config_filename):
           random_prefix_used = False
    sim_res_filename = sim_res_filename_pattern.format(sim_number)

    with open(sim_res_filename, "wb") as f:
        pickle.dump(sim_res, f)

    sim_res_config = configparser.ConfigParser()
    sim_res_config["simulation_params"] = {"sim_init_config_filename":
                                           sim_init_config_filename}
    sim_res_config["simulation_results"] = {"sim_res_filename":
                                            sim_res_filename}
    with open(sim_res_config_filename, "w") as f:
        sim_res_config.write(f)

    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
