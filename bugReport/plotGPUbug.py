import os
import traceback
import random
import pickle
import configparser
import torch
import plotly.graph_objects as go

import gcnu_common.utils.config_dict
import svGPFA.stats.svGPFAModelFactory
import svGPFA.stats.svEM
import svGPFA.utils.configUtils
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils


# config params
sim_res_number = 91450833
est_init_number = 551
n_repeats = 1
cuda_device_index = 0
est_init_config_filename_pattern = "../data/{:08d}_estimation_metaData.ini"
sim_res_config_filename_pattern = "../results/{:08d}_simulation_metaData.ini"
results_metadata_filename_pattern = "../results/{:08d}_gpuBugReport_metadata.ini"
results_filename_pattern = "../results/{:08d}_gpuBugReport.pickle"
fig_filename_pattern = "../figures/{:08d}_gpuBugReport.{:s}"

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

# Build configuration file parameter specifications
strings_dict = gcnu_common.utils.config_dict.GetDict(
    config=est_init_config).get_dict()
config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
    n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
    args_info=args_info)

# Build default parameter specificiations
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
        print("Device {:s}, Repeat {:d}: elapsed time {:f}, "
              "elapsed time per function call {:f}, "
              "lower bound {:f}".format(device, r, elapsed_times[d, r],
                                        elapsed_times[d, r]/num_fun_eval[d, r],
                                        lower_bounds[d, r]))
        r += 1


# plot results
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

fig.show()
