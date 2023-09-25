
import pdb
import sys
import torch
import plotly.io as pio
import pickle
import argparse
import configparser
import numpy as np

import gcnu_common.stats.pointProcesses.tests
import svGPFA.plot.plotUtils
import svGPFA.plot.plotUtilsPlotly
import svGPFA.utils.miscUtils
import svGPFA.utils.configUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_res_number", help="Simulation result number",
                        type=int)
    parser.add_argument("--latent_to_plot", help="Latent to plot",
                        type=int, default=0)
    parser.add_argument("--trial_to_plot", help="Trial to plot", type=int,
                        default=0)
    parser.add_argument("--neuron_to_plot", help="Neuron to plot", type=int,
                        default=0)
    parser.add_argument("--n_resamples_ksTest",
                        help="Number of resamples for KS test", type=int,
                        default=10)
    parser.add_argument("--sim_res_config_filename_pattern",
                        help="Simulation result configuration filename pattern",
                        type=str,
                        default="../results/{:08d}_simulation_metaData.ini")
    args = parser.parse_args()

    sim_res_number = args.sim_res_number
    latent_to_plot = args.latent_to_plot
    trial_to_plot = args.trial_to_plot
    neuron_to_plot = args.neuron_to_plot
    n_resamples_ksTest = args.n_resamples_ksTest
    sim_res_config_filename_pattern = args.sim_res_config_filename_pattern

    sim_res_config_filename = sim_res_config_filename_pattern.format(
        sim_res_number)
    sim_res_config = configparser.ConfigParser()
    sim_res_config.read(sim_res_config_filename)
    sim_init_config_filename = sim_res_config["simulation_params"]["sim_init_config_filename"]

    sim_init_config = configparser.ConfigParser()
    sim_init_config.read(sim_init_config_filename)
    n_time_steps_CIF = float(sim_init_config["control_variables"]["n_time_steps_CIF"])
    # CFilename = sim_init_config["embedding_params0"]["C0_filename"]
    # dFilename = sim_init_config["embedding_params0"]["d0_filename"]
    # C_np = np.genfromtxt(CFilename, delimiter=",")
    # C = torch.from_numpy(C_np).type(torch.double)
    # d_np = np.genfromtxt(dFilename, delimiter=",")
    # d = torch.from_numpy(d_np).type(torch.double).unsqueeze(dim=1)

    sim_res_filename = "../results/{:08d}_simRes.pickle".format(sim_res_number)
    latent_fig_filename_pattern = \
        "../figures/{:08d}_simulation_latent_trial{:03d}_latent{:03d}.{{:s}}".\
        format(sim_res_number, trial_to_plot, latent_to_plot)
    embedding_fig_filename_pattern = \
        "../figures/{:08d}_simulation_embedding_trial{:03d}_neuron{:03d}.{{:s}}".\
        format(sim_res_number, trial_to_plot, neuron_to_plot)
    cif_fig_filename_pattern = \
        "../figures/{:08d}_simulation_cif_trial{:03d}_neuron{:03d}.{{:s}}".\
        format(sim_res_number, trial_to_plot, neuron_to_plot)
    spikesTimes_fig_filename_pattern = \
        "../figures/{:08d}_simulation_spikesTimes_trial{:03d}.{{:s}}".format(
            sim_res_number, trial_to_plot)
    spikes_rates_fig_filename_pattern = \
        "../figures/{:08d}_simulation_spikesRates.{{:s}}".format(sim_res_number)
    ksTest_time_rescaling_fig_filename_pattern = \
        "../figures/{:08d}_simulation_ksTestTimeRescaling_trial{:03d}_neuron{:03d}.{{:s}}".format(sim_res_number, trial_to_plot, neuron_to_plot)
    roc_fig_filename_pattern = \
        "../figures/{:08d}_simulation_rocAnalysis_trial{:03d}_neuron{:03d}.{{:s}}".format(sim_res_number, trial_to_plot, neuron_to_plot)

    with open(sim_res_filename, "rb") as f:
        simRes = pickle.load(f)
    trials_times = simRes["trials_times"]
    latents_samples = simRes["latents_samples"]
    latents_means = simRes["latents_means"]
    latents_STDs = simRes["latents_STDs"]
    cif_values = simRes["cif_values"]
    if "spikes" in simRes:
        spikes_times = simRes["spikes"]
    elif "spikes_times" in simRes:
        spikes_times = simRes["spikes_times"]
    else:
        raise ValueError("spikes or spikes_times should be keys of "
                         f"{sim_res_filename}")

    pio.renderers.default = "browser"

    n_trials = len(trials_times)
    trials_times_to_plot = trials_times[trial_to_plot].squeeze()

    latent_samples_to_plot = latents_samples[trial_to_plot][latent_to_plot, :]
    latent_means_to_plot = latents_means[trial_to_plot][latent_to_plot, :]
    latents_STDs_to_plot = latents_STDs[trial_to_plot][latent_to_plot, :]
    title = "Trial {:d}, Latent {:d}".format(trial_to_plot, latent_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getSimulatedLatentPlot(
        times=trials_times_to_plot, latent_samples=latent_samples_to_plot,
        latent_means=latent_means_to_plot, latent_STDs=latents_STDs_to_plot,
        title=title)
    fig.write_image(latent_fig_filename_pattern.format("png"))
    fig.write_html(latent_fig_filename_pattern.format("html"))
    fig.show()

    # embedding_samples[r], embedding_means[r], embedding_STDs
    # \in n_neurons x nSamples
    embedding_samples = [torch.matmul(C, latents_samples[r])+d
                         for r in range(n_trials)]
    embedding_means = [torch.matmul(C, latents_means[r])+d
                       for r in range(n_trials)]
    embedding_STDs = [torch.matmul(C**2, latents_STDs[r]**2).sqrt()
                      for r in range(n_trials)]
    embedding_samples_to_plot = \
        embedding_samples[trial_to_plot][neuron_to_plot, :]
    embedding_means_to_plot = embedding_means[trial_to_plot][neuron_to_plot, :]
    embedding_STDs_to_plot = embedding_STDs[trial_to_plot][neuron_to_plot, :]
    title = "Trial {:d}, Neuron {:d}".format(trial_to_plot, neuron_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getSimulatedEmbeddingPlot(
        times=trials_times_to_plot, samples=embedding_samples_to_plot,
        means=embedding_means_to_plot, stds=embedding_STDs_to_plot,
        title=title)
    fig.write_image(embedding_fig_filename_pattern.format("png"))
    fig.write_html(embedding_fig_filename_pattern.format("html"))
    fig.show()

    cif_values_to_plot = cif_values[trial_to_plot][neuron_to_plot]
    title = "Trial {:d}, Neuron {:d}".format(trial_to_plot, neuron_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotCIF(times=trials_times_to_plot,
                                                 values=cif_values_to_plot,
                                                 title=title)
    fig.write_image(cif_fig_filename_pattern.format("png"))
    fig.write_html(cif_fig_filename_pattern.format("html"))
    fig.show()

    spikes_times_to_plot = spikes_times[trial_to_plot]
    title = "Trial {:d}".format(trial_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getSpikesTimesPlotOneTrial(
        spikes_times=spikes_times_to_plot, title=title)
    fig.write_image(spikesTimes_fig_filename_pattern.format("png"))
    fig.write_html(spikesTimes_fig_filename_pattern.format("html"))
    fig.show()

    spikes_rates = svGPFA.utils.miscUtils.computeSpikeRates(
        trials_times=trials_times, spikes_times=spikes_times)
    fig = svGPFA.plot.plotUtilsPlotly.\
        getPlotSpikeRatesForAllTrialsAndAllNeurons(spikes_rates=spikes_rates)
    fig.write_image(spikes_rates_fig_filename_pattern.format("png"))
    fig.write_html(spikes_rates_fig_filename_pattern.format("html"))
    fig.show()

    cif_values_to_plot = cif_values[trial_to_plot][neuron_to_plot]
    spikes_times_to_plot = spikes_times[trial_to_plot][neuron_to_plot]

    diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = \
        gcnu_common.stats.pointProcesses.tests.KSTestTimeRescalingNumericalCorrection(
            spikes_times=spikes_times_to_plot, cif_times=trials_times_to_plot,
            cif_values=cif_values_to_plot, gamma=n_resamples_ksTest)
    title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(
        trial_to_plot, neuron_to_plot, len(spikes_times_to_plot))
    fig = svGPFA.plot.plotUtilsPlotly.getPlotResKSTestTimeRescalingNumericalCorrection(
        diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx,
        estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb,
        title=title)
    fig.write_image(ksTest_time_rescaling_fig_filename_pattern.format("png"))
    fig.write_html(ksTest_time_rescaling_fig_filename_pattern.format("html"))
    fig.show()

    fpr, tpr, roc_auc = svGPFA.utils.miscUtils.computeSpikeClassificationROC(
        spikes_times=spikes_times_to_plot,
        cif_times=trials_times_to_plot,
        cif_values=cif_values_to_plot)
    title = "Trial {:d}, Neuron {:d}".format(trial_to_plot, neuron_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotResROCAnalysis(
        fpr=fpr, tpr=tpr, auc=roc_auc, title=title)
    fig.write_image(roc_fig_filename_pattern.format("png"))
    fig.write_html(roc_fig_filename_pattern.format("html"))
    fig.show()

    pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
