
import sys
import torch
import pickle
import argparse
import configparser

import gcnu_common.utils.config_dict
import gcnu_common.stats.pointProcesses.tests
import svGPFA.utils.configUtils
import svGPFA.utils.initUtils
import svGPFA.utils.miscUtils
import svGPFA.plot.plotUtilsPlotly


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("est_res_number", help="estimation result number",
                        type=int)
    parser.add_argument("--trial_to_plot", help="trial to plot", type=int,
                        default=0)
    parser.add_argument("--latent_to_plot", help="latent to plot", type=int,
                        default=0)
    parser.add_argument("--neuron_to_plot", help="neuron to plot", type=int,
                        default=0)
    parser.add_argument("--n_resamples_ksTest",
                        help="number of resamples in KS test",
                        type=int, default=10)
    parser.add_argument("--est_init_config_filename_pattern",
                        help="estimation initialization configuration "
                             "filename pattern",
                        type=str,
                        default="../data/{:08d}_estimation_metaData.ini")
    parser.add_argument("--est_res_metadata_filename_pattern",
                        help="estimation result metadata filename pattern",
                        type=str,
                        default="../results/{:08d}_estimation_metaData.ini")
    parser.add_argument("--model_save_filename_pattern",
                        help="filename pattern for model saving",
                        type=str,
                        default="../results/{:08d}_estimatedModel.pickle")
    parser.add_argument(
        "--lower_bound_hist_vs_iterNo_fig_filename_pattern_pattern",
        help="lower bound history versus iteration number figure filename "
             "pattern pattern",
        type=str,
        default="../figures/{:08d}_lowerBoundHistVsIterNo.{{:s}}")
    parser.add_argument(
        "--lower_bound_hist_vs_elapsed_time_fig_filename_pattern_pattern",
        help="lower bound history versus elapsed time figure filename "
             "pattern pattern",
        type=str,
        default="../figures/{:08d}_lowerBoundHistVsElapsedTime.{{:s}}")
    parser.add_argument(
        "--latents_fig_filename_pattern_pattern",
        help="latents figure filename pattern pattern",
        type=str,
        default="../figures/{:08d}_trueAndEstimatedLatents_latent{:03d}_trial{:03d}.{{:s}}")
    parser.add_argument(
        "--indPoints_mean_fig_filename_pattern_pattern",
        help="inducing points mean figure figure filename pattern pattern",
        type=str,
        default="../figures/{:08d}_trueAndEstimatedIndPointsMeans_latent{:03d}_trial{:03d}.{{:s}}")
    parser.add_argument(
        "--indPoints_cov_fig_filename_pattern_pattern",
        help="inducing points cov figure figure filename pattern pattern",
        type=str,
        default="../figures/{:08d}_trueAndEstimatedIndPointsCovs_latent{:03d}_trial{:03d}.{{:s}}")
    parser.add_argument(
        "--ksTest_time_rescaling_fig_filename_pattern_pattern",
        help="Kolmogorov-Smirnov time-rescaling test figure filename "
             "pattern pattern",
        type=str,
        default="../figures/{:08d}_ksTestTimeRescaling_numericalCorrection_trial{:03d}_neuron{:03d}.{{:s}}")
    parser.add_argument(
        "--true_and_estimated_CIFs_fig_filename_pattern_pattern",
        help="true and estimated CIFs figure filename pattern pattern",
        type=str,
        default="../figures/{:08d}_trueAndEstimatedCIFs_trial{:03d}_neuron{:03d}.{{:s}}")
    parser.add_argument(
        "--roc_fig_filename_pattern_pattern",
        help="ROC curve for spike classificaition based on the estimated CIF "
             "figure filename pattern pattern",
        type=str,
        default="../figures/{:08d}_rocAnalysis_trial{:03d}_neuron{:03d}.{{:s}}")
    parser.add_argument(
        "--kernels_params_fig_filename_pattern_pattern",
        help="kernels parameters figure filename pattern pattern",
        type=str,
        default="../figures/{:08d}_trueAndEstimatedKernelsParams.{{:s}}")
    parser.add_argument(
        "--embedding_params_fig_filename_pattern_pattern",
        help="embedding parameters figure filename pattern pattern",
        type=str,
        default="../figures/{:08d}_trueAndEstimatedEmbeddingParams.{{:s}}")
    parser.add_argument(
        "--embedding_fig_filename_pattern_pattern",
        help="embedding figure filename pattern pattern",
        type=str,
        default="../figures/{:08d}_trueAndEstimatedEmbedding_trial{:03d}_neuron{:03d}.{{:s}}")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    trial_to_plot = args.trial_to_plot
    neuron_to_plot = args.neuron_to_plot
    latent_to_plot = args.latent_to_plot
    n_resamples_ksTest = args.n_resamples_ksTest
    est_res_metadata_filename_pattern = args.est_res_metadata_filename_pattern
    model_save_filename_pattern = args.model_save_filename_pattern
    lower_bound_hist_vs_iterNo_fig_filename_pattern_pattern = \
        args.lower_bound_hist_vs_iterNo_fig_filename_pattern_pattern
    lower_bound_hist_vs_elapsed_time_fig_filename_pattern_pattern = \
        args.lower_bound_hist_vs_elapsed_time_fig_filename_pattern_pattern
    latents_fig_filename_pattern_pattern = \
        args.latents_fig_filename_pattern_pattern
    indPoints_mean_fig_filename_pattern_pattern = \
        args.indPoints_mean_fig_filename_pattern_pattern
    indPoints_cov_fig_filename_pattern_pattern = \
        args.indPoints_cov_fig_filename_pattern_pattern
    ksTest_time_rescaling_fig_filename_pattern_pattern = \
        args.ksTest_time_rescaling_fig_filename_pattern_pattern
    true_and_estimated_CIFs_fig_filename_pattern_pattern = \
        args.true_and_estimated_CIFs_fig_filename_pattern_pattern
    roc_fig_filename_pattern_pattern = args.roc_fig_filename_pattern_pattern
    kernels_params_fig_filename_pattern_pattern = \
        args.kernels_params_fig_filename_pattern_pattern
    embedding_params_fig_filename_pattern_pattern = \
        args.embedding_params_fig_filename_pattern_pattern
    embedding_fig_filename_pattern_pattern = \
        args.embedding_fig_filename_pattern_pattern

    est_res_metadata_filename = est_res_metadata_filename_pattern.format(
        est_res_number)
    model_save_filename = model_save_filename_pattern.format(est_res_number)
    lower_bound_hist_vs_iterNo_fig_filename_pattern = \
        lower_bound_hist_vs_iterNo_fig_filename_pattern_pattern.format(
            est_res_number)
    lower_bound_hist_vs_elapsed_time_fig_filename_pattern = \
        lower_bound_hist_vs_elapsed_time_fig_filename_pattern_pattern.format(
            est_res_number)
    latents_fig_filename_pattern = latents_fig_filename_pattern_pattern.format(
        est_res_number, latent_to_plot, trial_to_plot)
    indPoints_mean_fig_filename_pattern = \
        indPoints_mean_fig_filename_pattern_pattern.format(
            est_res_number, latent_to_plot, trial_to_plot)
    indPoints_cov_fig_filename_pattern = \
        indPoints_cov_fig_filename_pattern_pattern.format(
            est_res_number, latent_to_plot, trial_to_plot)
    ksTest_time_rescaling_fig_filename_pattern = \
        ksTest_time_rescaling_fig_filename_pattern_pattern.format(
            est_res_number, trial_to_plot, neuron_to_plot)
    true_and_estimated_CIFs_fig_filename_pattern = \
        true_and_estimated_CIFs_fig_filename_pattern_pattern.format(
            est_res_number, trial_to_plot, neuron_to_plot)
    roc_fig_filename_pattern = roc_fig_filename_pattern_pattern.format(
        est_res_number, trial_to_plot, neuron_to_plot)
    kernels_params_fig_filename_pattern = \
        kernels_params_fig_filename_pattern_pattern.format(est_res_number)
    embedding_params_fig_filename_pattern = \
        embedding_params_fig_filename_pattern_pattern.format(est_res_number)
    embedding_fig_filename_pattern = \
        embedding_fig_filename_pattern_pattern.format(est_res_number,
                                                      trial_to_plot,
                                                      neuron_to_plot)

    # ksTestTimeRescalingAnalyticalCorrectionFigFilename = "../figures/{:08d}_ksTestTimeRescaling_analyticalCorrection_trial{:03d}_neuron{:03d}.png".format(est_res_number, trial_to_plot, neuron_to_plot)
    # timeRescalingDiffCDFsFigFilename = "../figures/{:08d}_timeRescalingDiffCDFs_analyticalCorrection_trial{:03d}_neuron{:03d}.png".format(est_res_number, trial_to_plot, neuron_to_plot)
    # timeRescaling1LagScatterPlotFigFilename = "../figures/{:08d}_timeRescaling1LagScatterPlot_analyticalCorrection_trial{:03d}_neuron{:03d}.png".format(est_res_number, trial_to_plot, neuron_to_plot)
    # timeRescalingACFFigFilename = "../figures/{:08d}_timeRescalingACF_analyticalCorrection_trial{:03d}_neuron{:03d}.png".format(est_res_number, trial_to_plot, neuron_to_plot)

    est_res_config = configparser.ConfigParser()
    est_res_config.read(est_res_metadata_filename)
    sim_res_number = int(est_res_config["simulation_params"]["sim_res_number"])
    est_init_number = int(est_res_config["estimation_params"]["est_init_number"])
    sim_res_config_filename = "../results/{:08d}_simulation_metaData.ini".\
        format(sim_res_number)
    sim_res_config = configparser.ConfigParser()
    sim_res_config.read(sim_res_config_filename)

    sim_res_filename = sim_res_config["simulation_results"]["sim_res_filename"]
    with open(sim_res_filename, "rb") as f:
        sim_res = pickle.load(f)
    spikes_times = sim_res["spikes_times"]
    n_trials = len(spikes_times)
    trials_times = sim_res["trials_times"]
    true_latents_samples = sim_res["latents_samples"]
    true_latents_means = sim_res["latents_means"]
    true_latents_STDs = sim_res["latents_STDs"]
    true_cif_values = sim_res["cif_values"]
    true_C = sim_res["C"]
    true_d = sim_res["d"]
    true_ind_points_locs = sim_res["ind_points_locs"]

    sim_init_config_filename = \
        sim_res_config["simulation_params"]["sim_init_config_filename"]
    sim_init_config = configparser.ConfigParser()
    sim_init_config.read(sim_init_config_filename)
    n_latents = int(sim_init_config["control_variables"]["n_latents"])
    kernels_params0, kernels_types = \
        svGPFA.utils.initUtils.getKernelsParams0AndTypes(
            n_latents=n_latents, config_file_params_spec=sim_init_config)
    true_indPoints_means = svGPFA.utils.initUtils.getVariationalMean0(
        n_latents=n_latents, n_trials=n_trials,
        config_file_params_spec=sim_init_config)
    true_indPoints_covs = svGPFA.utils.initUtils.getVariationalCov0(
        n_latents=n_latents, n_trials=n_trials,
        config_file_params_spec=sim_init_config)

    with open(model_save_filename, "rb") as f:
        estResults = pickle.load(f)
    lowerBoundHist = estResults["lowerBoundHist"]
    elapsedTimeHist = estResults["elapsedTimeHist"]
    model = estResults["model"]

    # plot lower bound history
    fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(
        lowerBoundHist=lowerBoundHist)
    fig.write_image(lower_bound_hist_vs_iterNo_fig_filename_pattern.format(
        "png"))
    fig.write_html(lower_bound_hist_vs_iterNo_fig_filename_pattern.format(
        "html"))

    fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(
        elapsedTimeHist=elapsedTimeHist, lowerBoundHist=lowerBoundHist)
    fig.write_image(
        lower_bound_hist_vs_elapsed_time_fig_filename_pattern.format("png"))
    fig.write_html(
        lower_bound_hist_vs_elapsed_time_fig_filename_pattern.format("html"))

    # true and estimated latents
    testMuK, testVarK = model.predictLatents(times=trials_times)
    eIndPointsLocs = model.getIndPointsLocs()
    fig = svGPFA.plot.plotUtilsPlotly.getPlotTrueAndEstimatedLatents(
        tTimes=trials_times, tLatentsSamples=true_latents_samples,
        tLatentsMeans=true_latents_means, tLatentsSTDs=true_latents_STDs,
        tIndPointsLocs=true_ind_points_locs, eTimes=trials_times,
        eLatentsMeans=testMuK, eLatentsSTDs=torch.sqrt(testVarK),
        eIndPointsLocs=eIndPointsLocs, trialToPlot=trial_to_plot)
    fig.write_image(latents_fig_filename_pattern.format("png"))
    fig.write_html(latents_fig_filename_pattern.format("html"))

    # embedding params
    estimated_C, estimated_d = model.getSVEmbeddingParams()
    fig = svGPFA.plot.plotUtilsPlotly.getPlotTrueAndEstimatedEmbeddingParams(
        trueC=true_C.numpy(), trueD=true_d.numpy(),
        estimatedC=estimated_C.numpy(), estimatedD=estimated_d.numpy())
    fig.write_image(embedding_params_fig_filename_pattern.format("png"))
    fig.write_html(embedding_params_fig_filename_pattern.format("html"))

    # embedding
    true_embedding_samples = svGPFA.utils.miscUtils.getEmbeddingSamples(
        C=true_C, d=true_d, latents_samples=true_latents_samples)
    true_embedding_means = svGPFA.utils.miscUtils.getEmbeddingMeans(
        C=true_C, d=true_d, latents_means=true_latents_means)
    true_embedding_STDs = svGPFA.utils.miscUtils.getEmbeddingSTDs(
        C=true_C, latents_STDs=true_latents_STDs)
    true_embedding_samples_to_plot = \
        true_embedding_samples[trial_to_plot][neuron_to_plot, :]
    true_embedding_means_to_plot = \
        true_embedding_means[trial_to_plot][neuron_to_plot, :]
    true_embedding_STDs_to_plot = \
        true_embedding_STDs[trial_to_plot][neuron_to_plot, :]

    est_embedding_means, est_embedding_vars = \
        model.predictEmbedding(times=trials_times)

    est_means_to_plot = est_embedding_means[trial_to_plot, :, neuron_to_plot]
    est_STDs_to_plot = \
        est_embedding_vars[trial_to_plot, :, neuron_to_plot].sqrt()

    title = "Trial {:d}, Neuron {:d}".format(trial_to_plot, neuron_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotTrueAndEstimatedEmbedding(
        tTimes=trials_times[trial_to_plot, :, 0],
        tSamples=true_embedding_samples_to_plot,
        tMeans=true_embedding_means_to_plot,
        tSTDs=true_embedding_STDs_to_plot,
        eTimes=trials_times[trial_to_plot, :, 0],
        eMeans=est_means_to_plot,
        eSTDs=est_STDs_to_plot,
        title=title)
    fig.write_image(embedding_fig_filename_pattern.format("png"))
    fig.write_html(embedding_fig_filename_pattern.format("html"))

    # predict cif values
    with torch.no_grad():
        epcifValues = model.computeExpectedPosteriorCIFs(times=trials_times)
    spikes_times_to_plot = spikes_times[trial_to_plot][neuron_to_plot]
    trials_times_to_plot = trials_times[trial_to_plot, :, 0]
    cif_values_to_plot = epcifValues[trial_to_plot][neuron_to_plot]

    # CIF
    title = "Trial {:d}, Neuron {:d}".format(trial_to_plot, neuron_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotSimulatedAndEstimatedCIFs(
        tTimes=trials_times[trial_to_plot, :, 0],
        tCIF=true_cif_values[trial_to_plot][neuron_to_plot],
        tLabel="True",
        eMeanTimes=trials_times[trial_to_plot, :, 0],
        # eMeanCIF=emcifValues[trial_to_plot][neuron_to_plot],
        eMeanCIF=epcifValues[trial_to_plot][neuron_to_plot],
        eMeanLabel="Mean",
        # ePosteriorMeanTimes=oneTrialCIFTimes,
        # ePosteriorMeanCIF=epmcifValues[trial_to_plot][neuron_to_plot],
        # ePosteriorMeanLabel="Posterior Mean",
        title=title)
    fig.write_image(true_and_estimated_CIFs_fig_filename_pattern.format("png"))
    fig.write_html(true_and_estimated_CIFs_fig_filename_pattern.format("html"))

    # plot kernels
    tKernelsParams = kernels_params0
    mKernelsParams = model.getKernelsParams()
    fig = svGPFA.plot.plotUtilsPlotly.getPlotTrueAndEstimatedKernelsParams(
        kernelsTypes=kernels_types,
        trueKernelsParams=tKernelsParams,
        estimatedKernelsParams=mKernelsParams)
    fig.write_image(kernels_params_fig_filename_pattern.format("png"))
    fig.write_html(kernels_params_fig_filename_pattern.format("html"))

    title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(
        trial_to_plot, neuron_to_plot, len(spikes_times_to_plot))

    # inducing points means and covariances
    svPosteriorOnIndPointsParams = model.getSVPosteriorOnIndPointsParams()

    eIndPointsMeans = svPosteriorOnIndPointsParams[:n_latents]
    tIndPointsMeansToPlot = true_indPoints_means[trial_to_plot][latent_to_plot][ :, 0]
    eIndPointsMeansToPlot = eIndPointsMeans[latent_to_plot][trial_to_plot, :, 0]
    cholVecs = svPosteriorOnIndPointsParams[n_latents:]
    eIndPointsCovs = svGPFA.utils.miscUtils.buildCovsFromCholVecs(
        cholVecs=cholVecs)
    tIndPointsCovToPlot = true_indPoints_covs[latent_to_plot][trial_to_plot, :, :]
    tIndPointsSTDsToPlot = torch.diag(tIndPointsCovToPlot).sqrt()
    eIndPointsCovToPlot = eIndPointsCovs[latent_to_plot][trial_to_plot, :, :]
    eIndPointsSTDsToPlot = torch.diag(eIndPointsCovToPlot).sqrt()
    title = "Trial {:d}, Latent {:d}".format(trial_to_plot, latent_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotTrueAndEstimatedIndPointsMeansOneTrialOneLatent(trueIndPointsMeans=tIndPointsMeansToPlot, estimatedIndPointsMeans=eIndPointsMeansToPlot, trueIndPointsSTDs=tIndPointsSTDsToPlot, estimatedIndPointsSTDs=eIndPointsSTDsToPlot, title=title,)
    fig.write_image(indPoints_mean_fig_filename_pattern.format("png"))
    fig.write_html(indPoints_mean_fig_filename_pattern.format("html"))

    title = "Trial {:d}, Latent {:d}".format(trial_to_plot, latent_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotTrueAndEstimatedIndPointsCovsOneTrialOneLatent(
        trueIndPointsCov=tIndPointsCovToPlot,
        estimatedIndPointsCov=eIndPointsCovToPlot,
        title=title,
    )
    fig.write_image(indPoints_cov_fig_filename_pattern.format("png"))
    fig.write_html(indPoints_cov_fig_filename_pattern.format("html"))

    # KS test
    diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = \
        gcnu_common.stats.pointProcesses.tests.KSTestTimeRescalingNumericalCorrection(
            spikes_times=spikes_times_to_plot, cif_times=trials_times_to_plot,
            cif_values=cif_values_to_plot, gamma=n_resamples_ksTest)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotResKSTestTimeRescalingNumericalCorrection(
        diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx,
        estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb,
        title=title)
    fig.write_image(ksTest_time_rescaling_fig_filename_pattern.format("png"))
    fig.write_html(ksTest_time_rescaling_fig_filename_pattern.format("html"))

    # ROC predictive analysis
    fpr, tpr, roc_auc = svGPFA.utils.miscUtils.computeSpikeClassificationROC(
        spikes_times=spikes_times_to_plot.tolist(),
        cif_times=trials_times_to_plot,
        cif_values=cif_values_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotResROCAnalysis(
        fpr=fpr, tpr=tpr, auc=roc_auc, title=title)
    fig.write_image(roc_fig_filename_pattern.format("png"))
    fig.write_html(roc_fig_filename_pattern.format("html"))

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
