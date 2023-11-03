
import sys
import pickle
import argparse

import svGPFA.plot.plotUtilsPlotly


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("est_res_number", help="estimation result number",
                        type=int)
    parser.add_argument("--model_save_filename_pattern",
                        help="filename pattern for model saving",
                        type=str,
                        default="../results/{:08d}_estimatedModel.pickle")
    parser.add_argument("--all_params_fig_filename_pattern",
                        help="all parameters figure filename pattern",
                        type=str,
                        default="../figures/{:08d}_all_params.{{:s}}")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    model_save_filename_pattern = args.model_save_filename_pattern
    all_params_fig_filename_pattern = \
        args.all_params_fig_filename_pattern

    model_save_filename = model_save_filename_pattern.format(est_res_number)
    all_params_fig_filename_pattern = all_params_fig_filename_pattern.format(
        est_res_number)

    with open(model_save_filename, "rb") as f:
        estResults = pickle.load(f)
    model = estResults["model"]

    variational_params = model.getSVPosteriorOnIndPointsParams()
    n_latents = int(len(variational_params)/2)
    variational_mean_params = [variational_params[i] for i in range(n_latents)]
    variational_cov_params = [variational_params[i] for i in range(n_latents,
                                                                   2*n_latents)]

    embedding_params = model.getSVEmbeddingParams()
    C = embedding_params[0]
    d = embedding_params[1].squeeze()

    kernels_params = model.getKernelsParams()
    ind_points_locs = model.getIndPointsLocs()

    fig = svGPFA.plot.plotUtilsPlotly.getPlotAllParams(
        variational_means=variational_mean_params,
        variational_covs=variational_cov_params,
        C=C, d=d, kernels_params=kernels_params,
        ind_points_locs=ind_points_locs,
    )

    fig.write_image(all_params_fig_filename_pattern.format(
        "png"))
    fig.write_html(all_params_fig_filename_pattern.format(
        "html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
