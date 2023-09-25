import sys
import argparse
import numpy as np

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_filename", help="latent filename",
                        type=str,
                        default="../results/simulations/00000094_latent0_trial0_latent_mean_sigmoidal.npz")
    parser.add_argument("--kernel_params", help="kernel parameters",
                        type=str,
                        default="../results/simulations/00000094_latent0_trial0_latent_mean_sigmoidal.npz")


if __name__ == "__main__":
    main(sys.argv)
