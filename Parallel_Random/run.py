import argparse
from functions import trace_generation, frame_potential, frame_potential_uncorrelated, peo_finder

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help="Mode")
args = vars(parser.parse_args())


if __name__ == '__main__':
    if args['mode'] == 'trace_generation':
        trace_generation(directory = '../results/Parallel_Random/', k=3, threshold=1.2)
    elif args['mode'] == 'frame_potential':
        frame_potential(results_dir = '../results/Parallel_Random/')
    elif args['mode'] == 'frame_potential_uncorrelated':
        frame_potential_uncorrelated(results_dir = '../results/Parallel_Random/')
    elif args['mode'] == 'peo_finder':
        peo_finder()
    else:
        print("Invalid Argument")
