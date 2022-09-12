import argparse

from parallel_random import ansatze_config, trace_gen, peo_finder
from functions import simulation, frame_potential, bootstrapped_frame_potential

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help="Mode")
args = vars(parser.parse_args())


if __name__ == '__main__':
    if args['mode'] == 'simulation':
        simulation(ansatze_config, trace_gen, k=10, threshold=1.01)
    elif args['mode'] == 'frame_potential':
        frame_potential(ansatze_config)
    elif args['mode'] == 'bootstrapped_frame_potential':
        bootstrapped_frame_potential(ansatze_config)
    elif args['mode'] == 'peo_finder':
        peo_finder()
    else:
        print("Invalid Argument")