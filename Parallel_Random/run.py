import argparse
from PRU import trace_generation, frame_potential, train_PRU, peo_finder

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help="Mode")
args = vars(parser.parse_args())


if __name__ == '__main__':
    if args['mode'] == 'trace_generation':
        trace_generation(directory = './results/PRU/', k=2, threshold=1.01)
    elif args['mode'] == 'frame_potential':
        frame_potential(results_dir = './results/PRU/')
    elif args['mode'] == 'train_PRU':
        train_PRU()
    elif args['mode'] == 'peo_finder':
        peo_finder()
    else:
        print("Invalid Argument")