import argparse
from LRU import simulation, frame_potential

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help="Mode")
parser.add_argument('--id')
args = vars(parser.parse_args())


if __name__ == '__main__':
    if args['mode'] == 'simulation':
        simulation(directory = '../results/Local_Random/', k=3, threshold=1.1, nproc=64, id=args['id'])
    elif args['mode'] == 'frame_potential':
        frame_potential(results_dir = '../results/Local_Random/', k=5)
    else:
        print("Invalid Argument")