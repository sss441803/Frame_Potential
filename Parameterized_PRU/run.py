import argparse
from PRU import trace_generation, frame_potential, peo_finder #,train_PRU

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help="Mode")
args = vars(parser.parse_args())


if __name__ == '__main__':
    if args['mode'] == 'trace_generation':
        trace_generation(directory = '../results/Parameterized_PRU/', k=4, threshold=1.01)
    elif args['mode'] == 'frame_potential':
        frame_potential(results_dir = '../results/Parameterized_PRU/')
    #elif args['mode'] == 'train_PRU':
    #    train_PRU()
    elif args['mode'] == 'peo_finder':
        peo_finder()
    else:
        print("Invalid Argument")