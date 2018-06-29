import multiprocessing as mp
import os
import sys

if __name__ == "__main__":
    args = sys.argv
    instanceID = int(args[1])
    if len(args) == 3:
        nThread = int(args[2])
    else:
        nThread = mp.cpu_count()

    with open(f'../../../data/{instanceID}.log', 'w') as fp:
        print('Quest Job Array Unit Test', file=fp)
        print(f'Instance #{instanceID} running', file=fp)
        print(f'{sys.version}', file=fp)
        print(f'With {nThread} CPU threads available', file=fp)