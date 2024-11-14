# Copyright (C) TeNPy Developers, GNU GPLv3
import numpy as np
import timeit
import time
import pickle
import os
import sys
import logging


logger = logging.getLogger('benchmark')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def main(repeat_average=5):
    
    # ``python benchmark.py --help`` prints a summary of the options
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--modules', nargs='*', default=None,
                        help='Perform benchmarks for the given modules')
    parser.add_argument('-n', '--n_sites', type=int, default=4,
                        help='Maximum number of sites. Non-symmetric leg dim is 2**s')
    parser.add_argument('-l', '--n_legs', type=int, default=2,
                        help='Number of legs to contract. Tensors have 2*l legs each')
    parser.add_argument('-s', '--symmetry', type=str, default='SU(2)', choices=['SU(2)', 'U(1)', 'None'],
                        help='Which symmetry to enforce')
    parser.add_argument('-b', '--backend', type=str, default='fusion_tree',
                        choices=['fusion_tree', 'abelian', 'no_symmetry'],
                        help='A tenpy symmetry backend. Ignored for numpy benchmarks.')
    parser.add_argument('--bestof', type=int, default=5,
                        help='How often to repeat each benchmark to reduce the noise.')
    parser.add_argument('-f', '--outfolder', type=str, default='.',
                        help='Folder for output.')
    parser.add_argument('-t', '--maxtime', type=int, default=300,
                        help='Maximum runtime in seconds, for each module. Default 5 min.')
    args = parser.parse_args()

    if args.modules is None:
        raise ValueError('Need to specify --modules.')

    log_file = os.path.join(args.outfolder, 'log')
    logger.addHandler(logging.FileHandler(log_file, mode='w'))
    
    for module in args.modules:
        module = str(module)
        if module.endswith('.py'):
            module = module[:-3]
        perform_benchmark(module_name=module, outfolder=args.outfolder, max_time=args.maxtime,
                          symmetry=args.symmetry, symmetry_backend=args.backend,
                          max_n_sites=args.n_sites, n_legs=args.n_legs, seeds=[1, 42, 123],
                          repeat_avg=1, repeat_bestof=args.bestof)


def perform_benchmark(module_name: str,
                      outfolder: str = '.',
                      max_time: int = 300,
                      symmetry: str = 'SU(2)',
                      symmetry_backend: str = 'fusion_tree',
                      max_n_sites: int = 20,
                      n_legs: int = 2,
                      seeds: list[int] = [1, 42, 123],
                      repeat_avg: int = 1,
                      repeat_bestof: int = 5):
    block_backend = 'numpy'

    save_file = os.path.join(
        outfolder,
        f'{module_name}_l_{n_legs}_s_{symmetry}_b_{symmetry_backend}.pkl'
    )
    
    logger.info('\n'.join([
        '',
        '-' * 80,
        f'Benchmarking module {module_name}',
        f'symmetry : {symmetry}, backend : {symmetry_backend}',
        '-' * 80
    ]))
    t0 = time.time()
    results = {}  # results[n_sites] = best_runtime_seconds
    kwargs = dict(
        n_legs=n_legs, symmetry=symmetry, symmetry_backend=symmetry_backend,
        block_backend=block_backend
    )
    for n_sites in range(1, max_n_sites + 1):
        kwargs['n_sites'] = n_sites
        results_seeds = []
        for seed in seeds:
            kwargs['seed'] = seed
            setup_code = (
                f'import {module_name}\n'
                f'data = {module_name}.setup_benchmark(**{kwargs!r})'
            )
            timing_code = (
                f'{module_name}.benchmark(data)'
            )
            T = timeit.Timer(timing_code, setup_code)
            res = T.repeat(repeat_bestof, repeat_avg)
            results_seeds.append(min(res) / repeat_avg)
        res = np.mean(results_seeds)
        results[n_sites] = res
        
        logger.info(f'{n_sites=}  {res}s')
        
        with open(save_file, 'wb') as f:
            pickle.dump(results, f)
            
        time_passed = time.time() - t0
        if time_passed > max_time:
            logger.info(f'Time exceeded. {int(time_passed)}s > {max_time}s')
            break

if __name__ == '__main__':
    main()
