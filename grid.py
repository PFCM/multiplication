"""Do a stack of experiments"""
import os
import subprocess


# the collection of hyper parameters
common_args = [
    'python',
    'multiply.py',
    '--learning_rate=0.1',
    '--size=100',
    '--max_steps=100000',
    '--validation=0.0',
    '--binary=True'
]

all_cp_ranks = [1, 25, 50, 75, 100, 125, 200]
all_tt_ranks = [1, 8, 11, 14, 16, 18, 24]

operations = {
    'multiply': {
        'cp': all_cp_ranks,
        'tt': all_tt_ranks
    },
    'permute_multiply': {
        'cp': all_cp_ranks,
        'tt': all_tt_ranks
    },
    'correlate': {
        'cp': [50],
        'tt': [11],
        'values': [50, 25, 75, 45, 55]
    }
}

decomps = ['cp', 'tt']


def do_exp(all_args):
    print('-----------------------------------------------')
    subprocess.run(all_args, check=True)
    print('-----------------------------------------------')


def run():
    for experiment in operations:
        print('running: ' + experiment)
        for decomp in decomps:
            for rank in operations[experiment][decomp]:
                args = ['--decomposition=' + decomp,
                        '--rank={}'.format(rank),
                        '--operation=' + experiment]
                logfile = os.path.join('results', experiment, decomp)
                os.makedirs(logfile, exist_ok=True)
                if 'values' in operations[experiment]:
                    log_base = logfile
                    for value in operations[experiment]['values']:
                        logfile = os.path.join(
                            log_base, 'rank{}-{}vals.json'.format(rank, value))
                        args += ['--log_path='+logfile,
                                 '--values={}'.format(value)]
                        do_exp(common_args + args)
                else:
                    logfile = os.path.join(logfile, 'rank{}.json'.format(rank))
                    args += ['--log_path=' + logfile]
                    do_exp(common_args + args)


if __name__ == '__main__':
    run()
