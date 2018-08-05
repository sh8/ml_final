import pickle

import matplotlib.pyplot as plt

LINES = [
    'NORM_PG_2.0.p',
    'NORM_PG_4.0.p',
    'NORM_PG_6.0.p',
    'NORM_APG_2.0.p',
    'NORM_APG_4.0.p',
    'NORM_APG_6.0.p',
]

if __name__ == '__main__':
    lines = []
    r = range(1, 101)

    for LINE in LINES:
        with open(LINE, 'rb') as f:
            norm = pickle.load(f)
            line, = plt.plot(list(r), norm)
            lines.append(line)

    plt.legend(
        lines,
        [
            'PG(λ=2)',
            'PG(λ=4)',
            'PG(λ=6)',
            'APG(λ=2)',
            'APG(λ=4)',
            'APG(λ=6)',
        ])
    plt.yscale('log')
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Norm', fontsize=18)
    plt.subplots_adjust(hspace=0.4)
    plt.savefig('problem2_1_2_compare.png')
