import os
import matplotlib.pyplot as plt


X = ['50', '80', '100', '120', '150', '180', '200']
PLOT_PATH = 'plots'


# Peter GANILLA
peter_ganilla = {
    'title': 'Peter Rabbit GANILLA',
    'legend_loc': 2,
    'legend_size': 6,
    'lines': {
        'peter_ganilla_random_2':  # peter_ganilla_random_2
            {'fid': [165.23, 169.56, 183.91, 154.28, 151.22, 150.08, 146.33],
             'mse': [8960.73, 10133.38, 10082.73, 9480.68, 9155.82, 8975.09, 9201.85],
             'fwe': [0.005864, 0.006915, 0.007187, 0.006032, 0.005734, 0.005564, 0.005706],
             'color': 'tab:red'},
        'peter_ganilla_sequential_2':  # peter_ganilla_sequential_2
            {'fid': [170.92, 173.76, 174.38, 170.96, 165.79, 185.76, 187.54],
             'mse': [7815.85, 8513.14, 9638.09, 8444.15, 6599.15, 5269.74, 6537.52],
             'fwe': [0.004305, 0.005411, 0.006104, 0.004823, 0.003106, 0.001875, 0.002579],
             'color': 'tab:blue'},
        'peter_ganilla_diff_v1_1_frame1':  # peter_ganilla_diff_v1_1_frame1
            {'fid': [167.00, 167.71, 185.82, 197.20, 192.32, 167.96, 146.89],
             'mse': [7642.02, 7101.13, 8885.20, 9175.34, 8596.72, 8141.61, 6566.38],
             # 'fwe': [],  # TODO
             'color': 'tab:brown'},
        'peter_ganilla_diff_v1_1':  # peter_ganilla_diff_v1_1
            {'fid': [182.48, 169.67, 155.42, 149.82, 146.02, 149.84, 145.63],
             'mse': [6331.83, 5996.03, 7483.09, 6625.74, 6980.22, 7448.30, 7326.54],
             'fwe': [0.003340, 0.002533, 0.004381, 0.003430, 0.003653, 0.004303, 0.004188],
             'color': 'tab:green'},
        'peter_ganilla_seq_diff_v1_1':  # peter_ganilla_seq_diff_v1_1
            {'fid': [167.58, 165.05, 162.24, 169.00, 171.88, 152.75, 184.29],
             'mse': [9914.53, 9931.23, 10010.93, 8729.84, 8436.14, 6998.78, 8854.09],
             # 'fwe': [],  # TODO
             'color': 'tab:orange'},
        'peter_ganilla_diff_v2_1':  # peter_ganilla_diff_v2_1
            {'fid': [145.04, 153.82, 175.42, 166.70, 142.39, 148.93, 142.23],
             'mse': [7610.14, 6965.02, 6467.86, 7362.08, 7972.54, 7795.32, 7933.21],
             'fwe': [0.004074, 0.003731, 0.003804, 0.004825, 0.004878, 0.004771, 0.004901],
             'color': 'tab:purple'},
        # 'peter_ganilla_seq_diff_v2_1':  # peter_ganilla_seq_diff_v2_1 - TODO
        #     {'fid': [],
        #      'mse': [],
        #      'fwe': [],
        #      'color': 'tab:cyan'},
        'peter_ganilla_diff_v3_2':  # peter_ganilla_diff_v3_2
            {'fid': [162.94, 155.99, 151.59, 156.75, 149.36, 146.44, 146.72],
             'mse': [7064.61, 6808.67, 7157.06, 6356.23, 7024.37, 7174.03, 7304.93],
             'fwe': [0.003952, 0.003388, 0.003890, 0.003410, 0.003825, 0.004257, 0.004305],
             'color': 'tab:pink'},
        # 'peter_ganilla_seq_diff_v3_2':  # peter_ganilla_seq_diff_v3_2 - TODO
        #     {'fid': [],
        #      'mse': [],
        #      'fwe': [],
        #      'color': 'tab:gray'},
    }
}

# Peter CycleGAN
peter_cyclegan = {
    'title': 'Peter Rabbit CycleGAN',
    'legend_loc': 2,
    'legend_size': 6,
    'lines': {
        'peter_cyclegan_random_2':  # peter_cyclegan_random_2
            {'fid': [145.96, 146.85, 135.54, 132.00, 135.16, 136.86, 138.76],
             'mse': [7193.22, 8625.38, 7826.17, 7751.49, 8557.44, 8167.74, 8267.22],
             'fwe': [0.003993, 0.005508, 0.004478, 0.004351, 0.005069, 0.004597, 0.004671],
             'color': 'tab:red'},
        'peter_cyclegan_sequential_2':  # peter_cyclegan_sequential_2
            {'fid': [156.29, 176.47, 160.96, 146.43, 150.78, 154.06, 162.54],
             'mse': [8592.34, 9458.47, 6918.58, 8319.03, 10022.89, 8217.70, 7435.01],
             'fwe': [0.005265, 0.006381, 0.004090, 0.004400, 0.006533, 0.004624, 0.004435],
             'color': 'tab:blue'},
        'peter_cyclegan_diff_v1_1_frame1':  # peter_cyclegan_diff_v1_1_frame1
            {'fid': [156.25, 150.57, 148.26, 151.81, 152.51, 148.97, 150.73],
             'mse': [7149.68, 7783.61, 7037.82, 7903.04, 8347.09, 9095.60, 9148.22],
             # 'fwe': [],     # TODO
             'color': 'tab:brown'},
        'peter_cyclegan_diff_v1_3':  # peter_cyclegan_diff_v1_3
            {'fid': [143.62, 140.75, 148.37, 140.82, 146.39, 145.25, 145.00],
             'mse': [7522.38, 8090.81, 8406.33, 8847.03, 9030.39, 9534.88, 9727.44],
             'fwe': [0.004507, 0.005271, 0.005734, 0.006035, 0.006208, 0.006699, 0.006886],
             'color': 'tab:green'},
        # 'peter_cyclegan_seq_diff_v1_3':  # peter_cyclegan_seq_diff_v1_3 - TODO
        #     {'fid': [],
        #      'mse': [],
        #      'fwe': [],
        #      'color': 'tab:orange'},
        'peter_cyclegan_diff_v2_3':  # peter_cyclegan_diff_v2_3
            {'fid': [149.03, 144.71, 151.96, 152.97, 144.83, 151.23, 150.56],
             'mse': [6888.27, 6556.43, 7403.19, 7504.73, 8050.70, 8486.29, 8685.77],
             'fwe': [0.003790, 0.003415, 0.004378, 0.004464, 0.004717, 0.005230, 0.005454],
             'color': 'tab:purple'},
        'peter_cyclegan_seq_diff_v2_1':  # peter_cyclegan_seq_diff_v2_1
            {'fid': [162.60, 166.49, 169.20, 173.96, 180.61, 184.17, 172.62],
             'mse': [8758.63, 9117.28, 10399.49, 10097.91, 10707.93, 11244.93, 9340.88],
             # 'fwe': [],     # TODO
             'color': 'tab:cyan'},
        'peter_cyclegan_diff_v3_3':  # peter_cyclegan_diff_v3_3
            {'fid': [146.56, 137.82, 142.23, 137.27, 140.18, 147.15, 148.97],
             'mse': [7064.61, 6808.69, 7157.06, 6356.23, 7024.37, 7174.03, 7304.93],
             'fwe': [0.003924, 0.004396, 0.004594, 0.004492, 0.005111, 0.005695, 0.005722],
             'color': 'tab:pink'},
        # 'peter_cyclegan_seq_diff_v3_3':  # peter_cyclegan_seq_diff_v3_3 - TODO
        #     {'fid': [],
        #      'mse': [],
        #      'fwe': [],
        #      'color': 'tab:gray'},
    }
}

# Axel GANILLA
axel_ganilla = {
    'title': 'Axel Scheffler GANILLA',
    'legend_loc': 2,
    'legend_size': 6,
    'lines': {
        # 'axel_ganilla_random_2':  # axel_ganilla_random_2 - TODO
        #     {'fid': [],
        #      'mse': [],
        #      'fwe': [],
        #      'color': 'tab:red'},
        # 'axel_ganilla_diff_v3_1':  # axel_ganilla_diff_v3_1 - TODO
        #     {'fid': [],
        #      'mse': [],
        #      'fwe': [],
        #      'color': 'tab:blue'},
    }
}

# Axel CycleGAN
axel_cyclegan = {
    'title': 'Axel Scheffler CycleGAN',
    'legend_loc': 2,
    'legend_size': 6,
    'lines': {
        # 'axel_cyclegan_random_2':  # axel_cyclegan_random_2 - TODO
        #     {'fid': [],
        #      'mse': [],
        #      'fwe': [],
        #      'color': 'tab:red'},
        'axel_cyclegan_diff_v3_1':  # axel_cyclegan_diff_v3_1
            {'fid': [158.43, 160.20, 158.28, 155.23, 154.68, 158.20, 157.89],
             'mse': [4839.46, 5354.18, 5219.98, 5493.52, 5604.55, 5838.33, 5951.96],
             'fwe': [0.003329, 0.003948, 0.003884, 0.004091, 0.004011, 0.004206, 0.004343],
             'color': 'tab:blue'},
    }
}


# Plot
y = peter_ganilla  # change data

for score in ['fid']:  # ['fid', 'mse', 'fwe']
    plt.title('%s %s' % (y['title'], score.upper()))
    for k, v in y['lines'].items():
        plt.plot(X, v[score], label=k, color=v['color'])

    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend(loc=y['legend_loc'], prop={'size': y['legend_size']})
    # plt.savefig(os.path.join(PLOT_PATH, '%s %s.png' % (y['title'], score.upper())))
    # plt.close()
    plt.show()
