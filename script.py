import numpy as np
import h5py
import pylab as plt

coating = {'mea_1' : {'channels' : [ 0,  1,  2,  3,  5,  6,  8, 10, 12, 14, 16, 18, 20, 22, 23, 24, 26,
       28, 30, 32, 34, 36, 37, 39, 41, 43, 51, 52, 54, 56, 57],
                      'mapping' : 'mapping_mea1.txt',
                      'file' : 'MEA1.h5'},
           'mea_2' : {'channels' : [0, 1, 2, 4, 6, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 29, 31, 33, 35,
           37, 39, 43, 45, 52, 54, 55, 57],
            'mapping' : 'mapping_mea2.txt', 
            'file' : 'MEA2.h5'},
           'mea_3' : {'channels' : [0, 1, 3, 5, 7, 9, 12, 14, 16, 18, 20, 22, 24, 27, 28, 30, 32, 34, 37, 39, 41,
           43, 46, 52, 53, 55, 57],
            'mapping' : 'mapping_mea2.txt', 
            'file' : 'MEA3.h5'}
            }

mcs_mapping = h5py.File('MEA3.h5')['Data/Recording_0/AnalogStream/Stream_2/InfoChannel']['Label'].astype('int')

all_channels = np.delete(np.arange(60), [14, 44])

from circus.shared.parser import CircusParser
from circus.shared.files import load_data
from circus.shared.probes import get_nodes_and_edges

for key in ['mea_1', 'mea_2', 'mea_3']:

    mapping = np.loadtxt(coating[key]['mapping'])
    params = CircusParser(coating[key]['file'])
    params.get_data_file()

    mads = load_data(params, 'mads')
    whitening = np.linalg.inv(load_data(params, 'spatial_whitening'))
    thresholds = load_data(params, 'thresholds')

    coated_channels = []
    for i in mapping[coating[key]['channels'],1]:
       coated_channels += [np.where(mcs_mapping == i)[0]]

    nodes, edges = get_nodes_and_edges(params)

    inv_nodes = np.zeros(60, dtype=np.int32)
    inv_nodes[nodes] = np.arange(len(nodes))

    coated_channels = np.array(coated_channels).flatten()
    non_coated_channels = all_channels[~np.in1d(all_channels, coated_channels)]


    fig, ax = plt.subplots(3, 3, figsize=(15,10))

    ax[0, 0].violinplot([mads[inv_nodes[coated_channels]]], [0], showmeans=True)
    ax[0, 0].violinplot([mads[inv_nodes[non_coated_channels]]], [1], showmeans=True)

    ax[0, 0].set_ylabel('Noise level')
    ax[0, 0].spines['top'].set_visible(False)
    ax[0, 0].spines['right'].set_visible(False)
    ax[0, 0].set_xticks([])

    res = load_data(params, 'mua')
    coated_amplitudes = np.zeros(0, dtype=np.float32)

    for a in inv_nodes[coated_channels]:
        coated_amplitudes = np.concatenate((coated_amplitudes, res['amplitudes']['elec_%d' %a]/thresholds[a]))
        
    non_coated_amplitudes = np.zeros(0, dtype=np.float32)

    for a in inv_nodes[non_coated_channels]:
        non_coated_amplitudes = np.concatenate((non_coated_amplitudes, res['amplitudes']['elec_%d' %a]/thresholds[a]))


    # ax[0, 1].violinplot([coated_amplitudes], [0], showmeans=True)
    # ax[0, 1].violinplot([non_coated_amplitudes], [1], showmeans=True)
    # ax[0, 1].set_ylabel('normalized peak amplitude')
    # ax[0, 1].spines['top'].set_visible(False)
    # ax[0, 1].spines['right'].set_visible(False)
    # ax[0, 1].set_xticks([])

    coated_snrs = np.zeros(0, dtype=np.float32)

    for a in inv_nodes[coated_channels]:
        coated_snrs = np.concatenate((coated_snrs, [-res['amplitudes']['elec_%d' %a].min()/mads[a]]))
        
    non_coated_snrs = np.zeros(0, dtype=np.float32)

    for a in inv_nodes[non_coated_channels]:
        non_coated_snrs = np.concatenate((non_coated_snrs, [-res['amplitudes']['elec_%d' %a].min()/mads[a]]))

    ax[0, 1].violinplot([coated_snrs], [0], showmeans=True)
    ax[0, 1].violinplot([non_coated_snrs], [1], showmeans=True)
    ax[0, 1].set_ylabel('SNR')
    ax[0, 1].spines['top'].set_visible(False)
    ax[0, 1].spines['right'].set_visible(False)
    ax[0, 1].set_xticks([])

    gmin = min(coated_amplitudes.min(), non_coated_amplitudes.min())
    bins = np.linspace(gmin, -1, 20)

    x, y = np.histogram(coated_amplitudes, bins, density=True)
    ax[0, 2].semilogy(y[1:], x)

    x, y = np.histogram(non_coated_amplitudes, bins, density=True)
    ax[0, 2].semilogy(y[1:], x)
    ax[0, 2].legend(('coated', 'non coated'))
    ax[0, 2].set_xlabel('normalized peak amplitude')
    ax[0, 2].set_ylabel('probability density')
    ax[0, 2].spines['top'].set_visible(False)
    ax[0, 2].spines['right'].set_visible(False)
    ymin, ymax = ax[0, 2].get_ylim()
    ax[0, 2].plot([-1, -1], [ymin, ymax], 'k--')

    electrodes = load_data(params, 'electrodes')
    ax[2, 2].bar([0], [len(np.intersect1d(electrodes, coated_channels))])
    ax[2, 2].bar([1], [len(np.intersect1d(electrodes, non_coated_channels))])
    ax[2, 2].set_ylabel('Cells detected')
    ax[2, 2].spines['top'].set_visible(False)
    ax[2, 2].spines['right'].set_visible(False)
    ax[2, 2].set_xticks([])

    templates = load_data(params, 'templates')
    nb_templates = templates.shape[1]//2
    templates = templates[:,:nb_templates].toarray()
    templates = templates.reshape(58, 31, nb_templates)

    from circus.shared.probes import *
    nodes, positions = get_nodes_and_positions(params)

    norms = numpy.linalg.norm(templates, axis=1)
    ax[1, 0].violinplot(norms[inv_nodes[coated_channels]].flatten(), [0], showmeans=True)
    ax[1, 0].violinplot(norms[inv_nodes[non_coated_channels]].flatten(), [1], showmeans=True)
    ax[1, 0].set_ylabel('Energy of templates')
    ax[1, 0].spines['top'].set_visible(False)
    ax[1, 0].spines['right'].set_visible(False)
    ax[1, 0].set_xticks([])


    coms = np.dot(positions[:,:2].T, np.abs(templates[:,15, :]))/np.abs(templates[:,15,:]).sum(0)
    peaks = np.min(templates[:, 15, :], 0)

    ax[2, 0].scatter(positions[inv_nodes[coated_channels], 0], positions[inv_nodes[coated_channels], 1], c='C0')
    ax[2, 0].scatter(positions[inv_nodes[non_coated_channels], 0], positions[inv_nodes[non_coated_channels], 1], c='C1')
    ax[2, 0].scatter(coms[0], coms[1], c='k')
    ax[2, 0].spines['top'].set_visible(False)
    ax[2, 0].spines['right'].set_visible(False)
    ax[2, 0].spines['left'].set_visible(False)
    ax[2, 0].spines['bottom'].set_visible(False)
    ax[2, 0].set_xticks([])
    ax[2, 0].set_yticks([])
    ax[2, 0].set_title('MEA layout')

    # import sklearn.metrics.pairwise
    # distances_coated = sklearn.metrics.pairwise.distance.cdist(coms.T, positions[inv_nodes[coated_channels],:2])
    # distances_non_coated = sklearn.metrics.pairwise.distance.cdist(coms.T, positions[inv_nodes[non_coated_channels],:2])

    # gmax = min(distances_coated.max(), distances_non_coated.max())
    # bins = np.linspace(0, gmax, 20)

    # x, y = np.histogram(distances_coated, bins, density=True)

    # ax[2, 1].spines['top'].set_visible(False)
    # ax[2, 1].spines['right'].set_visible(False)
    # ax[2, 1].plot(y[1:], x)

    # x, y = np.histogram(distances_non_coated, bins, density=True)
    # ax[2, 1].plot(y[1:], x)

    # ax[2, 1].set_xlabel('Distances with COMs')
    # ax[2, 1].set_ylabel('probability density')

 
    purity = load_data(params, 'purity')
    import matplotlib
    cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=1)
    my_cmap = plt.get_cmap('winter')
    scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=my_cmap)

    results = load_data(params, 'results')
    for count, spikes in enumerate(results['spiketimes'].values()):
        colorVal = scalarMap.to_rgba(purity[count])
        ax[2, 1].scatter(spikes/params.data_file.sampling_rate, count*np.ones(len(spikes)), color=colorVal)
    ax[2, 1].set_xlabel('time (s)')
    ax[2, 1]


    ax[2, 1].spines['top'].set_visible(False)
    ax[2, 1].spines['right'].set_visible(False)

    ax[1, 1].spines['top'].set_visible(False)
    ax[1, 1].spines['right'].set_visible(False)

    ax[1, 2].spines['top'].set_visible(False)
    ax[1, 2].spines['right'].set_visible(False)

    for count, e in enumerate(electrodes):
        if e in coated_channels:
            ax[1, 1].plot(templates[e,:,count]/thresholds[e], c='C0')
        else:
            ax[1, 2].plot(templates[e,:,count]/thresholds[e], c='C1')

    ax[1, 1].plot([0, 31], [-1, -1], 'k--')
    ax[1, 2].plot([0, 31], [-1, -1], 'k--')
    ymin = min(ax[1, 1].get_ylim()[0], ax[1, 2].get_ylim()[0])
    ymax = max(ax[1, 1].get_ylim()[1], ax[1, 2].get_ylim()[1])
    ax[1, 1].set_ylim(ymin, ymax)
    ax[1, 2].set_ylim(ymin, ymax)
    ax[1, 1].set_xlabel('timesteps')
    ax[1, 2].set_xlabel('timesteps')
    ax[1, 1].set_ylabel('amplitude')
    ax[1, 2].set_ylabel('amplitude')

    plt.savefig(key + '.pdf')
    plt.tight_layout()
    plt.show()
    #plt.close()