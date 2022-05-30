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
mcs_factor = h5py.File('MEA3.h5')['Data/Recording_0/AnalogStream/Stream_2/InfoChannel']['ConversionFactor'][0] * 1e-6

all_channels = np.delete(np.arange(60), [14, 44])

from circus.shared.parser import CircusParser
from circus.shared.files import load_data
from circus.shared.probes import get_nodes_and_edges

save_pdf = True
use_monopole = True

def make_initial_guess_and_bounds(wf_ptp, local_contact_locations, max_distance_um=250):
    # constant for initial guess and bounds
    initial_z = 20

    ind_max = np.argmax(wf_ptp)
    max_ptp = wf_ptp[ind_max]
    max_alpha = max_ptp * max_distance_um

    # initial guess is the center of mass
    com = np.sum(wf_ptp[:, np.newaxis] * local_contact_locations, axis=0) / np.sum(wf_ptp)
    x0 = np.zeros(4, dtype='float32')
    x0[:2] = com
    x0[2] = initial_z
    initial_alpha = np.sqrt(np.sum((com - local_contact_locations[ind_max, :])**2) + initial_z**2) * max_ptp
    x0[3] = initial_alpha

    # bounds depend on initial guess
    bounds = ([x0[0] - max_distance_um, x0[1] - max_distance_um, 1, 0],
              [x0[0] + max_distance_um,  x0[1] + max_distance_um, max_distance_um*10, max_alpha])

    return x0, bounds


fig, ax = plt.subplots(3, 3, figsize=(15,10))

for key in ['mea_2']:

    mapping = np.loadtxt(coating[key]['mapping'])
    params = CircusParser(coating[key]['file'])

    params.write('data', 'suffix', '-raw')
    params.write('whitening', 'spatial', 'False')
    
    params = CircusParser(coating[key]['file'])
    params.get_data_file()

    mads = load_data(params, 'mads')
    thresholds = load_data(params, 'thresholds')

    mads *= mcs_factor
    thresholds *= mcs_factor

    coated_channels = []
    for i in mapping[coating[key]['channels'],1]:
       coated_channels += [np.where(mcs_mapping == i)[0]]

    nodes, edges = get_nodes_and_edges(params)

    inv_nodes = np.zeros(60, dtype=np.int32)
    inv_nodes[nodes] = np.arange(len(nodes))

    coated_channels = np.array(coated_channels).flatten()
    non_coated_channels = all_channels[~np.in1d(all_channels, coated_channels)]

    ax[0, 1].violinplot([mads[inv_nodes[coated_channels]]], [0], showmeans=True)
    ax[0, 1].violinplot([mads[inv_nodes[non_coated_channels]]], [1], showmeans=True)
    ax[0, 1].set_ylabel('Noise level ($\mathrm{\mu}V)$') 
    ax[0, 1].spines['top'].set_visible(False)
    ax[0, 1].spines['right'].set_visible(False)
    ax[0, 1].set_xticks([])
    ax[0, 1].set_title('MEA #1')

    res = load_data(params, 'mua')
    coated_amplitudes = np.zeros(0, dtype=np.float32)

    for a in inv_nodes[coated_channels]:
        coated_amplitudes = np.concatenate((coated_amplitudes, res['amplitudes']['elec_%d' %a]/thresholds[a]))
        
    coated_amplitudes *= mcs_factor

    non_coated_amplitudes = np.zeros(0, dtype=np.float32)

    for a in inv_nodes[non_coated_channels]:
        non_coated_amplitudes = np.concatenate((non_coated_amplitudes, res['amplitudes']['elec_%d' %a]/thresholds[a]))

    non_coated_amplitudes *= mcs_factor

    coated_snrs = np.zeros(0, dtype=np.float32)

    for a in inv_nodes[coated_channels]:
        coated_snrs = np.concatenate((coated_snrs, [-res['amplitudes']['elec_%d' %a].min()/mads[a]]))
        
    coated_snrs *= mcs_factor

    non_coated_snrs = np.zeros(0, dtype=np.float32)

    for a in inv_nodes[non_coated_channels]:
        non_coated_snrs = np.concatenate((non_coated_snrs, [-res['amplitudes']['elec_%d' %a].min()/mads[a]]))

    non_coated_snrs *= mcs_factor

    ax[0, 2].violinplot(20*np.log10(coated_snrs), [0], showmeans=True)
    ax[0, 2].violinplot(20*np.log10(non_coated_snrs), [1], showmeans=True)
    ax[0, 2].set_ylabel('SNR (dB)')
    ax[0, 2].spines['top'].set_visible(False)
    ax[0, 2].spines['right'].set_visible(False)
    ax[0, 2].set_xticks([])
    ax[0, 2].set_title('MEA #1')

    gmin = min(coated_amplitudes.min(), non_coated_amplitudes.min())
    bins = np.linspace(gmin, -1, 20)

    x, y = np.histogram(coated_amplitudes, bins, density=True)
    ax[1, 0].semilogy(y[1:], x)

    x, y = np.histogram(non_coated_amplitudes, bins, density=True)
    ax[1, 0].semilogy(y[1:], x)
    ax[1, 0].legend(('coated', 'non coated'))
    ax[1, 0].set_xlabel('normalized peak amplitude')
    ax[1, 0].set_ylabel('probability density')
    ax[1, 0].spines['top'].set_visible(False)
    ax[1, 0].spines['right'].set_visible(False)
    ymin, ymax = ax[1, 0].get_ylim()
    ax[1, 0].plot([-1, -1], [ymin, ymax], 'k--')
    ax[1, 0].set_title('MEA #1')


    params.write('data', 'suffix', '')
    params.write('whitening', 'spatial', 'True')
    params = CircusParser(coating[key]['file'])
    params.get_data_file()

    templates = load_data(params, 'templates')
    nb_templates = templates.shape[1]//2
    templates = templates[:,:nb_templates].toarray()
    templates = templates.reshape(len(nodes), 31, nb_templates)
    electrodes = load_data(params, 'electrodes')

    from circus.shared.probes import *
    nodes, positions = get_nodes_and_positions(params)

    ax[0, 0].scatter(positions[inv_nodes[coated_channels], 0], positions[inv_nodes[coated_channels], 1], c='C0')
    ax[0, 0].scatter(positions[inv_nodes[non_coated_channels], 0], positions[inv_nodes[non_coated_channels], 1], c='C1')
    ax[0, 0].spines['top'].set_visible(False)
    ax[0, 0].spines['right'].set_visible(False)
    ax[0, 0].spines['left'].set_visible(False)
    ax[0, 0].spines['bottom'].set_visible(False)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title('MEA layout')
 
    ax[1, 1].spines['top'].set_visible(False)
    ax[1, 1].spines['right'].set_visible(False)
    ax[1, 1].set_title('MEA #1')

    ax[1, 2].spines['top'].set_visible(False)
    ax[1, 2].spines['right'].set_visible(False)
    ax[1, 2].set_title('MEA #1')

    for count, e in enumerate(electrodes):
        if e in inv_nodes[coated_channels]:
            ax[1, 1].plot(templates[e, :, count]/mads[e], c='C0')
        else:
            ax[1, 2].plot(templates[e, :, count]/mads[e], c='C1')

    ax[1, 1].plot([0, 31], [-1, -1], 'k--')
    ax[1, 2].plot([0, 31], [-1, -1], 'k--')
    ymin = min(ax[1, 1].get_ylim()[0], ax[1, 2].get_ylim()[0])
    ymax = max(ax[1, 1].get_ylim()[1], ax[1, 2].get_ylim()[1])
    ax[1, 1].set_ylim(ymin, ymax)
    ax[1, 2].set_ylim(ymin, ymax)
    ax[1, 1].set_xlabel('timesteps')
    ax[1, 2].set_xlabel('timesteps')
    ax[1, 1].set_ylabel('amplitude ($\mu$V)')
    ax[1, 2].set_ylabel('amplitude ($\mu$V)')


results = {'snrs' : {'coated' : [], 'non coated' : []},
           'noise' : {'coated' : [], 'non coated' : []},
           'cells' : {'coated' : [], 'non coated' : []}}

for key in ['mea_1', 'mea_2', 'mea_3']:

    mapping = np.loadtxt(coating[key]['mapping'])
    params = CircusParser(coating[key]['file'])

    params.write('data', 'suffix', '-raw')
    params.write('whitening', 'spatial', 'False')
    
    params = CircusParser(coating[key]['file'])
    params.get_data_file()

    mads = load_data(params, 'mads')
    thresholds = load_data(params, 'thresholds')

    mads *= mcs_factor
    thresholds *= mcs_factor

    coated_channels = []
    for i in mapping[coating[key]['channels'],1]:
       coated_channels += [np.where(mcs_mapping == i)[0]]

    nodes, edges = get_nodes_and_edges(params)

    inv_nodes = np.zeros(60, dtype=np.int32)
    inv_nodes[nodes] = np.arange(len(nodes))

    coated_channels = np.array(coated_channels).flatten()
    non_coated_channels = all_channels[~np.in1d(all_channels, coated_channels)]

    results['noise']['coated'] += [mads[inv_nodes[coated_channels]].mean()]
    results['noise']['non coated'] += [mads[inv_nodes[non_coated_channels]].mean()]

    res = load_data(params, 'mua')
    coated_amplitudes = np.zeros(0, dtype=np.float32)

    for a in inv_nodes[coated_channels]:
        coated_amplitudes = np.concatenate((coated_amplitudes, res['amplitudes']['elec_%d' %a]/thresholds[a]))
        
    coated_amplitudes *= mcs_factor

    non_coated_amplitudes = np.zeros(0, dtype=np.float32)

    for a in inv_nodes[non_coated_channels]:
        non_coated_amplitudes = np.concatenate((non_coated_amplitudes, res['amplitudes']['elec_%d' %a]/thresholds[a]))

    non_coated_amplitudes *= mcs_factor

    coated_snrs = np.zeros(0, dtype=np.float32)

    for a in inv_nodes[coated_channels]:
        coated_snrs = np.concatenate((coated_snrs, [-res['amplitudes']['elec_%d' %a].min()/mads[a]]))
        
    coated_snrs *= mcs_factor

    non_coated_snrs = np.zeros(0, dtype=np.float32)

    for a in inv_nodes[non_coated_channels]:
        non_coated_snrs = np.concatenate((non_coated_snrs, [-res['amplitudes']['elec_%d' %a].min()/mads[a]]))

    non_coated_snrs *= mcs_factor

    results['snrs']['coated'] += [20*np.log10(coated_snrs).mean()]
    results['snrs']['non coated'] += [20*np.log10(non_coated_snrs).mean()]


for key in ['mea_1', 'mea_2', 'mea_3']:

    mapping = np.loadtxt(coating[key]['mapping'])
    params = CircusParser(coating[key]['file'])

    params.write('data', 'suffix', '')
    params.write('whitening', 'spatial', 'True')

    params = CircusParser(coating[key]['file'])
    params.get_data_file()


    templates = load_data(params, 'templates')
    nb_templates = templates.shape[1]//2
    templates = templates[:,:nb_templates].toarray()
    templates = templates.reshape(len(nodes), 31, nb_templates)

    best_elec = np.argmin(templates[:,15,:], axis=0)

    results['cells']['coated'] += [np.in1d(np.argmin(templates[:,15,:], axis=0), inv_nodes[coated_channels]).sum()]
    results['cells']['non coated'] += [np.in1d(np.argmin(templates[:,15,:], axis=0), inv_nodes[non_coated_channels]).sum()]



ax[2, 0].bar([0, 1], [np.mean(results['noise']['coated']), np.mean(results['noise']['non coated'])], 
    yerr=[np.std(results['noise']['coated']), np.std(results['noise']['non coated'])], color=['C0', 'C1'])
ax[2, 0].set_ylabel('Noise level ($\mathrm{\mu}V)$') 
ax[2, 0].spines['top'].set_visible(False)
ax[2, 0].spines['right'].set_visible(False)
ax[2, 0].set_xticks([])

ax[2, 1].bar([0, 1], [np.mean(results['snrs']['coated']), np.mean(results['snrs']['non coated'])], 
    yerr=[np.std(results['snrs']['coated']), np.std(results['snrs']['non coated'])], color=['C0', 'C1'])
ax[2, 1].set_ylabel('SNR (dB)')
ax[2, 1].spines['top'].set_visible(False)
ax[2, 1].spines['right'].set_visible(False)
ax[2, 1].set_xticks([])

ax[2, 2].bar([0, 1], [np.mean(results['cells']['coated']), np.mean(results['cells']['non coated'])], 
    yerr=[np.std(results['cells']['coated']), np.std(results['cells']['non coated'])], color=['C0', 'C1'])
ax[2, 2].set_ylabel('# cells')
ax[2, 2].spines['top'].set_visible(False)
ax[2, 2].spines['right'].set_visible(False)
ax[2, 2].set_xticks([])
