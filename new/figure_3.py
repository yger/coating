import numpy as np
import h5py
import pylab as plt

coating = {'mea_1' : {'channels' : [],
                      'mapping' : 'DIV21/mapping_mea5.txt',
                      'file' : 'DIV21/MEA5.h5'},
            'mea_2' : {'channels' : [],
                      'mapping' : 'DIV14/mapping_mea1.txt',
                      'file' : 'DIV14/MEA1.h5'},
            'mea_3' : {'channels' : [],
                        'mapping' : 'DIV14/mapping_mea4.txt', 
                        'file' : 'DIV14/MEA4.h5'},
            'mea_4' : {'channels' : [],
                      'mapping' : 'DIV25/mapping_mea1.txt',
                      'file' : 'DIV25/MEA1.h5'}
        }

mcs_mapping = h5py.File('DIV21/MEA5.h5')['Data/Recording_0/AnalogStream/Stream_2/InfoChannel']['Label'].astype('int')
mcs_factor = h5py.File('DIV21/MEA5.h5')['Data/Recording_0/AnalogStream/Stream_2/InfoChannel']['ConversionFactor'][0] * 1e-6

all_channels = np.delete(np.arange(60), [14, 44])

from circus.shared.parser import CircusParser
from circus.shared.files import load_data
from circus.shared.probes import get_nodes_and_edges

def make_initial_guess_and_bounds(wf_ptp, local_contact_locations, max_distance_um=250):
    # constant for initial guess and bounds
    initial_z = 1

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

def estimate_distance_error(vec, wf_ptp, local_contact_locations):
    # vec dims ar (x, y, z amplitude_factor)
    # given that for contact_location x=dim0 + z=dim1 and y is orthogonal to probe
    dist = np.sqrt(((local_contact_locations - vec[np.newaxis, :2])**2).sum(axis=1) + vec[2]**2)
    ptp_estimated = vec[3] / dist
    err = wf_ptp - ptp_estimated
    return err


import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

## Pierre : this is the NEW API
import spikeinterface.full as si
import sklearn.metrics

import plotting
import MEArec as mr

from generation import generation_params

result = {}

for mea_key in ['mea_1', 'mea_2', 'mea_3', 'mea_4']:

    mapping = np.loadtxt(coating[mea_key]['mapping'])
    params = CircusParser(coating[mea_key]['file'])

    params.write('data', 'suffix', '')
    params.write('whitening', 'spatial', 'True')

    params = CircusParser(coating[mea_key]['file'])
    params.get_data_file()

    mads = load_data(params, 'mads', '-merged')
    thresholds = load_data(params, 'thresholds', '-merged')
    purity = load_data(params, 'purity', '-merged')

    selection,  = np.where(purity > 0)

    coated_channels = []
    for i in mapping[coating[mea_key]['channels'],1]:
       coated_channels += [np.where(mcs_mapping == i)[0]]

    nodes, edges = get_nodes_and_edges(params)

    inv_nodes = np.zeros(60, dtype=np.int32)
    inv_nodes[nodes] = np.arange(len(nodes))

    coated_channels = np.array(coated_channels).flatten()
    non_coated_channels = all_channels[~np.in1d(all_channels, coated_channels)]

    electrodes = load_data(params, 'electrodes', '-merged')

    templates = load_data(params, 'templates', '-merged')
    nb_templates = templates.shape[1]//2
    templates = templates[:,:nb_templates].toarray()
    templates = templates.reshape(len(nodes), 31, nb_templates)
    
    import spikeinterface.full as si
    import scipy
    from circus.shared.probes import *
    nodes, positions = get_nodes_and_positions(params)

    positions[:,0] *= 100/40.

    result[mea_key] = {}
    result[mea_key]['coms'] = {'com' : np.zeros((2, 0)), 'mon' : np.zeros((2, 0))}
    
    for idx in selection:

        wf = templates[:,:,idx].T
        wf_ptp = wf.ptp(axis=0)

        #wf = templates_raw[idx]
        #wf_ptp = wf.ptp(axis=0)

        x0, bounds = make_initial_guess_and_bounds(wf_ptp, positions[:,:2], 1000)
        args = (wf_ptp, positions[:,:2])
        com = scipy.optimize.least_squares(estimate_distance_error, x0=x0, bounds=bounds, args = args)
        result[mea_key]['coms']['mon'] = np.hstack((result[mea_key]['coms']['mon'], com.x[:2,np.newaxis]))
            
        com = np.sum(wf_ptp[:, np.newaxis] * positions[:,:2], axis=0) / np.sum(wf_ptp)
        result[mea_key]['coms']['com'] = np.hstack((result[mea_key]['coms']['com'], com[:, np.newaxis]))
                        
    count = 0
    result[mea_key]['all_values'] = {}
    result[mea_key]['all_dist'] = {}

    for key in ['com', 'mon']:  
        nb_cells = len(result[mea_key]['coms'][key][0])      
        values = np.zeros((2, nb_cells))
        all_values = []
        all_energies = []
        all_dist = []
        for i in range(len(result[mea_key]['coms'][key][0])):
            dist = np.linalg.norm(result[mea_key]['coms'][key][:,i] - positions[:,:2], axis=1)
            min_dist = np.argmin(dist)
            values[:, i] = [dist.min(), -templates[min_dist, :, i].min()/thresholds[min_dist]]        
            local_values = -templates[:, :, i].min(1)/thresholds
            idx, = np.where(local_values != 0)
            all_values += local_values[idx].tolist()
            all_dist += dist[idx].tolist()

        result[mea_key]['all_values'][key] = np.array(all_values)
        result[mea_key]['all_dist'][key] = np.array(all_dist)

def plot_curve(data, bins, ax, key, label=None, **kwargs):
    means = []
    stds = []
    xaxis = []
    for i in range(len(bins) - 1):
        idx = np.where((data['all_dist'][key] > bins[i]) & (data['all_dist'][key] < bins[i+1]))[0]
        means += [data['all_values'][key][idx].mean()]
        stds += [data['all_values'][key][idx].std()]
        xaxis += [bins[i] + (bins[i+1] - bins[i])/2]
    xaxis = np.array(xaxis)
    means = np.array(means)
    stds = np.array(stds)
    print(xaxis)
    ax.plot(xaxis, means, lw=2, label=label, **kwargs)
    ax.fill_between(xaxis, means-stds, means+stds, alpha=0.5, **kwargs)
    return means, stds


layout = '''
            ACE
            BDF
        '''
        
fig = plt.figure(figsize=(20, 10))
axes, spec = plotting.panels(layout, fig=fig, simple_axis=True)
plotting.label_panels(axes)
plt.tight_layout()

for letter, key, color in zip(['A', 'B'], ['com', 'mon'], ['C3', 'C4']):
    nb_cells = len(result['mea_1']['coms'][key][0])
    axes[letter].scatter(result['mea_1']['coms'][key][0], result['mea_1']['coms'][key][1], c=color)
    axes[letter].scatter(positions[:,0], positions[:,1], s=100, alpha=0.5, c='k')
    axes[letter].spines['top'].set_visible(False)
    axes[letter].spines['right'].set_visible(False)
    axes[letter].spines['left'].set_visible(False)
    axes[letter].spines['bottom'].set_visible(False)
    axes[letter].set_xticks([],[])
    axes[letter].set_yticks([],[])
    #for i in range(len(result['mea_1']['coms'][key][0])):
    #    axes[letter].text(result['mea_1']['coms'][key][0,i], result['mea_1']['coms'][key][1,i], '%d' %i)

for letter, key, color in zip(['C', 'D'], ['com', 'mon'], ['C3', 'C4']):
    axes[letter].plot(result['mea_1']['all_dist'][key], result['mea_1']['all_values'][key], '.', c=color)
    axes[letter].set_ylim(-0.5, 10)
    xmin, xmax = axes[letter].get_xlim()
    axes[letter].plot([xmin, xmax], [0, 0], '0.5')
    if letter == 'D':
        axes[letter].set_xlabel('Distance [um]')
    axes[letter].set_ylabel('Normalized amplitude')

for i, mea_key in enumerate(['mea_1', 'mea_2', 'mea_3', 'mea_4']):
    plot_curve(result[mea_key], np.arange(0, 300, 25), axes['E'], key='com', label=mea_key, color='C%d' %i)
    xmin, xmax = axes[letter].get_xlim()
    axes['E'].plot([0, 300], [0, 0], '0.5')
    axes['E'].set_ylim(-0.5, 10)
    axes['E'].legend()
    #axes[letter].set_xlabel('Distance [um]')

for i, mea_key in enumerate(['mea_1', 'mea_2', 'mea_3', 'mea_4']):
    plot_curve(result[mea_key], np.arange(0, 300, 25), axes['F'], key='mon', label=mea_key, color='C%d' %i)
    axes['F'].set_xlabel('Distance [um]')
    xmin, xmax = axes[letter].get_xlim()
    axes['F'].plot([0, 300], [0, 0], '0.5')
    axes['F'].set_ylim(-0.5, 10)