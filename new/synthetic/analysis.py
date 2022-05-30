import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

## Pierre : this is the NEW API
import spikeinterface.full as si
import sklearn.metrics


import MEArec as mr

from generation import generation_params

file = 'rec0_50cells_5noise_0corr_5rate_Neuronexus-32.h5'

rec  = si.MEArecRecordingExtractor(file)
sorting_gt = si.MEArecSortingExtractor(file)

count = 0

gt_dict = {}
gt_dict['rec%d' %count] = (rec, sorting_gt)

#study = si.GroundTruthStudy.create('study', gt_dict, n_jobs=-1, chunk_memory='1G', progress_bar=True)
#study.run_sorters(['spykingcircus'], verbose=False)

mr_rec = mr.load_recordings(file)
cell_positions = mr_rec.template_locations[:, 1:3]

#sc_sorting = si.read_sorter_folder('study/sorter_folders/rec0/spykingcircus/')
#comp = si.compare_sorter_to_ground_truth(sorting_gt, sc_sorting)

#matches = comp.get_well_detected_units()
#data = sc_sorting.get_all_spike_trains()
#sorting = si.NumpySorting.from_times_labels(data[0][0], data[0][1], sc_sorting.get_sampling_frequency())

#from circus.shared.parser import CircusParser
#params = CircusParser('study/sorter_folders/rec0/spykingcircus/recording.npy')
#from circus.shared.files import load_data


raw_rec = si.MEArecRecordingExtractor(file)
gt_sorting = si.MEArecSortingExtractor(file)
#raw_rec = si.BinaryRecordingExtractor('data.raw', 30000, 32, 'float32')
#raw_rec.annotate(is_filtered=True)
#raw_rec = raw_rec.set_probegroup(rec.get_probegroup())
jobs_kwargs = {'chunk_size': 10000, "progress_bar": True, "n_jobs": -1}
waveforms = si.extract_waveforms(raw_rec, gt_sorting, f'waveforms', ms_before=1.5, 
   ms_after=1.5, load_if_exists=True, precompute_template=('average', 'median'), max_spikes_per_unit=500, return_scaled=False, **jobs_kwargs)
templates_raw = waveforms.get_all_templates(mode='median')

#waveforms_gt = si.extract_waveforms(raw_rec, sorting_gt, f'waveforms_gt', ms_before=1.5, 
#   ms_after=1.5, load_if_exists=True, precompute_template=('average', 'median'), max_spikes_per_unit=500, return_scaled=False, **jobs_kwargs)

#for i in gt_sorting.unit_ids:
#    templates_raw[i] = np.median(waveforms.get_waveforms(i), axis=0)

#templates = load_data(params, 'templates', '-merged')
#nb_templates = templates.shape[1]//2
#templates = templates[:,:nb_templates].toarray()
#templates = templates.reshape(32, 91, nb_templates)

import scipy
from circus.shared.probes import *

positions = mr_rec.channel_positions[:, 1:3]


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


#src_matches = []
#matches = np.array(matches)

#all_errors = {'coms' : [], 'monopoles' : []}

#for cell in matches:
#    src_matches += [int(comp.best_match_21[cell].strip('#'))]
#src_matches = np.array(src_matches)


estimations = {'com' : np.zeros((2, 0)), 'mon' : np.zeros((2, 0))}

for count, idx in enumerate(range(len(templates_raw))):


    #wf = templates[:,:,idx].T
    #wf_ptp = wf.ptp(axis=0)

    
    wf = templates_raw[idx]
    wf_ptp = wf.ptp(axis=0)

    thr = np.percentile(wf_ptp, 0)
    active = wf_ptp >= thr

    x0, bounds = make_initial_guess_and_bounds(wf_ptp[active], positions[active], 1000)
    y0 = np.concatenate((cell_positions[count], [1, 1]))
    args = (wf_ptp[active], positions[active])
    com = scipy.optimize.least_squares(estimate_distance_error, x0=y0, bounds=bounds, args = args, verbose=True)
    
    estimations['mon'] = np.hstack((estimations['mon'], com.x[:2,np.newaxis]))
    
    com = np.sum(wf_ptp[:, np.newaxis] * positions, axis=0) / np.sum(wf_ptp)
    estimations['com'] = np.hstack((estimations['com'], com[:, np.newaxis]))



all_errors = {'com' : np.linalg.norm(estimations['com'] - cell_positions.T, axis=0),
              'mon' : np.linalg.norm(estimations['mon'] - cell_positions.T, axis=0)}


import plotting
layout = '''
            ACE
            BDF
        '''
        
fig = plt.figure(figsize=(20, 10))
axes, spec = plotting.panels(layout, fig=fig, simple_axis=True)
plotting.label_panels(axes)
plt.tight_layout()


#fig, ax = plt.subplots(2, 3)

#si.plot_probe_map(raw_rec, ax=ax[0, 0])
axes['A'].scatter(cell_positions[:,0], cell_positions[:,1], c='k')
#ax['A'].scatter(coms[0], coms[1])
#ax[0, 0].scatter(monopoles[0], monopoles[1])
axes['A'].scatter(positions[:,0], positions[:,1], c='k', alpha=0.5, s=100)
axes['A'].set_xlabel('x (um)')
axes['A'].set_ylabel('y (um)')
#ax[0, 0].legend(('Original', 'Center of Mass', 'Monopole'))

si.plot_unit_templates(waveforms, unit_ids=['#1'], axes=axes['B'])
#si.plot_unit_templates(waveforms, unit_ids=[''], axes=axes['B'])


axes['E'].bar([0, 1], [all_errors['com'].mean(), all_errors['mon'].mean()], yerr=[all_errors['com'].std(), all_errors['mon'].std()], color=['C3', 'C4'])
axes['E'].set_ylabel('Mean Error (um)')


radius = range(50)
data = np.array([np.sum(all_errors['com'] < i) for i in radius])
axes['F'].plot(radius, data, c='C3')
data = np.array([np.sum(all_errors['mon'] < i) for i in radius])
axes['F'].plot(radius, data, c='C4')
axes['F'].set_xlabel('Tolerance radius (um)')
axes['F'].set_ylabel('# matches')

thresholds = si.get_noise_levels(raw_rec)

for letter, key, color in zip(['C', 'D'], ['com', 'mon'], ['C3', 'C4']):

    all_values = []
    all_dist = []

    for count in range(len(templates_raw)):
        real_dist = np.linalg.norm(estimations[key][:,count] - positions[:,:2], axis=1)
        local_values = -templates_raw[count, :, :].min(0)/thresholds
        idx, = np.where(local_values != 0)
        all_values += local_values[idx].tolist()
        all_dist += real_dist[idx].tolist()

    axes[letter].plot(all_dist, all_values, '.', c=color)
    axes[letter].set_xlabel('Distance [um]')
    axes[letter].set_ylabel('Normalized amplitude')


