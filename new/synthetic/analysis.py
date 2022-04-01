import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

## Pierre : this is the NEW API
import spikeinterface.full as si
import sklearn.metrics


import MEArec as mr

from generation import generation_params

file = 'rec0_50cells_0noise_0corr_5rate_Neuronexus-32.h5'

rec  = si.MEArecRecordingExtractor(file)
sorting_gt = si.MEArecSortingExtractor(file)

count = 0

gt_dict = {}
gt_dict['rec%d' %count] = (rec, sorting_gt)

#study = si.GroundTruthStudy.create('study', gt_dict, n_jobs=-1, chunk_memory='1G', progress_bar=True)
#study.run_sorters(['spykingcircus'], verbose=False)

mr_rec = mr.load_recordings(file)
cell_positions = mr_rec.template_locations[:, 1:3]

sc_sorting = si.read_sorter_folder('study/sorter_folders/rec0/spykingcircus/')
comp = si.compare_sorter_to_ground_truth(sorting_gt, sc_sorting)

matches = comp.get_well_detected_units()
data = sc_sorting.get_all_spike_trains()
sorting = si.NumpySorting.from_times_labels(data[0][0], data[0][1], sc_sorting.get_sampling_frequency())

from circus.shared.parser import CircusParser
params = CircusParser('study/sorter_folders/rec0/spykingcircus/recording.npy')
from circus.shared.files import load_data


raw_rec = si.BinaryRecordingExtractor('data.raw', 30000, 32, 'float32')
raw_rec.annotate(is_filtered=True)
jobs_kwargs = {'chunk_size': 10000, "progress_bar": True, "n_jobs": -1}
waveforms = si.extract_waveforms(raw_rec, sc_sorting, f'waveforms', ms_before=1.5, 
   ms_after=1.5, load_if_exists=True, precompute_template=('average', 'median'), max_spikes_per_unit=500, return_scaled=False, **jobs_kwargs)
templates_raw = waveforms.get_all_templates(mode='median')

for i in range(len(templates_raw)):
    templates_raw[i] = np.median(waveforms.get_waveforms(i), axis=0)

templates = load_data(params, 'templates', '-merged')
nb_templates = templates.shape[1]//2
templates = templates[:,:nb_templates].toarray()
templates = templates.reshape(32, 91, nb_templates)

import scipy
from circus.shared.probes import *

positions = mr_rec.channel_positions[:, 1:3]


def make_initial_guess_and_bounds(wf_ptp, local_contact_locations, max_distance_um=250):
    # constant for initial guess and bounds
    initial_z = 10

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


coms = np.zeros((2, 0))
monopoles = np.zeros((2, 0))

src_matches = []
matches = np.array(matches)

all_errors = {'coms' : [], 'monopoles' : []}

for cell in matches:
    src_matches += [int(comp.best_match_21[cell].strip('#'))]
src_matches = np.array(src_matches)


for count, idx in enumerate(matches):


    #wf = templates[:,:,idx].T
    #wf_ptp = wf.ptp(axis=0)

    wf = templates_raw[idx]
    wf_ptp = wf.ptp(axis=0)

    thr = np.percentile(wf_ptp, 0)
    active = wf_ptp >= thr

    x0, bounds = make_initial_guess_and_bounds(wf_ptp[active], positions[active,:2], 1000)
    args = (wf_ptp[active], positions[active,:2])
    com = scipy.optimize.least_squares(estimate_distance_error, x0=x0, bounds=bounds, args = args, verbose=True)
    
    monopoles = np.hstack((monopoles, com.x[:2,np.newaxis]))
    
    com = np.sum(wf_ptp[:, np.newaxis] * positions[:,:2], axis=0) / np.sum(wf_ptp)
    coms = np.hstack((coms, com[:, np.newaxis]))




all_errors = {'coms' : np.linalg.norm(coms - cell_positions[src_matches].T, axis=0),
              'monopoles' : np.linalg.norm(monopoles - cell_positions[src_matches].T, axis=0)}


fig, ax = plt.subplots(2, 3)

ax[0, 0].scatter(cell_positions[:,0], cell_positions[:,1])
ax[0, 0].scatter(coms[0], coms[1])
ax[0, 0].scatter(monopoles[0], monopoles[1])
ax[0, 0].scatter(positions[:,0], positions[:,1], c='k', alpha=0.5, s=100)
ax[0, 0].set_xlabel('x (um)')
ax[0, 0].set_xlabel('x (um)')
ax[0, 0].legend(('Original', 'Center of Mass', 'Monopole'))

ax[0, 1].bar([0, 1], [all_errors['coms'].mean(), all_errors['monopoles'].mean()], yerr=[all_errors['coms'].std(), all_errors['monopoles'].std()], color=['C1', 'C2'])
ax[0, 1].set_ylabel('Mean Error (um)')

all_values = []
values = []
all_dist = {'coms' : [], 'monopoles' : [], 'real' : []}
thresholds = load_data(params, 'thresholds', '-merged')

for count in range(len(matches)):
    coms_dist = np.linalg.norm(coms[:, count] - positions[:,:2], axis=1)
    mon_dist = np.linalg.norm(monopoles[:, count] - positions[:,:2], axis=1)
    real_dist = np.linalg.norm(cell_positions[src_matches[count]] - positions[:,:2], axis=1)
    local_values = -templates[:, :, matches[count]].min(1)/thresholds
    idx, = np.where(local_values != 0)
    all_values += local_values[idx].tolist()
    all_dist['coms'] += coms_dist[idx].tolist()
    all_dist['monopoles'] += mon_dist[idx].tolist()
    all_dist['real'] += real_dist[idx].tolist()

ax[1, 0].plot(all_dist['real'], all_values, '.', c='C0')
ax[1, 0].set_xlabel('Distance [um]')
ax[1, 0].set_ylabel('Normalized amplitude')

ax[1, 1].plot(all_dist['coms'], all_values, '.', c='C1')
ax[1, 1].set_xlabel('Distance [um]')
ax[1, 1].set_ylabel('Normalized amplitude')

ax[1, 2].plot(all_dist['monopoles'], all_values, '.', c='C2')
ax[1, 2].set_xlabel('Distance [um]')
ax[1, 2].set_ylabel('Normalized amplitude')


