import numpy as np
import h5py
import pylab as plt

coating = {'mea_1' : {'channels' : [],
                      'mapping' : 'DIV21/mapping_mea5.txt',
                      'file' : 'DIV21/MEA5.h5',
                      'image' : {'filename' : 'DIV21/SEM/Flou.jpg', 'channels' : list(range(0, 14)) + list(range(15, 44)) + list(range(45, 60))},
                      'coordinates' : {'image' : [(552, 223), (1317, 223), (935, 453)], 'channels' : [(555, 211), (1213, 180), (900, 480)], 'ids' : [24, 4, 54]}
                      },
            'mea_2' : {'channels' : [],
                      'mapping' : 'DIV14/mapping_mea1.txt',
                      'file' : 'DIV14/MEA1.h5',
                      'image' : {'filename' : 'DIV14/SEM/Flou.jpg', 'channels' : list(range(0, 14)) + list(range(15, 44)) + list(range(45, 60))},
                      'coordinates' : {'image' : [(552, 223), (1317, 223), (935, 453)], 'channels' : [(650, 300), (1308, 238), (1001, 557)], 'ids' : [24, 4, 54]}
                      },
            'mea_3' : {'channels' : [],
                        'mapping' : 'DIV14/mapping_mea4.txt', 
                        'file' : 'DIV14/MEA4.h5',
                        'image' : {'filename' : 'DIV14/SEM/Flou4.jpg', 'channels' : list(range(0, 14)) + list(range(15, 44)) + list(range(45, 60))},
                        'coordinates' : {'image' : [(552, 223), (1317, 223), (935, 453)], 'channels' : [(596, 102), (1254, 28), (959, 350)], 'ids' : [24, 4, 54]}
                      },
            'mea_4' : {'channels' : [],
                      'mapping' : 'DIV25/mapping_mea1.txt',
                      'file' : 'DIV25/MEA1.h5',
                       'image' : {'filename' : 'DIV25/SEM/Flou.jpg', 'channels' : list(range(0, 14)) + list(range(15, 44)) + list(range(45, 60))},
                       'coordinates' : {'image' : [(552, 223), (1317, 223), (935, 453)], 'channels' : [(424, 97), (1084, 26), (786, 346)], 'ids' : [24, 4, 54]}
                      }
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


    import findpeaks

    import skimage
    from PIL import Image
    SEM = Image.open(coating[mea_key]['image']['filename'])
    SEM = np.array(SEM.convert('L'))
    #plt.imread(coating[mea_key]['image']['filename'])
    fig, ax = plt.subplots(1, 1, figsize=(SEM.shape[1]/96, SEM.shape[0]/96))

    inv_channels = inv_nodes[np.array(coating[mea_key]['image']['channels'])]

    x_min = positions[inv_channels, 0].min()
    x_max = positions[inv_channels, 0].max()
    y_min = positions[inv_channels, 1].min()
    y_max = positions[inv_channels, 1].max()

    new_positions = positions.copy()
    new_positions[:, 0] -= x_min
    new_positions[:, 1] -= y_min

    import skimage.transform

    CHANNELS = Image.open(coating[mea_key]['image']['filename'])
    CHANNELS = np.array(CHANNELS.convert('L'))

    dst = np.array(coating[mea_key]['coordinates']['image'])
    src = np.array(coating[mea_key]['coordinates']['channels'])
    trans = skimage.transform.estimate_transform('affine', dst, src)
    I2 = skimage.transform.warp(SEM, trans)

    #ax.scatter(new_positions[value['channels'], 0], new_positions[value['channels'], 1], c='C1', s=800, alpha=1)
    #ax.imshow(1-J, cmap='Reds')
    #ax.imshow(I2, alpha=0.85, cmap='viridis')
    #ax.set_xlim(0, SEM.shape[1])
    #ax.set_ylim(SEM.shape[0], 0)
    I3 = I2 / I2.max()

    ### Segmentation
    from skimage.restoration import denoise_nl_means, estimate_sigma
    sigma_est = np.mean(estimate_sigma(I3))
    I4 = denoise_nl_means(I3, h=0.8*sigma_est, fast_mode=True)
    cells = I4 > 0.5
    
    from scipy import ndimage as ndi
    from skimage import color, feature, filters, measure, morphology, segmentation, util

    distance = ndi.distance_transform_edt(cells)
    local_max_coords = feature.peak_local_max(distance, min_distance=20)

    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)

    segmented_cells = segmentation.watershed(-distance, markers, mask=cells)

    for i in range(1, len(local_max_coords) + 1):
        x, y = np.where(segmented_cells == i)
        if len(x) < 10:
            segmented_cells[x, y] = 0

    cell_mask = segmented_cells.copy()
    cell_mask[cell_mask > 0] =  1

    b = findpeaks.findpeaks().fit(cell_mask)
    XB = np.array([i[0] for i in b['groups0']])

    fig, ax = plt.subplots(1, 1, figsize=(SEM.shape[1]/96, SEM.shape[0]/96))
    ax.scatter(new_positions[inv_channels, 0], new_positions[inv_channels, 1], c='C1', s=800)
    xmin, xmax = new_positions[inv_channels, 0].min(), new_positions[inv_channels, 0].max()
    ymin, ymax = new_positions[inv_channels, 1].min(), new_positions[inv_channels, 1].max()
    ax.set_xlim(xmin-20, xmax+20)
    ax.set_ylim(ymin-20, ymax+20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.savefig('tmp_channels.png', dpi=96)
    plt.close()

    found_positions = Image.open('tmp_channels.png')
    found_positions = np.array(found_positions.convert('L'))
    found_positions = 255 - found_positions
    found_positions[found_positions > 0] = 1
    plt.imshow(I2)
    plt.imshow(found_positions, alpha=0.5)
    plt.savefig('%s.png' %mea_key)
    plt.close()

    result[mea_key]['images'] = {'cells' : found_positions, 'mask' : cell_mask, 'tissue': I4, 'mon' : None, 'com' : None} 
    result[mea_key]['distances'] = {'com' : [], 'mon' : [], 'control' : []}

    XC = XB.copy()

    nb_random = 1000

    all_distances = []
    for src in range(nb_random):
        x = x_min + (x_max-x_min)*np.random.uniform()
        y = y_min + (y_max-y_min)*np.random.uniform()

        distances = np.linalg.norm(np.array([x, y]) - XC, axis=1)
        idx = np.argmin(distances)
        all_distances += [distances[idx]]
        #XC = np.delete(XC, idx, axis=0)

    result[mea_key]['distances']['control'] = all_distances

    for key in ['com', 'mon']:

        new_coms = result[mea_key]['coms'][key].copy()   
        new_coms[0] -= x_min
        new_coms[1] -= y_min

        #ax.scatter(new_positions[inv_channels, 0], new_positions[inv_channels, 1], c='C1', s=800)
        fig, ax = plt.subplots(1, 1, figsize=(SEM.shape[1]/96, SEM.shape[0]/96))
        ax.scatter(new_coms[0], new_coms[1], c='k', alpha=0.5, s=2000)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        x_min_2, x_max_2 = ax.get_xlim()
        y_min_2, y_max_2 = ax.get_ylim()

        xmin, xmax = new_positions[inv_channels, 0].min(), new_positions[inv_channels, 0].max()
        ymin, ymax = new_positions[inv_channels, 1].min(), new_positions[inv_channels, 1].max()

        ax.set_xlim(xmin-20, xmax+20)
        ax.set_ylim(ymin-20, ymax+20)

        plt.savefig('tmp_channels.png', dpi=96)
        plt.close()

        found_positions = Image.open('tmp_channels.png')
        found_positions = np.array(found_positions.convert('L'))

        result[mea_key]['images'][key] = 255 - found_positions

        found_positions = 255 - found_positions
        found_positions[found_positions > 0] = 1

        a = findpeaks.findpeaks().fit(found_positions)

        import scipy.spatial    
        XA = np.array([i[0] for i in a['groups0']])
        XC = XB.copy()

        all_distances = []
        for src in range(len(XA)):
            distances = np.linalg.norm(XA[src] - XC, axis=1)
            idx = np.argmin(distances)
            all_distances += [distances[idx]]
            XC = np.delete(XC, idx, axis=0)

        all_distances = np.array(all_distances)

        result[mea_key]['distances'][key] = all_distances


import plotting

layout = '''
            ABC
            ABC
            DEF
            DEG
        '''

fig = plt.figure(figsize=(20, 10))
axes, spec = plotting.panels(layout, fig=fig, simple_axis=True)
plotting.label_panels(axes)
plt.tight_layout()

axes['A'].imshow(255-result['mea_1']['images']['mon'], cmap='gray')
axes['A'].imshow(result['mea_1']['images']['cells'], alpha=0.5, cmap='Reds')
axes['B'].imshow(result['mea_1']['images']['tissue'], cmap='gray')
axes['C'].imshow(255-result['mea_1']['images']['mask'], cmap='gray')
for letter in ['A', 'B', 'C']:
    axes[letter].set_xticks([], [])
    axes[letter].set_yticks([], [])


# for letter, position in zip(['D', 'E'], [0, 1]):
#     count = 3
#     for key in ['com', 'mon', 'control']:
#         data = np.zeros(0)
#         for mea_key in result.keys():
#             data = np.concatenate((data, result[mea_key]['coms'][key][position]))
#             print(len(data), key, mea_key)
#         if position == 0:
#             center = (x_max - x_min) / 2 
#             bins = np.linspace(0, center, 10)
#             axes[letter].set_xlabel('x (um)')
#         elif position == 1:
#             center = (y_max - y_min) / 2 
#             bins = np.linspace(0, 2*center, 10)
#             axes[letter].set_xlabel('y (um)')
#         x, y = np.histogram(data, bins=bins)
#         axes[letter].plot(y[1:], x/float(x.sum()), 'C%d' %count, label=key)
#         count += 1
#         axes[letter].set_ylabel('Probability')
            

count = 3
for letter, key in zip(['D', 'E'], ['com', 'mon']):
    for mea_key in result.keys():
        x, y = np.histogram(result[mea_key]['distances'][key], bins=np.arange(0, 200, 20))
        if mea_key == 'mea_1':
            if key == 'com':
                label = 'Centre of Mass'
            elif key == 'mon':
                label = 'Monopole'
            axes[letter].plot(y[1:], x/float(x.sum()), 'C%d' %count, label=label)
        else:
            axes[letter].plot(y[1:], x/float(x.sum()), 'C%d' %count)

    x, y = np.histogram(result[mea_key]['distances']['control'], bins=np.arange(0, 200, 20))
    axes[letter].plot(y[1:], x/float(x.sum()), 'k', label='control')
    count += 1
    axes[letter].set_xlabel('Distance (um)')
    axes[letter].set_ylabel('Probability')
    axes[letter].legend()

count = 0

pooled = {'com' : np.zeros(0), 'mon' : np.zeros(0), 'control' : np.zeros(0)}
color = 'C3'

for mea_key in result.keys():
    print(count)
    for key in ['com', 'mon']:
        data = np.array(result[mea_key]['distances'][key])
        axes['F'].bar([count], [data.mean()], yerr=data.std()/np.sqrt(len(data)), color=color)
        if color == 'C3':
            color = 'C4'
        elif color == 'C4':
            color = 'C3'
        count += 1
        pooled[key] = np.concatenate((pooled[key], result[mea_key]['distances'][key]))
    count += 1

axes['F'].bar([count + 3], [pooled['com'].mean()], yerr=pooled['com'].std()/np.sqrt(len(pooled['com'])), color='C3')
axes['F'].bar([count + 4], [pooled['mon'].mean()], yerr=pooled['mon'].std()/np.sqrt(len(pooled['mon'])), color='C4')
axes['F'].set_xticks([], [])
count = 0
color = 'C3'

pooled = {'com' : np.zeros(0), 'mon' : np.zeros(0), 'control' : np.zeros(0)}
cutt_offs = {'com' : 100, 'mon' : 50, 'control' : np.inf}

for mea_key in result.keys():
    print(count)
    for key in ['com', 'mon']:
        cut_off = cutt_offs[key]
        data = np.array(result[mea_key]['distances'][key])
        idx = np.where(data < cut_off)[0]
        axes['G'].bar([count], [data[idx].mean()], yerr=data[idx].std()/np.sqrt(len(idx)), color=color)
        if color == 'C3':
            color = 'C4'
        elif color == 'C4':
            color = 'C3'
        count += 1
        pooled[key] = np.concatenate((pooled[key], data[idx]))
    count += 1

axes['G'].bar([count + 3], [pooled['com'].mean()], yerr=pooled['com'].std()/np.sqrt(len(pooled['com'])), color='C3')
axes['G'].bar([count + 4], [pooled['mon'].mean()], yerr=pooled['mon'].std()/np.sqrt(len(pooled['mon'])), color='C4')
axes['G'].set_xticks([0.5, 3.5, 6.5, 9.5, 15.5], ['mea #1', 'mea #2', 'mea #3', 'mea #4', 'pooled'])