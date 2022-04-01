import numpy as np
import h5py
import pylab as plt

coating = {'mea_1' : {'channels' : [],
                      'mapping' : 'mapping_mea1.txt',
                      'file' : 'MEA1.h5'},
           'mea_4' : {'channels' : [],
            'mapping' : 'mapping_mea4.txt', 
            'file' : 'MEA4.h5'}
            }

mcs_mapping = h5py.File('MEA1.h5')['Data/Recording_0/AnalogStream/Stream_2/InfoChannel']['Label'].astype('int')
mcs_factor = h5py.File('MEA1.h5')['Data/Recording_0/AnalogStream/Stream_2/InfoChannel']['ConversionFactor'][0] * 1e-6

all_channels = np.delete(np.arange(60), [14, 44])

from circus.shared.parser import CircusParser
from circus.shared.files import load_data
from circus.shared.probes import get_nodes_and_edges


save_pdf = True
use_monopole = False

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


images = {'0' : {'filename' : 'SEM neurons+Flou/0.png', 'channels' : list(range(0, 14)) + list(range(15, 44)) + list(range(45, 60))},
          '1' : {'filename' : 'SEM neurons+Flou/1.png', 'channels' : np.array([29, 32, 34,27,30,33,36,39,37,31,25,42,40,35,28,18])},  
          '2' : {'filename' : 'SEM neurons+Flou/2.png', 'channels' : np.array([24,26,29,22,23,27,30,19,20,21,25,16,17,15,18])},
          '3' : {'filename' : 'SEM neurons+Flou/3.png', 'channels' : np.array([13, 11, 8,12,10,5,58,9,7,1,55,6,3,0])},
          '4' : {'filename' : 'SEM neurons+Flou/4.png', 'channels' : np.array([6,3,0,57,53,4,2,59,56,54])},
          '5' : {'filename' : 'SEM neurons+Flou/5.png', 'channels' : np.array([4,6,3])},
          '6' : {'filename' : 'SEM neurons+Flou/6.png', 'channels' : np.array([38,41,43,48,45,47,46,55,51,50,49,57,53,52])}
          }

mappings = {'0' : {'image' : [(298, 61), (601, 36), (348, 589)], 'channels' : [(243, 98), (580, 98), (243, 510)], 'ids' : [24, 4, 54]},
            '1' : {'image' : [(78, 138), (430,111), (205, 587)], 'channels' : [(200, 136), (481, 133), (270, 497)], 'ids' : [24, 4, 54]},
            '2' : {'image' : [(385, 112), (711, 88), (503, 529)], 'channels' : [(340, 128), (622, 127), (416, 479)], 'ids' : [24, 4, 54]},
            '3' : {'image' : [(200, 93), (589, 57), (422, 416)], 'channels' : [(213, 130), (529, 130), (372, 361)], 'ids' : [24, 4, 54]},
            '4' : {'image' : [(187, 120), (680, 75), (278, 258)], 'channels' : [(252, 190), (633, 190), (320, 420)], 'ids' : [24, 4, 54]},
            '5' : {'image' : [(140, 128), (550, 96), (372, 474)], 'channels' : [(192, 190), (632, 190), (412, 420)], 'ids' : [24, 4, 54]},
            '6' : {'image' : [(159, 72), (540, 36), (377, 383)], 'channels' : [(291, 128), (615, 132), (453, 361)], 'ids' : [24, 4, 54]}}

compute_positions = True

for key in ['mea_1']:

    mapping = np.loadtxt(coating[key]['mapping'])
    params = CircusParser(coating[key]['file'])

    params.write('data', 'suffix', '')
    params.write('whitening', 'spatial', 'True')

    params = CircusParser(coating[key]['file'])
    params.get_data_file()


    mads = load_data(params, 'mads', '-merged')
    thresholds = load_data(params, 'thresholds', '-merged')
    purity = load_data(params, 'purity', '-merged')

    coated_channels = []
    for i in mapping[coating[key]['channels'],1]:
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
    
    import scipy
    from circus.shared.probes import *
    nodes, positions = get_nodes_and_positions(params)

    positions[:,0] *= 100/40.

    def estimate_distance_error(vec, wf_ptp, local_contact_locations):
        # vec dims ar (x, y, z amplitude_factor)
        # given that for contact_location x=dim0 + z=dim1 and y is orthogonal to probe
        dist = np.sqrt(((local_contact_locations - vec[np.newaxis, :2])**2).sum(axis=1) + vec[2]**2)
        ptp_estimated = vec[3] / dist
        err = wf_ptp - ptp_estimated
        return err

    if compute_positions:
        coms = np.zeros((2, 0))
        for idx in range(templates.shape[2]):
            wf = templates[:,:,idx].T
            wf_ptp = wf.ptp(axis=0)

            if use_monopole:
                x0, bounds = make_initial_guess_and_bounds(wf_ptp, positions[:,:2], 1000)
                args = (wf_ptp, positions[:,:2])
                com = scipy.optimize.least_squares(estimate_distance_error, x0=x0, bounds=bounds, args = args)
                coms = np.hstack((coms, com.x[:2,np.newaxis]))
            else:
                com = np.sum(wf_ptp[:, np.newaxis] * positions[:,:2], axis=0) / np.sum(wf_ptp)
                coms = np.hstack((coms, com[:, np.newaxis]))
            

        if use_monopole:
            np.save('coms_monopole_%s' %key, coms)
        else:
            np.save('coms_%s' %key, coms)
            
    else:
        if use_monopole:
            coms = np.load('coms_monopole_%s.npy' %key)
        else:
            coms = np.load('coms_%s.npy' %key)


    
    fig, ax = plt.subplots(2, 2)

    ax[0,0].scatter(coms[0], coms[1])
    ax[0,0].scatter(positions[:,0],positions[:,1],s=100,alpha=0.5)
    for i in range(len(coms[0])):
        ax[0,0].text(coms[0,i], coms[1,i], '%d' %i)
    
    values = np.zeros((2, len(coms[0])))
    all_values = []
    all_energies = []
    all_dist = []
    for i in range(len(coms[0])):
        dist = np.linalg.norm(coms[:,i] - positions[:,:2], axis=1)
        min_dist = np.argmin(dist)
        values[:, i] = [dist.min(), -templates[min_dist, :, i].min()/thresholds[min_dist]]        
        local_values = -templates[:, :, i].min(1)/thresholds
        idx, = np.where(local_values != 0)
        all_values += local_values[idx].tolist()
        all_dist += dist[idx].tolist()
        all_energies += np.linalg.norm(templates[:,:,i], axis=1)[idx].tolist()

    ax[0,1].plot(values[0], values[1], '.')
    ax[0,1].plot([values[0].min(), values[0].max()], [-1, -1], 'k--')
    ax[0,1].set_xlabel('Distance [um]')
    ax[0,1].set_ylabel('Normalized amplitude')

    ax[1,0].plot(all_dist, all_values, '.')
    ax[1,0].set_xlabel('Distance [um]')
    ax[1,0].set_ylabel('Normalized amplitude')

    ax[1,1].plot(all_dist, all_energies, '.')
    ax[1,1].set_xlabel('Distance [um]')
    ax[1,1].set_ylabel('Energy')

    if use_monopole:
        plt.savefig('stats_%s_monopole.pdf' %key)
    else:
        plt.savefig('stats_%s.pdf' %key)

    for key, value in images.items():

        import skimage
        SEM = plt.imread(value['filename'])
        fig, ax = plt.subplots(1, 1, figsize=(SEM.shape[1]/96, SEM.shape[0]/96))

        x_min = positions[inv_nodes[value['channels']], 0].min()
        x_max = positions[inv_nodes[value['channels']], 0].max()
        y_min = positions[inv_nodes[value['channels']], 1].min()
        y_max = positions[inv_nodes[value['channels']], 1].max()

        new_positions = positions.copy()
        new_positions[:, 0] -= x_min
        new_positions[:, 1] -= y_min 
        new_coms = coms.copy()   
        new_coms[0] -= x_min
        new_coms[1] -= y_min


        #ax.scatter(new_positions[inv_nodes[value['channels']], 0], new_positions[inv_nodes[value['channels']], 1], c='C1', s=800, alpha=1)
        ax.scatter(new_coms[0], new_coms[1], c='k', alpha=0.5, s=50)
        for i in range(len(coms[0])):
            plt.text(new_coms[0,i], new_coms[1,i], '%d' %i)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        x_min_2, x_max_2 = ax.get_xlim()
        y_min_2, y_max_2 = ax.get_ylim()
        print(x_min_2, x_max_2)
        print(y_min_2, y_max_2)

        xmin, xmax = new_positions[inv_nodes[value['channels']], 0].min(), new_positions[inv_nodes[value['channels']], 0].max()
        ymin, ymax = new_positions[inv_nodes[value['channels']], 1].min(), new_positions[inv_nodes[value['channels']], 1].max()

        ax.set_xlim(xmin-20, xmax+20)
        ax.set_ylim(ymin-20, ymax+20)        

        plt.savefig('channels_%s.png' %key, dpi=96)
        plt.close()


        fig, ax = plt.subplots(1, 1, figsize=(SEM.shape[1]/96, SEM.shape[0]/96))
        I = plt.imread(value['filename'].replace('tif', 'gif')).astype(np.float32)[:,:,1]


        import skimage.transform
        #import skimage.restoration
        #I2 = skimage.restoration.denoise_nl_means(I)

        J = plt.imread('channels_%s.png' %key).astype(np.float32)[:,:,1]

        src = np.array(mappings[key]['image'])
        dst = np.array(mappings[key]['channels'])

        trans = skimage.transform.estimate_transform('affine', dst, src)
        I2 = skimage.transform.warp(SEM, trans)

        #ax.scatter(new_positions[value['channels'], 0], new_positions[value['channels'], 1], c='C1', s=800, alpha=1)
        ax.imshow(J)


        ax.imshow(I2, alpha=0.75)

        H = 1 - J
        H[H < 1] = 0

        ax.imshow(H, cmap='Reds')
        ax.imshow(I2, alpha=0.85, cmap='viridis')

        #ax.set_xticks([])
        #ax.set_yticks([])
        #ax.set_title('MEA layout')
        #ax.set_ylim(-370,-270)
        if use_monopole:
            plt.savefig('registration_%s_monopole.jpg' %key)
        else:
            plt.savefig('registration_%s.jpg' %key)
        plt.show()