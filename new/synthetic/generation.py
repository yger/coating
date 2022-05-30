from pathlib import Path
import os
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import MEArec as mr
import neo
import numpy
import quantities as pq

# NEW API unique package
#import spikeinterface.full as si
from spikeinterface.comparison.comparisontools import make_matching_events

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def removeaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.get_yaxis().tick_right()
    ax.get_xaxis().tick_top()


generation_params = {
'probe' : 'Neuronexus-32',
'duration' : 5*60,
'n_cell' : 20,
'fs' : 30000.,
'lag_time' : 0.002,  # 2ms
'make_plots' : True,
'generate_recording' : True,
'noise_level' : 0,
'templates_seed' : 42,
'noise_seed' : 42,
'global_path' : os.path.abspath('../'),
'study_number' : 0,
'save_plots' : True,
'method' : 'brette',
'corr_level' : 0,
'rate_level' : 5, #Hz
'nb_recordings' : 5
}


def generate_recordings(params=generation_params):
    for i in range(params['nb_recordings']):
        generation_params['study_number'] = i
        generation_params['templates_seed'] = i
        generation_params['noise_seed'] = i
        generate_single_recording(generation_params)


def generate_single_recording(params=generation_params):

    paths = {}
    paths['basedir'] = params['global_path']
    paths['data'] = None

    if paths['data'] == None:
        paths['data'] = os.path.join(paths['basedir'], 'data')

    paths['templates'] =  os.path.join(paths['data'], 'templates')
    paths['recordings'] = os.path.join(paths['data'], 'recordings') 

    for i in paths.values():
        if not os.path.exists(i):
            os.makedirs(i)

    probe = params['probe']
    n_cell = params['n_cell']
    noise_level = params['noise_level']
    study_number = params['study_number']
    corr_level = params['corr_level']
    rate_level = params['rate_level']

    template_filename = os.path.join(paths['templates'], f'templates_{probe}_100.h5')
    recording_filename = os.path.join(paths['recordings'], f'rec{study_number}_{n_cell}cells_{noise_level}noise_{corr_level}corr_{rate_level}rate_{probe}.h5')
    plot_filename = os.path.join(paths['recordings'], f'rec{study_number}_{n_cell}cells_{noise_level}noise_{corr_level}corr_{rate_level}rate_{probe}.pdf')

    spikerate = params['rate_level']
    n_spike_alone = int(spikerate * params['duration'])

    print('Total target rate:', params['rate_level'], "Hz")
    print('Basal rate:', spikerate, "Hz")


    # collision lag range
    lag_sample = int(params['lag_time'] * params['fs'])

    refactory_period = 2 * params['lag_time']

    spiketimes = []

    if params['method'] == 'poisson':
        print('Spike trains generated as independent poisson sources')
        
        for i in range(params['n_cell']):
            
            #~ n = n_spike_alone + n_collision_by_pair * (params['n_cell'] - i - 1)
            n = n_spike_alone
            #~ times = np.random.rand(n_spike_alone) * params['duration']
            times = np.random.rand(n) * params['duration']
            
            times = np.sort(times)
            spiketimes.append(times)

    elif params['method'] == 'brette':
        import corr_spike_trains
        print('Spike trains generated as compound mixtures')
        C = np.ones((params['n_cell'], params['n_cell']))
        C = params['corr_level'] * np.maximum(C, C.T)
        #np.fill_diagonal(C, 0*np.ones(params['n_cell']))

        rates = rates = params['rate_level']*np.ones(params['n_cell'])

        cor_spk = corr_spike_trains.correlated_spikes(C, rates, params['n_cell'])
        cor_spk.find_mixture(iter=1e4)
        res = cor_spk.mixture_process(tauc=refactory_period/2, t=params['duration'])
        
        # make neo spiketrains
        for i in range(params['n_cell']):
            #~ print(spiketimes[i])
            mask = res[:, 0] == i
            times = res[mask, 1]
            times = np.sort(times)
            mask = (times > 0) * (times < params['duration'])
            times = times[mask]
            spiketimes.append(times)


    # remove refactory period
    for i in range(params['n_cell']):
        times = spiketimes[i]
        ind, = np.nonzero(np.diff(times) < refactory_period)
        ind += 1
        times = np.delete(times, ind)
        assert np.sum(np.diff(times) < refactory_period) ==0
        spiketimes[i] = times

    # make neo spiketrains
    spiketrains = []
    for i in range(params['n_cell']):
        mask = numpy.where(spiketimes[i] > 0)
        spiketimes[i] = spiketimes[i][mask] 
        spiketrain = neo.SpikeTrain(spiketimes[i], units='s', t_start=0*pq.s, t_stop=params['duration']*pq.s)
        spiketrain.annotate(cell_type='E')
        spiketrains.append(spiketrain)

    # check with sanity plot here
    if params['make_plots']:
        
        # count number of spike per units
        fig, axs = plt.subplots(2, 2)
        count = [st.size for st in spiketrains]
        ax = axs[0, 0]
        simpleaxis(ax)
        pairs = []
        collision_count_by_pair = []
        collision_count_by_units = np.zeros(n_cell)
        for i in range(n_cell):
            for j in range(i+1, n_cell):
                times1 = spiketrains[i].rescale('s').magnitude
                times2 = spiketrains[j].rescale('s').magnitude
                matching_event = make_matching_events((times1*params['fs']).astype('int64'), (times2*params['fs']).astype('int64'), lag_sample)
                pairs.append(f'{i}-{j}')
                collision_count_by_pair.append(matching_event.size)
                collision_count_by_units[i] += matching_event.size
                collision_count_by_units[j] += matching_event.size
        ax.plot(np.arange(len(collision_count_by_pair)), collision_count_by_pair)
        ax.set_xticks(np.arange(len(collision_count_by_pair)))
        ax.set_xticklabels(pairs)
        ax.set_ylim(0, max(collision_count_by_pair) * 1.1)
        ax.set_ylabel('# Collisions')
        ax.set_xlabel('Pairs')

        # count number of spike per units
        count_total = np.array([st.size for st in spiketrains])
        count_not_collision = count_total - collision_count_by_units

        ax = axs[1, 0]
        simpleaxis(ax)
        ax.bar(np.arange(n_cell).astype(numpy.int)+1, count_not_collision, color='g')
        ax.bar(np.arange(n_cell).astype(numpy.int)+1, collision_count_by_units, bottom =count_not_collision, color='r')
        ax.set_ylabel('# spikes')
        ax.set_xlabel('Cell id')
        ax.legend(('Not colliding', 'Colliding'), loc='best')

        # cross corrlogram
        ax = axs[0, 1]
        simpleaxis(ax)
        counts = []
        for i in range(n_cell):
            for j in range(i+1, n_cell):
                times1 = spiketrains[i].rescale('s').magnitude
                times2 = spiketrains[j].rescale('s').magnitude
                matching_event = make_matching_events((times1*params['fs']).astype('int64'), (times2*params['fs']).astype('int64'), lag_sample)
                
                #~ ax = axs[i, j]
                all_lag = matching_event['delta_frame']  / params['fs']
                count, bins  = np.histogram(all_lag, bins=np.arange(-params['lag_time'], params['lag_time'], params['lag_time']/20))
                #~ ax.bar(bins[:-1], count, bins[1] - bins[0])
                ax.plot(1000*bins[:-1], count, bins[1] - bins[0], c='0.5')
                counts += [count]
        counts = numpy.array(counts)
        counts = numpy.mean(counts, 0)
        ax.plot(1000*bins[:-1], counts, bins[1] - bins[0], c='r')
        ax.set_xlabel('Lags [ms]')
        ax.set_ylabel('# Collisions')

        ax = axs[1, 1]
        simpleaxis(ax)
        ratios = []
        for i in range(n_cell):
            nb_spikes = len(spiketrains[i])
            nb_collisions = 0
            times1 = spiketrains[i].rescale('s').magnitude
            for j in list(range(0, i)) + list(range(i+1, n_cell)):
                times2 = spiketrains[j].rescale('s').magnitude
                matching_event = make_matching_events((times1*params['fs']).astype('int64'), (times2*params['fs']).astype('int64'), lag_sample)
                nb_collisions += matching_event.size

            if nb_collisions > 0:
                ratios += [nb_spikes / nb_collisions]
            else:
                ratios += [0]

        ax.bar([0], [np.mean(ratios)], yerr=[np.std(ratios)])
        ax.set_ylabel('# spikes / # collisions')
        plt.tight_layout()

        if params['save_plots']:
            plt.savefig(plot_filename)
        else:
            plt.show()
        plt.close()

    if params['generate_recording']:
        spgen = mr.SpikeTrainGenerator(spiketrains=spiketrains)
        rec_params = mr.get_default_recordings_params()
        rec_params['recordings']['fs'] = params['fs']
        rec_params['recordings']['sync_rate'] = None
        rec_params['recordings']['sync_jitter'] = 5
        rec_params['recordings']['noise_level'] = params['noise_level']
        rec_params['recordings']['filter'] = False
        rec_params['spiketrains']['duration'] = params['duration']
        rec_params['spiketrains']['n_exc'] = params['n_cell']
        rec_params['spiketrains']['n_inh'] = 0
        rec_params['recordings']['chunk_duration'] = 10.
        rec_params['templates']['n_overlap_pairs'] = None
        rec_params['templates']['min_dist'] = 0.1
        rec_params['seeds']['templates'] = params['templates_seed']
        rec_params['seeds']['noise'] = params['noise_seed']
        recgen = mr.gen_recordings(params=rec_params, spgen=spgen, templates=template_filename, verbose=True)
        mr.save_recording_generator(recgen, filename=recording_filename)


