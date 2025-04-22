#!/usr/bin/env python
import caiman as cm
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
import pyximport
pyximport.install()
import scipy
from tensorflow.python.client import device_lib
from time import time
import pickle
from datetime import datetime
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from fiola.utilities import download_demo, load, to_2D, movie_iterator
import sys
from LSTM_Processing.model_creator import load_and_finetune_model, spikes_reader, train_model_from_scratch_with_kfold
from config import num_frames_total, num_frames_init, use_pretrained
import tensorflow as tf
tf.debugging.set_log_device_placement(True)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU device found")

logging.basicConfig(format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s][%(process)d] %(message)s",
                    level=logging.INFO)
logging.info(device_lib.list_local_devices())  # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.5.0
startTime = time()
import cv2
import glob
from time import time
from caiman.motion_correction import MotionCorrect #Subject to change
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.summary_images import local_correlations_movie_offline
from caiman.source_extraction.cnmf.utilities import get_file_size

#%%    
# Getting an intialization file by using CaImAn v1.9.13. The current bottleneck is the CaImAn motion correction. 
def run_caiman_init(fnames, pw_rigid=True, max_shifts=[6, 6], gnb=2, rf=15, K=5, gSig=[4, 4]):
    current_dir = os.getcwd()
    for filename in os.listdir(current_dir):
        if filename.endswith("_caiman_init.hdf5"):
            return filename
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)  # Adjust n_processes as needed

    timing = {}
    timing['start'] = time()

    # dataset dependent parameters
    display_images = False

    fr = 30  # imaging rate in frames per second
    decay_time = 0.4  # length of a typical transient in seconds
    dxy = (2., 2.)  # spatial resolution in x and y in (um per pixel)
    patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    overlaps = (24, 24)
    max_deviation_rigid = 3

    mc_dict = {
        'fnames': fnames,
        'fr': fr,
        'decay_time': decay_time,
        'dxy': dxy,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': 'copy',
    }

    opts = params.CNMFParams(params_dict=mc_dict)

    # Motion correction and memory mapping. To use Cuda for motion correction, set use_cuda to True. However, CaImAn does not seem to work correctly with GPU. 
    # This is the bottleneck of generate_init_result.py
    time_init = time()
    motion_opts = opts.get_group('motion')
    # motion_opts['use_cuda'] = True
    print(motion_opts)
    logging.info(motion_opts)
    mc = MotionCorrect(fnames, dview=dview, **motion_opts)
    mc.motion_correct(save_movie=True)
    logging.info('Finished mc, before cm.save_memmap')


    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C', border_to_0=border_to_0)  # exclude borders
    logging.info('Finished exclude borders')

    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    logging.info('Finished reshaping borders')

    # Restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)  # Adjust n_processes as needed
    logging.info('Finished restarting cluster')

    f_F_mmap = mc.mmap_file[0]
    Cns = local_correlations_movie_offline(f_F_mmap, remove_baseline=True, window=1000, stride=1000, winSize_baseline=100, quantil_min_baseline=10, dview=dview)
    logging.info('Finished local_correlations_movie_offline')
    Cn = Cns.max(axis=0)
    Cn[np.isnan(Cn)] = 0
    plt.imshow(Cn, vmax=0.5)

    # Parameters for source extraction and deconvolution
    p = 1
    merge_thr = 0.85
    stride_cnmf = 6
    method_init = 'greedy_roi'
    ssub = 2
    tsub = 2

    opts_dict = {'fnames': fnames,
                 'p': p,
                 'fr': fr,
                 'nb': gnb,
                 'rf': rf,
                 'K': K,
                 'gSig': gSig,
                 'stride': stride_cnmf,
                 'method_init': method_init,
                 'rolling_sum': True,
                 'merge_thr': merge_thr,
                 'n_processes': n_processes,
                 'only_init': True,
                 'ssub': ssub,
                 'tsub': tsub}

    opts.change_params(params_dict=opts_dict)
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)
    logging.info('Finished CNFM on patches')

    # Component evaluation
    min_SNR = 1.0
    rval_thr = 0.75
    cnn_thr = 0.3
    cnn_lowest = 0.0

    cnm.params.set('quality', {'decay_time': decay_time,
                               'min_SNR': min_SNR,
                               'rval_thr': rval_thr,
                               'use_cnn': False,
                               'min_cnn_thr': cnn_thr,
                               'cnn_lowest': cnn_lowest})
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    print(len(cnm.estimates.idx_components))
    time_patch = time()

    if display_images:
        cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components)

    cnm.estimates.select_components(use_object=True)
    cnm2 = cnm.refit(images, dview=dview)
    time_end = time()
    print('Total time until cnm refit is: ')
    print(time_end - time_init)

    min_SNR = 2
    rval_thr = 0.85
    cnn_thr = 0.15
    cnn_lowest = 0.0

    cnm2.params.set('quality', {'decay_time': decay_time,
                                'min_SNR': min_SNR,
                                'rval_thr': rval_thr,
                                'use_cnn': False,
                                'min_cnn_thr': cnn_thr,
                                'cnn_lowest': cnn_lowest})
    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
    print(len(cnm2.estimates.idx_components))

    cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)

    if display_images:
        cnm2.estimates.view_components(images, img=Cn, idx=cnm2.estimates.idx_components)
        cnm2.estimates.view_components(images, img=Cn, idx=cnm2.estimates.idx_components_bad)

    cnm2.estimates.select_components(use_object=True)
    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
    cnm2.mmap_F = f_F_mmap
    cnm2.estimates.Cn = Cn
    cnm2.estimates.template = mc.total_template_rig
    cnm2.estimates.shifts = mc.shifts_rig
    save_name = cnm2.mmap_file[:-5] + '_caiman_init.hdf5'


    # Finishing CaImAn initilazation  
    timing['end'] = time()
    print(timing)
    cnm2.save(save_name)
    print(save_name)
    output_file = save_name

    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
    plt.close('all')
    return output_file

#%%    
# Saving the fiola state to a pickle file, both locally and on persistant volume of kubernetes
def save_fiola_state(mc_nn_mov, trace_fiola, template, Ab, min_mov, params, filepath, frames_to_process, local):
    fio_state = {
        'params': params,
        'trace_fiola': trace_fiola,
        'template': template,
        'Ab': Ab,
        'min_mov': min_mov,
        'mov_data': mc_nn_mov,
        'frames_to_process': frames_to_process
    }
    if local:
        persistent_volume_path = "/persistent_storage"
    else:
        persistent_volume_path = "/persistent_storage"
    os.makedirs(persistent_volume_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join(persistent_volume_path, f"{filepath}_{timestamp}.pkl")

    with open(filename, 'wb') as f:
        pickle.dump(fio_state, f)

    # Save the latest filename to a known location
    latest_file_path = os.path.join(persistent_volume_path, 'latest_fiola_state.pkl')
    with open(latest_file_path, 'w') as f:
        f.write(filename)

    logging.info(f"Saved state to {filename}")
    logging.info(f"Updated latest state path to {latest_file_path}")

#%%    


def process_and_finetune(caiman_file, label_file=None, 
                         weights_path="/persistent_storage/pre_trained_model.h5", 
                         window_size=50, step_size=2, epochs=20, batch_size=32):
    spike_activity = spikes_reader(caiman_file)
    spike_activity = np.transpose(spike_activity)
    if label_file is None:
        labels = np.random.randint(0, 2, spike_activity.shape[0])
        logging.info('No label detected during finetuning. Fake labels used.')
    else:
        labels = np.load(label_file)
    if labels.shape[0] != spike_activity.shape[0]:
        raise ValueError(f"Mismatch: labels shape {labels.shape} and spike_activity shape {spike_activity.shape}")
    neurons = spike_activity.shape[1]
    if use_pretrained:
        logging.info('Using pre-trained model')
        # Load the pre-trained model
        load_and_finetune_model(spike_activity, weights_path, window_size, step_size, epochs, batch_size, labels, neurons)
    else: 
        logging.info('Training from scratch')
        # Train the model from scratch
        train_model_from_scratch_with_kfold(spike_activity, window_size, step_size, epochs, batch_size, labels, neurons)
# The main function wraper for generate_init_result.py. When it is being called by passing file name, it will run caiman intilization on that file and process it using fiola.
# The final fiola initialization result will be saved to kubernetes persistent volume in calcium mode.
def caiman_process(fnames, frames_to_process, local, label_file):
    mode = 'calcium'
    #folder = os.getenv('MOVIE_FOLDER', '/usr/src/app')
    # fnames = folder + '/test_sub.tif'

    # The parameters for processing. Subject to change. 
    fr = 30
    offline_batch = 5
    batch = 1
    flip = False
    detrend = False
    dc_param = 0.9995
    do_deconvolve = True
    ms = [3, 3]
    center_dims = None
    hals_movie = 'hp_thresh'
    n_split = 1
    nb = 1
    trace_with_neg = True
    lag = 5

    options = {
        'fnames': fnames,
        'fr': fr,
        'mode': mode, 
        'num_frames_init': frames_to_process,
        'num_frames_total': num_frames_total,
        'offline_batch': offline_batch,
        'batch': batch,
        'flip': flip,
        'detrend': detrend,
        'dc_param': dc_param,
        'do_deconvolve': do_deconvolve,
        'ms': ms,
        'hals_movie': hals_movie,
        'center_dims': center_dims,
        'n_split': n_split,
        'nb': nb,
        'trace_with_neg': trace_with_neg,
        'lag': lag
    }

    mov = cm.load(fnames, subindices=range(frames_to_process))
    fnames_init = fnames.split('.')[0] + '_init.tif'
    mov.save(fnames_init)

    print(tf.test.gpu_device_name())

    # Generating CaImAn initialization file
    caiman_file = run_caiman_init(fnames_init, pw_rigid=True, max_shifts=ms, gnb=nb, rf=15, K=4, gSig=[3, 3])
    logging.info(caiman_file)
    cnm2 = cm.source_extraction.cnmf.cnmf.load_CNMF(caiman_file)
    estimates = cnm2.estimates
    template = cnm2.estimates.template
    Cn = cnm2.estimates.Cn
    logging.info('Finished run caiman init')
    
    
    #Fine-tuning the model **Labels** should be send together with frames during initailization
    
    process_and_finetune(caiman_file, label_file=label_file)
    #-----------------------------------------------------------------------------------------------------------------
    
    # Apply fiola processing on the CaImAn initilaizaiton file
    motion_correct = True
    do_nnls = True
    if motion_correct:
        params = fiolaparams(params_dict=options)
        fio = FIOLA(params=params)
        mc_nn_mov, shifts_fiola, _ = fio.fit_gpu_motion_correction(mov, template, fio.params.mc_nnls['offline_batch'], min_mov=mov.min())
        plt.plot(shifts_fiola)
        plt.xlabel('frames')
        plt.ylabel('pixels')
        plt.legend(['x shifts', 'y shifts'])
    else:
        mc_nn_mov = mov
    logging.info('Finished mc')

    if do_nnls:
        params = fiolaparams(params_dict=options)
        fio = FIOLA(params=params)
        Ab = np.hstack((estimates.A.toarray(), estimates.b))
        trace_fiola, _ = fio.fit_gpu_nnls(mc_nn_mov, Ab, batch_size=fio.params.mc_nnls['offline_batch'])
        plt.plot(trace_fiola[:-nb].T)
        plt.xlabel('frames')
        plt.ylabel('fluorescence signal')

    else:
        if trace_with_neg:
            trace_fiola = np.vstack((estimates.C + estimates.YrA, estimates.f))
        else:
            trace_fiola = estimates.C + estimates.YrA
            trace_fiola[trace_fiola < 0] = 0
            trace_fiola = np.vstack((trace_fiola, estimates.f))
    logging.info('Finished nnls')

    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)
    Ab = np.hstack((estimates.A.toarray(), estimates.b))
    Ab = Ab.astype(np.float32)

    # Creating a fiola pipeline to make sure the processing is not damaged
    fio = fio.create_pipeline(mc_nn_mov, trace_fiola, template, Ab, min_mov=mov.min())
    
    # Saving the results to kubernetes persistent volume
    save_fiola_state(mc_nn_mov, trace_fiola, template, Ab, mov.min(), params, fnames.split('.')[0] + 'pkl', frames_to_process, local)
    print(f'The Total time for initialization is: {time() - startTime}')

