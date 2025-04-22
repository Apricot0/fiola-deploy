import asyncio
import os
import sys
import struct
import pickle
import logging
from collections import defaultdict
from time import time
import tifffile
import numpy as np
import warnings
import psutil
from fiola.fiola import FIOLA
from fiola.signal_analysis_online import SignalAnalysisOnlineZ
import io
from numba import njit, prange
import corelink
from corelink.resources.control import subscribe_to_stream
import receive_then_init
import queue
import argparse
from fiola.utilities import visualize
import numpy as np
from LSTM_Processing.model_creator import NeuralSpikeLSTM
import functools
from config import num_frames_init, num_frames_total, use_pretrained

warnings.filterwarnings("ignore", message="no queue or thread to delete")

# Define timing decorators for synchronous and asynchronous functions
def log_time_sync(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        elapsed = time() - start_time
        logging.info(f"{func.__name__} took {elapsed:.6f}s")
        return result
    return wrapper

def log_time_async(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time()
        result = await func(*args, **kwargs)
        elapsed = time() - start_time
        logging.info(f"{func.__name__} took {elapsed:.6f}s")
        return result
    return wrapper
batch = 1  # Number of frames processed at a time
time_per_step = []
online_trace = None
online_trace_deconvolved = None
start = None
window_size = 50
HEADER_SIZE = 14  # Timestamp (8 bytes) + frame number (2 bytes) + chunk index (2 bytes) + total chunks (2 bytes)
template = []

# Global main event loop reference (to be set later)
main_loop = None

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Dictionary to hold the incoming chunks for each frame
incoming_frames = defaultdict(lambda: {
    "timestamp": 0,
    "total_slices": 0,
    "received_slices": 0,
    "chunks": [],
    "start_time": time()
})

# Ensure the 'results' directory exists
os.makedirs('results', exist_ok=True)

LATEST_FIOLA_STATE_PATH = '/persistent_storage/latest_fiola_state.pkl'
LATEST_FINE_TUNED_MODEL_PATH = '/persistent_storage/trained_model_from_scratch.h5'
fio_objects = []

@log_time_sync
def print_nested_object(obj, indent=0, max_depth=3, max_array_print=5):
    """Recursively print attributes and their values for a given object."""
    spacing = " " * indent
    if indent // 4 >= max_depth:
        print(f"{spacing}Max depth reached, skipping deeper attributes...")
        return
    for attr in dir(obj):
        if not attr.startswith('__'):
            try:
                value = getattr(obj, attr)
                if isinstance(value, np.ndarray):
                    print(f"{spacing}{attr} (numpy array) = shape: {value.shape}, dtype: {value.dtype}")
                    logging.info(f"{attr} (numpy array) = shape: {value.shape}, dtype: {value.dtype}")
                    print(f"{spacing}Partial view:\n{value[:max_array_print]}")
                elif isinstance(value, dict):
                    print(f"{spacing}{attr} (dict) =")
                    logging.info(f"{attr} (dict) = {value}")
                    for k, v in value.items():
                        print(f"{spacing}  {k}: {v}")
                        logging.info(f"{spacing}  {k}: {v}")
                elif callable(value):
                    print(f"{spacing}{attr} (method) = {value}")
                    logging.info(f"{attr} (method) = {value}")
                elif isinstance(value, (object, list, tuple)) and not isinstance(value, (str, int, float)):
                    print(f"{spacing}{attr} (object) = {type(value).__name__}")
                    logging.info(f"{attr} (object) = {type(value).__name__}")
                    print_nested_object(value, indent + 4, max_depth, max_array_print)
                else:
                    print(f"{spacing}{attr} = {value}")
                    logging.info(f"{attr} = {value}")
            except Exception as e:
                print(f"{spacing}Could not retrieve {attr}: {e}")
                logging.error(f"Could not retrieve {attr}: {e}")

@log_time_sync
def load_fiola_state(filepath):
    global template, num_frames_init
    with open(filepath, 'rb') as f:
        fio_state = pickle.load(f)
    params = fio_state['params']
    trace_fiola = np.array(fio_state['trace_fiola'], dtype=np.float32)
    template = np.array(fio_state['template'], dtype=np.float32)
    Ab = np.array(fio_state['Ab'], dtype=np.float32)
    min_mov = fio_state['min_mov']
    mc_nn_mov = np.array(fio_state['mov_data'], dtype=np.float32)
    num_frames_init = fio_state['frames_to_process']
    fio = FIOLA(params=params)
    fio.create_pipeline(mc_nn_mov, trace_fiola, template, Ab, min_mov=min_mov)
    return fio

@log_time_sync
def read_tiff_data(buffer, offsets, bytecounts, dtype, shape):
    image_data = np.zeros(shape, dtype=dtype)
    for i in range(len(offsets)):
        buffer.seek(offsets[i])
        data = np.frombuffer(buffer.read(bytecounts[i]), dtype=dtype)
        image_data[i, ...] = data.reshape(shape[1:])
    return image_data

@njit(parallel=True)
def process_tiff_data(image_data, offsets, bytecounts, dtype, shape):
    for i in prange(len(offsets)):
        data = np.frombuffer(image_data[offsets[i]:offsets[i]+bytecounts[i]], dtype=dtype)
        image_data[i, ...] = data.reshape(shape[1:])
    return image_data

@log_time_async
async def memmap_from_buffer(tiff_buffer):
    buffer = io.BytesIO(tiff_buffer)
    try:
        with tifffile.TiffFile(buffer) as tif:
            tiff_series = tif.series[0]
            dtype = tiff_series.dtype
            shape = tiff_series.shape
            byte_order = tif.byteorder
            # Accept 1 frame input only
            shape = (1, *shape)
            logging.info(f"Shape: {shape}")
            image_data = np.zeros(shape, dtype=np.dtype(byte_order + dtype.char))
            offsets = []
            bytecounts = []
            for page in tif.pages:
                offsets.extend(page.dataoffsets)
                bytecounts.extend(page.databytecounts)
            image_data = read_tiff_data(buffer, offsets, bytecounts, np.dtype(byte_order + dtype.char), shape)
        if image_data.size != np.prod(shape):
            logging.error(f"Data array size {image_data.size} does not match expected size {np.prod(shape)}.")
            raise ValueError(f"Data array size {image_data.size} does not match expected size {np.prod(shape)}.")
        logging.info(f"Successfully memmapped buffer. Shape: {image_data.shape}, dtype: {image_data.dtype}")
        return image_data
    except Exception as e:
        logging.error(f"Error processing TIFF buffer: {e}")
        raise

@log_time_sync
def process_frame_data(memmap_image):
    frame_batch = memmap_image.astype(np.float32)
    return frame_batch

# Offload CPU-bound processing using an executor.
@log_time_async
async def process_frame_with_buffer(fio, frame_data, frame_idx, timestamp, processtimestamp, local, model):
    # First, asynchronously load the TIFF image (I/O-bound)
    memmap_image = await memmap_from_buffer(frame_data)
    # Then offload heavy CPU-bound processing to a separate thread.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,  # Default executor (ThreadPool)
        process_frame_cpu_bound,
        fio,
        memmap_image,
        frame_idx,
        timestamp,
        local,
        model
    )

def process_frame_cpu_bound(fio, memmap_image, frame_idx, timestamp, local, model):
    global online_trace, online_trace_deconvolved

    adjusted_frame_idx = frame_idx + num_frames_init
    start_time = time()

    # Ensure online_trace arrays are initialized
    if online_trace is None or online_trace.size == 0:
        total_neurons = fio.Ab.shape[-1]
        total_background = fio.params.hals['nb']
        online_trace = np.zeros((total_neurons, batch), dtype=np.float32)
        online_trace_deconvolved = np.zeros((total_neurons - total_background, batch), dtype=np.float32)

    # Pad the arrays if needed (only pad the difference)
    new_size = adjusted_frame_idx + batch
    current_size = online_trace.shape[1]
    if new_size > current_size:
        pad_width = new_size - current_size
        online_trace = np.pad(online_trace, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        online_trace_deconvolved = np.pad(online_trace_deconvolved, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

    # CPU-bound work: run FIOLA processing pipeline and update traces
    fio.fit_online_frame(memmap_image)
    trace_now = fio.pipeline.saoz.trace[:, adjusted_frame_idx: adjusted_frame_idx + batch]
    trace_window = fio.pipeline.saoz.trace_deconvolved[:, adjusted_frame_idx + batch - window_size: adjusted_frame_idx + batch]
    prediction = [[0]]
    logging.info(f"inference: {prediction[0][0]}")

    # Update the online traces
    online_trace[:, :batch] = trace_now
    online_trace_deconvolved[:, :batch] = fio.pipeline.saoz.trace_deconvolved[:, adjusted_frame_idx: adjusted_frame_idx + batch]

    if frame_idx == 100:
         print('Data at idx 100:')
         print_nested_object(fio, max_depth=3, max_array_print=5)

    end_time = time()
    total_time = end_time - (timestamp / 1000)  # Convert timestamp to seconds
    logging.info(f"Total time spent on frame {frame_idx} (from capture to finish): {total_time}, processing time: {end_time - start_time}")

    if not local:
         asyncio.run_coroutine_threadsafe(
             corelink.send(sender_id, f'Processed frame {frame_idx} with inference: {prediction[0][0]} using {total_time}'),
             main_loop
         )

@log_time_async
async def callback(data_bytes, streamID, header):
    global incoming_frames
    timestamp, frame_number, chunk_index, total_chunks = struct.unpack('>QHHH', data_bytes[:HEADER_SIZE])
    chunk_data = data_bytes[HEADER_SIZE:]
    arrival_time = time()
    frame = incoming_frames[frame_number]
    frame["timestamp"] = timestamp
    if frame["received_slices"] == 0:
        frame["total_slices"] = total_chunks
        frame["chunks"] = [None] * total_chunks
        frame["start_time"] = int(time() * 1000)
    if chunk_index < total_chunks and frame["chunks"][chunk_index] is None:
        frame["chunks"][chunk_index] = chunk_data
        frame["received_slices"] += 1
        if frame["received_slices"] == total_chunks:
            transmission_time = time() - timestamp / 1000
            logging.info(f"Frame {frame_number} transmission time: {transmission_time:.6f}s")
            frame_data = b''.join(frame["chunks"])
            asyncio.create_task(process_frame_with_buffer(
                fio_objects[0],
                frame_data,
                frame_number,
                frame["timestamp"],
                frame["start_time"],
                local=False,
                model=model
            ))
            del incoming_frames[frame_number]
            logging.info(f"Frame {frame_number} fully received at {arrival_time:.6f}, started processing at {time():.6f}")
    else:
        logging.info(f"Invalid or duplicate slice index: {chunk_index} for frame: {frame_number}")

@log_time_async
async def update(response, key):
    logging.info(f'Updating as new sender valid in the workspace: {response}')
    await subscribe_to_stream(response['receiverID'], response['streamID'])

@log_time_async
async def stale(response, key):
    logging.info(response)

@log_time_async
async def subscriber(response, key):
    logging.info(f"subscriber: {response}")

@log_time_async
async def dropped(response, key):    
    logging.info(f"dropped: {response}")

@log_time_async
async def processing():
    global fio_objects, sender_id, model, main_loop
    main_loop = asyncio.get_running_loop()
    if os.path.exists(LATEST_FIOLA_STATE_PATH):
        with open(LATEST_FIOLA_STATE_PATH, 'r') as f:
            latest_fiola_state_file = f.read().strip()
        if os.path.exists(latest_fiola_state_file):
            logging.info(f"Loading FIOLA state from {latest_fiola_state_file}")
            fio_objects.append(load_fiola_state(latest_fiola_state_file))
        else:
            logging.error(f"Latest FIOLA state file {latest_fiola_state_file} does not exist.")
            sys.exit(1)
    else:
        logging.info("Generating new FIOLA init file")
        terminate_event = asyncio.Event()
        await receive_then_init.receive_then_init(terminate_event)
        logging.info("Completed receive_then_init")
        if os.path.exists(LATEST_FIOLA_STATE_PATH):
            with open(LATEST_FIOLA_STATE_PATH, 'r') as f:
                latest_fiola_state_file = f.read().strip()
            if os.path.exists(latest_fiola_state_file):
                fio_objects.append(load_fiola_state(latest_fiola_state_file))
            else:
                logging.error(f"Failed to generate the FIOLA state file at {latest_fiola_state_file}")
                sys.exit(1)
        else:
            logging.error(f"Failed to generate the latest FIOLA state file path at {LATEST_FIOLA_STATE_PATH}")
            sys.exit(1)
   
    model = NeuralSpikeLSTM.load_trained_model(file_path = LATEST_FINE_TUNED_MODEL_PATH )
    await corelink.set_server_callback(update, 'update')
    await corelink.set_server_callback(stale, 'stale')
    await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
    await corelink.set_data_callback(callback)
    
    receiver_id = await corelink.create_receiver("FentonRaw1", "ws", data_type="description1", alert=True, echo=True)
    logging.info(f"Receiver ID: {receiver_id}")
    
    logging.info("Start receive process frames")
    await corelink.keep_open()
    
    await corelink.set_server_callback(subscriber, 'subscriber')
    await corelink.set_server_callback(dropped, 'dropped')
    sender_id = await corelink.create_sender("FentonRaw1", "ws", data_type="description2")
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        logging.info('Receiver terminated.')

@log_time_sync
def save_frame_as_tiff(frame_data, frame_idx, output_directory):
    output_filename = os.path.join(output_directory, f"frame_{frame_idx:04d}.tif")
    try:
        tifffile.imwrite(output_filename, frame_data)
        logging.info(f"Frame {frame_idx} saved as {output_filename}")
    except Exception as e:
        logging.error(f"Error saving frame {frame_idx}: {e}")

@log_time_async
async def local_test():
    global fio_objects
    tiff_movie_path = "./test_sub.tif"
    LOCAL_FIOLA_STATE_PATH = '/persistent_storage/latest_fiola_state.pkl'
    if os.path.exists(LOCAL_FIOLA_STATE_PATH):
        with open(LOCAL_FIOLA_STATE_PATH, 'r') as f:
            latest_fiola_state_file = f.read().strip()
        if os.path.exists(latest_fiola_state_file):
            logging.info(f"Loading FIOLA state from {latest_fiola_state_file}")
            fio_objects.append(load_fiola_state(latest_fiola_state_file))
    else:
        logging.error(f"No FIOLA state file found at {LOCAL_FIOLA_STATE_PATH}")
        logging.info("Generating new FIOLA init file")
        await receive_then_init.local_init(tiff_movie_path, num_frames_init)
        logging.info("Completed receive_then_init")
        if os.path.exists(LOCAL_FIOLA_STATE_PATH):
            with open(LOCAL_FIOLA_STATE_PATH, 'r') as f:
                latest_fiola_state_file = f.read().strip()
            if os.path.exists(latest_fiola_state_file):
                fio_objects.append(load_fiola_state(latest_fiola_state_file))
            else:
                logging.error(f"Failed to generate the FIOLA state file at {latest_fiola_state_file}")
                sys.exit(1)
        else:
            logging.error(f"Failed to generate the latest FIOLA state file path at {LATEST_FIOLA_STATE_PATH}")
            sys.exit(1)
    if not os.path.exists(tiff_movie_path):
        logging.error(f"TIFF movie file not found: {tiff_movie_path}")
        sys.exit(1)
    logging.info(f"Opening TIFF movie: {tiff_movie_path}")
    try:
        with tifffile.TiffFile(tiff_movie_path) as tif:
            count = 0
            loaded_model = NeuralSpikeLSTM.load_trained_model()
            for frame_idx, page in enumerate(tif.pages):
                count += 1
                if count == num_frames_total - num_frames_init:
                    break
                frame_data = page.asarray()
                logging.info(f"Processing frame {frame_idx + 1} of {len(tif.pages)}")
                frame_filename = f"./temp/frame_{frame_idx:04d}.tif"
                save_frame_as_tiff(frame_data, frame_idx, "./temp")
                timestamp = int(time() * 1000) 
                with open(frame_filename, "rb") as tiff_file:
                    frame_bytes = tiff_file.read() 
                    try:
                        await process_frame_with_buffer(
                            fio=fio_objects[0],
                            frame_data=frame_bytes,
                            frame_idx=frame_idx,
                            timestamp=timestamp,
                            processtimestamp=timestamp, 
                            local=True,
                            model=loaded_model
                        )
                    except Exception as e:
                        logging.error(f"Error processing frame {frame_idx}: {e}")
            fio.pipeline.saoz.online_trace = online_trace
            fio.pipeline.saoz.online_trace_deconvolved = online_trace_deconvolved
            fio_objects[0].compute_estimates()
        logging.info("All frames from the TIFF movie processed successfully.")
        fio = fio_objects[0]
        Ab = fio.estimates.Ab
        print(Ab)
        print(Ab.shape)
        print(fio.params.data['num_frames_init'])
        print(fio.params.data['num_frames_total'])
        print(fio.estimates)
        print(fio.estimates.trace)
        print(fio.estimates.trace.shape)
        print(fio.pipeline.saoz.online_trace)
        print(fio.pipeline.saoz.online_trace.shape)
        print(dir(fio.estimates.peak_to_std))
        indexes = list(range(Ab.shape[1]))[:-fio.params.hals['nb']]
        spikes = np.delete(fio.estimates.index[indexes], fio.estimates.index[indexes])
        print(spikes)
        print(type(spikes))
        np.save('spikes_over_time.npy', np.array(spikes))
        print_nested_object(fio, max_depth=3, max_array_print=5)
    except Exception as e:
        logging.error(f"Failed to read TIFF movie file: {e}")
        sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run FIOLA data processing pipeline.")
    parser.add_argument(
        "-lt", "--local-test",
        action="store_true",
        help="Run the script in local test mode with synthetic data."
    )
    # parser.add_argument(
    #     "-if", "--initial-frames",
    #     type=int,
    #     default=500,
    #     help="Number of initial frames to use."
    # )
    # parser.add_argument(
    #     "-tf", "--total-frames",
    #     type=int,
    #     default=2000,
    #     help="Total number of frames to process."
    # )
    # parser.add_argument(
    #     "-up", "--use-pretrained",
    #     action="store_true",
    #     help="Use a pretrained model if available."
    # )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    logging.info(f"Using pretrained model: {use_pretrained}")
    logging.info(f"Initial frames: {num_frames_init}")
    logging.info(f"Total frames: {num_frames_total}")
    if num_frames_init > num_frames_total:
        logging.error("Initial frames cannot be greater than total frames.")
        sys.exit(1)
    
    if args.local_test:
        logging.info("Running in local test mode.")
        asyncio.run(local_test())
    else:
        logging.info("Running in normal operation mode.")
        corelink.run(processing())

# Assign CPU affinity
p = psutil.Process(os.getpid())
numa_nodes = psutil.cpu_count(logical=False)
numa_cpus = list(range(numa_nodes))
p.cpu_affinity(numa_cpus)

