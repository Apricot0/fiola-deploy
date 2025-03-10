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
# import caiman as cm
import numpy as np
from LSTM_Processing.model_creator import NeuralSpikeLSTM

warnings.filterwarnings("ignore", message="no queue or thread to delete")

num_frames_init = 500 
num_frames_total = 2000
batch = 1  # Number of frames processed at a time
time_per_step = []
online_trace = None
online_trace_deconvolved = None
start = None
window_size = 50

HEADER_SIZE = 14  # Updated to include timestamp (8 bytes) + frame number (2 bytes) + chunk index (2 bytes) + total chunks (2 bytes)
template = []

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
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
LATEST_FINE_TUNED_MODEL_PATH = '/persistent_storage/fine_turned_model.h5'
fio_objects = []


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
                
                # Handle numpy arrays for clearer printing
                if isinstance(value, np.ndarray):
                    print(f"{spacing}{attr} (numpy array) = shape: {value.shape}, dtype: {value.dtype}")
                    logging.info(f"{attr} (numpy array) = shape: {value.shape}, dtype: {value.dtype}")
                    # Optionally, print part of the array
                    print(f"{spacing}Partial view:\n{value[:max_array_print]}")  # Adjust or remove [:max_array_print] as necessary for large arrays

                # Handle dictionaries
                elif isinstance(value, dict):
                    print(f"{spacing}{attr} (dict) =")
                    logging.info(f"{attr} (dict) = {value}")
                    for k, v in value.items():
                        print(f"{spacing}  {k}: {v}")
                        logging.info(f"{spacing}  {k}: {v}")

                # Handle methods or callable objects
                elif callable(value):
                    print(f"{spacing}{attr} (method) = {value}")
                    logging.info(f"{attr} (method) = {value}")

                # Handle nested objects (recursion)
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



# Loading FIOLA state from a pickle file 
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

    # print(f"mc_nn_mov: {type(mc_nn_mov)}, shape: {getattr(mc_nn_mov, 'shape', 'N/A')}, value: {mc_nn_mov}")
    # print(f"trace_fiola: {type(trace_fiola)}, shape: {getattr(trace_fiola, 'shape', 'N/A')}, value: {trace_fiola}")
    # print(f"template: {type(template)}, shape: {getattr(template, 'shape', 'N/A')}, value: {template}")
    # print(f"Ab: {type(Ab)}, shape: {getattr(Ab, 'shape', 'N/A')}, value: {Ab}")
    # print(f"min_mov: {type(min_mov)}, value: {min_mov}")

    fio.create_pipeline(mc_nn_mov, trace_fiola, template, Ab, min_mov=min_mov)
    
    # if not hasattr(fio.pipeline, 'saoz'):
    #     fio.pipeline.saoz = SignalAnalysisOnlineZ()
    
    return fio

# Reading each tiff frame in buffer, converting to the format that FIOLA wants
def read_tiff_data(buffer, offsets, bytecounts, dtype, shape):
    image_data = np.zeros(shape, dtype=dtype)
    for i in range(len(offsets)):
        buffer.seek(offsets[i])
        data = np.frombuffer(buffer.read(bytecounts[i]), dtype=dtype)
        image_data[i, ...] = data.reshape(shape[1:])
    return image_data

# Using numba to speed up numerical processing
@njit(parallel=True)
def process_tiff_data(image_data, offsets, bytecounts, dtype, shape):
    for i in prange(len(offsets)):
        data = np.frombuffer(image_data[offsets[i]:offsets[i]+bytecounts[i]], dtype=dtype)
        image_data[i, ...] = data.reshape(shape[1:])
    return image_data

# Process the frame data in buffer, to avoid disk io
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

            # Initialize image_data to hold the entire TIFF data
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

def process_frame_data(memmap_image):
    frame_batch = memmap_image.astype(np.float32)
    return frame_batch

async def process_frame_with_buffer(fio, frame_data, frame_idx, timestamp, processtimestamp, local, model):
    global online_trace, online_trace_deconvolved, time_per_step, start

    # Adjust frame_idx to account for the initialization frames
    adjusted_frame_idx = frame_idx + num_frames_init
    # print(adjusted_frame_idx)


    try:
        start_time = time()
        memmap_image = await memmap_from_buffer(frame_data)
        # print("mmemmap", memmap_image[0][0]) 
        buffer_time = time()
        # frame_batch = process_frame_data(memmap_image)
        proc_time = time()

        # Initialize online_trace and online_trace_deconvolved if they are None
        if online_trace is None:
            total_neurons = fio.Ab.shape[-1]
            total_background = fio.params.hals['nb']
            online_trace = np.zeros((total_neurons, batch), dtype=np.float32)
            online_trace_deconvolved = np.zeros((total_neurons - total_background, batch ), dtype=np.float32)
            # time_per_step = np.zeros(1 // batch)
            # start = time()

        # Resize online_trace and online_trace_deconvolved to accommodate new frames beyond initialization
        # if adjusted_frame_idx >= online_trace.shape[1]:
        fio.fit_online_frame(memmap_image)
        trace_now = fio.pipeline.saoz.trace[:,adjusted_frame_idx: adjusted_frame_idx + batch ]
        #TODO trace_window sometimes missing latest frame but sometimes have, might be concurrency issue
        trace_window = fio.pipeline.saoz.trace_deconvolved[:,adjusted_frame_idx + batch - window_size: adjusted_frame_idx + batch ]
        # prediction = model.predict(np.expand_dims( np.transpose(trace_window), axis=0))
        prediction = [[0]]
        logging.info(f"inference: {prediction[0][0]}")
        # print(trace_window.shape)
        # print(frame_idx, trace_now)
        # print(frame_idx, fio.pipeline.saoz.trace[:, 495:503])
        # print(frame_idx, fio.pipeline.saoz.trace.shape)
        # sys.exit()
        new_size = adjusted_frame_idx + batch
        online_trace = np.pad(online_trace, ((0, 0), (0, new_size)), mode='constant', constant_values=0)
        online_trace_deconvolved = np.pad(online_trace_deconvolved, ((0, 0), (0, new_size)), mode='constant', constant_values=0)

        # new_time_per_step_size = (new_size) // batch
        # time_per_step = np.pad(time_per_step, (0, new_time_per_step_size - time_per_step.shape[0]), mode='constant', constant_values=0)
        # print(online_trace.shape)
        # Calculate the current index for online_trace
        # current_idx = adjusted_frame_idx - num_frames_init

        # Ensure current_idx is non-negative
        # if current_idx < 0:
        #     logging.error(f"Negative current_idx: {current_idx}")
        #     return

        # Update online_trace and online_trace_deconvolved
        online_trace[:,  :batch] = trace_now
        online_trace_deconvolved[:, :batch] =   fio.pipeline.saoz.trace_deconvolved[:,adjusted_frame_idx: adjusted_frame_idx + batch ]


        # Record the time per step
        # time_per_step[current_idx // batch] = (time() - start)
        # fio.pipeline.saoz.online_trace = online_trace
        # fio.pipeline.saoz.online_trace_deconvolved = online_trace_deconvolved
        # Compute estimates after processing the frame

        
        if frame_idx == 100:
            print('Data at idx 100:')
            print_nested_object(fio, max_depth=3, max_array_print=5)
        # Log the processing times
        end_time = time()
        total_time = end_time - timestamp / 1000  # Convert timestamp back to seconds
        logging.info(f"Total time spent on frame {frame_idx} (from capture to finish): {total_time}, total time processing is {end_time - proc_time}, buffering time: {buffer_time - start_time}, start to finish: {end_time - start_time}")
        message = f'Processed frame {frame_idx} with inference: {prediction[0][0]} using {total_time}'
        if not local:
            await corelink.send(sender_id, message)
      

    except Exception as e:
        logging.error(f"Failed to process frame with buffer: {e}", exc_info=True)

async def callback(data_bytes, streamID, header):
    # logging.info(f"callback triggered")
    global incoming_frames
#    if streamID == sender_id:
 #       return
    # Extract the header information
    timestamp, frame_number, chunk_index, total_chunks = struct.unpack('>QHHH', data_bytes[:HEADER_SIZE])
    chunk_data = data_bytes[HEADER_SIZE:]
    arrival_time = time()

    frame = incoming_frames[frame_number]
    frame["timestamp"] = timestamp
    
    # Initialize frame entry if receiving the first chunk
    if frame["received_slices"] == 0:
        frame["total_slices"] = total_chunks
        frame["chunks"] = [None] * total_chunks
        frame["start_time"] = int(time() * 1000)
    # Store the chunk data in the correct position
    if chunk_index < total_chunks and frame["chunks"][chunk_index] is None:
        frame["chunks"][chunk_index] = chunk_data
        frame["received_slices"] += 1

        # Check if we have received all chunks for this frame
        if frame["received_slices"] == total_chunks:
            # Log transmission time
            transmission_time = time() - timestamp / 1000
            logging.info(f"Frame {frame_number} transmission time: {transmission_time:.6f}s")

            # Reconstruct the frame
            frame_data = b''.join(frame["chunks"])

            # Process the frame with the single FIOLA object
            asyncio.create_task(process_frame_with_buffer(fio_objects[0], frame_data, frame_number, frame["timestamp"], frame["start_time"], model = model, local = False))

            # Clean up the completed frame entry
            del incoming_frames[frame_number]

            # Log arrival time and processing start time
            logging.info(f"Frame {frame_number} fully received at {arrival_time:.6f}, started processing at {time():.6f}")
    else:
        logging.info(f"Invalid or duplicate slice index: {chunk_index} for frame: {frame_number}")

async def update(response, key):
    logging.info(f'Updating as new sender valid in the workspace: {response}')
    await subscribe_to_stream(response['receiverID'], response['streamID'])

async def stale(response, key):
    logging.info(response)

async def subscriber(response, key):
    logging.info(f"subscriber: {response}")

async def dropped(response, key):    
    logging.info(f"dropped: {response}")

async def processing():
    global fio_objects, sender_id, model

    # Read the latest FIOLA state path
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
   
    model = NeuralSpikeLSTM.load_trained_model()
    await corelink.set_server_callback(update, 'update')
    await corelink.set_server_callback(stale, 'stale')
    await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
    await corelink.set_data_callback(callback)
    
    receiver_id = await corelink.create_receiver("FentonRaw1", "ws",data_type="description1", alert=True, echo=True)
    logging.info(f"Receiver ID: {receiver_id}")
    
    logging.info("Start receive process frames")
    await corelink.keep_open()
    
    await corelink.set_server_callback(subscriber, 'subscriber')
    await corelink.set_server_callback(dropped, 'dropped')

    # await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
    sender_id = await corelink.create_sender("FentonRaw1", "ws",data_type= "description2")

    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        logging.info('Receiver terminated.')


def save_frame_as_tiff(frame_data, frame_idx, output_directory):
    # Generate a filename for each frame (e.g., frame_0001.tif)
    output_filename = os.path.join(output_directory, f"frame_{frame_idx:04d}.tif")
    
    # Save the frame as a separate TIFF file
    try:
        tifffile.imwrite(output_filename, frame_data)
        logging.info(f"Frame {frame_idx} saved as {output_filename}")
    except Exception as e:
        logging.error(f"Error saving frame {frame_idx}: {e}")

async def local_test():
    global fio_objects
    tiff_movie_path = "./test_sub.tif"
    LOCAL_FIOLA_STATE_PATH ='/persistent_storage/latest_fiola_state.pkl'
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
            #DEBUG
            count = 0
            ## load model
            loaded_model = NeuralSpikeLSTM.load_trained_model()
            for frame_idx, page in enumerate(tif.pages):
                count+=1
                if count == num_frames_total-num_frames_init:
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
                            model = loaded_model
                        )
                    except Exception as e:
                        logging.error(f"Error processing frame {frame_idx}: {e}")
            fio.pipeline.saoz.online_trace = online_trace
            fio.pipeline.saoz.online_trace_deconvolved = online_trace_deconvolved
            fio_objects[0].compute_estimates()
        logging.info("All frames from the TIFF movie processed successfully.")

        # states
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

        # save_spikes_to_file(fio_objects[0], indexes, 'spikes_over_time.npy')

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.local_test:
        logging.info("Running in local test mode.")
        asyncio.run(local_test())
    else:
        logging.info("Running in normal operation mode.")
        corelink.run(processing())

# Assigning CPU affinity
p = psutil.Process(os.getpid())

# Assign all CPUs to the process
numa_nodes = psutil.cpu_count(logical=False)
numa_cpus = list(range(numa_nodes))
p.cpu_affinity(numa_cpus)

