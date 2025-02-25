import os
import sys
import struct
import asyncio
import logging
from collections import defaultdict
from time import time, strftime, localtime
import tifffile
# import time 
import numpy as np
import io
import corelink
from corelink.resources.control import subscribe_to_stream
from generate_init_result import caiman_process
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"

HEADER_SIZE = 14  # Updated to include timestamp (8 bytes) + frame number (2 bytes) + chunk index (2 bytes) + total chunks (2 bytes)
INACTIVITY_TIMEOUT = 60  # seconds
FINISHED_INIT = False
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
	"last_update_time": time()
})

# List to keep track of sessions
sessions = []
current_session_frames = []
current_session_start_time = None
last_update_time = time()

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

# async def save_session_tiff(frames, session_start_time):
# 	tiff_filename = f'./data/session_{strftime("%Y%m%d_%H%M%S", localtime(session_start_time))}.tif'
# 	with tifffile.TiffWriter(tiff_filename, bigtiff=True) as tif_writer:
# 		for frame in frames:
#             # Convert the frame data from bytes to numpy array before writing
# 			image_data = np.array(tifffile.imread(io.BytesIO(frame)))
# 			tif_writer.write(image_data.squeeze(), contiguous=True)
# 	print(f"Session saved as {tiff_filename}")
# 	return tiff_filename
async def save_session_tiff(frames, session_start_time):
    tiff_filename = f'./data/session_{strftime("%Y%m%d_%H%M%S", localtime(session_start_time))}.tif'
    images = [Image.open(io.BytesIO(frame)) for frame in frames]
    images[0].save(tiff_filename, save_all=True, append_images=images[1:], bigtiff=True)
    print(f"Session saved as {tiff_filename}")
    return tiff_filename

async def check_inactivity(terminate_event):
	global current_session_frames, current_session_start_time, last_update_time, FINISHED_INIT
	while True:
		await asyncio.sleep(1)
		current_time = time()
		if current_session_start_time and (current_time - last_update_time > INACTIVITY_TIMEOUT):
			print(f"Session inactive for {INACTIVITY_TIMEOUT} seconds. Marking session as complete.")
			tiff_filename = await save_session_tiff(current_session_frames, current_session_start_time)
			print('The total pages are: ', len(tifffile.TiffFile(tiff_filename).pages))
			print('The shape of the tiff file is: ', tifffile.TiffFile(tiff_filename).asarray().shape)
			await asyncio.get_event_loop().run_in_executor(None, caiman_process, tiff_filename, len(tifffile.TiffFile(tiff_filename).pages))
			
			FINISHED_INIT = True
			terminate_event.set()  # Signal to terminate the main process
			return

async def callback(data_bytes, streamID, header):
	global incoming_frames, current_session_frames, current_session_start_time, last_update_time

	# Extract the header information
	timestamp, frame_number, chunk_index, total_chunks = struct.unpack('>QHHH', data_bytes[:HEADER_SIZE])
	chunk_data = data_bytes[HEADER_SIZE:]

	frame = incoming_frames[frame_number]
	frame["timestamp"] = timestamp

	# Update the last update time
	last_update_time = time()

	# Initialize session start time
	if not current_session_start_time:
		current_session_start_time = time()

	# Initialize frame entry if receiving the first chunk
	if frame["received_slices"] == 0:
		frame["total_slices"] = total_chunks
		frame["chunks"] = [None] * total_chunks

	# Store the chunk data in the correct position
	if chunk_index < total_chunks and frame["chunks"][chunk_index] is None:
		frame["chunks"][chunk_index] = chunk_data
		frame["received_slices"] += 1
		print(f"Received slice {chunk_index} for frame {frame_number}")

		# Check if we have received all chunks for this frame
		if frame["received_slices"] == total_chunks:
			# Reconstruct the frame
			frame_data = b''.join(frame["chunks"])
			current_session_frames.append(frame_data)
			print(f"Frame {frame_number} completed and added to session.")
			
			# Clean up the completed frame entry
			del incoming_frames[frame_number]
	else:
		print(f"Invalid or duplicate slice index: {chunk_index} for frame: {frame_number}")

async def update(response, key):
	print(f'Updating as new sender valid in the workspace: {response}')
	await subscribe_to_stream(response['receiverID'], response['streamID'])

async def stale(response, key):
	print(response)

async def receive_then_init(terminate_event):
	global FINISHED_INIT
	print('Started receive_then_init')
	asyncio.create_task(check_inactivity(terminate_event))  # Start inactivity check

	await corelink.set_data_callback(callback)
	await corelink.set_server_callback(update, 'update')
	await corelink.set_server_callback(stale, 'stale')
		
	await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
		
	receiver_id = await corelink.create_receiver("FentonRaw", "ws", alert=True, echo=True)
		
	print(f'Receiver ID: {receiver_id}')
	print("Start receiving initilization frames")
	await corelink.keep_open()
		
	while not FINISHED_INIT:
		await asyncio.sleep(100)
	await corelink.close()  # Close the corelink session
	print("Corelink init session closed.")

	print('Finished')

# corelink.run(main())

async def local_init(tiff_filename, frames_to_process):
    """
    Initializes the CaImAn processing pipeline with the first `frames_to_process` frames
    from the provided TIFF file.

    Args:
        tiff_filename (str): Path to the TIFF file.
        frames_to_process (int): Number of frames to process.
    """
    start_time = time()

    # Check if the TIFF file exists
    if not os.path.exists(tiff_filename):
        print(f"Error: The file {tiff_filename} does not exist.")
        return
    
    # Load the TIFF file and extract the specified frames
    try:
        with tifffile.TiffFile(tiff_filename) as tif:
            frames = [page.asarray() for page in tif.pages[:frames_to_process]]
            print(f"Loaded {len(frames)} frames for processing.")
    except Exception as e:
        print(f"Error reading TIFF file: {e}")
        return

    # Call the caiman_process function with the loaded frames
    caiman_process(tiff_filename, frames_to_process, local=True)

    print(f"Initialization completed in {time() - start_time:.2f} seconds.")

