import numpy as np
import time
from tensorflow.keras.models import load_model
import joblib
import tensorflow as tf

def realtime_inference_stream(model, scaler, stream_frames, window_size=30, stride=30, threshold=0.5, trigger_threshold=20):
    """
    Simulate real-time inference using a streaming buffer.
    
    For each window of frames (of length window_size) processed from the stream,
    the model outputs a prediction (binary: 1 for Light, 0 for Dark).
    
    To avoid noisy triggering, this function only "triggers" a state change event
    when a sustained change is observed over trigger_threshold consecutive windows.
    
    Parameters:
      model            : The pre-trained Keras model.
      scaler           : The pre-fitted StandardScaler.
      stream_frames    : An iterable (or list) of frames (each frame is a 1D array of features).
      window_size      : Number of frames to use for each prediction (e.g., 30).
      stride           : Number of new frames after which to make a new prediction.
                         (If stride == window_size, you get non-overlapping windows.)
      threshold        : Model output threshold for deciding Light (> threshold) vs. Dark.
      trigger_threshold: Number of consecutive windows with a new prediction required to trigger a state change.
      
    Returns:
      A list of predictions (each prediction is the model output for that sequence).
    """
    buffer = []
    predictions = []
    
    # Variables for managing the triggering of state changes.
    # last_triggered_state is the current stable state that has been triggered.
    # candidate_count counts how many consecutive windows have predicted a new state.
    last_triggered_state = None
    candidate_count = 0
    candidate_state = None

    for i, frame in enumerate(stream_frames):
        buffer.append(frame)
        # When we have enough frames, perform prediction.
        if len(buffer) >= window_size:
            sequence = np.array(buffer[-window_size:])  # shape: (window_size, n_features)
            # Scale the sequence using the pre-fitted scaler.
            sequence_scaled = scaler.transform(sequence)
            sequence_scaled = sequence_scaled.reshape(1, window_size, -1)
            # Get model prediction (assuming sigmoid activation for binary classification).
            pred_prob = model.predict(sequence_scaled)[0, 0]
            prediction = 1 if pred_prob > threshold else 0
            predictions.append(prediction)
            print(f"Frames {i-window_size+1}-{i}: Prediction = {'Light' if prediction==1 else 'Dark'} (Prob: {pred_prob:.2f})")
            
            # Triggering logic:
            # If no stable state has been set yet, we are waiting for the first sustained state.
            if last_triggered_state is None:
                # Set candidate state if needed.
                if candidate_state is None:
                    candidate_state = prediction
                    candidate_count = 1
                else:
                    # If the prediction matches the candidate, increase count; otherwise, reset candidate.
                    if prediction == candidate_state:
                        candidate_count += 1
                    else:
                        candidate_state = prediction
                        candidate_count = 1
                if candidate_count >= trigger_threshold:
                    last_triggered_state = candidate_state
                    print(f"*** TRIGGER: {('LIGHT' if last_triggered_state==1 else 'DARK')} condition detected (first trigger) ***")
                    candidate_state = None
                    candidate_count = 0
            else:
                # If we already have a stable state, check if the new window differs.
                if prediction != last_triggered_state:
                    candidate_count += 1
                    if candidate_count >= trigger_threshold:
                        last_triggered_state = prediction
                        print(f"*** TRIGGER: {('LIGHT' if last_triggered_state==1 else 'DARK')} condition detected ***")
                        candidate_count = 0
                else:
                    # If the prediction equals the current stable state, reset candidate count.
                    candidate_count = 0
            
            # Slide the window: remove the first 'stride' frames.
            buffer = buffer[stride:]
            # Simulate a small delay (e.g., frame interval) for demonstration purposes.
            time.sleep(0.05)
    
    return predictions

# -------------------------------------------------
# Example Simulation: Real-life testing without the true label
# -------------------------------------------------
if __name__ == '__main__':
    # Check for GPU availability.
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPUs available:", gpus)
    else:
        print("No GPU available, running on CPU.")

    # Load the saved model and scaler (adjust paths as needed).
    model_path = 'lstm_model.h5'
    scaler_path = 'scaler.save'
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Helper function to load frames from a MAT file.
    import scipy.io as sio
    def load_frames_from_mat(filepath, max_neurons):
        """
        Loads a MAT file and returns the frames.
        Each frame is a 1D array: [t, arena_x, arena_y, spks...],
        with the spike data padded to max_neurons columns.
        """
        data = sio.loadmat(filepath)
        t = data['t']              # shape: (n_frames, 1)
        arena_x = data['arena_x']  # shape: (n_frames, 1)
        arena_y = data['arena_y']  # shape: (n_frames, 1)
        spks = data['spks']        # shape: (n_neurons, n_frames)
        spks = spks.T             # shape: (n_frames, n_neurons)
        n_frames, n_neurons = spks.shape
        if n_neurons < max_neurons:
            padding = np.zeros((n_frames, max_neurons - n_neurons))
            spks = np.concatenate([spks, padding], axis=1)
        features = np.concatenate([t, arena_x, arena_y, spks], axis=1)
        return features  # shape: (n_frames, n_features)
    
    # Simulate streaming from a MAT file (for example, dark_1.mat).
    test_file = 'dark_1.mat'  # Adjust to your test file as needed.
    max_neurons = 149  # Use the same value as during training.
    frames = load_frames_from_mat(test_file, max_neurons)
    
    # Convert the frames into a list so that they are processed one-by-one.
    stream_frames = list(frames)
    
    # Run the real-time inference simulation.
    # Since we are in a real-life scenario, we do not provide the true label.
    realtime_predictions = realtime_inference_stream(
        model, scaler, stream_frames,
        window_size=30, stride=30, threshold=0.5,
        trigger_threshold=20  # Adjust as desired to filter noise.
    )
