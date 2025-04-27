import cv2
import numpy as np
import os
import json # Added for session management
import requests # For handling potential API errors
import matplotlib.pyplot as plt # For colormaps
from matplotlib.colors import Normalize # For colormap normalization
from evaluation_env import RemoteEvaluationEnv # Import the environment

# --- Configuration ---
MAP_FILENAME = 'train_data/walkable_mask.png'
SESSION_FILE = 'navigation_sessions.json' # File to store session data
GUESSES_FILE = 'guesses.json' # File to store transmitter guesses
SAMPLES_DIR = 'samples' # Directory to save RSSI maps
RSSI_MAP_INIT_VALUE = np.nan # Value for unvisited/unknown RSSI cells
RSSI_MIN_VIZ = -150 # Min RSSI for colormap normalization
RSSI_MAX_VIZ = -50    # Max RSSI for colormap normalization
COLORMAP_VIZ = plt.cm.viridis # Use viridis like data_loader.py
# Define the custom colormap: Cyan (#00FFFF) to Magenta (#FF00FF)
# CUSTOM_CMAP = mcolors.LinearSegmentedColormap.from_list(
#    "cyan_magenta", ['#00FFFF', '#FF00FF']
# )
# CUSTOM_CMAP.set_under('#808080') # Set color for values below min (grey)

# START_POS_COL = 2219 # i coordinate # No longer needed, server provides start
# START_POS_ROW = 1811 # j coordinate # No longer needed, server provides start
MOVE_STEP = 1       # How many pixels to move per key press (local visual only now)
DOT_COLOR = (0, 0, 255) # BGR format for red
DOT_RADIUS = 5 # Slightly larger dot
# Updated Window Title placeholder
WINDOW_NAME = "OpenCV Map Navigation" # Constant window name
WINDOW_TITLE_TEMPLATE = "{} ({} - Walk {}) | WASD: Move, +/-: Zoom, Q: Quit | Pos: {} | RSSI: {} | Zoom: {:.2f}x" # Template for the title text
TEAM_ID = "" # Replace with your actual team ID if different

# --- Zoom Configuration ---
INITIAL_ZOOM = 1.0
ZOOM_INCREMENT = 1.5 # Multiplicative factor for zoom
MIN_ZOOM = 0.25 # Allow zooming out
MAX_ZOOM = 16.0 # Example max zoom
DISPLAY_WINDOW_SIZE = 800 # Fixed size of the display window

# --- Action Mapping ---
KEY_TO_ACTION = {
    ord('w'): 1, # S
    ord('s'): 0, # N
    ord('d'): 2, # E
    ord('a'): 3, # W
}

# --- Colormap Normalization (updated) ---
norm_viz = Normalize(vmin=RSSI_MIN_VIZ, vmax=RSSI_MAX_VIZ, clip=False) # clip=False to use set_under

# --- Utility Functions ---
def load_sessions(filename):
    """Loads session data from a JSON file."""
    if not os.path.exists(filename):
        return {}
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {filename}. Starting fresh.")
        return {}
    except Exception as e:
        print(f"Error loading sessions from {filename}: {e}")
        return {}

def load_guesses(filename):
    """Loads transmitter guess data from a JSON file."""
    if not os.path.exists(filename):
        print(f"Warning: Guesses file not found at '{filename}'. Guessing on 'g' will be disabled.")
        return {}
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {filename}. Guessing on 'g' will be disabled.")
        return {}
    except Exception as e:
        print(f"Error loading guesses from {filename}: {e}. Guessing on 'g' will be disabled.")
        return {}

def save_sessions(filename, sessions_data, session_key, current_pos, transmitter_id, walk_id, rssi_map):
    """Saves the current session's state (including RSSI map path) to the JSON file."""
    if session_key: # Only save if a session was actually started/resumed
        # --- Save RSSI Map ---
        os.makedirs(SAMPLES_DIR, exist_ok=True) # Ensure samples directory exists
        rssi_map_filename = os.path.join(SAMPLES_DIR, f"{session_key}_rssi.npy")
        try:
            np.save(rssi_map_filename, rssi_map)
            print(f"RSSI map saved to {rssi_map_filename}")
        except Exception as e:
            print(f"Error saving RSSI map to {rssi_map_filename}: {e}")
            rssi_map_filename = None # Don't store path if saving failed

        # --- Save Session JSON ---
        sessions_data[session_key] = {
            'transmitter_id': transmitter_id,
            'walk_id': walk_id,
            'last_position': list(current_pos), # Ensure it's a list for JSON
            'rssi_map_path': rssi_map_filename # Store relative path to the map
        }
        try:
            with open(filename, 'w') as f:
                json.dump(sessions_data, f, indent=4)
            print(f"Session '{session_key}' saved to {filename}.")
        except Exception as e:
            print(f"Error saving sessions to {filename}: {e}")

# --- Helper function for new session ---
def handle_new_session(sessions):
    """Handles the logic for starting a new session."""
    # Simplified Transmitter ID input
    tx_num_str = input("Enter Transmitter Number (e.g., 0, 1, ...) [default: 0]: ")
    if tx_num_str.isdigit():
        transmitter_id = f"tx{int(tx_num_str)}"
    else:
        transmitter_id = "tx0" # Default if empty or invalid input
        print("Using default transmitter tx0.")

    # Find the next available walk_id for this transmitter
    max_walk_id = -1
    for data in sessions.values():
        if data.get('transmitter_id') == transmitter_id:
             max_walk_id = max(max_walk_id, data.get('walk_id', -1))
    walk_id = max_walk_id + 1 # Calculate the next walk_id

    session_key = f"{transmitter_id}_walk{walk_id}"
    print(f"Starting new session: {session_key} (TX: {transmitter_id}, Walk: {walk_id})")
    # Position is initialized after env.reset()
    # RSSI map is initialized after map dimensions are known
    return session_key, transmitter_id, walk_id, None # Return None for pos and rssi_map

# --- Helper function for resuming session ---
def handle_resume_session(sessions):
    """Handles the logic for resuming an existing session."""
    while True:
        try:
            select_idx_str = input(f"Enter number of session to resume (1-{len(sessions)}): ")
            select_idx = int(select_idx_str) - 1
            if 0 <= select_idx < len(sessions):
                session_key = list(sessions.keys())[select_idx]
                session_data = sessions[session_key]
                transmitter_id = session_data.get('transmitter_id', 'N/A')
                walk_id = session_data.get('walk_id', 'N/A')
                loaded_pos = session_data.get('last_position')
                rssi_map_path = session_data.get('rssi_map_path') # Get path from session

                if transmitter_id == 'N/A' or walk_id == 'N/A' or loaded_pos is None:
                   print("Error: Selected session data is incomplete (missing tx, walk, or pos). Cannot resume.")
                   return None, None, None, None, None # Return None for all

                current_pos = list(loaded_pos) # Ensure it's a mutable list
                print(f"Resuming session: {session_key} (TX: {transmitter_id}, Walk: {walk_id}) at {current_pos}")
                # RSSI map will be loaded later, after map dimensions are known. Pass the path for now.
                return session_key, transmitter_id, walk_id, current_pos, rssi_map_path
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyError:
             print("Error: Session data is missing expected keys. Cannot resume.")
             return None, None, None, None, None

# --- Helper function for window title ---
def update_window_title(transmitter_id, walk_id, current_pos, current_rssi):
    """Updates the OpenCV window title."""
    # Format RSSI nicely, handle non-numeric types like "N/A" or None
    if isinstance(current_rssi, (int, float)):
        rssi_str = f"{current_rssi:.2f}"
    elif isinstance(current_rssi, str):
        rssi_str = current_rssi # Keep "N/A" as is
    else:
        rssi_str = "Unknown" # Placeholder for other types

    # Format the title text using the template
    title_text = WINDOW_TITLE_TEMPLATE.format(WINDOW_NAME, transmitter_id, walk_id, current_pos, rssi_str, zoom_factor)
    # Set the title text for the constant window name
    cv2.setWindowTitle(WINDOW_NAME, title_text)

# --- Load Map ---
if not os.path.exists(MAP_FILENAME):
    print(f"Error: Map file not found at '{MAP_FILENAME}'")
    exit()

map_image = cv2.imread(MAP_FILENAME)
if map_image is None:
    print(f"Error: Could not load map image from '{MAP_FILENAME}'")
    exit()

# Ensure map is in 3-channel BGR format for color drawing
if len(map_image.shape) == 2: # Grayscale
    map_image = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)
elif map_image.shape[2] == 4: # BGRA
     map_image = cv2.cvtColor(map_image, cv2.COLOR_BGRA2BGR)

map_height, map_width = map_image.shape[:2]
print(f"Loaded map: {MAP_FILENAME} ({map_width}x{map_height})")

# --- Initialize State ---
# Load existing sessions
sessions = load_sessions(SESSION_FILE)
# Load transmitter guesses
transmitter_guesses = load_guesses(GUESSES_FILE)

print("--- Session Management ---")

current_pos = None
transmitter_id = None
walk_id = None
session_key = None # Will store the key for the current session
rssi_map = None # Initialize RSSI map variable
rssi_map_path_to_load = None # Temporary storage for path if resuming
zoom_factor = INITIAL_ZOOM # Initialize zoom factor

# Ask user to resume or start new
while True:
    print("Available sessions:")
    if not sessions:
        print("  (No saved sessions)")
    else:
        for i, (key, data) in enumerate(sessions.items()):
            # Ensure data has expected keys before accessing
            tx = data.get('transmitter_id', 'N/A')
            wk = data.get('walk_id', 'N/A')
            pos = data.get('last_position', ['N/A', 'N/A'])
            map_path = data.get('rssi_map_path', 'N/A') # Show saved map path
            print(f"  {i+1}. {key} (TX: {tx}, Walk: {wk}, Last Pos: {pos}, Map: {map_path})")

    choice = input("Start [n]ew session or [r]esume existing? (n/r): ").lower()

    if choice == 'n':
        session_key, transmitter_id, walk_id, current_pos = handle_new_session(sessions)
        # Initialize new RSSI map here, now that map dimensions are known
        print(f"Initializing new RSSI map ({map_height}x{map_width})")
        rssi_map = np.full((map_height, map_width), RSSI_MAP_INIT_VALUE, dtype=np.float32)
        break
    elif choice == 'r' and sessions:
        # Store the path returned by handle_resume_session
        session_key, transmitter_id, walk_id, current_pos, rssi_map_path_to_load = handle_resume_session(sessions)
        if session_key: # If resume was successful (basic checks passed)
            # Try loading the RSSI map
            if rssi_map_path_to_load and os.path.exists(rssi_map_path_to_load):
                try:
                    rssi_map = np.load(rssi_map_path_to_load)
                    print(f"Loaded existing RSSI map from {rssi_map_path_to_load}")
                    # Optional: Check if dimensions match
                    if rssi_map.shape != (map_height, map_width):
                         print(f"Warning: Loaded map shape {rssi_map.shape} does not match current map shape {(map_height, map_width)}. Re-initializing.")
                         rssi_map = np.full((map_height, map_width), RSSI_MAP_INIT_VALUE, dtype=np.float32)
                except Exception as e:
                    print(f"Error loading RSSI map from {rssi_map_path_to_load}: {e}. Initializing new map.")
                    rssi_map = np.full((map_height, map_width), RSSI_MAP_INIT_VALUE, dtype=np.float32)
            else:
                print(f"Warning: RSSI map file not found at '{rssi_map_path_to_load}'. Initializing new map.")
                rssi_map = np.full((map_height, map_width), RSSI_MAP_INIT_VALUE, dtype=np.float32)
            break # Exit loop once resume is handled
    elif choice == 'r' and not sessions:
        print("No sessions available to resume.")
    else:
        print("Invalid choice. Please enter 'n' or 'r'.")

if session_key is None or rssi_map is None: # Check if session setup failed
    print("Failed to initialize or resume a session. Exiting.")
    exit()

# --- Initialize Remote Environment ---
print("Initializing remote environment...")
env = RemoteEvaluationEnv(
    team_id=TEAM_ID,
    transmitter_id=transmitter_id,
    walk_id=walk_id
)

# --- Start/Reset Remote Walk ---
current_rssi = "N/A" # Initialize RSSI
try:
    # If current_pos is set, we are resuming, otherwise starting new
    is_resuming = current_pos is not None
    if is_resuming:
        print(f"Attempting to resume walk (TX: {transmitter_id}, Walk: {walk_id}) at {current_pos}...")
        # Try to get initial RSSI from loaded map
        try:
             initial_rssi_value = rssi_map[current_pos[1], current_pos[0]] # row, col
             if not np.isnan(initial_rssi_value):
                 current_rssi = float(initial_rssi_value)
                 print(f"Initial RSSI loaded from map: {current_rssi:.2f}")
             else:
                 print("Initial RSSI for resume position not found in map. Will fetch on first move.")
        except IndexError:
             print("Error: Resume position is outside the bounds of the loaded/initialized RSSI map.")
             # Decide how to handle this - maybe exit or just continue without initial RSSI
        except Exception as e:
             print(f"Error accessing initial RSSI from map: {e}")

        print("Skipping env.reset() for resume.")
        # Keep the loaded position
    else:
        print(f"Calling env.reset() to start new walk (TX: {transmitter_id}, Walk: {walk_id})...")
        reset_data = env.reset()
        # Use the position returned by the server as the definitive start
        current_pos = list(reset_data['ij'])
        raw_rssi = reset_data['rssi']
        print(f"Environment reset successful. Starting at {current_pos}, Raw RSSI: {raw_rssi}")
        # Update RSSI map with the initial value
        if isinstance(raw_rssi, (int, float)):
            current_rssi = float(raw_rssi)
            rssi_map[current_pos[1], current_pos[0]] = current_rssi # row, col
            print(f"Stored initial RSSI: {current_rssi:.2f}")
        else:
             print(f"Warning: Invalid initial RSSI type received: {type(raw_rssi)}. Storing NaN.")
             current_rssi = "N/A" # Keep display as N/A
             rssi_map[current_pos[1], current_pos[0]] = np.nan # Store NaN


except requests.exceptions.RequestException as e:
    # Check specifically for 400 Bad Request potentially indicating existing walk on START
    if e.response is not None and e.response.status_code == 400 and not is_resuming:
         print(f"Error during initial env.reset(): {e}")
         print("This might indicate the walk ID already exists on the server.")
         print("Proceeding with local navigation only for visualization.")
         env = None # Disable further remote calls
    elif is_resuming:
        # If an error occurs during resume *attempt* (though we skip reset now), handle it.
        # This block might be less relevant now we skip reset on resume.
        print(f"Error connecting to evaluation server while verifying resume: {e}")
        print("Proceeding with local navigation only for visualization.")
        env = None # Disable further remote calls
    else: # General connection error during initial reset
        print(f"Error connecting to evaluation server during initial reset: {e}")
        print("Proceeding with local navigation only for visualization.")
        env = None # Disable further remote calls
except Exception as e:
    print(f"An unexpected error occurred during environment initialization: {e}")
    env = None

# --- Main Loop ---
print("Starting OpenCV navigation.")
print("Controls: W=Up, A=Left, S=Down, D=Right, Q=Quit & Save")
# Initial window title update
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) # Use the constant name
cv2.resizeWindow(WINDOW_NAME, DISPLAY_WINDOW_SIZE, DISPLAY_WINDOW_SIZE) # Use the constant name and fixed size
# Update title with potentially loaded initial RSSI
update_window_title(transmitter_id, walk_id, current_pos, current_rssi) # Set initial title text

while True:
    # --- Create Display Image with RSSI Overlay ---
    # Start with a fresh copy of the base map for calculations
    overlay_image = map_image.copy()

    # Find valid RSSI data points
    valid_mask = ~np.isnan(rssi_map)
    rows, cols = np.where(valid_mask)

    if rows.size > 0: # Only proceed if there are valid points
        valid_rssi = rssi_map[rows, cols]

        # Normalize RSSI values and apply colormap
        # Note: norm_viz clips values outside the defined range
        colors_rgba = COLORMAP_VIZ(norm_viz(valid_rssi))

        # Convert RGBA (0-1) to BGR (0-255) for OpenCV
        colors_bgr = (colors_rgba[:, [2, 1, 0]] * 255).astype(np.uint8)

        # Apply the colors to the overlay image at the valid locations
        overlay_image[rows, cols] = colors_bgr

    # --- Calculate Zoomed ROI ---
    # Size of the region to extract from the full overlay_image
    roi_size_on_map = int(DISPLAY_WINDOW_SIZE / zoom_factor)

    # Ensure roi_size is at least 1 pixel
    roi_size_on_map = max(1, roi_size_on_map)

    # Calculate top-left corner of the ROI, centered on current_pos
    center_x, center_y = int(current_pos[0]), int(current_pos[1])
    roi_x = center_x - roi_size_on_map // 2
    roi_y = center_y - roi_size_on_map // 2

    # Clamp ROI coordinates to stay within map bounds
    roi_x = max(0, min(roi_x, map_width - roi_size_on_map))
    roi_y = max(0, min(roi_y, map_height - roi_size_on_map))

    # Extract the ROI
    roi = overlay_image[roi_y:roi_y + roi_size_on_map, roi_x:roi_x + roi_size_on_map]

    # Resize ROI to the fixed display window size (Zoom effect)
    # Use INTER_NEAREST for pixelated zoom
    if roi.shape[0] > 0 and roi.shape[1] > 0: # Check if ROI is valid
        zoomed_display = cv2.resize(roi, (DISPLAY_WINDOW_SIZE, DISPLAY_WINDOW_SIZE), interpolation=cv2.INTER_NEAREST)
    else:
        # Handle invalid ROI (e.g., if roi_size_on_map was 0 or negative, though checks should prevent this)
        # Create a blank image as fallback
        zoomed_display = np.zeros((DISPLAY_WINDOW_SIZE, DISPLAY_WINDOW_SIZE, 3), dtype=np.uint8)
        print("Warning: Could not create valid ROI for zooming.")


    # --- Draw Current Position on Zoomed Image ---
    # Calculate position relative to the ROI's top-left corner
    marker_x_relative = center_x - roi_x
    marker_y_relative = center_y - roi_y

    # Scale the relative position to the zoomed display window size
    marker_x_zoomed = int(marker_x_relative * zoom_factor)
    marker_y_zoomed = int(marker_y_relative * zoom_factor)

    # Clamp marker position within the zoomed window bounds (safety)
    marker_x_zoomed = max(0, min(marker_x_zoomed, DISPLAY_WINDOW_SIZE - 1))
    marker_y_zoomed = max(0, min(marker_y_zoomed, DISPLAY_WINDOW_SIZE - 1))

    # Draw the marker on the zoomed image
    cv2.circle(zoomed_display, (marker_x_zoomed, marker_y_zoomed), DOT_RADIUS, DOT_COLOR, -1) # -1 fills the circle


    # Display the zoomed image
    # Update title just before showing
    update_window_title(transmitter_id, walk_id, current_pos, current_rssi)
    cv2.imshow(WINDOW_NAME, zoomed_display) # Show the zoomed view

    # Wait for key press
    key = cv2.waitKey(0) & 0xFF

    # --- Process Key Press ---
    if key == ord('q') or key == 27: # 'q' or ESC key
        print("Quit key pressed. Saving session and exiting.")
        break

    # --- Handle Movement Keys (WASD) ---
    if key in KEY_TO_ACTION:
        action = KEY_TO_ACTION[key]
        action_char = chr(key)
        print(f"Attempting move: {action_char.upper()} (Action: {action})")

        if env: # Only attempt remote step if environment was initialized
            try:
                # --- Define the guess parameters ---
                # guess_location = [930, 2071] # Fixed location
                # guess_radius = 20 # Fixed radius
                # circle_guess = (guess_location[0], guess_location[1], guess_radius)
                # print(f"  Attempting move: {action_char.upper()} (Action: {action}) with Circle Guess: {circle_guess}")

                # Call the remote step function WITHOUT the circle guess for WASD
                step_data = env.step(action)
                new_pos_server = list(step_data['ij'])
                raw_rssi = step_data['rssi'] # Get raw RSSI

                print(f"  Server Response: New Pos: {new_pos_server}, Raw RSSI: {raw_rssi}")

                # Update local position ONLY if the server confirmed the move
                position_updated = False # Reset for this specific action
                if new_pos_server != current_pos:
                    print(f"  Position updated: {current_pos} -> {new_pos_server}")
                    current_pos = new_pos_server
                    position_updated = True # Mark as updated
                else:
                    print(f"  Position unchanged according to server.")

                # Update current RSSI display value and store in map
                if isinstance(raw_rssi, (int, float)):
                    new_rssi = float(raw_rssi)
                    current_rssi = new_rssi
                    try:
                        rssi_map[current_pos[1], current_pos[0]] = new_rssi
                        print(f"  Stored RSSI: {new_rssi:.2f} at {current_pos}")
                    except IndexError:
                        print(f"  Error: Position {current_pos} is out of bounds for RSSI map. Cannot store RSSI.")
                    except Exception as e:
                        print(f"  Error storing RSSI value {new_rssi} into map: {e}")
                else:
                    print(f"  Warning: Invalid RSSI type received from step: {type(raw_rssi)}. Storing NaN.")
                    try:
                        rssi_map[current_pos[1], current_pos[0]] = np.nan
                        print(f"  Stored NaN at {current_pos} due to invalid RSSI type.")
                    except IndexError:
                        print(f"  Error: Position {current_pos} is out of bounds for RSSI map. Cannot store NaN.")
                    except Exception as e:
                        print(f"  Error storing NaN into map: {e}")

                # Title is updated before imshow

            except ValueError as e: # Illegal move detected by server
                print(f"  Server rejected move (North with guess): {e}")
            except requests.exceptions.RequestException as e:
                print(f"  Error connecting to evaluation server during guess step: {e}")
            except Exception as e:
                print(f"  An unexpected error occurred during guess step: {e}")
        else:
            print("  (Environment not available, cannot perform North move with guess)")

    # --- Handle Guess Key ('g') ---
    elif key == ord('g'):
        print("Guess key 'g' pressed. Attempting North move with guess.")
        action = 0 # North

        if env: # Only attempt remote step if environment was initialized
            # Retrieve guess for the current transmitter
            current_guess_data = transmitter_guesses.get(transmitter_id)
            circle_guess = None

            if current_guess_data:
                try:
                    guess_i = int(current_guess_data['i'])
                    guess_j = int(current_guess_data['j'])
                    guess_r = float(current_guess_data['r'])
                    circle_guess = (guess_i, guess_j, guess_r)
                    print(f"  Found guess for {transmitter_id}: {circle_guess}")
                except (KeyError, ValueError, TypeError) as e:
                    print(f"  Warning: Invalid guess data format for {transmitter_id} in {GUESSES_FILE}: {e}. Moving North without guess.")
            else:
                print(f"  Warning: No guess found for {transmitter_id} in {GUESSES_FILE}. Moving North without guess.")

            try:
                # Call step with action 0 (North) and the retrieved circle_guess (or None)
                print(f"  Calling env.step(action=0, circle={circle_guess})")
                step_data = env.step(action=0, circle=circle_guess)
                new_pos_server = list(step_data['ij'])
                raw_rssi = step_data['rssi']

                print(f"  Server Response: New Pos: {new_pos_server}, Raw RSSI: {raw_rssi}")

                # Update local position ONLY if the server confirmed the move
                position_updated = False # Reset for this specific action
                if new_pos_server != current_pos:
                    print(f"  Position updated: {current_pos} -> {new_pos_server}")
                    current_pos = new_pos_server
                    position_updated = True # Mark as updated
                else:
                    print(f"  Position unchanged according to server.")

                # Update current RSSI display value and store in map
                if isinstance(raw_rssi, (int, float)):
                    new_rssi = float(raw_rssi)
                    current_rssi = new_rssi
                    try:
                        rssi_map[current_pos[1], current_pos[0]] = new_rssi
                        print(f"  Stored RSSI: {new_rssi:.2f} at {current_pos}")
                    except IndexError:
                        print(f"  Error: Position {current_pos} is out of bounds for RSSI map. Cannot store RSSI.")
                    except Exception as e:
                        print(f"  Error storing RSSI value {new_rssi} into map: {e}")
                else:
                    print(f"  Warning: Invalid RSSI type received from step: {type(raw_rssi)}. Storing NaN.")
                    try:
                        rssi_map[current_pos[1], current_pos[0]] = np.nan
                        print(f"  Stored NaN at {current_pos} due to invalid RSSI type.")
                    except IndexError:
                        print(f"  Error: Position {current_pos} is out of bounds for RSSI map. Cannot store NaN.")
                    except Exception as e:
                        print(f"  Error storing NaN into map: {e}")

                # Title is updated before imshow

            except ValueError as e: # Illegal move detected by server
                print(f"  Server rejected move (North with guess): {e}")
            except requests.exceptions.RequestException as e:
                print(f"  Error connecting to evaluation server during guess step: {e}")
            except Exception as e:
                print(f"  An unexpected error occurred during guess step: {e}")
        else:
            print("  (Environment not available, cannot perform North move with guess)")

    # --- Handle Zoom Keys ---
    elif key == ord('+') or key == ord('='): # '+' key (often requires shift, so check '=' too)
        zoom_factor = min(MAX_ZOOM, zoom_factor * ZOOM_INCREMENT)
        print(f"Zoom factor increased to: {zoom_factor:.2f}x")
    elif key == ord('-'): # '-' key
        zoom_factor = max(MIN_ZOOM, zoom_factor / ZOOM_INCREMENT)
        print(f"Zoom factor decreased to: {zoom_factor:.2f}x")

    else:
        print(f"Ignoring key code: {key}")
        continue # Don't update position if key is not WASD or Q/ESC

    # --- Boundary and Collision Checks Removed (Server handles validity) ---
    # The drawing happens at the start of the loop with the current_pos,
    # which is updated based on server response (or local simulation if server fails).

# --- Cleanup ---
# Save the final state before destroying windows
# Make sure rssi_map exists before saving (it should, based on init logic)
if rssi_map is not None and session_key is not None:
    save_sessions(SESSION_FILE, sessions, session_key, current_pos, transmitter_id, walk_id, rssi_map)
else:
    print("Warning: Cannot save session state (session_key or rssi_map is None).")


cv2.destroyAllWindows()
print("OpenCV window closed.") 