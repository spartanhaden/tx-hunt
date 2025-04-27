import pandas as pd
from collections import defaultdict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pickle
import time
import json # Added for session data

# --- Configuration ---
CSV_FILENAME = 'train_data/training_walks.csv'
MAP_FILENAME = 'train_data/walkable_mask.png'
CACHE_FILENAME = 'processed_data_cache.pkl'
SESSION_FILENAME = 'navigation_sessions.json' # Added for session loading
SAMPLES_DIR = 'samples' # Added to locate session .npy files
RSSI_MIN_FOR_COLORMAP = -150 # Weakest signal mapped to colormap start
RSSI_MAX_FOR_COLORMAP = -50  # Strongest signal mapped to colormap end
NO_SIGNAL_RSSI = -1000.0
NO_SIGNAL_COLOR_HEX = '#000000' # Black
MID_GREY_COLOR_HEX = '#808080' # Added for very weak signals
COLORMAP = plt.cm.viridis

# --- Helper Functions ---

def get_rssi_color(rssi):
    """Calculates the hex color for a given RSSI value."""
    # Handle explicit no signal first
    if rssi == NO_SIGNAL_RSSI:
        return NO_SIGNAL_COLOR_HEX
    # Handle very weak signals below the new threshold
    if rssi < -1500:
        return MID_GREY_COLOR_HEX

    # Normalize RSSI for colormap
    norm = mcolors.Normalize(vmin=RSSI_MIN_FOR_COLORMAP, vmax=RSSI_MAX_FOR_COLORMAP)
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=COLORMAP)

    # Clamp RSSI to the colormap range before mapping
    clamped_rssi = max(RSSI_MIN_FOR_COLORMAP, min(rssi, RSSI_MAX_FOR_COLORMAP))
    rgba_color = scalar_map.to_rgba(clamped_rssi)
    
    # Convert RGBA (0-1) to RGB hex string
    hex_color = mcolors.to_hex(rgba_color[:3])
    return hex_color

# --- Session Data Loading ---

def load_and_merge_session_data(processed_data, transmitter_location_map):
    """Loads data from navigation sessions and merges it into processed_data."""
    print("--- Loading and Merging Session Data ---")
    if not os.path.exists(SESSION_FILENAME):
        print(f"Session file '{SESSION_FILENAME}' not found. Skipping session data.")
        return processed_data # Return original data if no session file

    try:
        with open(SESSION_FILENAME, 'r') as f:
            sessions = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {SESSION_FILENAME}. Skipping session data.")
        return processed_data
    except Exception as e:
        print(f"Error loading session file {SESSION_FILENAME}: {e}. Skipping session data.")
        return processed_data

    target_transmitters = {f"tx{i}" for i in range(5, 10)} # tx5 to tx9
    loaded_session_walks = 0

    for session_key, session_data in sessions.items():
        transmitter_id = session_data.get('transmitter_id')
        walk_id_str = f"session_walk{session_data.get('walk_id', 'unknown')}" # Use session walk_id, make unique key
        rssi_map_path = session_data.get('rssi_map_path')

        if not transmitter_id or not rssi_map_path:
            print(f"Warning: Skipping session '{session_key}' due to missing transmitter_id or rssi_map_path.")
            continue

        if transmitter_id not in target_transmitters:
            # print(f"Skipping session '{session_key}' (transmitter {transmitter_id} not in target range tx5-tx9).")
            continue # Only load tx5-tx9 for now

        if not os.path.exists(rssi_map_path):
             print(f"Warning: RSSI map file not found for session '{session_key}' at path '{rssi_map_path}'. Skipping.")
             continue

        # Ensure the transmitter exists in the base data (needed for location later, though not strictly required here)
        if transmitter_id not in processed_data:
             processed_data[transmitter_id] = {} # Initialize if not present from CSV/cache
             print(f"Note: Transmitter '{transmitter_id}' from session not found in initial data. Added.")
             # We might not have a location for this transmitter if it wasn't in the CSV.
             # The visualization server handles missing locations gracefully.

        try:
            print(f"Loading RSSI map for session: {session_key} ({rssi_map_path})")
            rssi_map = np.load(rssi_map_path)

            # Extract valid points (where RSSI is not NaN)
            valid_rows, valid_cols = np.where(~np.isnan(rssi_map))
            session_walk_data = []
            for r, c in zip(valid_rows, valid_cols):
                rssi = float(rssi_map[r, c]) # Ensure float
                color = get_rssi_color(rssi)
                # IMPORTANT: Store as (col, row, rssi, color) matching CSV format
                session_walk_data.append((int(c), int(r), rssi, color)) 

            if session_walk_data:
                # Add the extracted walk data under the unique session walk ID
                processed_data[transmitter_id][walk_id_str] = session_walk_data
                loaded_session_walks += 1
                print(f"  -> Added walk '{walk_id_str}' for transmitter '{transmitter_id}' with {len(session_walk_data)} points.")
            else:
                 print(f"  -> No valid RSSI points found in map for session '{session_key}'.")


        except Exception as e:
             print(f"Error processing session '{session_key}' (path: {rssi_map_path}): {e}")

    print(f"--- Finished loading session data. Added {loaded_session_walks} walks. ---")
    return processed_data

# --- Main Data Loading Function ---

def load_data():
    """Loads and processes walk data from the CSV, using a cache if available."""
    print("--- Checking for Cached Data ---")

    cache_exists = os.path.exists(CACHE_FILENAME)
    csv_exists = os.path.exists(CSV_FILENAME)
    cache_valid = False

    if cache_exists and csv_exists:
        try:
            cache_mtime = os.path.getmtime(CACHE_FILENAME)
            csv_mtime = os.path.getmtime(CSV_FILENAME)
            if cache_mtime >= csv_mtime:
                cache_valid = True
                print(f"Cache file '{CACHE_FILENAME}' is up-to-date.")
            else:
                print(f"Cache file '{CACHE_FILENAME}' is older than '{CSV_FILENAME}'. Regenerating.")
        except Exception as e:
            print(f"Warning: Could not check file modification times. Will regenerate cache. Error: {e}")
            cache_valid = False # Force regeneration if time check fails
    elif cache_exists:
         print(f"Found cache file '{CACHE_FILENAME}', but source CSV '{CSV_FILENAME}' is missing. Using cache.")
         cache_valid = True # Allow using cache if source is gone
    else:
        print(f"Cache file '{CACHE_FILENAME}' not found. Processing data from scratch.")


    if cache_valid:
        try:
            with open(CACHE_FILENAME, 'rb') as f:
                processed_data, transmitter_location_map, map_info = pickle.load(f)
            print(f"Successfully loaded data from cache: {CACHE_FILENAME}")
            # Basic check if loaded data seems okay
            if not processed_data or not transmitter_location_map or not map_info:
                 print("Warning: Cached data seems incomplete. Regenerating.")
                 cache_valid = False # Force regeneration if cache is empty/corrupt
            else:
                # Ensure map path is correctly set (might not be needed if pickled correctly)
                map_info['path'] = MAP_FILENAME
                print(f"Loaded {len(processed_data)} transmitters from cache.")
                # --- Merge Session Data ---
                processed_data = load_and_merge_session_data(processed_data, transmitter_location_map)
                # --- End Merge ---
                return processed_data, transmitter_location_map, map_info
        except (pickle.UnpicklingError, EOFError, FileNotFoundError, TypeError) as e:
            print(f"Error loading from cache file '{CACHE_FILENAME}': {e}. Regenerating data.")
            cache_valid = False # Force regeneration on load error
        except Exception as e: # Catch other potential errors during load
            print(f"An unexpected error occurred loading cache: {e}. Regenerating data.")
            cache_valid = False

    # --- Proceed with loading from CSV if cache is not used ---
    if not cache_valid:
        print("--- Loading and Processing Data from CSV ---")
        try:
            df = pd.read_csv(CSV_FILENAME)
            print(f"Successfully loaded CSV: {CSV_FILENAME}")
        except FileNotFoundError:
            print(f"Error: CSV file '{CSV_FILENAME}' not found.")
            return None, None, None
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None, None, None

        processed_data = defaultdict(lambda: defaultdict(list))
        transmitter_location_map = {}
        map_info = {'path': MAP_FILENAME, 'width': 0, 'height': 0}

        # Load map to get dimensions
        try:
            with Image.open(MAP_FILENAME) as img:
                map_info['width'], map_info['height'] = img.size
            print(f"Loaded map info: {MAP_FILENAME} ({map_info['width']}x{map_info['height']})")
        except FileNotFoundError:
            print(f"Warning: Map file '{MAP_FILENAME}' not found. Proceeding without map dimensions.")
            # Keep default width/height 0
        except Exception as e:
            print(f"Warning: Error loading map file '{MAP_FILENAME}': {e}. Proceeding without map dimensions.")
            # Keep default width/height 0


        # Iterate through the DataFrame rows
        print("Processing walk data points...")
        start_time = time.time() # Start timer
        for row in df.itertuples(index=False):
            transmitter = row.transmitter
            walk = row.walk

            # Map CSV (i, j) to image (col, row)
            img_col = int(row.i)
            img_row = int(row.j)
            rssi = float(row.rssi)

            # Get color for this point
            color = get_rssi_color(rssi)

            # Store receiver location, RSSI, and color
            data_tuple = (img_col, img_row, rssi, color)
            processed_data[transmitter][walk].append(data_tuple)

            # Store transmitter location if not already present
            if transmitter not in transmitter_location_map:
                transmitter_location_map[transmitter] = (int(row.tx_location_i), int(row.tx_location_j))

        # Convert defaultdicts back to regular dicts for pickling
        processed_data = {k: dict(v) for k, v in processed_data.items()}
        end_time = time.time() # End timer
        print(f"Data processing complete in {end_time - start_time:.2f} seconds.")
        print(f"Found {len(processed_data)} transmitters.")
        print(f"Mapped transmitter locations: {transmitter_location_map}")

        # --- Merge Session Data ---
        processed_data = load_and_merge_session_data(processed_data, transmitter_location_map)
        # --- End Merge ---

        # Save the *merged* processed data to the cache file
        try:
            print(f"Attempting to save merged data to cache: {CACHE_FILENAME}")
            with open(CACHE_FILENAME, 'wb') as f:
                pickle.dump((processed_data, transmitter_location_map, map_info), f)
            print(f"Successfully saved data to cache: {CACHE_FILENAME}")
        except Exception as e:
            print(f"Error saving data to cache file '{CACHE_FILENAME}': {e}")

        return processed_data, transmitter_location_map, map_info

if __name__ == '__main__':
    # Example usage if run directly
    walk_data, tx_locs, map_inf = load_data()
    if walk_data:
        print("\n--- Data Loading Test ---")
        print(f"Loaded data for transmitters: {list(walk_data.keys())}")
        # print(f"Transmitter Locations: {tx_locs}")
        print(f"Map Info: {map_inf}")
        # Example: Access data for the first transmitter and first walk
        first_tx = list(walk_data.keys())[0]
        first_walk = list(walk_data[first_tx].keys())[0]
        print(f"\nExample data for {first_tx}, walk {first_walk} (first 5 points):")
        print(walk_data[first_tx][first_walk][:5])

        # Example: Get color for a specific RSSI
        print(f"\nExample color for RSSI -70: {get_rssi_color(-70)}")
        print(f"Example color for RSSI -150: {get_rssi_color(-150)}")
        print(f"Example color for RSSI -1000: {get_rssi_color(-1000)}") 