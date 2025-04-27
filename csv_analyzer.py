import pandas as pd
from IPython import embed
from collections import defaultdict
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Directly load the CSV file
df = pd.read_csv('train_data/training_walks.csv')
print(f"Successfully loaded CSV data")

# --- Process Data ---
print("\n--- Processing Data ---")
# Using defaultdict for easier nested structure creation
processed_data = defaultdict(lambda: defaultdict(list))
transmitter_location_map = {}

# Iterate through the DataFrame rows
# Using itertuples for efficiency
for row in df.itertuples(index=False): 
    transmitter = row.transmitter
    walk = row.walk
    # Store only receiver location and RSSI
    data_tuple = (row.i, row.j, row.rssi)
    processed_data[transmitter][walk].append(data_tuple)
    
    # Store transmitter location if not already present
    if transmitter not in transmitter_location_map:
        transmitter_location_map[transmitter] = (row.tx_location_i, row.tx_location_j)

# Convert defaultdicts back to regular dicts for cleaner inspection (optional)
processed_data = {k: dict(v) for k, v in processed_data.items()}
print("Data processing complete.")
print("Transmitter locations mapped.")

# --- Calculate and Print RSSI Stats ---
print("\n--- RSSI Statistics per Transmitter and Walk ---")
for transmitter, walks in processed_data.items():
    print(f"\nTransmitter: {transmitter}")
    for walk, data_points in walks.items():
        # Extract RSSI values
        rssi_values = [point[2] for point in data_points]

        if rssi_values: # Check if there are any data points
            # Convert to pandas Series for easy stats
            rssi_series = pd.Series(rssi_values)
            stats = rssi_series.describe() # Gets count, mean, std, min, 25%, 50%, 75%, max

            # Print the desired stats
            print(f"  Walk: {walk}")
            print(f"    Count: {int(stats['count'])}")
            print(f"    Min:   {stats['min']:.2f}")
            print(f"    Max:   {stats['max']:.2f}")
            print(f"    Mean:  {stats['mean']:.2f}")
            print(f"    Std:   {stats['std']:.2f}")
        else:
            print(f"  Walk: {walk} - No data points")

# --- Prepare for Interactive Session ---
print("\n--- Data Overview ---")
print(f"Total records: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\nTransmitter Location Map:")
print(transmitter_location_map)

# print("\n--- Starting Interactive IPython Session ---")
# print("Available variables:")
# print("  df: pandas DataFrame with the loaded CSV data")
# print("  processed_data: Nested dictionary {transmitter: {walk: [(i, j, rssi), ...]}}")
# print("  transmitter_location_map: Dictionary {transmitter: (tx_i, tx_j)}")
# print("\nType 'exit' or press Ctrl+D to quit.")

# # Embed IPython shell
# embed()

# print("\nExited IPython session.")

# --- Generate Images ---
print("\n--- Generating Walk Images ---")

# Config
map_filename = 'train_data/walkable_mask.png'
output_dir = 'sample_data'
walkable_color = (255, 255, 255)  # White
non_walkable_color = (200, 200, 200) # Light Grey
no_signal_color = (0, 0, 0) # Black for RSSI = -1000
rssi_min_for_colormap = -150 # Weakest signal mapped to colormap start
rssi_max_for_colormap = -50  # Strongest signal mapped to colormap end
colormap = plt.cm.viridis

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
print(f"Created base output directory: {output_dir}")

# Load the background map
try:
    base_map_bw = Image.open(map_filename).convert('L') # Load as grayscale
    base_map_rgb = Image.new("RGB", base_map_bw.size, non_walkable_color) # Start with grey background

    # Make walkable areas (white in original) white in the RGB map
    walkable_pixels = np.array(base_map_bw) > 128 # Find pixels brighter than mid-grey
    base_map_rgb.paste(walkable_color, mask=Image.fromarray(walkable_pixels.astype(np.uint8) * 255))

    print(f"Successfully loaded and prepared background map: {map_filename}")
    map_width, map_height = base_map_rgb.size
    print(f"Map dimensions: {map_width}x{map_height}")

except FileNotFoundError:
    print(f"Error: Map file '{map_filename}' not found. Cannot generate images.")
    exit()
except Exception as e:
    print(f"Error loading or processing map file: {e}")
    exit()

# Normalize RSSI values for colormap
norm = mcolors.Normalize(vmin=rssi_min_for_colormap, vmax=rssi_max_for_colormap)
scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=colormap)

# --- Original Generate Images section follows ---
# Process and save images
total_walks = sum(len(walks) for walks in processed_data.values())
processed_count = 0

for transmitter, walks in processed_data.items():
    tx_dir = os.path.join(output_dir, transmitter)
    os.makedirs(tx_dir, exist_ok=True)
    print(f"  Processing transmitter: {transmitter} (Output: {tx_dir})")

    for walk, data_points in walks.items():
        # Create a fresh copy of the map for this walk
        walk_map = base_map_rgb.copy()
        pixels = walk_map.load() # Access pixel data

        for i, j, rssi in data_points:
            # --- Coordinate Mapping --- 
            # CSV (i, j) -> Image (col, row)
            # Based on visual inspection (temp_5_swap_axes.png), the correct mapping is:
            # CSV 'i' corresponds to image column (x-coordinate, increasing right)
            # CSV 'j' corresponds to image row (y-coordinate, increasing down)
            col = i
            row = j
            # --------------------------

            # Ensure coordinates are within map bounds
            if 0 <= col < map_width and 0 <= row < map_height:
                if rssi == -1000.0:
                    color_tuple = no_signal_color
                else:
                    # Clamp RSSI to the colormap range before mapping
                    clamped_rssi = max(rssi_min_for_colormap, min(rssi, rssi_max_for_colormap))
                    rgba_color = scalar_map.to_rgba(clamped_rssi)
                    # Convert RGBA (0-1) to RGB (0-255) tuple
                    color_tuple = tuple(int(c * 255) for c in rgba_color[:3])

                try:
                    pixels[col, row] = color_tuple
                except IndexError:
                     # This shouldn't happen with the bounds check, but just in case
                     print(f"Warning: Coordinate ({col}, {row}) out of bounds for map size ({map_width}, {map_height}). Skipping point.")
            else:
                 # Optional: Warn if points are outside map bounds
                 # print(f"Warning: Coordinate ({col}, {row}) outside map bounds ({map_width}, {map_height}). Skipping point.")
                 pass # Silently skip points outside the map


        # Save the image
        output_filename = os.path.join(tx_dir, f"walk_{walk:02d}.png")
        walk_map.save(output_filename)

        processed_count += 1
        if processed_count % 32 == 0: # Print progress every 32 walks (approx once per transmitter)
             print(f"    Generated {processed_count}/{total_walks} images...")


print(f"\nImage generation complete. {total_walks} images saved in '{output_dir}'.")