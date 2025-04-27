import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Configuration
csv_file = 'train_data/training_walks.csv'
map_file = 'train_data/walkable_mask.png'
output_image_file = 'transmitter_locations.png'

# --- Load Data ---
print(f"Loading CSV data from: {csv_file}")
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_file}")
    exit()

print(f"Loading map image from: {map_file}")
try:
    img = Image.open(map_file)
except FileNotFoundError:
    print(f"Error: Map file not found at {map_file}")
    exit()

# --- Flip the map image vertically ---
print("Flipping map image vertically...")
img = img.transpose(Image.FLIP_TOP_BOTTOM)

# --- Find Unique Transmitter Locations ---
print("Finding unique transmitter locations...")
# Select relevant columns and drop duplicates
unique_transmitters = df[['transmitter', 'tx_location_i', 'tx_location_j']].drop_duplicates()
print(f"Found {len(unique_transmitters)} unique transmitter locations.")
# print(unique_transmitters) # Optional: Print the found locations

# --- Create Plot ---
print("Generating plot...")
fig, ax = plt.subplots(figsize=(12, 12 * img.height / img.width)) # Adjust figure size based on image aspect ratio
image_height = img.height # Get image height for coordinate flipping

# Display the map image
ax.imshow(img)

# Plot each transmitter location and add a label
for _, row in unique_transmitters.iterrows():
    transmitter_name = row['transmitter']
    tx_i = row['tx_location_i'] # x-coordinate (pixels from left)
    original_tx_j = row['tx_location_j'] # y-coordinate (pixels from top, before flip)
    tx_j = image_height - original_tx_j # Flipped y-coordinate (pixels from bottom)

    # Plot a marker (e.g., red circle) at the flipped location
    ax.plot(tx_i, tx_j, 'ro', markersize=5, label=transmitter_name if _ == 0 else "") # Only label once for legend

    # Add text label near the marker at the flipped location
    ax.text(tx_i + 5, tx_j + 5, transmitter_name, color='red', fontsize=8, ha='left', va='bottom')

# --- Customize and Save Plot ---
ax.set_title('Transmitter Locations on Walkable Mask (Y-axis Flipped)')
ax.set_xticks([]) # Hide x-axis ticks
ax.set_yticks([]) # Hide y-axis ticks
# ax.legend() # Optional: Add a legend if desired (might get cluttered)

plt.tight_layout()
print(f"Saving plot to: {output_image_file}")
plt.savefig(output_image_file, dpi=300, bbox_inches='tight')

print("Plot saved successfully.")
# plt.show() # Optional: Display the plot interactively 