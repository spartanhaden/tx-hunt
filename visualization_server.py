from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware # Import CORS middleware
import uvicorn
import os
import sys

# Import the data loading function
from data_loader import load_data, MAP_FILENAME

app = FastAPI()

# --- CORS Configuration ---
# Allow requests from your frontend development server
# Replace "http://localhost:8000" or "http://127.0.0.1:8000" if your frontend runs elsewhere
# Or use ["*"] for open access (less secure)
origins = [
    "http://localhost",
    "http://localhost:8000", # Common port for simple servers
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    # Add the origin where you serve your index.html if different
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)


# --- Data Loading ---
# Load data when the server starts
walk_data, transmitter_locations, map_info = load_data()

if walk_data is None:
    print("Critical Error: Failed to load walk data. Server cannot start.")
    sys.exit(1) # Exit if data loading failed

# --- API Endpoints ---

@app.get("/api/transmitters")
async def get_transmitters():
    """Returns a list of available transmitter IDs."""
    return list(walk_data.keys())

@app.get("/api/map/info")
async def get_map_info():
    """Returns the map dimensions."""
    if not map_info or map_info['width'] == 0:
        raise HTTPException(status_code=404, detail="Map info not available")
    return {"width": map_info['width'], "height": map_info['height']}

@app.get("/api/map/image")
async def get_map_image():
    """Returns the base map image file."""
    map_path = map_info.get('path', MAP_FILENAME) # Use path from map_info or default
    if not os.path.exists(map_path):
        raise HTTPException(status_code=404, detail=f"Map image not found at {map_path}")
    # Set cache-control headers to encourage browser caching
    headers = {"Cache-Control": "public, max-age=3600"} # Cache for 1 hour
    return FileResponse(map_path, media_type="image/png", headers=headers)

@app.get("/api/walk_data/{transmitter_id}")
async def get_walk_data(transmitter_id: str):
    """Returns walk data (points with col, row, rssi, color) for a specific transmitter."""
    if transmitter_id not in walk_data:
        raise HTTPException(status_code=404, detail=f"Transmitter '{transmitter_id}' not found")
    
    # Get the specific transmitter's location
    location = transmitter_locations.get(transmitter_id)
    if location is None:
         # Handle case where location might be missing for a valid transmitter
         # Option 1: Raise error
         # raise HTTPException(status_code=404, detail=f"Location for transmitter '{transmitter_id}' not found")
         # Option 2: Return null or an empty object for location
         location_data = None
    else:
        # Ensure location is in a consistent format (e.g., [col, row])
        # Assuming transmitter_locations stores it as a tuple or list [col, row]
        location_data = list(location) # Convert to list if it's a tuple

    # Prepare the response data
    response_data = {
        "walks": walk_data[transmitter_id],
        "transmitter_location": location_data # Include the location
    }

    # Return the pre-processed data including colors and location
    return JSONResponse(content=response_data)

# --- Static Files ---
# Mount the 'static' directory to serve HTML, CSS, JS
# This needs to be AFTER the API routes if you have overlapping paths (like '/')
# But before the root path handler if you want '/' to serve index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    """Serves the main index.html file."""
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found in static directory")
    return FileResponse(index_path)

# --- Run Server ---
if __name__ == "__main__":
    print("--- Starting Visualization Server ---")
    print("Access the visualization at: http://127.0.0.1:8000")
    # Use reload=True for development to automatically restart on code changes
    uvicorn.run("visualization_server:app", host="127.0.0.1", port=8000, reload=True) 