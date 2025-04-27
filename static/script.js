document.addEventListener('DOMContentLoaded', () => {
    const transmitterSelect = document.getElementById('transmitter-select');
    const walkTogglesContainer = document.getElementById('walk-toggles');
    const baseMapImg = document.getElementById('base-map-img');
    const walkCanvas = document.getElementById('walk-canvas');
    const mapContent = document.getElementById('map-content');
    const mapContainer = document.getElementById('map-container');
    const ctx = walkCanvas.getContext('2d');

    let currentTransmitterData = null; // Store walk data for the selected transmitter
    let currentTransmitterLocation = null; // Store location of the selected transmitter
    let visibleWalks = {}; // Store visibility state for walks { walkId: boolean }
    let mapWidth = 0;
    let mapHeight = 0;
    // let panzoomInstance = null; // --- REMOVE panzoom instance

    // --- Custom Pan/Zoom State ---
    let scale = 1;
    let translateX = 0;
    let translateY = 0;
    let isDragging = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let initialTranslateX = 0;
    let initialTranslateY = 0;
    const minScale = 0.1;
    const maxScale = 10;

    // --- Fetch Initial Data ---

    // Fetch transmitters
    fetch('/api/transmitters')
        .then(response => response.json())
        .then(transmitters => {
            transmitterSelect.innerHTML = '<option value="">-- Select Transmitter --</option>'; // Clear loading text
            transmitters.forEach(tx => {
                const option = document.createElement('option');
                option.value = tx;
                option.textContent = tx;
                transmitterSelect.appendChild(option);
            });
        })
        .catch(error => console.error('Error fetching transmitters:', error));

    // Function to apply the current transform
    function applyTransform() {
        // Ensure scale stays within bounds
        scale = Math.max(minScale, Math.min(maxScale, scale));
        // TODO: Add bounding logic for translateX/Y if needed later
        mapContent.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
    }

    // Fetch map info and load map image
    fetch('/api/map/info')
        .then(response => response.json())
        .then(info => {
            mapWidth = info.width;
            mapHeight = info.height;
            baseMapImg.src = '/api/map/image'; // Load the image
            baseMapImg.onload = () => {
                // Set canvas dimensions once the image is loaded
                walkCanvas.width = mapWidth;
                walkCanvas.height = mapHeight;
                // Ensure map-content div matches image size initially
                mapContent.style.width = `${mapWidth}px`;
                mapContent.style.height = `${mapHeight}px`;

                // Set transform origin for scaling (Set only once here)
                mapContent.style.transformOrigin = '0 0';

                // Apply initial transform (centered or default)
                // Optional: Center initially (adjust as needed)
                // translateX = (mapContainer.offsetWidth - mapWidth * scale) / 2;
                // translateY = (mapContainer.offsetHeight - mapHeight * scale) / 2;
                applyTransform(); // Apply initial state

                // --- REMOVE panzoom initialization ---
                // if (!panzoomInstance) {
                //     panzoomInstance = panzoom(mapContent, {
                //         maxZoom: 10,
                //         minZoom: 0.1, // Adjust as needed, might depend on map size
                //         bounds: true, // Prevent panning outside the container
                //         boundsPadding: 0.05, // Small padding
                //         // Autocenter: true, // Center the content initially
                //     });
                //      // Center the map initially after panzoom is setup
                //      panzoomInstance.moveTo(0, 0);
                //      panzoomInstance.zoomAbs(0, 0, 1); // Reset zoom
                //      // You might need a more robust centering logic depending on container size
                //      // panzoomInstance.moveTo(-(mapWidth / 2) + mapContainer.offsetWidth / 2, -(mapHeight / 2) + mapContainer.offsetHeight / 2);

                //      // REMOVE panzoom wheel listener
                //      // mapContainer.addEventListener('wheel', (event) => {
                //      //    // Prevent the page from scrolling
                //      //    event.preventDefault();
                //      //    // Zoom in/out based on wheel direction
                //      //    if (event.deltaY < 0) {
                //      //        panzoomInstance.zoomIn(); // Zoom in
                //      //    } else {
                //      //        panzoomInstance.zoomOut(); // Zoom out
                //      //    }
                //      // });
                // }
                // --- END REMOVE panzoom initialization ---
            };
            baseMapImg.onerror = () => {
                console.error('Error loading base map image.');
                // Handle map loading error (e.g., display a message)
            };
        })
        .catch(error => console.error('Error fetching map info:', error));


    // --- Event Listeners ---

    transmitterSelect.addEventListener('change', (event) => {
        const transmitterId = event.target.value;
        if (transmitterId) {
            fetchWalkData(transmitterId);
        } else {
            // Clear layers if no transmitter is selected
            currentTransmitterData = null;
            currentTransmitterLocation = null; // Clear location too
            walkTogglesContainer.innerHTML = '';
            clearCanvas();
        }
    });

    // --- Custom Pan/Zoom Event Listeners ---

    // Zoom Listener (Wheel) - attached to the container
    mapContainer.addEventListener('wheel', (event) => {
        event.preventDefault(); // Prevent page scroll

        const rect = mapContainer.getBoundingClientRect();
        // Mouse position relative to container
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;

        // --- Calculate Offset --- 
        const containerWidth = mapContainer.offsetWidth;
        const containerHeight = mapContainer.offsetHeight;
        const offsetX = containerWidth * 2 / 3;
        const offsetY = containerHeight * 5 / 6;

        // --- Apply Offset to Mouse Position --- 
        const effectiveMouseX = mouseX + offsetX;
        const effectiveMouseY = mouseY + offsetY;

        // Point on the image under the *effective* mouse position before zoom
        const imageX = (effectiveMouseX - translateX) / scale;
        const imageY = (effectiveMouseY - translateY) / scale;

        // Determine zoom factor
        const zoomFactor = event.deltaY < 0 ? 1.02 : 1 / 1.02; // Reduced zoom factor for smoothness
        const newScale = scale * zoomFactor;

        // Calculate new translation to keep image point under *effective* mouse pos
        translateX = effectiveMouseX - imageX * newScale;
        translateY = effectiveMouseY - imageY * newScale;
        scale = newScale; // Update scale

        applyTransform();
    }, { passive: false }); // Need passive: false to call preventDefault

    // Pan Listeners (Mouse) - attached to the content div that gets transformed
    mapContent.addEventListener('mousedown', (event) => {
        event.preventDefault(); // Prevent text selection/image drag
        isDragging = true;
        dragStartX = event.clientX;
        dragStartY = event.clientY;
        initialTranslateX = translateX;
        initialTranslateY = translateY;
        mapContent.style.cursor = 'grabbing'; // Indicate dragging
    });

    // Attach move/up listeners to the window/document to catch events
    // even if the cursor moves outside the mapContent during a drag.
    document.addEventListener('mousemove', (event) => {
        if (!isDragging) return;

        const deltaX = event.clientX - dragStartX;
        const deltaY = event.clientY - dragStartY;

        translateX = initialTranslateX + deltaX;
        translateY = initialTranslateY + deltaY;

        applyTransform();
    });

    document.addEventListener('mouseup', () => {
        if (!isDragging) return;
        isDragging = false;
        mapContent.style.cursor = 'grab'; // Reset cursor
    });

     // Optional: Reset cursor if mouse leaves the window while dragging
    document.addEventListener('mouseleave', () => {
        if (isDragging) {
            isDragging = false;
            mapContent.style.cursor = 'grab';
        }
    });

    // Set initial cursor style
    mapContent.style.cursor = 'grab';


    // --- Functions ---

    function fetchWalkData(transmitterId) {
        fetch(`/api/walk_data/${transmitterId}`)
            .then(response => response.json())
            .then(data => {
                currentTransmitterData = data.walks; // Extract walk data
                currentTransmitterLocation = data.transmitter_location; // Extract location
                visibleWalks = {}; // Reset visibility state
                createWalkToggles();
                drawWalkPoints(); // Draw points and potentially the transmitter location
            })
            .catch(error => console.error(`Error fetching walk data for ${transmitterId}:`, error));
    }

    function createWalkToggles() {
        walkTogglesContainer.innerHTML = ''; // Clear previous toggles
        if (!currentTransmitterData) return;

        const walkIds = Object.keys(currentTransmitterData).sort((a, b) => parseInt(a) - parseInt(b));

        walkIds.forEach(walkId => {
            visibleWalks[walkId] = true; // Default to visible

            const div = document.createElement('div');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `walk-${walkId}`;
            checkbox.value = walkId;
            checkbox.checked = true;
            checkbox.addEventListener('change', handleToggleChange);

            const label = document.createElement('label');
            label.htmlFor = `walk-${walkId}`;
            // Get number of points for the label
            const pointCount = currentTransmitterData[walkId]?.length || 0;
            label.textContent = ` Walk ${walkId} (${pointCount} points)`;

            div.appendChild(checkbox);
            div.appendChild(label);
            walkTogglesContainer.appendChild(div);
        });
    }

    function handleToggleChange(event) {
        const walkId = event.target.value;
        visibleWalks[walkId] = event.target.checked;
        drawWalkPoints(); // Redraw canvas based on new visibility
    }

    function clearCanvas() {
        ctx.clearRect(0, 0, walkCanvas.width, walkCanvas.height);
    }

    function drawWalkPoints() {
        clearCanvas();
        if (!mapWidth || !mapHeight) return; // Need map dimensions

        // Ensure canvas size is correct
        if (walkCanvas.width !== mapWidth || walkCanvas.height !== mapHeight) {
             walkCanvas.width = mapWidth;
             walkCanvas.height = mapHeight;
        }

        // Settings for drawing points (adjust size as needed)
        const pointSize = 2; // Diameter of the points
        // const transmitterMarkerSize = 3; // REMOVED - No longer needed for square
        const transmitterCircleRadius = 20; // Radius for the dashed circle (was 10)
        const transmitterCircleDashStep = 2; // Draw every 2nd pixel approx (was 5)

        // Draw Walk Points (if data exists)
        if (currentTransmitterData) {
            Object.keys(visibleWalks).forEach(walkId => {
                if (visibleWalks[walkId] && currentTransmitterData[walkId]) {
                    currentTransmitterData[walkId].forEach(point => {
                        const [col, row, rssi, color] = point;
                        ctx.fillStyle = color;
                        ctx.beginPath();
                        ctx.arc(col, row, pointSize / 2, 0, 2 * Math.PI);
                        ctx.fill();
                    });
                }
            });
        }

        // Draw Transmitter Location (if data exists)
        if (currentTransmitterLocation) {
            const [txCol, txRow] = currentTransmitterLocation;

            // 1. Draw the single red center pixel
            ctx.fillStyle = 'red';
            ctx.fillRect(txCol, txRow, 1, 1); // Draw a 1x1 pixel

            // 2. Draw the dashed hot pink circle
            ctx.fillStyle = 'orange'; // Changed from 'hotpink'
            const radius = transmitterCircleRadius;
            const circumference = 2 * Math.PI * radius;
            const numSteps = Math.round(circumference); // Number of steps around the circle

            for (let i = 0; i < numSteps; i++) {
                // Only draw if it's part of the "dash"
                if (i % transmitterCircleDashStep === 0) {
                    const angle = (i / numSteps) * 2 * Math.PI;
                    const x = txCol + Math.round(radius * Math.cos(angle));
                    const y = txRow + Math.round(radius * Math.sin(angle));
                    ctx.fillRect(x, y, 1, 1); // Draw a 1x1 pixel for the dash
                }
            }
        }
    }

    // Optional: Handle window resize to potentially recenter map or adjust bounds
    // window.addEventListener('resize', () => {
    //     // Recalculate bounds or centering if needed
    // });
}); 