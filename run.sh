#!/bin/bash

# Function to check and install dependencies
install_dependencies() {
   echo "Checking and installing required dependencies..."
   pip install -r requirement.txt || { echo "Failed to install dependencies!"; exit 1; }
}

echo "Using Python environment: $(which python)"
python --version

# Step 2: Run model downloader first
echo "Running model downloader..."

# Run the model downloader script and check if it completes successfully
python modeldownloader.py || { echo "Model downloader failed!"; exit 1; }

# Step 3: After model downloader completes, navigate to the directory containing QueryProcessing.py
cd Query || { echo "Query directory not found!"; exit 1; }

# Step 4: Start FastAPI in the background
echo "Starting FastAPI app..."
uvicorn QueryProcessing:app --host 0.0.0.0 --port 8088 &

# Step 5: Capture PID to potentially stop later
FASTAPI_PID=$!

# Step 6: Wait for the FastAPI server to become available
echo "Waiting for FastAPI to be available..."
until curl -s http://localhost:8088/docs > /dev/null; do
  sleep 1
done

# Step 7: Navigate to FrontEndDesign to open the HTML page
cd ../FrontEndDesign || { echo "FrontEndDesign directory not found!"; exit 1; }

echo "Opening index.html in your default browser..."
xdg-open index.html 2>/dev/null || open index.html 2>/dev/null || start index.html

# Step 8: Keep the process alive (or allow CTRL+C to stop FastAPI)
echo "FastAPI running in background (PID: $FASTAPI_PID). Press Ctrl+C to stop."

# Optional: Wait for user input to kill the FastAPI process
wait $FASTAPI_PID
