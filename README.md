# Floorball Referee - Ball Detection and Goal Counting System

## Setup Requirements
- Python 3.6 or higher
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- A webcam or camera connected to your computer

## Installation
```bash
pip install opencv-python numpy
```

## Running the Script
1. Connect your camera/webcam to the computer
2. Open a terminal/command prompt
3. Navigate to the script directory:
   ```bash
   cd C:/projects/albia/floorball-referee
   ```
4. Run the script:
   ```bash
   python innebandyDomare.py
   ```

## Camera Selection
- By default, the script tries to use camera index 1 (usually an external webcam)
- If this fails, it will fall back to camera index 0 (typically a built-in webcam)
- To manually change the camera, edit the `camera_index` variable in the script
- The script now uses DirectShow (CAP_DSHOW) for better Windows compatibility

## Instructions for Use
1. **Select Goal Area**: 
   - When the application starts, you'll see a video feed
   - Click on two opposite corners of the goal area to define a rectangle
   - Press 'c' to confirm and proceed to goal detection mode

2. **Goal Detection**: 
   - The script will track the ball in real-time
   - When the ball enters the defined goal area, it registers a goal
   - A slow-motion replay will show after each goal
   - Goal information is displayed on screen and logged to `goal_log.txt`

3. **Control Keys**:
   - Press 'q' to exit the program
   - Press 'd' to toggle debug view (shows contours and masks)
   - Press 'a' to toggle alternative ball detection method for difficult lighting

## Troubleshooting
- **Ball Not Being Detected**:
  - Toggle the alternative detection method with 'a' key
  - The script now shows debugging views to help adjust detection parameters
  - If needed, adjust the HSV values (`lower_white` and `upper_white`) for your specific ball/lighting
  - Ensure good lighting with minimal shadows for best detection
  
- **Camera Errors**:
  - The script now has automatic camera recovery for most common errors
  - If you see "Failed to grab frame" messages, the script will attempt to reset the connection
  - Ensure no other applications are using your camera
  - Try disconnecting and reconnecting the camera if issues persist

- **Detection Sensitivity**:
  - For fine-tuning, adjust `min_ball_radius`, `max_ball_radius`, and `ball_detection_confidence` variables

## Log Files
The script generates a log file at `C:/projects/albia/floorball-referee/goal_log.txt` 
containing timestamps of detected goals.

## Tips for Best Performance
- Use good, consistent lighting
- Maximize contrast between the ball and background
- For a beige floor surface, try increasing the brightness of your scene
- Position the camera to minimize glare on the playing surface
