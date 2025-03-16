# Floorball Referee - Ball Detection and Goal Counting System

## Setup Requirements
- Python 3.6 or higher
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- Pygame (`pygame`) for audio playback
- A webcam or camera connected to your computer

## Installation
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install opencv-python numpy pygame
```

## Setting Up Goal Sound
1. Place an MP3 file named `goal_sound.mp3` in the same directory as the script
2. The sound will automatically play when a goal is detected
3. The sound can be any length, but a 20-30 second celebration sound works well
4. You can test the sound by pressing the 's' key during runtime

## Running the Script
1. Connect your camera/webcam to the computer
2. Open a terminal/command prompt
3. Navigate to the script directory:
   ```bash
   cd C:/projects/albia/floorball-referee
   ```
4. Run the script:
   ```bash
   python floorball_video_assist.py
   ```

## Camera Selection
- By default, the script tries to use camera index 1 (usually an external webcam)
- If this fails, it will fall back to camera index 0 (typically a built-in webcam)
- To manually change the camera, edit the `camera_index` variable in the script
- The script uses DirectShow (CAP_DSHOW) for better Windows compatibility

## Instructions for Use
1. **Select Goal Area**: 
   - When the application starts, you'll see a video feed
   - Click on two opposite corners of the goal area to define a rectangle
   - Press 'c' to confirm and proceed to goal detection mode

2. **Goal Detection**: 
   - The script will track the ball in real-time
   - When the ball enters the defined goal area, it registers a goal
   - A celebration sound will play when a goal is detected
   - For a goal to count, the ball must:
     - Enter the goal area
     - Have been outside the goal area for at least 5 seconds
     - Occur after the goal cooldown period has passed
   - The system will continue recording for a moment after the goal
   - A slow-motion replay will show twice after each goal
   - Goal information is displayed on screen and logged to `goal_log.txt`

3. **Control Keys**:
   - `q` - Exit the program
   - `d` - Toggle debug mode (shows/hides all debug windows)
   - `1-4` - Toggle individual debug windows (when debug mode is on)
   - `p` - Toggle performance mode (reduces processing load)
   - `c` - Toggle motion-based detection (not recommended)
   - `u` - Toggle blue ball detection mode
   - `+/-` - Increase/decrease color detection weight (vs. motion weight)
   - `r` - Manually reset ball tracking status
   - `b` - Reset the background model when lighting changes
   - `s` - Test play the goal sound

## Ball Detection Options

### Ball Color Options
The system can detect different colored balls:

1. **White Ball (Default)**: Works well in most conditions but may be challenging on light-colored floors
   - Best used with the pure color detection mode (enabled by default)
   - May struggle with reflections and bright spots

2. **Blue Ball Mode** (press `u` to toggle): Using a blue ball can dramatically improve detection accuracy
   - Provides better contrast against beige/light floors
   - Much less affected by light patches and reflections
   - Recommended option for reliable detection

### Detection Methods
After extensive testing, the following settings work best:

1. **Pure Color Detection** (default): Uses only color information to detect the ball
   - Color weight: 1.0, Motion weight: 0.0
   - Best performance for both white and blue balls
   - Works well even with slow-moving balls
   - Activated by default for best results

2. **Mixed Detection**: Combines color and motion information
   - Not recommended - tends to miss the ball or generate false detections
   - Use only in specific circumstances where color alone doesn't work

## Why Motion Detection Didn't Help
Motion-based detection was found to be less helpful than expected because:

1. It tends to miss slow-moving balls
2. The background subtraction can get confused by minor camera movement
3. It adds processing overhead without improving accuracy
4. Pure color-based tracking proved more reliable, especially with good color contrast

## Recommended Setup for Best Results
- **Use a blue ball** on light-colored floors
- Keep pure color detection mode enabled (default)
- Ensure even lighting without strong direct sunlight or shadows
- Position the camera to minimize reflections on the playing surface
- Reset the background model (`b` key) if lighting conditions change

## Handling Sunlight and Reflections
The system includes special handling for static bright areas like sunlight patches:

- Position stability tracking helps filter out stationary bright spots
- Blue ball detection mode is much less susceptible to light interference
- Draw curtains or blinds to create more consistent lighting if possible

## Performance Optimization
- By default, debug windows are hidden for better performance
- If experiencing slow frame rates:
  1. Make sure debug mode is off (press 'd' to toggle off if needed)
  2. Press 'p' to enable higher performance mode
  3. Close any unnecessary applications running in the background

## Troubleshooting
- **Ball Not Being Detected**:
  - Try using a blue ball and press 'u' to enable blue ball detection
  - Reset the background model with the 'b' key
  - Ensure good lighting with minimal shadows
  
- **False Detections**:
  - Switch to a blue ball for better contrast against the floor
  - Reset the background model with 'b' key after lighting changes
  - Adjust camera position to minimize direct sunlight in the frame
  
- **Poor Performance/Low FPS**:
  - Turn off debug mode by pressing 'd'
  - Enable performance mode by pressing 'p'
  
- **Camera Errors**:
  - The script has automatic camera recovery for most common errors
  - If you see "Failed to grab frame" messages, the script will attempt to reset the connection
  - Ensure no other applications are using your camera
  - Try disconnecting and reconnecting the camera if issues persist

## Log Files
The script generates a log file at `C:/projects/albia/floorball-referee/goal_log.txt` 
containing timestamps of detected goals.
