# Floorball Referee - Technical Overview

## Program Purpose
The Floorball Referee program is an automated assistant for detecting goals in floorball games. It tracks a ball in real-time using computer vision, detects when the ball enters a user-defined goal area, and records goals with sound alerts and replay.

## Core Algorithms and Techniques

### Ball Detection System
The program uses a multi-stage approach to detect the ball reliably:

1. **Color-Based Detection**
   - Ball is identified using HSV (Hue-Saturation-Value) color filtering
   - Separate color ranges for white ball and blue ball options
   - Morphological operations (erosion, dilation) clean up the binary mask

2. **Hough Circle Detection**
   - Mathematical algorithm to find circular shapes in the image
   - Provides robust shape-based detection complementary to color detection
   - Parameters tuned to detect ball-sized circles while filtering noise

3. **Contour Analysis**
   - Detects connected regions in the binary mask
   - Analyzes properties like circularity, area, and radius
   - Penalizes non-circular shapes to filter out noise

4. **Temporal Consistency**
   - Requires multiple consecutive detections of ball in similar positions
   - Prevents random spots from being falsely identified as the ball
   - Maintains ball tracking through brief occlusions or detection failures

5. **Stability Analysis**
   - Monitors position variance to distinguish between moving balls and stationary objects
   - Specifically targets sunlight spots and reflections that might appear ball-like
   - Penalizes objects that remain in the same position for extended periods

### Scoring System
Each ball candidate is scored based on multiple factors:
- **Shape score**: How circular the object is (0-1)
- **Size score**: How close the object's radius is to expected ball size
- **Color score**: How well the object's color matches ball color
- **Motion score**: Whether the object is moving (vs stationary objects)
- **Temporal score**: Consistency with previous ball positions
- **Stability score**: Penalizes objects that don't move over time

### Goal Detection Logic

1. **Goal Area Definition**
   - User defines goal area by clicking two opposite corners
   - Creates a rectangular region of interest

2. **Goal Validation Rules**
   - Ball must enter the defined goal area
   - Ball must have been outside goal area for at least 5 seconds
   - Previous goal's cooldown period must have elapsed
   - Celebration sound must have finished playing

3. **Post-Goal Process**
   - Sound effect plays
   - System records additional frames after goal
   - Slow-motion replay shown twice
   - Goal logged with timestamp for record-keeping

## Integration of Components

1. **Input Processing**
   - Camera frame is captured and converted to HSV and grayscale
   - Various image masks are generated for detection

2. **Detection Pipeline**
   - Color filter identifies potential ball regions
   - Shape analysis confirms circular objects
   - Temporal tracking prevents false positives
   - Final candidate scoring determines most likely ball

3. **Game State Machine**
   - Tracks ball's relation to goal area over time
   - Maintains state for pre-goal requirements, goal events, and post-goal celebration
   - Controls goal readiness based on timing rules

4. **User Interface**
   - Displays detected ball, goal area, and system status
   - Shows "GOAL READY" indicator when system can detect goals
   - Provides debug views for analyzing detection performance

## Optimizations

1. **Performance Tuning**
   - Frame processing rate adjusted based on system capability
   - Debug windows scaled down for better performance
   - Motion detection limited to essential frames

2. **False Positive Reduction**
   - Multi-frame consistency checks filter momentary noise
   - Stricter circularity and size requirements
   - Position variance analysis to eliminate static objects

3. **Detection Success Rate**
   - Pure color mode for slow-moving balls
   - Blue ball option for better contrast on light floors
   - Adjustable parameters for different playing environments
