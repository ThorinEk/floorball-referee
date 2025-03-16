import cv2
import numpy as np
import time
import logging
import os
import pygame  # Add this import for sound playback
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("goal_log.txt"),
        logging.StreamHandler()
    ]
)

class FloorballReferee:
    def __init__(self):
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        self.goal_sound = None
        self.sound_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goal_sound.mp3")
        
        # Add property to track when goal celebration is complete
        self.celebration_complete = True  # Initially true since no celebration is happening
        self.goal_cooldown = 5  # seconds between possible goals (base cooldown)
        self.sound_duration = 0  # Will be set when sound is loaded
        
        # Try to load the goal sound
        try:
            self.goal_sound = pygame.mixer.Sound(self.sound_file)
            # Get sound duration in seconds
            self.sound_duration = self.goal_sound.get_length()
            print(f"Goal sound loaded successfully: {self.sound_file} (Duration: {self.sound_duration:.1f}s)")
        except Exception as e:
            print(f"Could not load goal sound: {e}")
            print(f"Please ensure '{self.sound_file}' exists")
        
        # Flag to track if sound is currently playing
        self.sound_playing = False
        
        self.camera = None
        self.camera_index = 1  # Try external camera first
        self.goal_area = None
        self.goal_count = 0
        self.last_goal_time = 0
        # Increase goal cooldown to prevent rapid succession goals
        self.goal_cooldown = 5  # seconds between possible goals
        self.show_replay = False
        self.replay_frames = []
        self.replay_index = 0
        self.replay_count = 0
        self.max_replay_count = 2  # Only show replay twice
        
        # Parameters for ball detection
        self.min_ball_radius = 5
        self.max_ball_radius = 30
        self.ball_detection_confidence = 0.75
        
        # Create color ranges that work better with beige background
        # HSV range for white ball (wider range to account for lighting)
        self.lower_white = np.array([0, 0, 160])  # Lower value threshold for better detection
        self.upper_white = np.array([180, 60, 255])  # Higher saturation threshold
        
        # Alternative detection parameters
        self.use_alternative_detection = False
        self.ball_history = []  # Track recent ball positions for smoothing

        # Add post-goal recording duration
        self.post_goal_frames = 15  # Frames to record after goal detection
        self.recording_post_goal = False
        self.post_goal_counter = 0

        # Add variables to track when ball leaves goal area
        self.ball_left_goal_area = True
        self.ball_left_goal_time = 0
        self.required_time_outside_goal = 5  # seconds ball must be outside goal before new goal counts

        # Add background subtraction for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=16, detectShadows=False)
        self.prev_frame = None
        self.min_movement_area = 20  # Minimum area for movement detection
        self.ball_confidence_threshold = 0.5  # Lowered threshold to detect ball more easily
        self.motion_weight = 0.5  # Reduced weight for motion (was 0.7) to detect slower movements
        self.frame_counter = 0  # Counter for processing every N frames
        self.process_every_n_frames = 2  # Process every 2nd frame for motion detection
        self.last_valid_detection = None
        self.frames_since_valid_detection = 0
        self.max_frames_without_detection = 15  # Increased to maintain tracking longer
        
        # Add performance monitoring
        self.fps_values = []
        self.last_fps_update = time.time()
        self.fps = 0
        self.fps_update_interval = 1.0  # Update FPS every second
        
        # Debug window control
        self.debug_mode = False  # Set to False by default to hide debug windows
        self.show_motion_mask = False
        self.show_color_mask = False
        self.show_combined_mask = False
        self.show_candidates = False
        self.debug_scale = 0.5  # Scale factor for debug windows (smaller = better performance)

        # Add parameters to control detection balance between color and motion
        self.use_pure_color_detection = False  # Toggle to use only color-based detection
        self.color_detection_weight = 0.9  # Default to higher color weight (better for white on beige)
        self.motion_detection_weight = 0.1  # Lower motion weight
        
        # Add tracking for stationary objects to filter out sunlight spots
        self.stationary_objects = []  # List to track potentially stationary objects
        self.stationary_threshold = 10  # Number of frames to consider an object stationary
        self.position_stability_radius = 10  # Maximum movement (in pixels) still considered "same position"
        self.stationary_penalty = 0.5  # Penalty factor for stationary objects
        self.recent_positions = []  # Track recent positions to detect movement pattern
        self.frames_to_track = 15  # Number of frames to track for movement analysis

        # Update default values based on user experience
        self.color_detection_weight = 1.0  # Start with pure color detection by default
        self.motion_detection_weight = 0.0  # No motion weight by default
        self.use_pure_color_detection = True  # Enable pure color mode by default
        
        # Add blue ball detection capability
        self.use_blue_ball = False  # Toggle for blue ball detection
        # HSV range for blue ball
        self.lower_blue = np.array([100, 50, 50])  # Blue hue, moderate saturation and value
        self.upper_blue = np.array([130, 255, 255])  # Upper blue range

        # Add additional parameters for stricter ball detection and false positive reduction
        self.min_circularity = 0.65  # Minimum circularity for a ball (0-1 where 1 is perfect circle)
        self.consecutive_detections_required = 3  # Number of consecutive frames to confirm detection
        self.consistent_detections = []  # Track recent consistent detections
        self.use_hough_circles = True  # Enable Hough circle detection for better shape filtering
        self.hough_param1 = 50  # First parameter of Hough Circle detection (higher = fewer circles)
        self.hough_param2 = 30  # Second parameter (lower = more circles)
        self.hough_min_dist = 20  # Minimum distance between detected circles
        self.min_detection_confidence = 0.7  # Minimum confidence score to display a detection

    def initialize_camera(self):
        """Initialize the camera with error handling and retry capability"""
        try:
            # Try the preferred camera index
            self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)  # Use DirectShow API
            
            # Check if camera opened successfully
            if not self.camera.isOpened():
                print(f"Could not open camera at index {self.camera_index}. Trying fallback camera...")
                self.camera_index = 0
                self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                
            # If still not opened, raise exception
            if not self.camera.isOpened():
                raise Exception("Could not open any camera")
                
            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to get recent frames
            
            print(f"Successfully opened camera at index {self.camera_index}")
            return True
        
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False

    def reset_camera(self):
        """Reset the camera if it's encountering issues"""
        if self.camera is not None:
            self.camera.release()
        
        time.sleep(1)  # Wait before reconnecting
        return self.initialize_camera()

    def select_goal_area(self):
        """Let user select the goal area with mouse clicks"""
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                if len(points) == 2:
                    cv2.destroyWindow("Select Goal Area")
        
        cv2.namedWindow("Select Goal Area")
        cv2.setMouseCallback("Select Goal Area", mouse_callback)
        
        while len(points) < 2:
            ret, frame = self.camera.read()
            if not ret:
                if not self.reset_camera():
                    print("Failed to reset camera. Exiting.")
                    return False
                continue
                
            # Draw points already selected
            for pt in points:
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)
            
            cv2.putText(frame, "Click to select two corners of the goal area", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Select Goal Area", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
        
        # Create the goal area from the two points
        x1, y1 = points[0]
        x2, y2 = points[1]
        self.goal_area = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        
        return True

    def detect_ball(self, frame):
        """Detect the white ball using improved methods with motion detection and shape analysis"""
        # Record time for performance measurement
        start_time = time.time()
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Increment frame counter
        self.frame_counter += 1
        
        # Define kernel for morphological operations first, so it's available everywhere
        kernel = np.ones((3, 3), np.uint8)
        
        # Apply background subtraction to detect motion
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Process motion detection every N frames to reduce noise
        if self.frame_counter % self.process_every_n_frames == 0:
            # Apply morphological operations to clean up the mask
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Compute absolute difference between current and previous frame for motion detection
        motion_mask = np.zeros_like(gray)
        if self.prev_frame is not None:
            frame_diff = cv2.absdiff(gray, self.prev_frame)
            # Lower threshold to detect slower movements (was 15)
            motion_mask = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)[1]
            motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)
            
            # Combine motion detection with background subtraction - use OR instead of AND to be more permissive
            if not self.use_pure_color_detection:
                # Use weighted combination instead of strict AND
                # This allows detection when either motion OR background subtraction detects something
                fg_mask = cv2.addWeighted(fg_mask, 0.7, motion_mask, 0.3, 0)
                # Apply threshold to make it binary again
                _, fg_mask = cv2.threshold(fg_mask, 30, 255, cv2.THRESH_BINARY)
        
        # Update previous frame
        self.prev_frame = gray.copy()
        
        # Generate appropriate color mask based on ball color
        if self.use_blue_ball:
            # Use HSV range for blue ball
            color_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
            # Show indicator that we're detecting blue ball
            if self.debug_mode:
                print("Using blue ball detection mode")
        else:
            # Standard HSV-based mask for white ball - more permissive for beige background
            color_mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Improve detection with morphological operations
        # Note: kernel is already defined above
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        # If using pure color detection mode, just use the color mask
        # Otherwise combine color and motion information
        if self.use_pure_color_detection:
            combined_mask = color_mask
        else:
            # Create weighted combination of color and motion masks
            # Higher weight to color for white ball on beige background
            combined_mask = cv2.addWeighted(
                color_mask, self.color_detection_weight, 
                fg_mask, self.motion_detection_weight, 0
            )
            # Convert back to binary
            _, combined_mask = cv2.threshold(combined_mask, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Only show debug windows if debug mode is enabled
        if self.debug_mode:
            # Scale down debug windows for better performance
            debug_scale = self.debug_scale
            
            if self.show_motion_mask:
                small_fg_mask = cv2.resize(fg_mask, (0, 0), fx=debug_scale, fy=debug_scale)
                cv2.imshow("Motion Mask", small_fg_mask)
            
            if self.show_color_mask:
                small_color_mask = cv2.resize(color_mask, (0, 0), fx=debug_scale, fy=debug_scale)
                cv2.imshow("Color Mask", small_color_mask)
            
            if self.show_combined_mask:
                small_combined_mask = cv2.resize(combined_mask, (0, 0), fx=debug_scale, fy=debug_scale)
                cv2.imshow("Combined Mask", small_combined_mask)
            
            if self.show_candidates:
                debug_frame = frame.copy()
        
        # Sort contours by their potential to be the ball
        ball_candidates = []
        
        # Create a copy of the original frame for Hough circle detection
        original_gray = gray.copy()
        
        # Two-stage detection:
        # 1. Find potential candidates from contour analysis (existing)
        ball_candidates = []
        
        # First stage: Find candidates based on contour analysis
        for contour in contours:
            # Filter contours that could be the ball
            area = cv2.contourArea(contour)
            if area < 10:  # Ignore very small contours
                continue
                
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Apply stricter circularity threshold to reduce floor noise
            if circularity < self.min_circularity:
                continue  # Skip non-circular objects
                
            # Find the minimum enclosing circle
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            
            # Skip unreasonably tiny detections as they're likely noise
            if radius < 7:  # Slightly increase minimum radius to filter floor spots
                continue
                
            # Score the candidate based on multiple factors
            shape_score = circularity  # Perfect circle has circularity = 1
            size_score = 1 - abs(radius - 12) / 15  # Normalize size score (optimal radius around 12px)
            
            # Calculate motion score based on average pixel intensity in motion mask
            # at the ball position
            mask_roi = fg_mask[max(0, int(y-radius)):min(frame.shape[0], int(y+radius)), 
                              max(0, int(x-radius)):min(frame.shape[1], int(x+radius))]
            if mask_roi.size > 0:
                motion_score = np.mean(mask_roi) / 255
            else:
                motion_score = 0
            
            # Calculate color score based on selected ball color
            hsv_roi = hsv[max(0, int(y-radius)):min(frame.shape[0], int(y+radius)), 
                         max(0, int(x-radius)):min(frame.shape[1], int(x+radius))]
            if hsv_roi.size > 0:
                if self.use_blue_ball:
                    # For blue ball: look for blue hue with good saturation
                    blue_pixels = np.sum((hsv_roi[:,:,0] > 100) & (hsv_roi[:,:,0] < 130) & 
                                        (hsv_roi[:,:,1] > 50)) / (hsv_roi.shape[0] * hsv_roi.shape[1])
                    color_score = blue_pixels
                else:
                    # For white ball: look for low saturation and high value
                    white_pixels = np.sum((hsv_roi[:,:,1] < 60) & (hsv_roi[:,:,2] > 160)) / (hsv_roi.shape[0] * hsv_roi.shape[1])
                    color_score = white_pixels
            else:
                color_score = 0
            
            # Calculate temporal consistency score if we have history
            temporal_score = 0
            if self.ball_history:
                last_pos = self.ball_history[-1]
                # Calculate distance to last known position
                distance = np.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)
                # Normalize distance score - closer is better
                max_expected_distance = 50  # Maximum expected movement between frames
                temporal_score = max(0, 1 - (distance / max_expected_distance))
            else:
                temporal_score = 0.5  # Neutral score if no history
            
            # Weighted total score
            total_score = (0.3 * shape_score + 0.25 * size_score + 
                          0.15 * motion_score + 0.2 * color_score + 
                          0.1 * temporal_score)
            
            # Calculate position stability score
            position = (int(x), int(y))
            stability_score = self.calculate_position_stability(position)
            
            # Adjusted weights for beige floor - prioritize shape and color
            # Include stability score to penalize stationary objects
            total_score *= stability_score  # Multiply by stability score as a penalty factor
            
            ball_candidates.append({
                'position': (int(x), int(y), int(radius)),
                'score': total_score,
                'area': area,
                'circularity': circularity,
                'stability': stability_score
            })
        
        # Second stage: Also try Hough Circle detection for better shape analysis
        if self.use_hough_circles:
            # Apply blur to reduce noise for Hough detection
            blurred = cv2.GaussianBlur(original_gray, (5, 5), 0)
            
            # Detect circles using Hough transform
            circles = cv2.HoughCircles(
                blurred, 
                cv2.HOUGH_GRADIENT, 
                dp=1, 
                minDist=self.hough_min_dist,
                param1=self.hough_param1,
                param2=self.hough_param2,
                minRadius=self.min_ball_radius,
                maxRadius=self.max_ball_radius
            )
            
            # If circles are found, add them to candidates
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                for (x, y, radius) in circles:
                    # For each Hough circle, check if it matches our color criteria
                    if radius < self.min_ball_radius or radius > self.max_ball_radius:
                        continue
                    
                    # Calculate color score based on selected ball color
                    hsv_roi = hsv[max(0, y-radius):min(frame.shape[0], y+radius), 
                                 max(0, x-radius):min(frame.shape[1], x+radius)]
                    
                    if hsv_roi.size > 0:
                        if self.use_blue_ball:
                            blue_pixels = np.sum((hsv_roi[:,:,0] > 100) & (hsv_roi[:,:,0] < 130) & 
                                                (hsv_roi[:,:,1] > 50)) / (hsv_roi.shape[0] * hsv_roi.shape[1])
                            color_score = blue_pixels
                        else:
                            white_pixels = np.sum((hsv_roi[:,:,1] < 60) & (hsv_roi[:,:,2] > 160)) / (hsv_roi.shape[0] * hsv_roi.shape[1])
                            color_score = white_pixels
                    else:
                        color_score = 0
                    
                    # If the circle doesn't match our color criteria well, skip it
                    if color_score < 0.3:  # Require minimum color match for Hough circles
                        continue
                    
                    # Give Hough circles a high circularity score since they're algorithmically detected as circles
                    circularity = 0.9
                    
                    # Calculate temporal consistency score
                    temporal_score = 0
                    if self.ball_history:
                        last_pos = self.ball_history[-1]
                        distance = np.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)
                        max_expected_distance = 50
                        temporal_score = max(0, 1 - (distance / max_expected_distance))
                    else:
                        temporal_score = 0.5
                    
                    # Calculate position stability score
                    position = (int(x), int(y))
                    stability_score = self.calculate_position_stability(position)
                    
                    # Hough circles get a good default shape and size score
                    shape_score = 0.9  # High shape score for Hough circles
                    size_score = 1 - abs(radius - 12) / 15
                    
                    # Overall score emphasizes color and shape for Hough circles
                    total_score = (0.25 * shape_score + 0.25 * size_score + 
                                   0.1 * temporal_score + 0.4 * color_score) * stability_score
                    
                    # Add as a candidate
                    ball_candidates.append({
                        'position': (int(x), int(y), int(radius)),
                        'score': total_score,
                        'area': np.pi * radius * radius,  # Approximate area
                        'circularity': circularity,
                        'stability': stability_score,
                        'source': 'hough'  # Mark the source for debugging
                    })
        
        # Debug drawing - only if in debug mode and candidates window is enabled
        if self.debug_mode and self.show_candidates and 'debug_frame' in locals():
            for i, candidate in enumerate(ball_candidates):
                x, y, radius = candidate['position']
                score = candidate['score']
                color = (0, 255 * score, 255 * (1 - score))  # Color based on score
                cv2.circle(debug_frame, (x, y), radius, color, 2)
                cv2.putText(debug_frame, f"{score:.2f}", (x, y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Scale down debug frame for better performance
            small_debug_frame = cv2.resize(debug_frame, (0, 0), fx=debug_scale, fy=debug_scale)
            cv2.imshow("Ball Candidates", small_debug_frame)
        
        # Select the best candidate with improved filtering
        ball_position = None
        candidates_with_info = []  # Store candidates with scores for access from main loop
        
        if ball_candidates:
            # Sort by score
            ball_candidates.sort(key=lambda x: x['score'], reverse=True)
            best_candidate = ball_candidates[0]
            
            # Apply stricter filtering to reduce false positives
            if best_candidate['score'] >= self.ball_confidence_threshold:
                # Get the position of the best candidate
                new_pos = best_candidate['position']
                
                # Use temporal consistency across multiple frames to validate detection
                # Store this detection in the consistent_detections list
                self.consistent_detections.append(new_pos)
                
                # Keep only the most recent detections
                if len(self.consistent_detections) > self.consecutive_detections_required:
                    self.consistent_detections.pop(0)
                
                # If we have enough consecutive detections, check consistency
                if len(self.consistent_detections) >= self.consecutive_detections_required:
                    # Check if the positions are relatively stable (not jumping around)
                    # Calculate average position
                    if self.check_detection_stability(self.consistent_detections, max_distance=30):
                        # Only assign ball_position if we have stable consecutive detections
                        ball_position = new_pos
                        self.last_valid_detection = ball_position
                        self.frames_since_valid_detection = 0
                        
                        # Save all candidate info for use in main loop
                        candidates_with_info = ball_candidates.copy()
                        
                        # Update position history with valid detection
                        self.ball_history.append(ball_position)
                        if len(self.ball_history) > 10:
                            self.ball_history.pop(0)
                            
                        # Update recent positions for stability tracking
                        self.recent_positions.append((ball_position[0], ball_position[1]))
                        if len(self.recent_positions) > self.frames_to_track:
                            self.recent_positions.pop(0)
                
                # Even if we don't have enough consistent detections yet, store this for debugging
                if new_pos[2] > 7 and best_candidate['score'] > 0.7:  # Only for reasonable candidates
                    candidates_with_info = ball_candidates.copy()
            else:
                # Reset consecutive detections if no good detection in this frame
                self.consistent_detections = []
                self.frames_since_valid_detection += 1
        else:
            # Reset consecutive detections if no candidates found
            self.consistent_detections = []
            self.frames_since_valid_detection += 1
        
        # If we've lost tracking but had a recent valid detection, use that
        if (ball_position is None and self.last_valid_detection is not None and 
            self.frames_since_valid_detection < self.max_frames_without_detection):
            ball_position = self.last_valid_detection
        
        # Update FPS calculation
        processing_time = time.time() - start_time
        self.fps_values.append(1.0 / max(processing_time, 0.001))  # Avoid division by zero
        
        # Calculate average FPS over time
        current_time = time.time()
        if current_time - self.last_fps_update > self.fps_update_interval:
            self.fps = np.mean(self.fps_values)
            self.fps_values = []
            self.last_fps_update = current_time
            
        return ball_position, candidates_with_info  # Return both position and candidate data

    def calculate_position_stability(self, current_position):
        """
        Calculate a score that penalizes objects that stay in the same position
        Returns 1.0 for moving objects and a lower score for stationary objects
        """
        if not self.recent_positions:
            return 1.0  # No history, assume moving
        
        # Check if the current position is similar to previous positions
        x, y = current_position
        similar_positions = 0
        total_positions = len(self.recent_positions)
        
        for pos in self.recent_positions:
            prev_x, prev_y = pos
            distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            if distance < self.position_stability_radius:
                similar_positions += 1
        
        # Calculate stability ratio - how often the object has been in this position
        stability_ratio = similar_positions / total_positions if total_positions > 0 else 0
        
        # If the object has been in roughly the same position for a while, penalize it
        if stability_ratio > 0.8:  # 80% of recent frames in the same position
            # Calculate movement variance to differentiate between totally static and slightly moving objects
            if len(self.recent_positions) >= 3:
                positions = np.array(self.recent_positions)
                variance = np.var(positions, axis=0).sum()  # Sum of variance in x and y directions
                
                # If almost no variance, likely a static bright spot like sunlight
                if variance < 5:  # Very low variance threshold
                    return 0.2  # Heavily penalize static objects
                elif variance < 20:  # Low variance but some movement
                    return 0.5  # Moderate penalty for slow-moving objects
                else:
                    return 0.8  # Slight penalty for objects that move but return to same area
            
            return 0.5  # Default penalty if not enough history
        
        return 1.0  # No penalty for moving objects

    def check_detection_stability(self, detections, max_distance=30):
        """Check if a series of detections are stable (not jumping around)"""
        if len(detections) < 2:
            return True
            
        # Calculate pairwise distances between consecutive detections
        for i in range(1, len(detections)):
            prev_x, prev_y, _ = detections[i-1]
            curr_x, curr_y, _ = detections[i]
            
            # Calculate distance
            distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            
            # If any consecutive detections are too far apart, consider unstable
            if distance > max_distance:
                return False
        
        # All distances are within acceptable range
        return True

    def is_goal(self, ball_position):
        """Check if the ball is in the goal area"""
        if not ball_position or not self.goal_area:
            return False
            
        x, y, _ = ball_position
        x1, y1, x2, y2 = self.goal_area
        
        return x1 <= x <= x2 and y1 <= y <= y2

    def show_goal_replay(self):
        """Show slow-motion replay of the goal, but only twice"""
        if not self.replay_frames:
            return
            
        # Display replay frames at a slower pace
        frame = self.replay_frames[self.replay_index]
        
        # Add goal text
        cv2.putText(frame, "GOAL!", (frame.shape[1]//2 - 70, frame.shape[0]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        
        # Display replay count
        cv2.putText(frame, f"Replay {self.replay_count + 1}/2", 
                   (frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Floorball Referee", frame)
        
        # Slower replay speed (higher wait time)
        cv2.waitKey(150)  # Increased from 100 to 150ms for slower replay
        
        self.replay_index += 1
        if self.replay_index >= len(self.replay_frames):
            self.replay_index = 0
            self.replay_count += 1
            
            # Only show replay twice
            if self.replay_count >= self.max_replay_count:
                self.show_replay = False
                self.replay_frames = []
                self.replay_index = 0
                self.replay_count = 0

    def play_goal_sound(self):
        """Play the goal celebration sound"""
        if self.goal_sound and not self.sound_playing:
            try:
                pygame.mixer.stop()  # Stop any currently playing sounds
                self.goal_sound.play()
                self.sound_playing = True
                self.celebration_complete = False  # Start celebration period
                print("Playing goal sound!")
            except Exception as e:
                print(f"Error playing sound: {e}")

    def run(self):
        """Main function to run the application"""
        if not self.initialize_camera():
            print("Failed to initialize camera. Exiting.")
            return
        
        if not self.select_goal_area():
            print("Goal area selection canceled. Exiting.")
            self.camera.release()
            cv2.destroyAllWindows()
            return
        
        print("Starting goal detection...")
        
        # Initialize replay buffer - increase size for longer replay
        replay_buffer_size = 60  # Doubled from 30 to 60 for longer replay
        replay_buffer = []
        
        # Flag to track if ball is in goal area (to prevent multiple counts)
        ball_in_goal = False
        
        # Toggle for debug view
        show_debug = False
        
        # Allow background subtractor to learn initial background
        print("Learning background... Please wait.")
        for _ in range(30):  # Learn from 30 frames
            ret, frame = self.camera.read()
            if ret:
                self.bg_subtractor.apply(frame)
        
        # When initializing
        print("Starting goal detection with optimized settings...")
        print(f"Default color weight: {self.color_detection_weight:.1f}, motion weight: {self.motion_detection_weight:.1f}")
        print("Press 'c' to toggle pure color detection mode")
        print("Press '+'/'-' to adjust color/motion balance")
        print("Press 'd' to show debug windows")
        
        # Before entering main loop, add instructions for the recommended setting
        print("TIP: For best detection of white ball on beige floor, try:")
        print("  - Pure color mode (press 'c')")
        print("  - Or color weight 0.9 (press '+' a few times)")
        print("  - Reset background model if lighting changes (press 'b')")
        
        print("Starting goal detection with improved false-positive filtering")
        if self.use_hough_circles:
            print("Using Hough circle detection for better shape analysis")
        print("Requiring consistent detections across multiple frames")
        
        while True:
            try:
                # Check if sound is done playing
                if self.sound_playing and not pygame.mixer.get_busy():
                    self.sound_playing = False
                    print("Goal sound finished playing")
                    self.celebration_complete = True  # Mark the celebration as complete
                
                if self.show_replay:
                    self.show_goal_replay()
                    # No need for additional waitKey here as it's in the replay function
                    continue
                
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to grab frame. Attempting to reset camera...")
                    if not self.reset_camera():
                        print("Could not recover camera. Exiting.")
                        break
                    continue
                
                # Store frame in replay buffer
                replay_buffer.append(frame.copy())
                if len(replay_buffer) > replay_buffer_size:
                    replay_buffer.pop(0)
                
                # Detect the ball - updated to receive both position and candidates
                detection_result = self.detect_ball(frame)
                if isinstance(detection_result, tuple) and len(detection_result) == 2:
                    ball_position, ball_candidates = detection_result
                else:
                    # Handle case if detect_ball wasn't updated yet
                    ball_position = detection_result
                    ball_candidates = []
                
                # Draw the goal area
                if self.goal_area:
                    x1, y1, x2, y2 = self.goal_area
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw the detected ball
                if ball_position:
                    x, y, radius = ball_position
                    
                    # Find the best candidate's score if available
                    best_score = 0
                    best_source = "unknown"
                    if ball_candidates:
                        for candidate in ball_candidates:
                            if candidate['position'] == ball_position:
                                best_score = candidate['score']
                                if 'source' in candidate:
                                    best_source = candidate['source']
                                break
                        
                    # Only display high confidence detections to reduce flickering
                    if best_score >= self.min_detection_confidence or self.frames_since_valid_detection < self.max_frames_without_detection:
                        cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)
                        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                        
                        # In debug mode, show the detection source and score
                        if self.debug_mode:
                            cv2.putText(frame, f"{best_score:.2f} {best_source}", (x + radius + 5, y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                                       
                            # When drawing detected ball, show stability score
                            if ball_candidates:
                                # Find the stability score if we have it
                                for candidate in ball_candidates:
                                    if candidate['position'] == ball_position and 'stability' in candidate:
                                        stability_text = f"S:{candidate['stability']:.2f}"
                                        cv2.putText(frame, stability_text, (x + radius + 5, y + 15), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                                        break
                
                # Show FPS on the main frame
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 120), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Check for goal with improved logic
                current_time = time.time()
                ball_in_goal_now = ball_position and self.is_goal(ball_position)
                
                # Track when ball leaves the goal area
                if ball_in_goal and not ball_in_goal_now:
                    self.ball_left_goal_area = True
                    self.ball_left_goal_time = current_time
                
                # Reset ball_left_goal_area flag if ball re-enters goal without waiting
                if not self.ball_left_goal_area and ball_in_goal_now:
                    self.ball_left_goal_area = False
                
                # Handle recording post-goal footage
                if self.recording_post_goal:
                    # Continue recording post-goal action
                    self.post_goal_counter += 1
                    
                    # When we've collected enough frames after the goal
                    if self.post_goal_counter >= self.post_goal_frames:
                        self.recording_post_goal = False
                        self.post_goal_counter = 0
                        
                        # Now start the replay
                        self.replay_frames = replay_buffer.copy()
                        self.show_replay = True
                        self.replay_index = 0
                        self.replay_count = 0
                # Check if enough time has passed since ball left goal area
                # AND ball is now back in goal area AND we're not in cooldown
                elif (ball_in_goal_now and not ball_in_goal and 
                      self.ball_left_goal_area and
                      current_time - self.ball_left_goal_time >= self.required_time_outside_goal and
                      current_time - self.last_goal_time > self.goal_cooldown):
                    
                    # Goal conditions met
                    self.goal_count += 1
                    self.last_goal_time = current_time
                    # Reset ball_left_goal_area flag since we're counting a new goal
                    self.ball_left_goal_area = False
                    
                    # Log the goal
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    logging.info(f"GOAL! Total count: {self.goal_count} at {timestamp}")
                    
                    # Play goal sound
                    self.play_goal_sound()
                    
                    # Start post-goal recording
                    self.recording_post_goal = True
                    self.post_goal_counter = 0
                    
                    # Visual indicator that goal was detected
                    cv2.putText(frame, "GOAL DETECTED!", (frame.shape[1]//2 - 100, frame.shape[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Update ball_in_goal flag for next iteration
                ball_in_goal = ball_in_goal_now
                
                # Display status information
                cv2.putText(frame, f"Goals: {self.goal_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show time until next possible goal
                goal_ready = (self.ball_left_goal_area and 
                              (current_time - self.ball_left_goal_time >= self.required_time_outside_goal) and 
                              self.celebration_complete)  # Added celebration check
                
                if not goal_ready and self.ball_left_goal_area:
                    # Show time since ball left goal area
                    time_since_left = current_time - self.ball_left_goal_time
                    time_remaining = max(0, self.required_time_outside_goal - time_since_left)
                    
                    if not self.celebration_complete:
                        # If celebration is ongoing, show that message instead
                        cv2.putText(frame, "Celebration in progress...", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                    else:
                        # Otherwise, show normal countdown
                        cv2.putText(frame, f"Ball outside goal: {time_since_left:.1f}s / {self.required_time_outside_goal}s", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                elif not self.ball_left_goal_area:
                    cv2.putText(frame, "Ball must leave goal area", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cooldown_remaining = max(0, self.goal_cooldown - (current_time - self.last_goal_time))
                if cooldown_remaining > 0:
                    cv2.putText(frame, f"Cooldown: {cooldown_remaining:.1f}s", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Goal readiness indicator - now also checks celebration status
                goal_status_color = (0, 255, 0) if (goal_ready and cooldown_remaining == 0 and self.celebration_complete) else (0, 0, 255)
                cv2.putText(frame, "GOAL READY" if (goal_ready and cooldown_remaining == 0 and self.celebration_complete) else "NOT READY", 
                          (frame.shape[1] - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, goal_status_color, 2)
                
                # If we're recording post-goal, show an indicator
                if self.recording_post_goal:
                    cv2.putText(frame, f"Recording goal... {self.post_goal_counter}/{self.post_goal_frames}", 
                                (frame.shape[1] - 300, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show the frame
                cv2.imshow("Floorball Referee", frame)
                
                # Check for keypresses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    # Toggle debug mode
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                    
                    # If turning off debug mode, close all debug windows
                    if not self.debug_mode:
                        cv2.destroyWindow("Motion Mask")
                        cv2.destroyWindow("Color Mask")
                        cv2.destroyWindow("Combined Mask")
                        cv2.destroyWindow("Ball Candidates")
                    else:
                        # If turning on debug mode, enable all windows by default
                        self.show_motion_mask = True
                        self.show_color_mask = True
                        self.show_combined_mask = True
                        self.show_candidates = True
                
                elif key == ord('1'):
                    # Toggle motion mask window
                    self.show_motion_mask = not self.show_motion_mask
                    if not self.show_motion_mask and self.debug_mode:
                        cv2.destroyWindow("Motion Mask")
                
                elif key == ord('2'):
                    # Toggle color mask window
                    self.show_color_mask = not self.show_color_mask
                    if not self.show_color_mask and self.debug_mode:
                        cv2.destroyWindow("Color Mask")
                
                elif key == ord('3'):
                    # Toggle combined mask window
                    self.show_combined_mask = not self.show_combined_mask
                    if not self.show_combined_mask and self.debug_mode:
                        cv2.destroyWindow("Combined Mask")
                
                elif key == ord('4'):
                    # Toggle ball candidates window
                    self.show_candidates = not self.show_candidates
                    if not self.show_candidates and self.debug_mode:
                        cv2.destroyWindow("Ball Candidates")
                        
                elif key == ord('p'):
                    # Toggle performance mode - adjust processing parameters for better performance
                    self.process_every_n_frames = 3 if self.process_every_n_frames == 2 else 2
                    self.debug_scale = 0.3 if self.debug_scale == 0.5 else 0.5
                    print(f"Performance mode: Processing every {self.process_every_n_frames} frames, Debug scale: {self.debug_scale}")
                
                elif key == ord('a'):
                    # Toggle alternative detection method
                    self.use_alternative_detection = not self.use_alternative_detection
                    print(f"Alternative detection: {'ON' if self.use_alternative_detection else 'OFF'}")
                
                elif key == ord('c'):
                    # Toggle pure color detection (no motion requirement)
                    self.use_pure_color_detection = not self.use_pure_color_detection
                    # When disabling pure color detection, use the weights that worked best previously
                    if not self.use_pure_color_detection:
                        self.color_detection_weight = 0.9
                        self.motion_detection_weight = 0.1
                    else:
                        self.color_detection_weight = 1.0
                        self.motion_detection_weight = 0.0
                    
                    print(f"Pure color detection: {'ON' if self.use_pure_color_detection else 'OFF'}")
                    print(f"Current settings: Color weight: {self.color_detection_weight:.1f}, Motion weight: {self.motion_detection_weight:.1f}")
                
                elif key == ord('u'):
                    # Toggle blue ball detection
                    self.use_blue_ball = not self.use_blue_ball
                    print(f"Blue ball detection: {'ON' if self.use_blue_ball else 'OFF'}")
                    if self.use_blue_ball:
                        print("Now detecting blue ball - use this mode with a blue ball for better contrast")
                    else:
                        print("Now detecting white ball")
                    
                    # Reset any existing detection history when switching ball color
                    self.ball_history = []
                    self.recent_positions = []
                    self.last_valid_detection = None
                
                elif key == ord('r'):
                    # Manual reset of ball tracking status
                    self.ball_left_goal_area = True
                    self.ball_left_goal_time = current_time - self.required_time_outside_goal
                    print("Manual reset of ball tracking status")
                
                elif key == ord('b'):
                    # Reset background model
                    print("Resetting background model...")
                    self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=16, detectShadows=False)
                    # Allow background subtractor to learn initial background
                    for _ in range(10):  # Learn from 10 frames
                        ret, temp_frame = self.camera.read()
                        if ret:
                            self.bg_subtractor.apply(temp_frame)
                
                elif key == ord('+') or key == ord('='):
                    # Increase color detection weight
                    self.color_detection_weight = min(1.0, self.color_detection_weight + 0.1)
                    self.motion_detection_weight = 1.0 - self.color_detection_weight
                    print(f"Color weight: {self.color_detection_weight:.1f}, Motion weight: {self.motion_detection_weight:.1f}")
                elif key == ord('-') or key == ord('_'):
                    # Decrease color detection weight
                    self.color_detection_weight = max(0.0, self.color_detection_weight - 0.1)
                    self.motion_detection_weight = 1.0 - self.color_detection_weight
                    print(f"Color weight: {self.color_detection_weight:.1f}, Motion weight: {self.motion_detection_weight:.1f}")
                elif key == ord('s'):
                    # Test goal sound
                    self.play_goal_sound()
                    print("Testing goal sound")
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                # Try to recover and continue
                time.sleep(0.5)
        
        # Clean up
        pygame.mixer.quit()  # Properly shut down pygame mixer
        logging.info(f"Session ended. Total goals: {self.goal_count}")
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    referee = FloorballReferee()
    referee.run()
