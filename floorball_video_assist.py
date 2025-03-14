import cv2
import numpy as np
import time
import logging
import os
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
        self.lower_white = np.array([0, 0, 180])  # Low saturation, high value
        self.upper_white = np.array([180, 50, 255])  # Allow any hue, low saturation, high value
        
        # Alternative detection parameters
        self.use_alternative_detection = False
        self.ball_history = []  # Track recent ball positions for smoothing

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
        """Detect the white ball using improved methods"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Standard HSV-based mask
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Improve detection with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # If standard detection is struggling, use alternative method
        if self.use_alternative_detection:
            # Enhance contrast
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2)
                
            # Combine methods
            mask = cv2.bitwise_or(mask, binary)
        
        # Find contours to identify the ball
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug visualization
        debug_frame = frame.copy()
        cv2.drawContours(debug_frame, contours, -1, (0, 255, 0), 2)
        cv2.imshow("Debug Mask", mask)
        cv2.imshow("Debug Contours", debug_frame)
        
        ball_position = None
        max_confidence = 0
        
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
            
            # Find the minimum enclosing circle
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            
            # Check if the detected object matches our criteria for a ball
            if (self.min_ball_radius <= radius <= self.max_ball_radius and 
                circularity > 0.5):  # More circular shapes have values closer to 1
                
                confidence = circularity * (1 - abs(radius - 15) / 15)  # Ideal size around 15px
                
                if confidence > max_confidence:
                    max_confidence = confidence
                    ball_position = (int(x), int(y), int(radius))
        
        # If using position tracking, smooth the detection
        if ball_position and self.ball_history:
            # Average with recent positions for smoothing
            recent_positions = self.ball_history[-3:] if len(self.ball_history) >= 3 else self.ball_history
            x_avg = sum([pos[0] for pos in recent_positions]) / len(recent_positions)
            y_avg = sum([pos[1] for pos in recent_positions]) / len(recent_positions)
            
            # Check if new detection is reasonable (not too far from recent positions)
            if abs(x_avg - ball_position[0]) < 50 and abs(y_avg - ball_position[1]) < 50:
                # Blend new detection with average for smoother tracking
                ball_position = (
                    int(0.7 * ball_position[0] + 0.3 * x_avg),
                    int(0.7 * ball_position[1] + 0.3 * y_avg),
                    ball_position[2]
                )
        
        # Update position history
        if ball_position:
            self.ball_history.append(ball_position)
            if len(self.ball_history) > 10:
                self.ball_history.pop(0)
        
        return ball_position

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
        
        while True:
            try:
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
                
                # Detect the ball
                ball_position = self.detect_ball(frame)
                
                # Draw the goal area
                if self.goal_area:
                    x1, y1, x2, y2 = self.goal_area
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw the detected ball
                if ball_position:
                    x, y, radius = ball_position
                    cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                
                # Check for goal with improved logic
                current_time = time.time()
                ball_in_goal_now = ball_position and self.is_goal(ball_position)
                
                # Only trigger goal when ball enters goal area (not already in)
                # and after cooldown period has passed
                if (ball_in_goal_now and not ball_in_goal and 
                    current_time - self.last_goal_time > self.goal_cooldown):
                    
                    self.goal_count += 1
                    self.last_goal_time = current_time
                    
                    # Log the goal
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    logging.info(f"GOAL! Total count: {self.goal_count} at {timestamp}")
                    
                    # Prepare replay
                    self.replay_frames = replay_buffer.copy()
                    self.show_replay = True
                    self.replay_index = 0
                    self.replay_count = 0
                
                # Update ball_in_goal flag for next iteration
                ball_in_goal = ball_in_goal_now
                
                # Display status information
                cv2.putText(frame, f"Goals: {self.goal_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show time until next possible goal
                cooldown_remaining = max(0, self.goal_cooldown - (current_time - self.last_goal_time))
                if cooldown_remaining > 0:
                    cv2.putText(frame, f"Cooldown: {cooldown_remaining:.1f}s", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Show the frame
                cv2.imshow("Floorball Referee", frame)
                
                # Check for keypresses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    show_debug = not show_debug
                elif key == ord('a'):
                    # Toggle alternative detection method
                    self.use_alternative_detection = not self.use_alternative_detection
                    print(f"Alternative detection: {'ON' if self.use_alternative_detection else 'OFF'}")
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                # Try to recover and continue
                time.sleep(0.5)
        
        # Clean up
        logging.info(f"Session ended. Total goals: {self.goal_count}")
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    referee = FloorballReferee()
    referee.run()
