import cv2
import time
import sys
import traceback

def list_available_cameras():
    """Check available camera indices and return a list of working ones."""
    available_cameras = []
    
    print("Starting camera detection...")
    
    # Test camera indices from 0 to 9
    for i in range(10):
        try:
            print(f"Attempting to open camera index {i}...")
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow API on Windows
            
            if cap.isOpened():
                print(f"✓ Camera index {i} opened successfully")
                
                # Try to read a frame
                print(f"  Attempting to read a frame from camera {i}...")
                ret, frame = cap.read()
                
                if ret:
                    print(f"  ✓ Successfully read frame from camera {i}")
                    print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
                    
                    # Display the camera feed briefly to identify it
                    window_name = f"Camera {i}"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 640, 480)
                    cv2.imshow(window_name, frame)
                    cv2.waitKey(500)  # Wait a bit longer to make sure window shows
                    print(f"  Displaying preview for camera {i} - look at your screen")
                    time.sleep(2)  # Wait for the user to see the preview
                    
                    available_cameras.append(i)
                else:
                    print(f"  ✗ Failed to read frame from camera {i} (camera might be in use by another application)")
                
                # Release the camera
                print(f"  Releasing camera {i}...")
                cap.release()
                cv2.destroyWindow(window_name)
                print(f"  Camera {i} released")
            else:
                print(f"✗ Failed to open camera index {i}")
        
        except Exception as e:
            print(f"ERROR with camera {i}: {str(e)}")
            traceback.print_exc()
        
        print("-" * 50)  # Separator between camera attempts
            
    return available_cameras

if __name__ == "__main__":
    print("=" * 80)
    print("CAMERA DETECTION UTILITY")
    print("This script will attempt to detect all connected cameras")
    print("If no windows appear, try running the script as administrator")
    print("=" * 80)
    
    try:
        print("\nOpenCV version:", cv2.__version__)
        print("Python version:", sys.version)
        print("\nChecking for available cameras...\n")
        
        available = list_available_cameras()
        
        print("\n" + "=" * 80)
        if available:
            print(f"RESULTS: Found {len(available)} available camera(s):")
            print("Available camera indices:", available)
            print("\nTo use a specific camera, set 'camera_index' in innebandyDomare.py to one of these values.")
        else:
            print("RESULTS: No cameras found!")
            print("Possible issues:")
            print("1. Cameras are not properly connected")
            print("2. Cameras are being used by another application")
            print("3. You need administrator privileges to access cameras")
            print("4. Camera drivers are not installed correctly")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nUNHANDLED ERROR: {str(e)}")
        traceback.print_exc()
    
    print("\nPress Enter to exit...")
    input()
    
    cv2.destroyAllWindows()
