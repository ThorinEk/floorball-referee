import cv2
import time
import os

def test_camera(index):
    """Test a single camera and display its feed."""
    print(f"\nTrying to access camera {index}...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Try DirectShow
    
    if not cap.isOpened():
        print(f"Failed to open camera {index} with DirectShow, trying default...")
        cap = cv2.VideoCapture(index)  # Try default
        
    if not cap.isOpened():
        print(f"Could not open camera {index}")
        return False
    
    print(f"Successfully opened camera {index}")
    print("Press 'q' to close this camera and try next one")
    print("Press 'ESC' to exit the program")
    
    window_name = f"Camera {index} Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Show 200 frames or until user quits
    for _ in range(200):
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame from camera {index}")
            break
            
        # Display resolution and index on frame
        height, width = frame.shape[:2]
        cv2.putText(frame, f"Camera {index}: {width}x{height}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return None  # Signal to exit program
    
    cap.release()
    cv2.destroyWindow(window_name)
    return True

print("SIMPLE CAMERA TEST UTILITY")
print("This will attempt to open each camera index one by one")

# Clear terminal (works on Windows and Unix)
os.system('cls' if os.name == 'nt' else 'clear')

for i in range(10):  # Test first 10 camera indices
    result = test_camera(i)
    if result is None:  # User pressed ESC
        print("User canceled testing")
        break

print("\nCamera testing completed.")
print("Press any key to exit...")
input()
cv2.destroyAllWindows()
