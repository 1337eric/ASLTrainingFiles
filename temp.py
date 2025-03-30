import sys
import json
import cv2
import numpy as np
import base64
import traceback

def process_frame():
    for line in sys.stdin:
        try:
            data = json.loads(line)
            frame_b64 = data.get('frame', '').replace("data:image/webp;base64,","")
            
            if not frame_b64:
                print("Warning: Empty frame data received")
                continue
                
            # Check if there's padding and add if needed
            padding = len(frame_b64) % 4
            if padding:
                frame_b64 += '=' * (4 - padding)
                
            # Decode base64 string to image
            try:
                img_bytes = base64.b64decode(frame_b64)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                # Display the frame
                if frame is not None:
                    cv2.imshow('Frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("Failed to decode frame: OpenCV couldn't parse the image data")
                    print(f"Image bytes length: {len(img_bytes)}")
            except Exception as e:
                print(f"Error during image decoding: {e}")
                print(f"Base64 string length: {len(frame_b64)}")
                print(f"First 100 chars of base64: {frame_b64[:100]}...")
                traceback.print_exc()
            
            sys.stdout.flush()
        except Exception as e:
            print(f"Error processing input: {e}")
            traceback.print_exc()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_frame()