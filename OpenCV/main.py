import cv2
from vehicle_detector import VehicleDetector

# Load video file
video_path = 'path_to_your_video.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

vd = VehicleDetector()
lane = []
count = 0

def syncLane(lane):
    baseTimer = 20  # Base timer value
    timeLimits = [2, 5]
    t = [(i / sum(lane)) * baseTimer if timeLimits[0] < (i / sum(lane)) * baseTimer < timeLimits[1]
          else min(timeLimits, key=lambda x: abs(x - (i / sum(lane)) * baseTimer)) for i in lane]
    print("--------------------------")
    print("Lane 1 waiting time:", t[0])
    print("Lane 2 waiting time:", t[1])
    print("Lane 3 waiting time:", t[2])
    print("Lane 4 waiting time:", t[3])
    print("Total average waiting time:", sum(t))
    print("--------------------------")
    w = int(sum(t)) * 1000
    return w

# Main loop for video processing
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break  # Exit the loop if video ends

    # Process the current frame
    vehicle_boxes = vd.detect_vehicles(img)
    vehicle_count = len(vehicle_boxes)

    for box in vehicle_boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (25, 0, 180), 2)
        cv2.putText(img, "Vehicles Detected: " + str(vehicle_count), (20, 50), 0, 1, (100, 200, 0), 2)

    print("Total vehicle count:", vehicle_count)
    lane.append(vehicle_count)
    
    # If processing every 4 frames for waiting time
    if (count % 4 == 0 and count != 0):
        waitTime = syncLane(lane)
        cv2.waitKey(waitTime)
        lane.clear()
        count = 0

    cv2.imshow("Cars", img)
    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q'
        break

# For last set of lane
waitTime = syncLane(lane)

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
