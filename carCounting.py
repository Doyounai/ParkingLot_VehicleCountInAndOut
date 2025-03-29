import cv2
import numpy as np
import requests
import os

hostname = 'http://localhost:3000/parking/events/'

class DirectionalCarTracker:
    def __init__(self, weights_path, config_path, names_path, entry_zone, exit_zone):
        # Load YOLO
        self.net = cv2.dnn.readNet(weights_path, config_path)
        
        # Load class names
        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Get output layer names
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        
        # Tracking parameters
        self.car_tracks = {}
        self.next_id = 0
        
        # Tracking thresholds
        self.iou_threshold = 0.5
        self.confidence_threshold = 0.5
        self.tracking_threshold = 0.3
        
        # Counting zones
        self.entry_zone = entry_zone
        self.exit_zone = exit_zone
        
        # Tracking statistics
        self.cars_entered = 0
        self.cars_exited = 0
        
        # Advanced tracking states
        self.car_journey_states = {}

    def detect_cars(self, frame):
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        # Detect and filter cars
        car_detections = self._process_detections(outs, width, height)
        
        # Track cars and update their states
        tracked_cars = self._track_cars(car_detections, width, height)
        
        # Count cars with directional tracking
        self._update_directional_counts(tracked_cars)
        
        return tracked_cars

    def _process_detections(self, outs, width, height):
        boxes = []
        confidences = []
        class_ids = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for cars
                if confidence > self.confidence_threshold and self.classes[class_id] == 'car':
                    # Bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.iou_threshold)
        
        # Filter detected cars
        detected_cars = []
        if len(indexes) > 0:
            for i in range(len(indexes)):
                detected_cars.append({
                    'box': boxes[indexes[i]],
                    'confidence': confidences[indexes[i]]
                })
        
        return detected_cars

    def _track_cars(self, detections, width, height):
        tracked_cars = {}
        
        for detection in detections:
            x, y, w, h = detection['box']
            
            # Find best matching track
            best_match = self._find_best_track(x, y, w, h)
            
            if best_match is not None:
                # Update existing track
                track_id = best_match
                self.car_tracks[track_id] = (x, y, w, h)
                tracked_cars[track_id] = (x, y, w, h)
            else:
                # Create new track
                new_id = self.next_id
                self.next_id += 1
                self.car_tracks[new_id] = (x, y, w, h)
                tracked_cars[new_id] = (x, y, w, h)
                
                # Initialize car journey state
                self.car_journey_states[new_id] = {
                    'in_entry_zone': False,
                    'in_exit_zone': False,
                    'entered_from_start': False,
                    'exited_from_start': False
                }
        
        return tracked_cars

    def _find_best_track(self, x, y, w, h):
        best_match = None
        best_iou = 0
        
        for track_id, (tx, ty, tw, th) in self.car_tracks.items():
            iou = self._calculate_iou((x, y, w, h), (tx, ty, tw, th))
            if iou > best_iou and iou > self.tracking_threshold:
                best_match = track_id
                best_iou = iou
        
        return best_match

    def _update_directional_counts(self, tracked_cars):
        for car_id, (x, y, w, h) in tracked_cars.items():
            car_center_x = x + w // 2
            car_center_y = y + h // 2
            
            # Current car state
            current_state = self.car_journey_states[car_id]
            
            # Check entry zone
            is_in_entry_zone = self._point_in_zone(car_center_x, car_center_y, self.entry_zone)
            
            # Check exit zone
            is_in_exit_zone = self._point_in_zone(car_center_x, car_center_y, self.exit_zone)
            
            # Directional counting logic
            if is_in_entry_zone and not current_state['in_entry_zone']:
                current_state['in_entry_zone'] = True
                
                # If car was previously in exit zone, count as entered
                if current_state['in_exit_zone'] and not current_state['entered_from_start']:
                    self.cars_entered += 1
                    response = requests.post(hostname, json={"eventType": "enter", "vehicleId": str(car_id), "lotId": 1})
                    print(response.status_code, response.text)
                    current_state['entered_from_start'] = True
            
            if is_in_exit_zone and not current_state['in_exit_zone']:
                current_state['in_exit_zone'] = True
                
                # If car was previously in entry zone, count as exited
                if current_state['in_entry_zone'] and not current_state['exited_from_start']:
                    self.cars_exited += 1
                    response = requests.post(hostname, json={"eventType": "exit", "vehicleId": str(car_id), "lotId": 1})
                    print(response.status_code, response.text)
                    current_state['exited_from_start'] = True

    def _point_in_zone(self, x, y, zone):
        # Zone is a polygon defined by points
        return cv2.pointPolygonTest(np.array(zone), (x, y), False) >= 0

    def _calculate_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Compute intersection coordinates
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        # Check for intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Compute union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area

def main():
    # YOLO configuration paths
    weights_path = './yolov4.weights'
    config_path = './yolov4.cfg'
    names_path = './coco.names'
    
    # Video path
    video_path = './cars.mp4'
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define advanced zones (polygon zones)
    # Adjust these zones based on your specific video layout
    entry_zone = [
        (0, frame_height // 3),
        (frame_width, frame_height // 3),
        (frame_width, frame_height // 3 + 50),
        (0, frame_height // 3 + 50)
    ]
    
    exit_zone = [
        (0, frame_height * 2 // 3),
        (frame_width, frame_height * 2 // 3),
        (frame_width, frame_height * 2 // 3 + 50),
        (0, frame_height * 2 // 3 + 50)
    ]
    
    # Initialize directional tracker
    tracker = DirectionalCarTracker(weights_path, config_path, names_path, entry_zone, exit_zone)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and track cars
        tracked_cars = tracker.detect_cars(frame)
        
        # Draw tracking visualization
        for car_id, (x, y, w, h) in tracked_cars.items():
            # Determine box color based on car state
            car_state = tracker.car_journey_states[car_id]
            
            # Color coding
            color = (0, 255, 0)  # Default green
            if car_state['in_entry_zone'] and car_state['in_exit_zone']:
                color = (255, 0, 255)  # Magenta for completed journey
            elif car_state['in_entry_zone']:
                color = (255, 0, 0)  # Blue for entry zone
            elif car_state['in_exit_zone']:
                color = (0, 0, 255)  # Red for exit zone
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'Car {car_id}', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Draw zones
        cv2.polylines(frame, [np.array(entry_zone)], True, (255, 0, 0), 2)
        cv2.polylines(frame, [np.array(exit_zone)], True, (0, 0, 255), 2)
        
        # Display counts
        cv2.putText(frame, f'Entered: {tracker.cars_entered}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'Exited: {tracker.cars_exited}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow('Directional Car Tracking', frame)
        
        # Exit option
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print(f"Total Cars Entered: {tracker.cars_entered}")
    print(f"Total Cars Exited: {tracker.cars_exited}")

if __name__ == '__main__':
    main()