import cv2
import time
from datetime import datetime

# Import modules from your project structure
from config import (
    VIDEO_PATH, YOLO_MODEL_PATH, OCR_ROI, IN_ZONE, OUT_ZONE, 
    IN_TIME_WINDOW_END_SEC, OUT_TIME_WINDOW_START_SEC,
    LOG_FILE_PATH, CSV_EXPORT_PATH, FRAME_SKIP, OUTPUT_VIDEO_PATH,
    DRAW_DEBUG_ZONES
)
from models.yolo_detector import YOLODetector
from models.tracker import PersonTracker, TrackedPerson, _calculate_time_difference_in_seconds
from models.activity_classifier import ActivityClassifier
from utils.ocr_extractor import OCRExtractor
from utils.video_processor import VideoProcessor
from utils.data_logger import DataLogger

def run_office_tracking():
    """
    Main function to run the CCTV office tracking system.
    This orchestrates video processing, person detection, tracking, activity classification,
    OCR time extraction, and comprehensive data logging based on the project requirements.
    """
    print("\n--- Initializing Office Tracking System ---")
    
    # 1. Initialize all necessary components
    try:
        video_processor = VideoProcessor(VIDEO_PATH, OUTPUT_VIDEO_PATH)
        yolo_detector = YOLODetector()
        person_tracker = PersonTracker()
        activity_classifier = ActivityClassifier()
        ocr_extractor = OCRExtractor()
        data_logger = DataLogger(LOG_FILE_PATH, CSV_EXPORT_PATH)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize one or more components. Please check configurations and file paths. Details: {e}")
        # Release resources if any were opened before exiting
        if 'video_processor' in locals() and video_processor:
            video_processor.release()
        return

    frame_idx = 0
    start_processing_time = time.time() # For overall performance measurement

    print("\n--- Starting Video Processing Loop ---")
    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            print("End of video stream or failed to read frame. Exiting loop.")
            break

        frame_idx += 1
        current_video_time_sec = video_processor.get_current_time_seconds()

        # Skip frames if FRAME_SKIP is set to process video faster
        if FRAME_SKIP > 1 and frame_idx % FRAME_SKIP != 0:
            # If skipping frames, just write the original frame to maintain video length
            video_processor.write_frame(frame) 
            continue

        # 2. Extract CCTV Time via OCR from a defined ROI
        ocr_time = ocr_extractor.extract_time(frame)
        if ocr_time == "N/A":
            print(f"Warning: OCR failed to extract time at frame {frame_idx} (Video Time: {current_video_time_sec:.2f}s). Using last valid time if available, or 'N/A'.")
            # If OCR fails, we'll try to use the last known OCR time for tracked persons.
            # For new events (IN/OUT/START_WORKING), if OCR is N/A, these events might be missed or logged with N/A.

        # 3. Detect Persons in the current frame using YOLO
        detections = yolo_detector.detect(frame)
        
        # 4. Update Person Tracker with new detections
        # This will match detections to existing persons, create new ones, or mark existing as missing.
        tracked_persons = person_tracker.update(detections, ocr_time, current_video_time_sec)

        # 5. Process Each Tracked Person for IN/OUT/Activity/Working Time
        for person in tracked_persons:
            # Always update the last known OCR time for this person if a valid one is available
            if ocr_time != "N/A":
                person.last_ocr_time = ocr_time
            
            # Calculate centroid of the person's bounding box
            cx = (person.bbox[0] + person.bbox[2]) / 2
            cy = (person.bbox[1] + person.bbox[3]) / 2

            # --- IN Time Detection (First 20 seconds of video) ---
            # A person's IN time is recorded if they are in the IN_ZONE during the first 20 seconds,
            # and they don't already have an IN time recorded.
            if current_video_time_sec <= IN_TIME_WINDOW_END_SEC and person.in_time is None and ocr_time != "N/A":
                if IN_ZONE[0] <= cx <= IN_ZONE[2] and IN_ZONE[1] <= cy <= IN_ZONE[3]:
                    person.in_time = ocr_time
                    person.in_frame_time_sec = current_video_time_sec
                    data_logger.log_event(person.id, "IN", ocr_time, current_video_time_sec, "Person entered office.")
                    print(f"-> IN Event: {person.id} entered at {ocr_time}")

            # --- Activity Classification (After the first 20 seconds) ---
            # After the initial IN time window, classify activity (standing/working).
            if current_video_time_sec > IN_TIME_WINDOW_END_SEC:
                new_activity = activity_classifier.classify(person.bbox)
                
                # Check if activity has changed to manage working sessions
                if person.activity != new_activity:
                    prev_activity = person.activity
                    person.update_activity(new_activity, ocr_time, current_video_time_sec) # This updates person.activity and manages session start/end
                    
                    # Log the activity change
                    data_logger.log_event(
                        person.id, "ACTIVITY_CHANGE", ocr_time, current_video_time_sec, 
                        f"Changed from '{prev_activity}' to '{new_activity}'."
                    )
                    
                    # Log explicit WORKING_START/WORKING_END events
                    if new_activity == "working" and ocr_time != "N/A":
                        data_logger.log_event(person.id, "WORKING_START", ocr_time, current_video_time_sec, "Person started working (sitting).")
                        print(f"-> Working Event: {person.id} started working at {ocr_time}")
                    elif prev_activity == "working" and new_activity == "standing" and ocr_time != "N/A":
                        data_logger.log_event(person.id, "WORKING_END", ocr_time, current_video_time_sec, "Person stopped working (stood up).")
                        print(f"-> Working Event: {person.id} stopped working at {ocr_time}")

                # Accumulate total working seconds for currently active working sessions
                # The `person.total_working_seconds` is cumulatively updated when a session *ends* (in `update_activity`).
                # For *displaying* the current total, `VideoProcessor` will calculate the duration of the ongoing session
                # and add it to `person.total_working_seconds`. So no direct update here.
                pass 

            # --- OUT Time Detection (After 30 seconds of video) ---
            # A person's OUT time is recorded if they are in the OUT_ZONE after 30 seconds,
            # have an IN time, and don't already have an OUT time recorded.
            if current_video_time_sec >= OUT_TIME_WINDOW_START_SEC and \
               person.in_time is not None and person.out_time is None and ocr_time != "N/A":
                if OUT_ZONE[0] <= cx <= OUT_ZONE[2] and OUT_ZONE[1] <= cy <= OUT_ZONE[3]:
                    person.out_time = ocr_time
                    person.out_frame_time_sec = current_video_time_sec
                    data_logger.log_event(person.id, "OUT", ocr_time, current_video_time_sec, "Person exited office.")
                    print(f"-> OUT Event: {person.id} exited at {ocr_time}")

                    # If the person was working when they exited, end their working session
                    if person.is_working and person.current_working_session_start_time:
                        duration = _calculate_time_difference_in_seconds(person.current_working_session_start_time, ocr_time)
                        person.total_working_seconds += duration # Add remaining duration
                        data_logger.log_event(person.id, "WORKING_END", ocr_time, current_video_time_sec, "Person stopped working (exited office).")
                        person.current_working_session_start_time = None
                        person.is_working = False


        # 6. Visualize Results on the frame
        annotated_frame = video_processor.draw_annotations(frame, tracked_persons, ocr_time)
        
        # Display the annotated frame
        cv2.imshow("Office Tracking System - Press 'q' to quit", annotated_frame)
        video_processor.write_frame(annotated_frame)

        # Check for 'q' key press to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User requested to quit. Exiting loop.")
            break

    # 7. Finalize and Export Data after video processing loop ends
    end_processing_time = time.time()
    total_processing_duration = end_processing_time - start_processing_time
    print(f"\n--- Video Processing Finished ---")
    print(f"Total frames processed: {frame_idx}")
    print(f"Total processing time: {total_processing_duration:.2f} seconds.")
    
    # Ensure any ongoing working sessions are finalized before exporting the report
    for person in tracked_persons:
        if person.is_working and person.current_working_session_start_time and person.last_ocr_time != "N/A":
            duration = _calculate_time_difference_in_seconds(person.current_working_session_start_time, person.last_ocr_time)
            person.total_working_seconds += duration
            data_logger.log_event(person.id, "WORKING_END", person.last_ocr_time, video_processor.get_current_time_seconds(), "Person stopped working (video ended).")

    data_logger.export_to_csv(tracked_persons) # Pass tracked_persons for the final report

    # 8. Release all resources (video capture, video writer, OpenCV windows)
    video_processor.release()
    cv2.destroyAllWindows()
    print("All resources released. Office Tracking System shut down.")

if __name__ == "__main__":
    run_office_tracking()
