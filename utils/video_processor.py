import cv2
import numpy as np
import os
from config import OCR_ROI, IN_ZONE, OUT_ZONE, DRAW_BBOX, DRAW_LABELS, DRAW_TIME, DRAW_DEBUG_ZONES
# Import the helper function from the tracker module
from models.tracker import _calculate_time_difference_in_seconds 

class VideoProcessor:
    """
    Provides utility functions for loading videos, drawing annotations on frames,
    and saving processed videos. This class is responsible for all visual output.
    """
    def __init__(self, video_path, output_path=None):
        """
        Initializes the VideoProcessor by opening the video file.

        Args:
            video_path (str): Path to the input video file.
            output_path (str, optional): Path to save the processed video. If None, video won't be saved.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Error: Video file not found at: {video_path}")
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Error: Cannot open video file: {video_path}. Check path and codecs.")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video loaded: '{video_path}'")
        print(f"Resolution: {self.width}x{self.height}, FPS: {self.fps}, Frames: {self.frame_count}")

        self.writer = None
        if output_path:
            # Define the codec for the output video (e.g., MP4V for .mp4, XVID for .avi)
            # 'mp4v' is a good cross-platform choice for .mp4 files.
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
            if not self.writer.isOpened():
                print(f"Warning: Could not open video writer for {output_path}. Output video will not be saved.")
                self.writer = None # Reset writer if it failed to open
            else:
                print(f"Output video writer initialized for: '{output_path}'")

    def read_frame(self):
        """
        Reads the next frame from the video.

        Returns:
            tuple: (ret, frame) where ret is True if frame is read successfully, False otherwise.
                   frame (numpy.ndarray): The read frame.
        """
        return self.cap.read()

    def get_current_frame_number(self):
        """Returns the current frame number (1-indexed)."""
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def get_current_time_seconds(self):
        """Returns the current video time in seconds."""
        return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    def draw_annotations(self, frame, tracked_persons, ocr_time):
        """
        Draws all necessary annotations on the frame:
        - OCR-extracted CCTV time
        - Debug zones (IN_ZONE, OUT_ZONE, OCR_ROI) if enabled
        - Bounding boxes, IDs, activity, IN/OUT times, and accumulated working time for each person.

        Args:
            frame (numpy.ndarray): The frame to draw on.
            tracked_persons (list): List of `TrackedPerson` objects currently being tracked.
            ocr_time (str): The current CCTV timestamp extracted via OCR.
        """
        # Create a copy to draw on, so the original frame remains untouched
        annotated_frame = frame.copy()

        # --- Draw OCR extracted time ---
        if DRAW_TIME:
            # Position the OCR time label slightly above the OCR_ROI or at a fixed position
            text_pos = (OCR_ROI[0], OCR_ROI[1] - 10 if OCR_ROI[1] > 20 else 10)
            cv2.putText(annotated_frame, f"CCTV Time: {ocr_time}", text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            # Draw a rectangle around the OCR_ROI for visualization if debug zones are enabled
            if DRAW_DEBUG_ZONES:
                cv2.rectangle(annotated_frame, (OCR_ROI[0], OCR_ROI[1]), (OCR_ROI[2], OCR_ROI[3]), (0, 255, 255), 1)


        # --- Draw debug zones if enabled ---
        if DRAW_DEBUG_ZONES:
            # IN_ZONE
            cv2.rectangle(annotated_frame, (IN_ZONE[0], IN_ZONE[1]), (IN_ZONE[2], IN_ZONE[3]), (0, 255, 0), 2)
            cv2.putText(annotated_frame, "IN Zone", (IN_ZONE[0] + 5, IN_ZONE[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            
            # OUT_ZONE
            cv2.rectangle(annotated_frame, (OUT_ZONE[0], OUT_ZONE[1]), (OUT_ZONE[2], OUT_ZONE[3]), (0, 0, 255), 2)
            cv2.putText(annotated_frame, "OUT Zone", (OUT_ZONE[0] + 5, OUT_ZONE[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

        # --- Draw annotations for each tracked person ---
        for person in tracked_persons:
            x1, y1, x2, y2 = person.bbox
            
            # Ensure coordinates are integers for drawing functions
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Choose color based on person's state
            color = (0, 255, 0) # Default green for tracked persons
            if person.is_working:
                color = (0, 255, 128) # Orange for 'working' (sitting)
            elif person.in_time and not person.out_time: 
                color = (255, 0, 0) # Blue for persons who have entered but not yet exited
            elif person.out_time:
                color = (0, 0, 255) # Red for persons who have exited

            # Draw bounding box
            if DRAW_BBOX:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw labels (ID, activity, times)
            if DRAW_LABELS:
                label_lines = [f"ID: {person.id}"]
                label_lines.append(f"Activity: {person.activity.upper()}") # Show current activity

                if person.in_time:
                    label_lines.append(f"IN: {person.in_time}")
                if person.out_time:
                    label_lines.append(f"OUT: {person.out_time}")
                
                # Show accumulated working time if person has any
                # Need to calculate current ongoing session duration for display if person is_working
                display_total_working_seconds = person.total_working_seconds
                if person.is_working and person.current_working_session_start_time and ocr_time != "N/A":
                    current_session_duration = _calculate_time_difference_in_seconds(
                        person.current_working_session_start_time, ocr_time
                    )
                    display_total_working_seconds += current_session_duration # Add ongoing session duration for display
                    
                if display_total_working_seconds > 0:
                    hours, remainder = divmod(int(display_total_working_seconds), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    duration_str = f"{hours:02}:{minutes:02}:{seconds:02}"
                    label_lines.append(f"Work Time: {duration_str}")
                
                # Stack labels vertically above the bounding box
                for i, line in enumerate(label_lines):
                    text_y = y1 - 10 - (len(label_lines) - 1 - i) * 20 # Adjust Y for multiple lines
                    # Ensure text is not drawn outside the top of the frame
                    if text_y < 10: 
                        text_y = y1 + 20 + i * 20 # Move text inside/below bbox if too high

                    cv2.putText(annotated_frame, line, (x1, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        return annotated_frame

    def write_frame(self, frame):
        """
        Writes the processed frame to the output video if a writer is initialized.

        Args:
            frame (numpy.ndarray): The frame to write.
        """
        if self.writer:
            self.writer.write(frame)

    def release(self):
        """
        Releases the video capture and writer objects to free up resources.
        """
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        print("Video capture and writer released.")