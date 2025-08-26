import numpy as np
from collections import deque
from config import MAX_DIST_PERSON, MAX_MISSING_FRAMES

def _calculate_time_difference_in_seconds(start_time_str, end_time_str):
    """
    Calculates the difference between two CCTV time strings in seconds.
    Handles HH:MM:SS or HH:MM format.
    Assumes times are within a 24-hour period.
    """
    def parse_time_to_seconds(time_str):
        try:
            parts = [int(p) for p in time_str.split(':')]
            if len(parts) == 3: # HH:MM:SS
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2: # HH:MM
                return parts[0] * 3600 + parts[1] * 60
            return 0
        except ValueError:
            return 0
    
    start_sec = parse_time_to_seconds(start_time_str)
    end_sec = parse_time_to_seconds(end_time_str)

    # Handle time rollover (e.g., crossing midnight). Assume it's the next day.
    if end_sec < start_sec:
        end_sec += 24 * 3600 # Add 24 hours (86400 seconds)
    
    return end_sec - start_sec

class TrackedPerson:
    """
    Represents a single tracked person with their unique ID, current bounding box,
    activity status, and various timestamps related to their presence and work.
    """
    def __init__(self, id, bbox, initial_cctv_time=None, initial_frame_time_sec=None):
        """
        Initializes a new tracked person.

        Args:
            id (str): A unique identifier for this person (e.g., "Person 1").
            bbox (list): Initial bounding box coordinates [x1, y1, x2, y2].
            initial_cctv_time (str, optional): The CCTV time when this person was first detected.
            initial_frame_time_sec (float, optional): The video frame time (in seconds)
                                                      when this person was first detected.
        """
        self.id = id
        self.bbox = bbox # Current bounding box [x1, y1, x2, y2]
        self.centroid = self._get_centroid(bbox)
        self.missing_frames = 0 # Counter for how many frames the person has not been detected
        self.activity = "standing" # Initial activity status
        
        # --- Time Logging ---
        self.in_time = initial_cctv_time # CCTV time when person entered
        self.in_frame_time_sec = initial_frame_time_sec # Video time when person entered
        self.out_time = None # CCTV time when person exited
        self.out_frame_time_sec = None # Video time when person exited
        
        self.is_working = False # True if currently in a 'working' (sitting) state
        self.current_working_session_start_time = None # CCTV time when the current working session started
        self.total_working_seconds = 0.0 # Accumulates total working time across all sessions
        
        self.last_ocr_time = initial_cctv_time # Last known CCTV time this person was seen
        self.last_frame_time_sec = initial_frame_time_sec # Last known video time this person was seen


    def _get_centroid(self, bbox):
        """Calculates the center point (centroid) of a bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update_bbox(self, new_bbox, ocr_time, frame_time_sec):
        """
        Updates the person's bounding box and centroid, and resets the missing frames counter.
        Also updates the last seen time.

        Args:
            new_bbox (list): The new bounding box coordinates.
            ocr_time (str): The current CCTV time.
            frame_time_sec (float): The current video frame time in seconds.
        """
        self.bbox = new_bbox
        self.centroid = self._get_centroid(new_bbox)
        self.missing_frames = 0 # Reset missing frames as person is detected
        self.last_ocr_time = ocr_time
        self.last_frame_time_sec = frame_time_sec

    def update_activity(self, new_activity, ocr_time, frame_time_sec):
        """
        Updates the person's activity status and manages the start/end of working sessions.
        This method will update `total_working_seconds` when a working session ends.

        Args:
            new_activity (str): The newly classified activity ("standing" or "working").
            ocr_time (str): The current CCTV timestamp.
            frame_time_sec (float): The current video frame time in seconds.
        """
        if self.activity != new_activity:
            print(f"Person {self.id}: Activity changed from '{self.activity}' to '{new_activity}' at {ocr_time}")

            # If the person was previously working and now is not
            if self.is_working and self.current_working_session_start_time:
                # End the current working session and add its duration to total
                duration = _calculate_time_difference_in_seconds(self.current_working_session_start_time, ocr_time)
                self.total_working_seconds += duration
                print(f"Person {self.id}: Ended working session at {ocr_time}. Session duration: {duration:.2f}s. Total work: {self.total_working_seconds:.2f}s")
                self.current_working_session_start_time = None # Reset for next session
            
            self.activity = new_activity # Update to the new activity

            # If the person just started working
            if self.activity == "working":
                self.is_working = True
                self.current_working_session_start_time = ocr_time # Mark start of new session
            else:
                self.is_working = False # No longer working
        
        # If still working, continuously update the current session duration
        # This is not directly updated in total_working_seconds here,
        # but `main.py` will use `_calculate_time_difference_in_seconds` 
        # to update `total_working_seconds` for *currently active* sessions.

    def increment_missing(self):
        """Increments the count of frames the person has been missing."""
        self.missing_frames += 1

    def is_too_old(self):
        """Checks if the person has been missing for too many frames, indicating track loss."""
        return self.missing_frames > MAX_MISSING_FRAMES

class PersonTracker:
    """
    Manages the assignment and persistence of unique IDs to detected persons across frames.
    It uses a simple centroid-based tracking algorithm to associate new detections
    with existing tracked persons.
    """
    def __init__(self):
        """
        Initializes the PersonTracker with an empty list of currently tracked persons
        and a counter for assigning new unique IDs.
        """
        self.tracked_persons = []
        self.next_person_id = 1 # Starts with "Person 1"
        print(f"Person Tracker initialized. Max association distance: {MAX_DIST_PERSON}px, Max missing frames before loss: {MAX_MISSING_FRAMES}")

    def _get_distance(self, centroid1, centroid2):
        """Calculates Euclidean distance between two 2D centroids."""
        return np.linalg.norm(np.array(centroid1) - np.array(centroid2))

    def update(self, detections, ocr_time, frame_time_sec):
        """
        Updates the tracker with new detections from the current frame.
        It performs the following steps:
        1. Marks all existing tracked persons as potentially missing.
        2. Tries to match new detections with existing tracked persons based on centroid distance.
        3. Updates matched persons with new bounding boxes and resets their missing frame count.
        4. Creates new `TrackedPerson` objects for any unmatched detections.
        5. Removes persons whose tracks have been lost (missing for too many frames).
        6. Manages the `total_working_seconds` for persons whose working sessions might end
           due to disappearance.

        Args:
            detections (list): A list of new detections (each a dict with 'bbox', 'confidence').
            ocr_time (str): The current CCTV timestamp from OCR.
            frame_time_sec (float): The current video frame time in seconds.

        Returns:
            list: The updated list of `TrackedPerson` objects currently being tracked.
        """
        # Step 1: Mark all existing persons as missing by default for this frame
        for person in self.tracked_persons:
            person.increment_missing()

        # Keep track of which new detections have been matched
        matched_detection_indices = [False] * len(detections)

        # Step 2: Try to match existing persons with new detections
        for i, person in enumerate(self.tracked_persons):
            min_dist = float('inf')
            best_match_idx = -1

            for j, det in enumerate(detections):
                if not matched_detection_indices[j]: # Only consider detections not yet matched
                    det_centroid = person._get_centroid(det['bbox'])
                    dist = self._get_distance(person.centroid, det_centroid)

                    if dist < min_dist and dist < MAX_DIST_PERSON:
                        min_dist = dist
                        best_match_idx = j
            
            if best_match_idx != -1:
                # Step 3: Match found, update the person's state
                person.update_bbox(detections[best_match_idx]['bbox'], ocr_time, frame_time_sec)
                matched_detection_indices[best_match_idx] = True
            else:
                # No match for this person in the current frame.
                # If they were working and now disappeared, finalize their working session.
                if person.is_working and person.current_working_session_start_time:
                    # End working session using the *last known* OCR time if current is N/A
                    session_end_time = person.last_ocr_time if ocr_time == "N/A" else ocr_time
                    duration = _calculate_time_difference_in_seconds(person.current_working_session_start_time, session_end_time)
                    person.total_working_seconds += duration
                    print(f"Person {person.id}: Ended working session due to disappearance at {session_end_time}. Duration: {duration:.2f}s. Total work: {person.total_working_seconds:.2f}s")
                    self.current_working_session_start_time = None
                    person.is_working = False


        # Step 4: Create new `TrackedPerson` objects for un-matched detections
        for j, det in enumerate(detections):
            if not matched_detection_indices[j]:
                new_person = TrackedPerson(f"Person {self.next_person_id}", det['bbox'], ocr_time, frame_time_sec)
                self.tracked_persons.append(new_person)
                self.next_person_id += 1

        # Step 5: Remove persons that have been missing for too many frames (track lost)
        # Before removing, ensure any ongoing working session is finalized
        persons_to_keep = []
        for person in self.tracked_persons:
            if not person.is_too_old():
                persons_to_keep.append(person)
            else:
                # If a person's track is being lost and they were working, finalize their session
                if person.is_working and person.current_working_session_start_time:
                    # Use the last known OCR time for the session end
                    session_end_time = person.last_ocr_time 
                    duration = _calculate_time_difference_in_seconds(person.current_working_session_start_time, session_end_time)
                    person.total_working_seconds += duration
                    print(f"Person {person.id}: Track lost and working session ended at {session_end_time}. Duration: {duration:.2f}s. Total work: {person.total_working_seconds:.2f}s")
                    self.current_working_session_start_time = None
                    person.is_working = False
        
        self.tracked_persons = persons_to_keep

        return self.tracked_persons