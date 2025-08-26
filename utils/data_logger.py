import os
import pandas as pd
from datetime import datetime

class DataLogger:
    """
    Manages the storage of event data (IN, OUT, WORKING_START, WORKING_END, ACTIVITY_CHANGE)
    for each person and provides functionality to export this data to a comprehensive CSV file.
    It focuses on logging distinct events and then compiling a final report.
    """
    def __init__(self, log_file_path, csv_export_path):
        """
        Initializes the DataLogger with paths for a text log file and a CSV export file.

        Args:
            log_file_path (str): Path to the text log file for raw event logging.
            csv_export_path (str): Path to the CSV file for exporting the final aggregated report.
        """
        self.log_file_path = log_file_path
        self.csv_export_path = csv_export_path
        self.events = [] # Stores all raw event data as dictionaries
        
        # Ensure the directory for logs exists
        log_dir = os.path.dirname(log_file_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Clear previous log content on initialization
        with open(self.log_file_path, 'w') as f:
            f.write(f"--- Office Tracking Log Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

        print(f"Data Logger initialized. Text log: '{self.log_file_path}', CSV report: '{self.csv_export_path}'")

    def _write_to_txt_log(self, message):
        """Internal method to append a timestamped message to the text log file."""
        try:
            with open(self.log_file_path, 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        except IOError as e:
            print(f"Error writing to text log file {self.log_file_path}: {e}")

    def log_event(self, person_id, event_type, cctv_time_str, video_frame_time_sec, details=""):
        """
        Records a significant event for a specific person.

        Args:
            person_id (str): Unique identifier for the person (e.g., "Person 1").
            event_type (str): Type of event ("IN", "OUT", "WORKING_START", "WORKING_END", "ACTIVITY_CHANGE").
            cctv_time_str (str): The CCTV timestamp extracted via OCR (e.g., "14:30:05").
            video_frame_time_sec (float): The actual video frame time in seconds when the event occurred.
            details (str, optional): Additional context or details about the event.
        """
        event_entry = {
            "timestamp_utc": datetime.now().isoformat(), # Timestamp when the event was logged by the system
            "person_id": person_id,
            "event_type": event_type,
            "cctv_time_str": cctv_time_str,
            "video_frame_time_sec": video_frame_time_sec,
            "details": details
        }
        self.events.append(event_entry)
        self._write_to_txt_log(
            f"Person {person_id}: Event='{event_type}', CCTV Time='{cctv_time_str}', "
            f"Video Time='{video_frame_time_sec:.2f}s', Details='{details}'"
        )
        print(f"Logged Event: Person {person_id}, Type: {event_type}, CCTV Time: {cctv_time_str}")

    def export_to_csv(self, tracked_persons):
        """
        Exports a comprehensive report to a CSV file. This report aggregates all
        information for each person, including their IN/OUT times and total working hours.
        This method takes the final state of `tracked_persons` to ensure up-to-date working times.

        Args:
            tracked_persons (list): The final list of `TrackedPerson` objects from the tracker.
        """
        data_for_df = []
        for person in tracked_persons:
            # Format total working seconds into HH:MM:SS
            hours, remainder = divmod(int(person.total_working_seconds), 3600)
            minutes, seconds = divmod(remainder, 60)
            total_working_formatted = f"{hours:02}:{minutes:02}:{seconds:02}"
            
            # Gather all working periods from the raw events for this person
            working_periods_details = []
            current_start = None
            for event in self.events:
                if event["person_id"] == person.id:
                    if event["event_type"] == "WORKING_START":
                        current_start = event["cctv_time_str"]
                    elif event["event_type"] == "WORKING_END" and current_start is not None:
                        working_periods_details.append(f"{current_start}-{event['cctv_time_str']}")
                        current_start = None
            
            # If a working session was ongoing when the video ended, add it as 'Ongoing'
            if person.is_working and person.current_working_session_start_time:
                 working_periods_details.append(f"{person.current_working_session_start_time}-Ongoing (Video End)")

            data_for_df.append({
                "Person ID": person.id,
                "IN Time (CCTV)": person.in_time if person.in_time else "N/A",
                "OUT Time (CCTV)": person.out_time if person.out_time else "N/A",
                "Total Working Hours": total_working_formatted,
                "Working Periods (Start-End)": "; ".join(working_periods_details) if working_periods_details else "N/A"
            })
        
        if not data_for_df:
            print("No data available to export to CSV.")
            return

        df = pd.DataFrame(data_for_df)
        try:
            df.to_csv(self.csv_export_path, index=False)
            print(f"Aggregated report successfully exported to CSV: '{self.csv_export_path}'")
        except IOError as e:
            print(f"Error exporting data to CSV file {self.csv_export_path}: {e}")
