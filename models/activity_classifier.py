from config import SITTING_THRESHOLD_HEIGHT_RATIO

class ActivityClassifier:
    """
    Classifies a person's activity as "standing" or "working" (which implies sitting in this context)
    based on the aspect ratio (height / width) of their bounding box.
    """
    def __init__(self):
        """
        Initializes the ActivityClassifier with the `SITTING_THRESHOLD_HEIGHT_RATIO`
        defined in `config.py`.
        """
        self.sitting_threshold = SITTING_THRESHOLD_HEIGHT_RATIO
        print(f"Activity Classifier initialized. Sitting height/width aspect ratio threshold: {self.sitting_threshold}")
        print("Note: This classification is heuristic (rule-based) and may require tuning for different camera angles/body types.")

    def classify(self, bbox):
        """
        Classifies the activity of a person based on their bounding box dimensions.
        
        A person is generally classified as "working" (sitting) if their bounding box
        height-to-width ratio falls below a certain threshold. Otherwise, they are "standing".

        Args:
            bbox (list): Bounding box coordinates [x1, y1, x2, y2].

        Returns:
            str: "working" (if sitting) or "standing". Returns "standing" for invalid bboxes.
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Handle invalid bounding box dimensions
        if width <= 0 or height <= 0:
            return "standing" # Cannot classify, assume standing or unknown

        # Calculate the aspect ratio (height divided by width)
        aspect_ratio = height / width

        # A common heuristic: standing individuals typically have a higher height/width ratio
        # than sitting individuals, whose height is reduced relative to their width.
        # You will need to fine-tune `SITTING_THRESHOLD_HEIGHT_RATIO` in `config.py`
        # by observing the aspect ratios of people in your specific video.
        if aspect_ratio < self.sitting_threshold:
            return "working" # Interpreted as sitting
        else:
            return "standing"