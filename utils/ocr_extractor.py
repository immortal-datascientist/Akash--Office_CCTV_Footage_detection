import cv2
import pytesseract
import re
from PIL import Image
import numpy as np # Import numpy for array operations
from config import OCR_ROI, TESSERACT_CMD

class OCRExtractor:
    """
    Handles the extraction of timestamp text from a specified region of interest (ROI)
    in a video frame using OCR (Pytesseract).
    """
    def __init__(self):
        """
        Initializes the OCRExtractor with the predefined OCR_ROI from config.py
        and sets the tesseract command path.
        """
        self.roi = OCR_ROI
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        print(f"OCR Extractor initialized. ROI: {self.roi}, Tesseract CMD: {TESSERACT_CMD}")

    def _preprocess_image_for_ocr(self, image_roi):
        """
        Applies advanced image preprocessing steps to enhance OCR accuracy,
        especially for potentially noisy CCTV footage.
        Steps include: Grayscale, Denoising, Thresholding, and sometimes Dilation/Erosion.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Non-Local Means Denoising - often more effective than Gaussian blur for text in noisy images
        # The parameters (h, hColor, templateWindowSize, searchWindowSize) need tuning for optimal results.
        # h: filter strength, hColor: same as h but for color images (not applicable for grayscale here),
        # templateWindowSize: odd size of patch to compute weight, searchWindowSize: odd size of window to search similar patches.
        denoised = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21) # Adjusted parameters for potentially better text clarity

        # Apply adaptive thresholding to convert to binary image.
        # This helps in segmenting text from background in varying lighting conditions.
        # cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU is good for dark text on a light background, or vice-versa.
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Optional: Dilation/Erosion for character thickening/thinning. 
        # Can be useful if characters are too thin or too thick after thresholding.
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)) # Small kernel for fine adjustments
        # thresh = cv2.dilate(thresh, kernel, iterations=1)
        # thresh = cv2.erode(thresh, kernel, iterations=1)

        return thresh

    def extract_time(self, frame):
        """
        Extracts the timestamp string from the specified ROI in the given frame using OCR.
        It specifically looks for a `DD/MM/YYYY HH:MM:SS AM/PM` pattern and then extracts
        the `HH:MM:SS AM/PM` portion.

        Args:
            frame (numpy.ndarray): The current video frame.

        Returns:
            str: The extracted time string (e.g., "02:41:08 PM"), 
                 or "N/A" if extraction fails or is invalid.
        """
        # Validate OCR_ROI configuration
        if not (isinstance(self.roi, tuple) and len(self.roi) == 4 and all(isinstance(x, int) for x in self.roi)):
            print(f"Error: OCR_ROI is not correctly defined as (x1, y1, x2, y2) in config.py. Current: {self.roi}")
            return "N/A"

        x1, y1, x2, y2 = self.roi
        
        # Ensure ROI coordinates are within frame dimensions to prevent errors
        h, w, _ = frame.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, h) # Ensure y2 is within frame height

        if x2 <= x1 or y2 <= y1:
            print(f"Warning: Invalid OCR_ROI coordinates: {self.roi}. Ensure x2 > x1 and y2 > y1 and within frame bounds (W:{w}, H:{h}).")
            return "N/A"

        # Crop the frame to the defined ROI for targeted OCR
        image_roi = frame[y1:y2, x1:x2]

        if image_roi.size == 0:
            print(f"Warning: Cropped image ROI is empty for coordinates: {self.roi}. Skipping OCR.")
            return "N/A"

        # Preprocess the cropped image to optimize for OCR
        preprocessed_image = self._preprocess_image_for_ocr(image_roi)

        # Convert the OpenCV image (numpy array) to a PIL Image object, which pytesseract prefers
        pil_image = Image.fromarray(preprocessed_image)

        # Perform OCR using Pytesseract
        try:
            # `config` parameter helps Tesseract achieve better results for specific layouts.
            # `--psm 6`: Page Segmentation Mode 6 - Assume a single uniform block of text.
            # `--oem 3`: OCR Engine Mode 3 - Use both LSTM (neural network) and legacy engine.
            # -c tessedit_char_whitelist="..." helps to restrict characters to only digits, slashes, colons, spaces, A, P, M.
            # This can significantly reduce gibberish if the time format is strict.
            tesseract_config = '--psm 6 --oem 3 -c tessedit_char_whitelist="0123456789/: APM"'
            full_text = pytesseract.image_to_string(pil_image, config=tesseract_config)
            
            # Clean up the extracted text: remove newlines, extra spaces, and leading/trailing whitespace
            cleaned_text = full_text.replace('\n', ' ').strip()
            
            # Use a robust regular expression to find the full date and time pattern:
            # DD/MM/YYYY HH:MM:SS AM/PM
            # Then extract only the time part (HH:MM:SS AM/PM)
            # Regex breakdown:
            # \d{2}/\d{2}/\d{4} : Matches DD/MM/YYYY
            # \s             : Matches a single space
            # \d{1,2}:\d{2}:\d{2} : Matches HH:MM:SS (1 or 2 digits for hour)
            # \s             : Matches a single space
            # (?:AM|PM)      : Matches AM or PM (non-capturing group)
            full_timestamp_match = re.search(r'\d{2}/\d{2}/\d{4}\s(\d{1,2}:\d{2}:\d{2}\s(?:AM|PM))', cleaned_text)
            
            if full_timestamp_match:
                return full_timestamp_match.group(1) # Return only the time part
            else:
                # If the full pattern isn't found, try to find just a robust time pattern (HH:MM:SS AM/PM or HH:MM AM/PM)
                time_only_match = re.search(r'(\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})\s(?:AM|PM)', cleaned_text)
                if time_only_match:
                    return time_only_match.group(0) # Return the full time string including AM/PM
                else:
                    return cleaned_text if cleaned_text else "N/A" # Fallback
        except Exception as e:
            # Catch any exceptions during OCR (e.g., Tesseract not found, image issues)
            print(f"Error during OCR extraction: {e}")
            return "N/A"