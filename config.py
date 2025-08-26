import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(BASE_DIR, 'sample_video', 'office_cctv_footage.mp4') 

OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, 'logs', 'processed_office_video.mp4')


FRAME_SKIP = 1 


YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolo11m.pt') 

CONFIDENCE_THRESHOLD = 0.5


NMS_THRESHOLD = 0.4

TARGET_CLASSES = [0]


OCR_ROI = (750, 1180, 1070, 1210) # Estimated for a down-right timestamp in 1080x1224


TESSERACT_CMD = 'tesseract'


MAX_DIST_PERSON = 70 

MAX_MISSING_FRAMES = 15 


SITTING_THRESHOLD_HEIGHT_RATIO = 1.4 


IN_ZONE = (0, 0, 300, 1224)     
OUT_ZONE = (780, 0, 1080, 1224) 


IN_TIME_WINDOW_END_SEC = 16    

OUT_TIME_WINDOW_START_SEC = 43 


LOGS_DIR = os.path.join(BASE_DIR, 'logs')
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

LOG_FILE_PATH = os.path.join(LOGS_DIR, 'office_tracking_log.txt')

CSV_EXPORT_PATH = os.path.join(LOGS_DIR, 'person_activity_report.csv')


DRAW_BBOX = True
DRAW_LABELS = True
DRAW_TIME = True

DRAW_DEBUG_ZONES = True 