from ultralytics import YOLO
import cv2
from sort import *
from util import get_car, read_license_plate, write_csv

results = {}
mot_tracker = Sort()

# Carregar os modelos utilizados
coco_model = YOLO('yolov8n.pt')
# dataset treinado
license_plate_detector = YOLO('c:/Users/louis/Desktop/CODIGOS/TCC/best.pt')

#carregar video
cap = cv2.VideoCapture('C:/Users/louis/Desktop/CODIGOS/TCC/Video1.mp4')
vehicles = [2, 3, 5, 7]

# leitura dos frames do video
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret and frame_nmr < 10:
        results[frame_nmr] = {}

# detecção de veículos
detections = coco_model(frame)[0]
detections_ = []
for detection in detections.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = detection
    if int(class_id) in vehicles:
        detections_.append([x1, y1, x2, y2, score])

#rastreio de veículos
track_ids = mot_tracker.update(np.asarray(detections_))

#detecção de placas
license_plates = license_plate_detector(frame)[0]
for license_plate in license_plates.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = license_plate

#conexão placa;carro
xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

#cortar placa
license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

#processar placa
license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
_, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64,255, cv2.THRESH_BINARY_INV)

#leitura dos dígitos
license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

if license_plate_text is not None:
    results[frame_nmr][car_id] = {'car': {'bbox':[xcar1, ycar1, xcar2, ycar2]},
                                'license_plate': {'bbox': [x1, y1, x2, y2], 
                                                'text': license_plate_text,
                                                'bbox_score': score,
                                                'text_score': license_plate_text_score}}
#resultados
write_csv(results,'./test.csv')
