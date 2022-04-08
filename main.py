import re
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np
from matplotlib import pyplot as plt
import easyocr
import csv
from datetime import datetime
import shortuuid
import os

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


def load_labels(path='labels.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = np.expand_dims((image - 255) / 255, axis=0)


def get_output_tensor(interpreter, index):
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    # Get all output details
    boxes = get_output_tensor(interpreter, 1)
    classes = get_output_tensor(interpreter, 3)
    scores = get_output_tensor(interpreter, 0)
    count = int(get_output_tensor(interpreter, 2))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'detection_boxes': boxes[i],
                'detection_classes': classes[i],
                'detection_scores': scores[i]
            }
            results.append(result)
    return results

detection_threshold = 0.7
region_threshold = 0.6

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0] * region.shape[1]

    plate = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate

def ocr_it(image, detections, detection_threshold, region_threshold):
    # Scores, boxes and classes above threshold
    scores = list(filter(lambda x: x > detection_threshold, detections['bounding_box']))
    boxes = detections['bounding_box'][:len(scores)]
    classes = detections['class_id'][:len(scores)]

    # Full image dimensions
    width = image.shape[1]
    height = image.shape[0]

    # Apply ROI filtering and OCR
    for idx, box in enumerate(boxes):
        roi = box * [height, width, height, width]
        region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]
        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(region)

        text = filter_text(region, ocr_result, region_threshold)

        plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        plt.show()
        return text, region

def save_results(text, region, csv_filename, folder_path):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    uuid = shortuuid.uuid()

    fileName = current_time + "-" + uuid


    img_name = '{}.jpg'.format(fileName)

    cv2.imwrite(os.path.join(folder_path, img_name), region)

    with open(csv_filename, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name, text])

def main():
    labels = load_labels()
    interpreter = Interpreter('detect.tflite')
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    cap = cv2.VideoCapture(1)

    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320, 320))
        res = detect_objects(interpreter, img, 0.8)

        print(res)
        for result in res:
        
            # ymin, xmin, ymax, xmax = result['bounding_box']
            # xmin = int(max(1, xmin * CAMERA_WIDTH))
            # xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
            # ymin = int(max(1, ymin * CAMERA_HEIGHT))
            # ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
            #
            # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            # cv2.putText(frame, labels[int(result['class_id'])], (xmin, min(ymax, CAMERA_HEIGHT - 20)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            text, region = ocr_it(img, res[1], detection_threshold, region_threshold)
            save_results(text, region, 'realtimeresults.csv', 'Detection_Images')

            '''
            try:
                text, region = ocr_it(img, res, detection_threshold, region_threshold)
                save_results(text, region, 'realtimeresults.csv', 'Detection_Images')
            except:
                pass
            '''
        cv2.imshow('Jetson Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()



if __name__ == "__main__":
    main()