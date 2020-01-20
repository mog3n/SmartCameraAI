import time
from pydarknet import Detector, Image
import cv2


def main():
    vid = cv2.VideoCapture('video3.mp4')

    average_time = 0
    net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"),
                   bytes("models/yolov3.weights", encoding="utf-8"),
                   0,
                   bytes("models/coco.data", encoding="utf-8"))

    while vid.isOpened():
        r, frame = vid.read()
        if not r:
            print("Something wrong with the file")
            break

        start_time = time.time()

        clean_frame = Image(frame)
        results = net.detect(clean_frame)

        # FPS Check
        end_time = time.time()
        average_time = average_time * 0.8 + (end_time - start_time) * 0.2
        fps = 1 / (end_time - start_time)
        print(fps)

        end_time = time.time()
        average_time = average_time * 0.8 + (end_time - start_time) * 0.2

        for cat, score, bounds in results:
            x, y, w, h = bounds
            cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0))
            cv2.putText(frame, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

        cv2.imshow("Darknet Results", frame)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()
