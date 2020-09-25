from PersonReid.inference import Inference
from detector.darknet_images import image_detection
from detector.darknet import bbox2points, load_network
import cv2
from threading import Thread
from PersonReid.torchreid.data import ImageDataManager
from PersonReid.torchreid.data.datasets.image.ghtk import GHTKDataset
from PersonReid.torchreid import models, utils, data
from queue import Queue

data.register_image_dataset('ghtk_dataset', GHTKDataset)

def video_capture(frame_queue, cap):
    """
    Get images from video and put to queue
    :param frame_queue: Queue contains images to be proceesed
    :param cap: a VideoCapture instance
    """
    while cap.isOpened():
        print('cap')
        _, frame = cap.read()
        if frame is None:
            break
        frame_queue.put(frame)
    cap.release()


def get_detections(frame_queue, detections_queue, network, classnames, class_colors, thresh, cap):
    """
    Get image from frame_queue and detect instances.Put that instances to queue
    :param frame_queue: Queue contains images to be processed
    :param detections_queue: Queue contains resized image and detection's results
    :param network: detections model
    :param classnames: classnames of detection model
    :param class_colors: color for each class detected by network
    :param thresh: threshold for detection model
    :param cap: a VideoCapture instance
    """
    while cap.isOpened():
        if len(frame_queue.queue) == 0:
            continue
        print('detect')
        frame = frame_queue.get()
        detections = image_detection(
            image_path=frame,
            network=network,
            class_names=classnames,
            class_colors=class_colors,
            thresh=thresh
        )
        detections_queue.put(detections)
    cap.release()


def crop_image(detection_queue, crop_queue, cap):
    """
    Crop instances from image
    :param detection_queue: Queue contains resized image and detection's results
    :param crop_queue: Queue contains resized image and cropped images with its bounding box coordinate
    :param cap: a VideoCapture instance
    """
    while cap.isOpened():
        if len(detection_queue.queue) == 0:
            continue
        print('crop')
        image, detections = detection_queue.get()
        instances = []
        bbox = []
        for label, confidence, box in detections:
            if label == 'box':
                continue
            xmin, ymin, xmax, ymax = list(map(int, bbox2points(box)))
            tmp_img = image[ymin:ymax, xmin:xmax]
            tmp_img = cv2.resize(tmp_img, (256, 128))
            bbox.append([xmin, ymin, xmax, ymax])
            instances.append(tmp_img)
        crop_queue.put((image, bbox, instances))
    cap.release()


def reid(crop_queue, ids_queue, model, cap):
    """
    Do reid with instances
    :param crop_queue: Queue contains resized image and cropped images with its bounding box coordinate
    :param ids_queue: Queue contains resized image and id of instance with its bounding box coordinate
    :param model: reid model
    :param cap: a VideoCapture instance
    """
    while cap.isOpened():
        if len(crop_queue.queue) == 0:
            continue
        print('reid')
        image, bbox, instances = crop_queue.get()
        ids = model.inference(instances)
        ids_queue.put((image, bbox, ids))
    cap.release()


def draw(ids_queue, cap):
    """
    Visualize the id into image and write it into a video.
    :param ids_queue: Queue contains resized image and id of instance with its bounding box coordinate
    :param cap: a VideoCapture instance
    """
    video = cv2.VideoWriter('result_%s' %out_filename, cv2.VideoWriter_fourcc(*'MJPG'), 15, (1080, 720))
    while cap.isOpened():
        if len(ids_queue.queue) == 0:
            continue
        print('draw')
        image, bbox, ids = ids_queue.get()
        if len(bbox) != 0:
            for i in range(len(bbox)):
                box = bbox[i]
                id = ids[i]
                image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                image = cv2.putText(image, '%02d' %id, (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        image = cv2.resize(image, (1080, 720))
        video.write(image)
    cap.release()


if __name__ == '__main__':
    vid_path = 'test5.mp4'
    out_filename = 'BestMonster.avi'

    # Hyper-parameters for detection
    d_weights = 'detector/backup/box_best.weights'
    d_config_file = 'detector/cfg/box.cfg'
    d_data_file = 'detector/cfg/box.data'
    d_thresh = 0.25

    #Hyper-parameters for reid
    r_weights = '/home/vietnb/reid/log/resnet50-triplet-ghtk-plus/model/model.pth.tar-100'

    cap = cv2.VideoCapture(vid_path)

    network, class_names, class_colors = load_network(
        d_config_file,
        d_data_file,
        d_weights,
        batch_size=1
    )

    reid_datamanager = ImageDataManager(
        root='dataset',
        sources=['ghtk_dataset'],
        height=256,
        width=128,
        combineall=False,
        train_sampler='RandomIdentitySampler'
    )
    model = models.build_model(
        name='resnet50',
        num_classes=reid_datamanager.num_train_pids,
        loss='triplet'
    )
    utils.load_pretrained_weights(model, r_weights)

    infer_model = Inference(datamanager=reid_datamanager, model=model)

    frame_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    crop_queue = Queue(maxsize=1)
    ids_queue = Queue(maxsize=1)

    Thread(target=video_capture, args=(frame_queue, cap)).start()
    Thread(target=get_detections, args=(frame_queue, detections_queue, network, class_names, class_colors, d_thresh, cap)).start()
    Thread(target=crop_image, args=(detections_queue, crop_queue, cap)).start()
    Thread(target=reid, args=(crop_queue, ids_queue, infer_model, cap)).start()
    Thread(target=draw, args=(ids_queue, cap)).start()