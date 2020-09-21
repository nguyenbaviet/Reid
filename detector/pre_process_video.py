import os
import cv2
from sklearn.model_selection import train_test_split
import shutil
import random
import glob
import xml.etree.ElementTree as ET

def video2images(video,save_dir, rd=True):
    """
    Capture video to images
    """
    def get_random_value():
        return random.randint(10, 30)
    basename = os.path.splitext(os.path.split(video)[-1])[0]
    save_dir = os.path.join(save_dir, basename)
    if os.path.isdir(save_dir):
        print('Directory %s existed' %save_dir)
        return
    os.mkdir(save_dir)
    vid = cv2.VideoCapture(video)
    id = 0
    dt = 0
    if rd:
        rd_value = get_random_value()
    while True:
        _, frame = vid.read()
        if frame is None:
            break
        if rd:
            if rd_value == dt:
                dt = 0
                rd_value = get_random_value()
                save_name = os.path.join(save_dir, basename + '%.06d.jpg' % id)
                cv2.imwrite(save_name, frame)
                id += 1

        else:
            save_name = os.path.join(save_dir, basename + '%.06d.jpg' % id)
            cv2.imwrite(save_name, frame)
            id += 1
        dt += 1

# base_dir = 'C:/Users/PC/Desktop/vid_2508'
# save_dir = 'C:/Users/PC/Desktop/processed_imgs'
# list_vid = [os.path.join(base_dir, v) for v in os.listdir(base_dir)]
# for v in list_vid:
#     video2images(v, save_dir)
# base_dir = 'C:/Users/PC/Desktop/vid_2608'
# list_vid = [os.path.join(base_dir, v) for v in os.listdir(base_dir)]
# for v in list_vid:
#     video2images(v, save_dir)

def split_train_test(src, dest, test_size=0.2):
    """
    src: list img folders
    dest: folders contain train test folder
    """
    list_folder = [os.path.join(src, f) for f in os.listdir(src)]
    list_img = []
    for folder in list_folder:
        tmp_imgs = [os.path.join(folder, img) for img in os.listdir(folder)]
        list_img += tmp_imgs
    imgs_train, imgs_test = train_test_split(list_img, test_size=test_size)
    train_dest = os.path.join(dest, 'train')
    test_dest = os.path.join(dest, 'test')
    for img in imgs_train:
        shutil.copy(img, train_dest)
    for img in imgs_test:
        shutil.copy(img, test_dest)
# src = 'C:/Users/PC/Desktop/processed_imgs'
# dest = 'C:/Users/PC/Desktop/dataset'
# split_train_test(src, dest)

def make_train_test_file(src):
    list_file = [os.path.split(f)[-1] for f in glob.glob(src + '/*.jpg' )]
    random.shuffle(list_file)
    train_file = open('dataset/train.txt', 'w')
    test_file = open('dataset/test.txt', 'w')
    for f in list_file[:-500]:
        train_file.write('dataset/train/%s\n' %f)
    for f in list_file[-500:]:
        test_file.write('dataset/train/%s\n' %f)
src = 'C:/Users/PC/Desktop/dataset/train'
make_train_test_file(src=src)

def convert_xml_to_txt(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    name = root.find('./filename').text.replace('jpg', 'txt')
    file = open('dataset/' + name, 'w')
    width = int(root.find('./size/width').text)
    height = int(root.find('./size/height').text)
    for element in root.findall('./object'):
        classname = 0 if element.find('./name').text == 'box' else 1
        xmin = int(element.find('./bndbox/xmin').text)
        xmax = int(element.find('./bndbox/xmax').text)
        ymin = int(element.find('./bndbox/ymin').text)
        ymax = int(element.find('./bndbox/ymax').text)
        x = (xmax + xmin) / 2 / width
        y = (ymax + ymin) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        file.write('%d %.6f %.6f %.6f %.6f\n' %(classname, x, y, w, h))

# for xml_file in glob.glob('C:/Users/PC/Desktop/dataset/train/*.xml'):
#     convert_xml_to_txt(xml_file)