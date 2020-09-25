#Person reid

Using YoloV4 to detect people in each image and then extract features by resnet50 model to reid each person.

#1.Requirements
You should follow https://github.com/AlexeyAB/darknet to install darknet and https://github.com/KaiyangZhou/deep-person-reid to install torchreid.

#2.Detection model
- To create data, please follow https://github.com/tzutalin/labelImg to install LabelImg and using it to create data with Yolo format.
- Dataset for box and person is located in /home/vietnb/darknet/dataset
- To train model: run sh train.sh in detector's folder
- To inference video: run sh test.sh in detector's folder
- To test model with map score: run sh compute_map.sh in detector's folder
- To specific anchor sizes for your dataset: run sh compute_anchors.sh in detector's folder. The result will be used for your config file instead of default anchor sizes. Note that the size of anchors at 1st-yolo-layer should
be smaller than 30x30, at 2nd-yolo-layer should be smaller than 60x60  and the last yolo-layer should be the remaining. The filters in front of each yolo-layer 
should be equal to (num_classes + 5) * number of anchors.
- For more improvements of yolov4, please follow https://github.com/AlexeyAB/darknet#how-to-improve-object-detection

#3.Person reid
- Dataset for person in GHTK is located in /home/vietnb/reid/dataset/ghtk_dataset. It's still very small (about 230 images with 56 people). You need to create more data
 or join with some other datasets like market1501, dukemtmc,...
- You can find the instructions for torch reid in https://kaiyangzhou.github.io/deep-person-reid/user_guide
- To train model: run main.py in PersonReid's directory
- To visualize heatmap: run actmap.py
- To inference video: run main.py in root's directory
#4.To do
- Write some Data Augmentation for Person reid: flip, rotate, cutout, hue,...
- Write reid's inference for 2 cameras
- Try some other models in torchreid