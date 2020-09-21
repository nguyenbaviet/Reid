cfg=cfg/box.cfg
data=cfg/box.data
weights=yolov4.conv.137
./darknet detector train $data $cfg $weights -dont_show -map