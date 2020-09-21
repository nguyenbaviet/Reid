cfg=cfg/box.cfg
data=cfg/box.data
weights=backup/box_best.weights
video=test3.mp4
output=bestmonster.avi
./darknet detector demo $data $cfg $weights $video -out_filename $output