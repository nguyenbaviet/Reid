data=cfg/box.data
cfg=cfg/box.cfg
weights=backup/box_best.weights
thresh=0.25
input=dataset/train.txt
./darknet detector test $data $cfg $weights -thresh $thresh -dont_show -save_labels < $input