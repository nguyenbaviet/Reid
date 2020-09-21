data=cfg/box.data
num_cluster=9
width=416
height=416
./darknet detector calc_anchors $data -num_of_clusters $num_cluster -width $width -height $height