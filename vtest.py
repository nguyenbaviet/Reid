import os, shutil


base_dir = 'C:/Users/PC/Desktop/reid_data'
save_dir = 'C:/Users/PC/Desktop/temp'
for folder in os.listdir(base_dir):
    label = int(folder) + 1 # 0 is background
    dir = os.path.join(base_dir, folder)
    i = 0
    for id, file in enumerate(os.listdir(dir)):
        # i += 1
        # cam_id = id % 2 + 1
        # name = '%04d_c%d_%04d.jpg' %(label, cam_id, i)
        # os.rename(os.path.join(dir, file), os.path.join(dir, name))
        shutil.copy(os.path.join(dir, file), save_dir)