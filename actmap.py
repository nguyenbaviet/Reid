from PersonReid.tools.visualize_actmap import visactmap
from PersonReid.torchreid import models, data, engine
import torch
from PersonReid.torchreid.utils import load_pretrained_weights
from PersonReid.torchreid.data.datasets.image.ghtk import GHTKDataset

data.register_image_dataset('ghtk_dataset', GHTKDataset)
datamanager = data.ImageDataManager(
    root='dataset',
    sources=['ghtk_dataset', 'market1501'],
    targets='ghtk_dataset',
    height=256,
    width=128,
    combineall=False,
    batch_size_train=128,
    batch_size_test=128,
    num_instances=4,
    train_sampler='RandomIdentitySampler'
)

model = models.build_model(
    name='resnet50',
    num_classes=datamanager.num_train_pids,
    loss='triplet'
)

weights = '/home/vietnb/reid/log/resnet50-triplet-ghtk-plus/model/model.pth.tar-100'

use_gpu = torch.cuda.is_available()

if use_gpu:
    model = model.cuda()
load_pretrained_weights(model, weights)

visactmap(
    model, datamanager.test_loader, 'heatmap', 128, 256, use_gpu
)

