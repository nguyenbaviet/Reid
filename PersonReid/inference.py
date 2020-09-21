import os, sys
sys.path.append(os.getcwd())

from PersonReid.torchreid import models, data, utils, metrics
from PersonReid.torchreid import engine
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image

class Inference(engine.Engine):
    def __init__(self, datamanager, model, dist_metric='euclidean', image_size=(256, 128), device='cuda'):
        super(Inference, self).__init__(datamanager=datamanager)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.gallery = datamanager.test_dataset['ghtk_dataset']['gallery']
        # self.g_features = self._feature_extraction(datamanager.test_loader['ghtk_dataset']['gallery'])
        # torch.save(self.g_features, 'gallery.pt')
        self.g_features = torch.load('gallery.pt')
        self.dist_metric = dist_metric
        self.image_size = image_size

    def preprocess_image(self, images, pixel_mean=[0.485, 0.456, 0.406], pixel_std=[0.229, 0.224, 0.225], pixel_norm=True):
        def pre_process():
            transforms = []
            transforms += [T.Resize(self.image_size)]
            transforms += [T.ToTensor()]
            if pixel_norm:
                transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
            return T.Compose(transforms)
        data = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = T.ToPILImage()(image)
            image = pre_process()(image)
            data.append(image)
        data = torch.stack(data, dim=0)
        data = data.to(self.device)
        return data
    def _feature_extraction(self, data_loader):
        f_, pids_, camids_ = [], [], []
        for batch_idx, data in enumerate(data_loader):
            imgs, pids, camids = self.parse_data_for_eval(data)
            if self.use_gpu:
                imgs = imgs.cuda()
            features = self.extract_features(imgs)
            features = features.data.cpu()
            f_.append(features)
            # pids_.extend(pids)
            # camids_.extend(camids)
        f_ = torch.cat(f_, 0)
        f_ = f_.to(self.device)
        # pids_ = np.asarray(pids_)
        # camids_ = np.asarray(camids_)
        return f_, pids_, camids_
    def inference(self, queries):
        assert isinstance(queries, list), 'Query must be a list of images!!!'
        ids = []
        d = []

        features = self.model(self.preprocess_image(queries))
        distmat = metrics.compute_distance_matrix(features, self.g_features[0])
        distmat = distmat.cpu().detach().numpy()
        indices = np.argsort(distmat, axis=1)
        for i, id in enumerate(indices[:, 0]):
            ids.append(self.gallery[id][1])
            # d.append(distmat[i][id])
        return ids


if __name__ == '__main__':
    datamanager = data.ImageDataManager(
        root='dataset',
        sources='market1501',
        height=256,
        width=128,
        combineall=False,
        batch_size_train=128,
        batch_size_test=32,
        num_instances=4,
        train_sampler='RandomIdentitySampler'
    )
    model = models.build_model(
        name='resnet50',
        num_classes=datamanager.num_train_pids,
        loss='triplet'
    )
    weights = '/home/vietnb/reid/log/resnet50-triplet-market1501/model/model.pth.tar-10'
    utils.load_pretrained_weights(model, weight_path=weights)

    query = ['/home/vietnb/reid/1000_c1s1_003.jpg.jpg', '/home/vietnb/reid/1000_c1s1_003.jpg.jpg']
    infer = Inference(datamanager=datamanager, model=model)
    infer.inference(query)