import os, sys
sys.path.append(os.getcwd())

from PersonReid.torchreid import models, data, optim, utils
from PersonReid.torchreid import engine
from PersonReid.torchreid.data.datasets.image.ghtk import GHTKDataset

data.register_image_dataset('ghtk_dataset', GHTKDataset)

datamanager = data.ImageDataManager(
    root='dataset',
    sources=['ghtk_dataset'],
    targets=['ghtk_dataset'],
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
model = model.cuda()


optimizer = optim.build_optimizer(
    model, optim='adam', lr=3e-4
)
scheduler = optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)
path = '/home/vietnb/reid/log/resnet50-triplet-ghtk/model/model.pth.tar-50'
start_epoch = utils.load_pretrained_weights(model, path, optimizer, scheduler, transfer_learning=False)

# start_epoch = utils.resume_from_checkpoint(
#     path,
#     model,
#     optimizer
# )

my_engine = engine.ImageTripletEngine(
    datamanager, model, optimizer, margin=0.3,
    weight_t=0.7, weight_x=1, scheduler=scheduler
)

my_engine.run(
    save_dir='log/resnet50-triplet-ghtk',
    max_epoch=50,
    # start_epoch=start_epoch,
    print_freq=20,
    eval_freq=10,
    fixbase_epoch=20,
    open_layers='classifier',
    test_only=True,
    visrank_topk=5,
    visrank=True
)
# my_engine.run(
#     max_epoch=50,
#     save_dir='log/resnet50-triplet-ghtk',
#     print_freq=10,
#     test_only=False
# )