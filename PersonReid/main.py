import os, sys
sys.path.append(os.getcwd())

from PersonReid.torchreid import models, data, optim, utils
from PersonReid.torchreid import engine
from PersonReid.torchreid.data.datasets.image.ghtk import GHTKDataset

#with a new dataset, you must to register it
data.register_image_dataset('ghtk_dataset', GHTKDataset)

datamanager = data.ImageDataManager(
    root='dataset',
    sources=['ghtk_dataset', 'market1501'],
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
    model, optim='adam', lr=1e-4
)
scheduler = optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

my_engine = engine.ImageTripletEngine(
    datamanager, model, optimizer, margin=0.3,
    weight_t=0.7, weight_x=1, scheduler=scheduler
)
# if you want to do transfer-learning after add more data to dataset, set transfer_learning to True.
# or not, if you just want to re-train model without adding data, set transfer_learning to False

path = '/home/vietnb/reid/log/resnet50-triplet-ghtk/model/model.pth.tar-1000'
start_epoch = utils.load_pretrained_weights(model, path, optimizer, scheduler, transfer_learning=False)

start_epoch = utils.resume_from_checkpoint(
    path,
    model,
    optimizer
)



my_engine.run(
    save_dir='log/resnet50-triplet-ghtk',
    max_epoch=50,
    start_epoch=start_epoch,
    print_freq=20,
    eval_freq=10,
    fixbase_epoch=20,
    open_layers='classifier',
    test_only=False,
    visrank_topk=5,
    visrank=True
)

# if you want to train from scratch, just comment upper block and uncomment under command

# my_engine.run(
#     max_epoch=100,
#     save_dir='log/resnet50-triplet-ghtk-plus',
#     print_freq=10,
#     test_only=False
# )