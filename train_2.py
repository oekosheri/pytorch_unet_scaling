import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
# from torch.utils.tensorboard import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torchvision import datasets, transforms, models
import horovod.torch as hvd
from sklearn.model_selection import train_test_split
import os, sys
import math
from tqdm import tqdm
from models import Unet
from metric_losses import jaccard_coef
from torch.optim import Adam
from dataset import Segmentation_dataset
import time

parser = argparse.ArgumentParser(description='Elastic PyTorch UNET training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
#                     help='path to training data')
# parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/validation'),
#                     help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--batches-per-commit', type=int, default=100,
                    help='number of batches processed before calling `state.commit()`; '
                         'commits prevent losing progress if an error occurs, but slow '
                         'down training.')
parser.add_argument('--batches-per-host-check', type=int, default=10,
                    help='number of batches processed before calling `state.check_host_updates()`; '
                         'this check is very fast compared to state.commit() (which calls this '
                         'as part of the commit process), but because still incurs some cost due '
                         'to broadcast, so we may not want to perform it every batch.')
parser.add_argument('--batch-size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=16,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.001,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument(
    "--image_dir",
    type=str,
    help="directory of images",
    default=str(os.environ["WORK"]) + "/images_collective",
)
parser.add_argument(
    "--mask_dir",
    type=str,
    help="directory of masks",
    default=str(os.environ["WORK"]) + "/masks_collective",
)
parser.add_argument("--repeat", type=int, help="for dataset repeat", default=2)
parser.add_argument("--augment", type=int, help="0 is False, 1 is True", default=0)
lossFunc = BCEWithLogitsLoss()

def custom_lr(optimizer, epoch, lr=0.001, num_workers=1):
    # optimised for 150 epochs
    for pm in optimizer.param_groups:
        # print(epoch, pm["lr"])
        if epoch < 100:
            pm["lr"] = lr * num_workers
        # if epoch == 0:
        #     pm["lr"] = lr
        # elif epoch > 0 and epoch < 10:
        #     increment = ((lr * num_workers) - lr) / 10
        #     pm["lr"] = pm["lr"] + increment
        # elif epoch >= 10 and epoch < 40:
        #     pm["lr"] = lr * num_workers
        elif epoch >= 100 and epoch < 120:
            pm["lr"] = lr / 2 * num_workers
        elif epoch >= 120 and epoch <= 150:
            pm["lr"] = lr / 4 * num_workers
        # elif epoch >= 70:
        #     pm["lr"] = lr / 8 * num_workers

def get_lr(optimizer):
    for pm in optimizer.param_groups:
        return pm["lr"]

def dataset(args, image_dir, mask_dir):

    images = sorted(os.listdir(image_dir))
    masks = sorted(os.listdir(mask_dir))
    print(len(images))
    # split in train and test
    tr_images, ts_images, tr_masks, ts_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )
    # repeat dataset
    repeat = args.repeat
    train_im = tr_images * repeat
    train_ma = tr_masks * repeat
    test_im = ts_images  # * repeat
    test_ma = ts_masks  # * repeat

    if args.augment == 0:
        augment = False
    else:
        augment = True

    # seg datasets
    tr_set = Segmentation_dataset(
        image_dir, mask_dir, train_im, train_ma, augment=augment
    )
    ts_set = Segmentation_dataset(
        image_dir, mask_dir, test_im, test_ma, augment=augment
    )
    print(len(tr_set), len(ts_set))

    def get_loader(ds, args, distribute=True, val=False):
        ds_sampler = None
        if distribute:
            ds_sampler = hvd.elastic.ElasticSampler(
                dataset=ds,
            )

        data_loader = torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=args.val_batch_size if val else allreduce_batch_size,
            #pin_memory=args.use_gpu,
            shuffle=ds_sampler is None,
            sampler=ds_sampler,
            drop_last=True,
        )

        return ds_sampler, data_loader

    train_sampler, train_dataloader = get_loader(tr_set, args, distribute=True)
    test_sampler, test_dataloader = get_loader(ts_set, args, distribute=True, val=True)

    return train_sampler, train_dataloader, test_sampler, test_dataloader

def train(state):
    model.train()
    epoch = state.epoch
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    tic = time.time()

    batch_offset = state.batch
    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for idx, (data, target) in enumerate(train_loader):
            state.batch = batch_idx = batch_offset + idx
            if args.batches_per_commit > 0 and \
                    state.batch % args.batches_per_commit == 0:
                state.commit()
            elif args.batches_per_host_check > 0 and \
                    state.batch % args.batches_per_host_check == 0:
                state.check_host_updates()

            custom_lr(optimizer, epoch + 1, lr=args.base_lr, num_workers=hvd.size())

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                train_accuracy.update(accuracy(output, target_batch))
                loss = lossFunc(output, target_batch)
                train_loss.update(loss)
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()

            state.train_sampler.record_batch(idx, allreduce_batch_size)

            optimizer.step()
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})

            t.update(1)

    toc = time.time()

    if log_writer:
        # log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        # log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)
        # log_writer.add_scalar('train/time', toc-tic, epoch)
        return train_loss.avg, toc-tic
    state.commit()



def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    size = next(iter(val_loader))[0].shape
    B, C, H, W = size[0], size[1], size[2], size[3]
    inputs = torch.zeros((len(val_loader) * B, C, H, W))
    labels = torch.zeros((len(val_loader) * B, C, H, W))
    predicts = torch.zeros((len(val_loader) * B, C, H, W))
    s = 0
    tic = time.time()

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                inputs[s : s + B] = data.cpu()
                labels[s : s + B] = target.cpu()
                predicts[s : s + B] = output.cpu()

                val_loss.update(lossFunc(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)
                s += B

    labels_n = labels  # .type(torch.int)
    labels_n2 = hvd.allgather(labels_n).numpy()
    preds_n = (torch.sigmoid(predicts) > 0.5).type(torch.int)
    preds_n2 = hvd.allgather(preds_n).numpy()
    IOU = jaccard_coef(labels_n2, preds_n2)
    print("IOU ", IOU, " ", len(labels_n), ' ', len(labels_n2))

    toc = time.time()

    if log_writer:
        # log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        # log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        # log_writer.add_scalar('val/IOU', IOU, epoch)
        # log_writer.add_scalar('val/time', toc-tic, epoch)
        return val_loss.avg, IOU



# From horovod documentation. Not in use.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


def end_epoch(state):
    state.epoch += 1
    state.batch = 0
    state.train_sampler.set_epoch(state.epoch)
    state.commit()


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


@hvd.elastic.run
def full_train(state):
    import pandas as pd
    df = pd.DataFrame(columns=('time_per_epoch','loss','val_loss','iou'))
    while state.epoch < args.epochs:
        loss, time = train(state)
        hvd.allreduce(torch.tensor([0]), name="Barrier")
        val_loss, iou = validate(state.epoch)
        df.loc[state.epoch] = [time, loss, val_loss, iou]
        #save_checkpoint(state.epoch)
        end_epoch(state)
    hvd.allreduce(torch.tensor([0]), name="Barrier")
    df.to_csv("./log.csv", sep=",", float_format="%.6f")

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    verbose = 1 if hvd.rank() == 0 else 0

    # log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None
    log_writer = True if hvd.rank() == 0 else None
    torch.set_num_threads(4)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    image_dir = args.image_dir
    mask_dir = args.mask_dir
    images = sorted(os.listdir(image_dir))
    masks = sorted(os.listdir(mask_dir))
    train_sampler, train_loader, val_sampler, val_loader = dataset(args, image_dir, mask_dir)

    print("data loader size:", len(train_loader), len(val_loader))
    print("horovod size, rank:", hvd.size(), hvd.rank())
    sys.stdout.flush()
    model = Unet(num_class=1)

    lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

    if args.cuda:
        model.cuda()
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()

    optimizer = Adam(model.parameters(), lr=args.base_lr)
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=args.batches_per_allreduce,
        op=hvd.Adasum if args.use_adasum else hvd.Average,
        gradient_predivide_factor=args.gradient_predivide_factor)

    resume_from_epoch = 0
    if hvd.rank() == 0:
        for try_epoch in range(args.epochs, 0, -1):
            if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
                resume_from_epoch = try_epoch
                break

        if resume_from_epoch > 0:
            filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
            checkpoint = torch.load(filepath)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    state = hvd.elastic.TorchState(model=model,
                                   optimizer=optimizer,
                                   train_sampler=train_sampler,
                                   val_sampler=val_sampler,
                                   epoch=resume_from_epoch,
                                   batch=0)

    full_train(state)