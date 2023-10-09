import os
import sys, glob
import math
import argparse
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
from datetime import timedelta
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from models import Unet
from metric_losses import jaccard_coef
from dataset import Segmentation_dataset
# import ssl

# # ssl._create_default_https_context = ssl._create_unverified_context


def custom_lr(optimizer, epoch, lr=0.001, num_workers=1):
    # optimised for 150 epochs
    for pm in optimizer.param_groups:
        # print(epoch, pm["lr"])
        if epoch < 80:
            pm["lr"] = lr * num_workers
        # if epoch == 0:
        #     pm["lr"] = lr
        # elif epoch > 0 and epoch < 10:
        #     increment = ((lr * num_workers) - lr) / 10
        #     pm["lr"] = pm["lr"] + increment
        # elif epoch >= 10 and epoch < 40:
        #     pm["lr"] = lr * num_workers
        elif epoch >= 80 and epoch < 120:
            pm["lr"] = lr / 2 * num_workers
        elif epoch >= 120 and epoch < 150:
            pm["lr"] = lr / 4 * num_workers
        elif epoch >= 150:
            pm["lr"] = lr / 8 * num_workers


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

    def get_loader(ds, args, distribute=True):
        ds_sampler = None
        if distribute:
            ds_sampler = DistributedSampler(
                dataset=ds,
                shuffle=True,
                num_replicas=args.world_size,
                rank=args.world_rank,
            )

        data_loader = DataLoader(
            dataset=ds,
            batch_size=args.local_batch_size if distribute else args.global_batch_size,
            pin_memory=args.use_gpu,
            shuffle=ds_sampler is None,
            sampler=ds_sampler,
            drop_last=True,
        )

        return data_loader

    train_dataloader = get_loader(tr_set, args, distribute=True)
    test_dataloader = get_loader(ts_set, args, distribute=True)

    return train_dataloader, test_dataloader


def train(args, train_dataloader, test_dataloader):
    # size = next(iter(train_dataloader))[0].shape
    # B, C, H, W = size[0], size[1], size[2], size[3]
    # print("Train B,C,H,W", B, C, H, W)

    # get model
    model = Unet(num_class=1).to(args.device)
    # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.distributed:
        dist.barrier()
        model = (
            DDP(model, device_ids=[args.local_rank], bucket_cap_mb=args.bucket_cap_mb)
            if args.use_gpu
            else DDP(model, bucket_cap_mb=args.bucket_cap_mb)
        )

    # initialize loss function and optimizer
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(model.parameters(), lr=args.lr)

    train_loss = []
    # test_loss = []
    time_per_epoch = []
    lr_save = []
    trainSteps = len(train_dataloader)
    # testSteps = len(test_dataloader)
    if args.world_rank == 0:
        print(trainSteps)
        sys.stdout.flush()
    # loop over epochs
    train_time = time.time()

    print("[INFO]  training the network...")

    for e in range(args.epoch):

        model.train()
        totalTrainLoss = 0

        # epoch time
        elapsed_train = time.time()
        # loop over the training set
        for i, (x, y) in enumerate(train_dataloader):
            # send the input to the device
            (x, y) = (
                x.to(args.device, non_blocking=True),
                y.to(args.device, non_blocking=True),
            )
            # perform a forward pass and calculate the training loss
            pred = model(x)
            loss = lossFunc(pred, y)
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far
            totalTrainLoss += loss.item()

        elapsed_train = time.time() - elapsed_train
        lr_save.append(get_lr(opt))

        custom_lr(opt, e + 1, lr=args.lr, num_workers=args.world_size)

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps


        # print the model training and time every epoch
        if args.world_rank == 0:
            train_loss.append(avgTrainLoss)
            # test_loss.append(avgTestLoss)
            time_per_epoch.append(elapsed_train)

            print("[INFO] EPOCH: {}/{}".format(e + 1, args.epoch))
            print(
                "Train loss: {:.6f}, elapsed time: {}".format(
                    avgTrainLoss, elapsed_train
                )
            )
            sys.stdout.flush()
        # custom_lr(opt, e + 1, lr=args.lr, num_workers=args.world_size)
        # lr.append(get_lr(opt))

    total_train_time = time.time() - train_time
    df_save = pd.DataFrame()
    if args.world_rank == 0:

        df_save["time_per_epoch"] = time_per_epoch
        df_save["loss"] = train_loss
        df_save["lr"] = lr_save
        df_save["training_time"] = total_train_time
        print("Elapsed execution time: " + str(total_train_time) + " sec")
        sys.stdout.flush()


    return model, df_save


def test(args, model, test_dataloader, df_save):
    size = next(iter(test_dataloader))[0].shape
    B, C, H, W = size[0], size[1], size[2], size[3]
    # print("Test B,C,H,W", B, C, H, W)
    testSteps = len(test_dataloader)

    # saving the validation set in eval loop

    inputs = torch.zeros((len(test_dataloader) * B, C, H, W))
    labels = torch.zeros((len(test_dataloader) * B, C, H, W))
    predicts = torch.zeros((len(test_dataloader) * B, C, H, W))

    lossFunc = BCEWithLogitsLoss()
    test_loss = []
    totalTestLoss = 0
    # switch off autograd
    s = 0
    elapsed_eval = time.time()

    # if args.distributed:
    #     model.module.eval()
    # else:
    #     model.eval()

    with torch.no_grad():
        model.eval()
        # loop over the validation set
        for x, y in test_dataloader:
            # send the input to the device
            (x, y) = (x.to(args.device), y.to(args.device))
            # make the predictions and calculate the validation loss
            # if args.distributed:
            #     pred = model.module(x)
            # else:
            #     pred = model(x)
            pred = model(x)

            totalTestLoss += lossFunc(pred, y).item()
            # filling empty valid set tensors
            # print("x,y shape:", x.shape, y.shape)
            inputs[s : s + B] = x.cpu()
            labels[s : s + B] = y.cpu()
            predicts[s : s + B] = pred.cpu()
            s += B
    avgTestLoss = totalTestLoss / testSteps
    # test_loss.append(avgTestLoss)
    elapsed_eval = time.time() - elapsed_eval
    # print(test_loss, len(test_loss))
    # torch tensor outs
    labels_n = labels.numpy()  # .type(torch.int)
    preds_n = (torch.sigmoid(predicts) > 0.5).type(torch.int).numpy()

    IOU = jaccard_coef(labels_n, preds_n)
    # df_save["val_loss"] = test_loss
    df_save["test_time"] = elapsed_eval
    df_save["iou"] = IOU

    print("Test loss: {:.6f}, elapsed time: {}".format(avgTestLoss, elapsed_eval))
    print("IOU on test: {}".format(IOU))
    sys.stdout.flush()

    return df_save


def main(args):

    torch.manual_seed(1235)

    # IP address of master node to initiate processes
    # url = "tcp://" + args.node + ".hpc.itc.rwth-aachen.de:29500"
    url = "tcp://" + args.node + ':29500'
    print(url)
    sys.stdout.flush()

    args.distributed = False
    args.world_size = 1
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.distributed = args.world_size > 1

    args.world_rank = args.local_rank = 0

    if args.distributed:
        args.world_rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(
            args.backend,
            timeout=timedelta(seconds=120),
            rank=args.world_rank,
            world_size=args.world_size,
            init_method=url,
        )
        dist.barrier()  # wait until all ranks have arrived
    args.local_batch_size = math.ceil(args.global_batch_size / args.world_size)

    print(
        "world size, world rank, local rank",
        args.world_size,
        args.world_rank,
        args.local_rank,
    )
    # Device configuration
    args.use_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True  # enable built-in cuda auto tuner
        torch.cuda.set_device(args.local_rank)
        torch.manual_seed(42)
        args.device = torch.device("cuda:%d" % args.local_rank)
    else:
        args.device = "cpu"

    if args.world_rank == 0:
        print("PyTorch Settings:")
        settings_map = vars(args)
        for name in sorted(settings_map.keys()):
            print("--" + str(name) + ": " + str(settings_map[name]))
        print("")
        sys.stdout.flush()

    image_dir = args.image_dir
    mask_dir = args.mask_dir


    if args.distributed:
        dist.barrier()

    train_dataloader, test_dataloader = dataset(args, image_dir, mask_dir)
    if args.world_rank == 0:
        print(len(train_dataloader), len(test_dataloader))
        sys.stdout.flush()

    model, df_save = train(args, train_dataloader, test_dataloader)


    df_save = test(args, model, test_dataloader, df_save)

    if args.world_rank == 0:
        df_save.to_csv("./log.csv", sep=",", float_format="%.6f")
    # destory the process group again
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training args")
    parser.add_argument("--global_batch_size", type=int, help="8 or 16 or 32")
    parser.add_argument("--lr", type=float, help="ex. 0.001", default=0.001)
    parser.add_argument("--repeat", type=int, help="for dataset repeat", default=2)
    parser.add_argument("--epoch", type=int, help="iterations")
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
    parser.add_argument("--augment", type=int, help="0 is False, 1 is True", default=0)
    parser.add_argument(
        "--bucket_cap_mb",
        required=False,
        help="max message bucket size in mb",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--backend",
        required=False,
        help="Backend used by torch.distribute",
        type=str,
        default="nccl",
    )
    parser.add_argument(
        "--node",
        required=True,
        type=str,
        default=0,
    )
    args = parser.parse_args()

    main(args)
