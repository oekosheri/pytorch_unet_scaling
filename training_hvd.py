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
from dataset import Segmentation_dataset
import time
from datetime import timedelta
from models import Unet
from metric_losses import jaccard_coef
import horovod.torch as hvd



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
                #shuffle=True,
                num_replicas=hvd.size(),
                rank=hvd.rank(),
            )

        data_loader = DataLoader(
            dataset=ds,
            batch_size=args.local_batch_size if distribute else args.global_batch_size,
            pin_memory=args.use_gpu,
            #shuffle=True,
            sampler=ds_sampler,
            drop_last=True,
        )

        return data_loader

    train_dataloader = get_loader(tr_set, args, distribute=True)
    test_dataloader = get_loader(ts_set, args, distribute=True)

    return train_dataloader, test_dataloader


def train(args, train_dataloader, test_dataloader):


    # get model
    print("Creating the model ...")
    sys.stdout.flush()
    model = Unet(num_class=1).to(args.device)

    # initialize loss function and optimizer
    print("Creating the loss and optimizer ...")
    sys.stdout.flush()
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(model.parameters(), lr=args.lr)
    # wrapping the optimizer
    opt = hvd.DistributedOptimizer(opt, named_parameters=model.named_parameters(), op=hvd.Average)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(opt, root_rank=0)


    train_loss = []
    test_loss = []
    time_per_epoch = []
    lr_save = []
    trainSteps = len(train_dataloader)
    testSteps = len(test_dataloader)
    if hvd.rank() == 0:
        print(trainSteps)
        sys.stdout.flush()
    # loop over epochs
    train_time = time.time()

    print("[INFO]  training the network...")
    sys.stdout.flush()

    for e in range(args.epoch):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0
        # epoch time
        elapsed_train = time.time()
        # loop over the training set
        for (i, (x, y)) in enumerate(train_dataloader):
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
        custom_lr(opt, e + 1, lr=args.lr, num_workers=hvd.size())

        # my_lr_scheduler.step()
        # print(my_lr_scheduler.get_last_lr())
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        # avgTestLoss = totalTestLoss / testSteps
        # update our training history

        # print the model training and time every epoch
        if hvd.rank() == 0:
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
    if hvd.rank() == 0:

        df_save["time_per_epoch"] = time_per_epoch
        df_save["loss"] = train_loss
        df_save["lr"] = lr_save
        df_save["training_time"] = total_train_time
        print("Elapsed execution time: " + str(total_train_time) + " sec")
        sys.stdout.flush()

    # with open("tr_loss.txt", "w") as f:
    #     print(loss_dict, file=f)
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

    with torch.no_grad():
        model.eval()
        # loop over the validation set
        for (x, y) in test_dataloader:
            # send the input to the device
            (x, y) = (x.to(args.device), y.to(args.device))
            # make the predictions
            pred = model(x)
            # calculate the validation loss
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

    hvd.init()
    hvd.allreduce(torch.tensor([0]), name="Barrier")

    args.local_batch_size = math.ceil(args.global_batch_size / hvd.size())

    args.distributed = hvd.size() > 0


    # Device configuration
    print(f"Cuda available: {torch.cuda.is_available()} - Device count: {torch.cuda.device_count()}")
    args.use_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True  # enable built-in cuda auto tuner
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(42)
        args.device = torch.device("cuda:%d" % hvd.local_rank())

    if hvd.rank() == 0:
        print("PyTorch Settings:")
        settings_map = vars(args)
        for name in sorted(settings_map.keys()):
            print("--" + str(name) + ": " + str(settings_map[name]))
        print("")
        sys.stdout.flush()

    image_dir = args.image_dir
    mask_dir = args.mask_dir


    if args.distributed:
        hvd.allreduce(torch.tensor([0]), name="Barrier")
    print(hvd.rank())
    sys.stdout.flush()

    train_dataloader, test_dataloader = dataset(args, image_dir, mask_dir)
    if hvd.rank() == 0:
        print(len(train_dataloader), len(test_dataloader))
        sys.stdout.flush()

    model, df_save = train(args, train_dataloader, test_dataloader)



    if args.distributed:
        hvd.allreduce(torch.tensor([0]), name="Barrier")

    if hvd.rank() == 0:
        df_save = test(args, model, test_dataloader, df_save)

    df_save.to_csv("./log.csv", sep=",", float_format="%.6f")
    # destory the process group again
    if args.distributed:
        hvd.allreduce(torch.tensor([0]), name="Barrier")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training args")
    parser.add_argument("--global_batch_size", type=int, help="8 or 16 or 32")
    # parser.add_argument("--device", type=str, help="cuda" or "cpu", default="cuda")
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

    args = parser.parse_args()

    main(args)
