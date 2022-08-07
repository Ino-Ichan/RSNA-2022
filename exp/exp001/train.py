import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random

from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import cv2

import albumentations
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import confusion_matrix

from tqdm import tqdm
import argparse
import os, sys, yaml

sys.path.append('/workspace/')
from src.utils import plot_sample_images


# import neptune.new as neptune
import wandb

import time
from contextlib import contextmanager

import timm
from timm.scheduler import CosineLRScheduler

import segmentation_models_pytorch as smp
from loguru import logger

import warnings


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def seed_torch(seed=516):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def ousm_loss(error, k=2):
    # ousm, drop large k sample
    bs = error.shape[0]
    if len(error.shape) == 2:
        error = error.mean(1)
    _, idxs = error.topk(bs - k, largest=False)
    error = error.index_select(0, idxs)
    return error


# Freeze batchnorm 2d
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


# =============================================================================
# Config
# =============================================================================
target_columns = [
    "target"
]

class Cfg:
    debug = False
    project_name = "uwmgi-2022"
    # exp_name = os.getcwd().split('/')[-1]
    device_id = 0

    df_train_path = "/workspace/data/df_train2_cut.csv"

    model_name = "timm-efficientnet-b7"
    in_channels = 3
    in_sep = 2
    num_classes = 3
    img_size = 640
    batch_size = 16
    n_workers = 10
    n_epochs = 60
    start_epoch = 61
    transform = True
    hold_out = [4]
    load_checkpoint = [
        "",
        "",
        "",
        "/workspace/output/exp511/model/cv3_weight_checkpoint_last.pth",
        "/workspace/output/exp511/model/cv4_weight_checkpoint_last.pth",
    ]
    accumulation_steps = 1
    early_stopping_steps = 100
    freeze_bn = False
    use_amp = True

    initial_lr = 5e-5
    final_lr = 1e-5
    warmup_lr_init = 1e-5
    warmup_t = 1

    fold_name = 'cv'

    best_direction = "min"





# =============================================================================
# Model and Losses
# =============================================================================

class Net(nn.Module):
    def __init__(
        self,
        name: str = "resnet18",
    ):
        super(Net, self).__init__()
        self.model = timm.create_model(name, pretrained=True, num_classes=len(target_columns), in_chans=3)

    def forward(self, x, mode="train"):
        x = self.model(x)
        return x

JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss    = smp.losses.DiceLoss(mode='multilabel')
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)

def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

def criterion(y_pred, y_true):
    return 0.5*BCELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)

# def criterion(y_pred, y_true):
#     # print(f"TverskyLoss(y_pred, y_true): {TverskyLoss(y_pred, y_true)}, DiceLoss(y_pred, y_true): {DiceLoss(y_pred, y_true)}")
#     return 0.5*TverskyLoss(y_pred, y_true) + 0.5*DiceLoss(y_pred, y_true)

# =============================================================================
# Dataset
# =============================================================================
def load_img(path, size):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (size, size))
    img = img / img.max()
    return img[:,:,None]

def load_msk(path, size):
    msk = np.load(path)
    msk = cv2.resize(msk, (size, size))
    msk = msk.astype('float32')
    msk/=255.0
    return msk

class CustomDataset(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 in_channels,
                 in_sep,
                 transform=None,
                 mode="train",
                 ):

        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.in_channels = in_channels
        self.in_sep = in_sep
        self.transform = transform

        self.mode = mode
        self.cols = target_columns

    def __len__(self):
        return self.df.shape[0]

    def get_mask_path(self, path):
        return path.replace('train', 'mask/np/uw-madison-gi-tract-image-segmentation/train').replace('.png', '.npy')

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = row.image_path
        mask_path = self.get_mask_path(image_path)

        images = load_img(image_path, self.image_size)
        masks = load_msk(mask_path, self.image_size)

        if self.in_channels > 1:
            seq_n = (self.in_channels - 1) // 2
            for i in range(1 * self.in_sep, (1 + seq_n) * self.in_sep, self.in_sep):
                # print("os.path.exists(image_path_pre{i:02})", os.path.exists(row[f"image_path_pre{i:02}"]))
                # print("os.path.exists(image_path_next{i:02})", os.path.exists(row[f"image_path_next{i:02}"]))
                images = np.concatenate([
                    load_img(row[f"image_path_pre{i:02}"], self.image_size),
                    images,
                    load_img(row[f"image_path_next{i:02}"], self.image_size),
                ], axis=-1)
                # pre_mask_path = self.get_mask_path(row[f"image_path_pre{i:02}"])
                # next_mask_path = self.get_mask_path(row[f"image_path_next{i:02}"])
                # masks = np.concatenate([
                #     load_msk(pre_mask_path, self.image_size),
                #     masks,
                #     load_msk(next_mask_path, self.image_size),
                # ], axis=-1)


        if self.transform is not None:
            aug = self.transform(image=images, mask=masks)
            images = aug['image'].astype(np.float32).transpose(2, 0, 1)
            masks = aug['mask'].astype(np.float32).transpose(2, 0, 1)
        else:
            images = images.transpose(2, 0, 1)

        return {
            "image": torch.tensor(images, dtype=torch.float),
            "target": torch.tensor(masks, dtype=torch.float),
        }


# =============================================================================
# one epoch
# =============================================================================
# @logger.catch
def train_one_epoch(train_dataloader, model, device, criterion, scheduler, wandb, cfg, e, cv, mode="train"):

    train_time = time.time()
    logger.info("")
    logger.info("+" * 30)
    logger.info(f"+++++  Epoch {e} at CV {cv}")
    logger.info("+" * 30)
    logger.info("")
    progress_bar = tqdm(train_dataloader, dynamic_ncols=True)
    iters = len(train_dataloader)

    model.train()
    torch.set_grad_enabled(True)

    # freeze bach norm
    if cfg.freeze_bn:
        model = model.apply(set_bn_eval)

    loss_list = []


    for step_train, data in enumerate(progress_bar):
        if cfg.debug:
            if step_train == 2:
                break

        inputs = data["image"].to(device)
        target = data["target"].to(device)

        bs = inputs.shape[0]

        with autocast(enabled=cfg.use_amp):
            output = model(inputs)
            loss = criterion(output, target).mean()

        if cfg.accumulation_steps > 1:
            loss_bw = loss / cfg.accumulation_steps
            scaler.scale(loss_bw).backward()
            if (step_train + 1) % cfg.accumulation_steps == 0 or step_train == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                # scheduler.step(e + step_train / iters)
                scheduler.step(e)
                optimizer.zero_grad()
        else:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # scheduler.step(e + step_train / iters)
            scheduler.step(e)

        loss_list.append(loss.item())
        text_progress_bar = f"loss: {loss.item()} loss(avg): {np.array(loss_list).mean()}"
        progress_bar.set_description(text_progress_bar)

    logger.info(f"Train loss: {np.array(loss_list).mean()}")
    logger.info(f"Train time: {(time.time() - train_time) / 60:.3f} min")

    wandb.log({
        f"epoch": e,
        f"Loss/train_cv{cv}": np.array(loss_list).mean(),
    })

# @logger.catch
def val_one_epoch(val_dataloader, model, device, wandb, cfg, e, cv, mode="val"):

    val_time = time.time()
    progress_bar = tqdm(val_dataloader, dynamic_ncols=True)

    model.eval()
    torch.set_grad_enabled(False)

    loss_list = []
    dice_coef_list = []
    pred_list = []
    target_list = []

    for step_val, data in enumerate(progress_bar):
        if cfg.debug:
            if step_val == 2:
                break

        inputs = data["image"].to(device)
        target = data["target"].to(device)

        with autocast(enabled=cfg.use_amp):
            output = model(inputs)
            loss = criterion(output, target).mean()
            dice = dice_coef(target, output)

        loss_list.append(loss.item())
        # dice_coef_list.extend(dice.cpu().numpy().tolist())
        dice_coef_list.append(dice.item())
        # pred_list.extend(output.detach().cpu().numpy().tolist())
        # target_list.extend(target.cpu().numpy().tolist())

        text_progress_bar = f"loss: {loss.item()} loss(avg): {np.array(loss_list).mean()} dice(avg): {np.array(dice_coef_list).mean()}"
        progress_bar.set_description(text_progress_bar)


    logger.info(f"Val loss: {np.array(loss_list).mean()}")
    logger.info(f"Val Dice: {np.array(dice_coef_list).mean()}")
    logger.info(f"Val time: {(time.time() - val_time) / 60:.3f} min")

    log_dict = {
        f"epoch": e,
        f"Loss/val_cv{cv}": np.array(loss_list).mean(),
        f"Dice/val_cv{cv}": np.array(dice_coef_list).mean(),
    }
    wandb.log(log_dict)

    return np.mean(loss_list), np.mean(dice_coef_list)


def get_train_transforms(image_size):
    return albumentations.Compose([
    albumentations.Resize(image_size, image_size),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    # albumentations.RandomBrightness(limit=0.3, p=0.75),
    # albumentations.RandomContrast(limit=0.3, p=0.75),

    # albumentations.OneOf([
    #     # albumentations.OpticalDistortion(distort_limit=1.),
    #     albumentations.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
    #     albumentations.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
    # ], p=0.25),

    # albumentations.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.75),
    albumentations.ShiftScaleRotate(shift_limit=0., scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.5),
    albumentations.RandomCrop(384, 384),
    # albumentations.CoarseDropout(max_holes=10, max_width=10, max_height=4),
    # ToTensorV2(p=1)
])


def get_val_transforms(image_size):
    return albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        # albumentations.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        # ),
        # ToTensorV2(p=1)
])


if __name__ == "__main__":
    print('Start!!!')
    warnings.simplefilter('ignore')

    cfg = Cfg()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e', '--exp_name',  type=str,
                        help='experiment name')
    args = parser.parse_args()

    # seed_everythin
    seed_torch()

    # output
    # exp_name = cfg.exp_name  # os.path.splitext(os.path.basename(__file__))[0]
    exp_name = args.exp_name  # os.path.splitext(os.path.basename(__file__))[0]
    output_path = os.path.join("/workspace/output", exp_name)
    # path
    model_path = output_path + "/model"
    plot_path = output_path + "/plot"
    oof_path = output_path + "/oof"
    sample_img_path = output_path + "/sample_img"

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(output_path + "/log", exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(oof_path, exist_ok=True)
    os.makedirs(sample_img_path, exist_ok=True)

    # logger
    log_path = os.path.join(output_path, "log/log.txt")
    logger.add(log_path)
    logger.info("config")
    logger.info(cfg)
    logger.info('')

    debug = cfg.debug
    if debug:
        logger.info("Debug!!!!!")

    # params
    device_id = cfg.device_id
    try:
        device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"
    except Exception as e:
        logger.info('GPU is not available, {}'.format(e))
        sys.exit()

    print(device)

    #######################################
    ## CV
    #######################################
    df = pd.read_csv(cfg.df_train_path)

    # df["frame"] = [int(i.split("-")[1]) for i in df.image_id]
    # df = df[df.frame > 5].reset_index(drop=True)


    cv_list = cfg.hold_out if cfg.hold_out else [0, 1, 2, 3, 4]
    oof = np.zeros((len(df), len(target_columns)))
    best_eval_score_list = []

    for cv in cv_list:

        logger.info('# ===============================================================================')
        logger.info(f'# Start CV: {cv}')
        logger.info('# ===============================================================================')

        # wandb
        wandb.init(config=cfg, tags=[exp_name, f"cv{cv}", cfg.model_name],
                   project=cfg.project_name, entity='inoichan',
                   name=f"{exp_name}_cv{cv}_{cfg.model_name}", reinit=True)

        df_train = df[df[cfg.fold_name] != cv].reset_index(drop=True)
        df_val = df[df[cfg.fold_name] == cv].reset_index(drop=True)
        val_index = df[df[cfg.fold_name] == cv].index

        #######################################
        ## Dataset
        #######################################
        # transform
        train_transform = get_train_transforms(cfg.img_size)
        val_transform = get_val_transforms(cfg.img_size)

        train_dataset = CustomDataset(df=df_train, image_size=cfg.img_size, in_sep=cfg.in_sep,
                                      in_channels=cfg.in_channels, transform=train_transform, mode="train")
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                      pin_memory=True, num_workers=cfg.n_workers, drop_last=True)
        # plot sample image
        # plot_sample_images(train_dataset, sample_img_path, "train", normalize="imagenet")
        # plot_sample_images(train_dataset, sample_img_path, "train", normalize=None)

        val_dataset = CustomDataset(df=df_val, image_size=cfg.img_size, in_sep=cfg.in_sep,
                                   in_channels=cfg.in_channels, transform=val_transform, mode="val")
        val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                                    pin_memory=True, num_workers=cfg.n_workers, drop_last=False)

        # plot_sample_images(val_dataset, sample_img_path, "val",  normalize="imagenet")
        # plot_sample_images(val_dataset, sample_img_path, "val", normalize=None)

        # ==== INIT MODEL
        device = torch.device(device)
        # model = Net(name=cfg.model_name).to(device)
        model = smp.Unet(
            encoder_name=cfg.model_name,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=cfg.in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=cfg.num_classes,        # model output channels (number of classes in your dataset)
            activation=None,
        ).to(device)


        optimizer = optim.AdamW(model.parameters(), lr=float(cfg.initial_lr), eps=1e-7)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs * len(train_dataloader), eta_min=float(cfg.final_lr))
        scheduler = CosineLRScheduler(
            optimizer, t_initial=cfg.n_epochs, lr_min=float(cfg.final_lr),
            warmup_t=cfg.warmup_t, warmup_lr_init=cfg.warmup_lr_init, warmup_prefix=True
        )
        # scheduler = optim.lr_scheduler.OneCycleLR(
        #     optimizer, max_lr=float(cfg.initial_lr), 
        #     steps_per_epoch=len(train_dataloader), epochs=cfg.n_epochs
        # )

        # criterion = nn.BCEWithLogitsLoss(reduction='none')
        scaler = GradScaler(enabled=cfg.use_amp)

        # load weight
        load_checkpoint = cfg.load_checkpoint[cv]
        logger.info("-" * 10)
        if os.path.exists(load_checkpoint):
            weight = torch.load(load_checkpoint, map_location=device)
            model.load_state_dict(weight["state_dict"])
            logger.info(f"Successfully loaded model, model path: {load_checkpoint}")
            # optimizer.load_state_dict(["optimizer"])
            # for state in optimizer.state.values():
            #     for k, v in state.items():
            #         if isinstance(v, torch.Tensor):
            #             state[k] = v.to(device)
        else:
            logger.info(f"Training from scratch..")
        logger.info("-" * 10)

        # wandb misc
        wandb.watch(model)

        # ==== TRAIN LOOP

        best = -1e5
        best_score = -1e5

        best_epoch = 0
        early_stopping_cnt = 0

        for e in range(cfg.start_epoch , cfg.start_epoch + cfg.n_epochs):
            if e > 0:
                wandb.log({
                    "Learning Rate": optimizer.param_groups[0]["lr"],
                    "epoch": e
                })
                # scheduler_warmup.step(e-1)
                train_one_epoch(train_dataloader, model, device, criterion, scheduler, wandb, cfg, e, cv)

            val_loss, score = val_one_epoch(val_dataloader, model, device, wandb, cfg, e, cv)

            logger.info('Saving last model ...')
            model_save_path = os.path.join(model_path, f"cv{cv}_weight_checkpoint_last.pth")

            torch.save({
                "state_dict": model.state_dict(),
                # "optimizer": optimizer.state_dict()
            }, model_save_path)

            if best < score:
                logger.info(f'Best score update: {best:.5f} --> {score:.5f}')
                best = score
                best_epoch = e

                logger.info('Saving best model ...')
                model_save_path = os.path.join(model_path, f"cv{cv}_weight_checkpoint_best.pth")

                torch.save({
                    "state_dict": model.state_dict(),
                    # "optimizer": optimizer.state_dict()
                }, model_save_path)

                early_stopping_cnt = 0
            else:
                # early stopping
                early_stopping_cnt += 1
                if early_stopping_cnt >= cfg.early_stopping_steps:
                    logger.info(f"Early stopping at Epoch {e}")
                    break

            logger.info('-' * 20)
            logger.info(f'Best val score: {best}, at epoch {best_epoch} cv{cv} exp {exp_name}')
            logger.info('-' * 20)

            best_eval_score_list.append(best)
            wandb.log({
                "epoch": e,
                "Best AUC": best,
            })

    #######################################
    ## Save oof
    #######################################
    mean_score = np.mean(best_eval_score_list)
    std_score = np.std(best_eval_score_list)
    logger.info('-' * 20)
    logger.info(f'Oof score: {mean_score} ± {std_score}')
    logger.info('-' * 20)