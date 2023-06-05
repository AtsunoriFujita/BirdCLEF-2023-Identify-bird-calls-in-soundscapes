import argparse
import importlib
import os
import pickle
import random
import sys
import warnings
from copy import copy

import audiomentations as AA
import librosa
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
import torch_audiomentations as TAA
from sklearn import metrics
from timm.scheduler import CosineLRScheduler
from torch.cuda.amp import GradScaler, autocast
from torch.distributions import Beta
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from tqdm import tqdm
from utils import AverageMeter

sys.path.append('../configs')
sys.path.append('./samplers')
sys.path.append('./pcen')
from pcen import StreamingPCENTransform
from sampler import MultilabelBalancedRandomSampler

warnings.filterwarnings("ignore")
TRAIN_DATA_PATH = '../input/'


parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)
CFG = copy(importlib.import_module(parser_args.config).cfg)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_logger(log_file='train.log'):
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def load_wave_and_crop(filename, period, start=None):

    waveform_orig, sample_rate = librosa.load(filename, sr=32000, mono=True, duration=60)

    wave_len = len(waveform_orig)
    waveform = np.concatenate([waveform_orig, waveform_orig, waveform_orig])

    effective_length = sample_rate * period
    while len(waveform) < (period * sample_rate * 3):
        waveform = np.concatenate([waveform, waveform_orig])
    if start is not None:
        start = start - (period - 5) / 2 * sample_rate
        while start < 0:
            start += wave_len
        start = int(start)
    else:
        if wave_len < effective_length:
            start = np.random.randint(effective_length - wave_len)
        elif wave_len > effective_length:
            start = np.random.randint(wave_len - effective_length)
        elif wave_len == effective_length:
            start = 0

    waveform_seg = waveform[start: start + int(effective_length)]

    return waveform_orig, waveform_seg, sample_rate, start


tr_transforms = AA.Compose(
    [
        AA.OneOf([
            AA.Gain(min_gain_in_db=-15, max_gain_in_db=15, p=1.0),
            AA.GainTransition(
                min_gain_in_db=-24.0,
                max_gain_in_db=6.0,
                min_duration=0.2,
                max_duration=6.0,
                p=1.0
            )
        ], p=0.5,),
        AA.OneOf([
            AA.AddGaussianNoise(p=1.0),
            AA.AddGaussianSNR(p=1.0),
        ], p=0.3,),
        AA.OneOf([
            AA.AddShortNoises(
                sounds_path="../input/ff1010bird_nocall/nocall",
                min_snr_in_db=0,
                max_snr_in_db=3,
                p=1.0,
                lru_cache_size=10,
                min_time_between_sounds=4.0,
                max_time_between_sounds=16.0,
            ),
            AA.AddShortNoises(
                sounds_path="../input/esc50/use_label",
                min_snr_in_db=0,
                max_snr_in_db=3,
                p=1.0,
                lru_cache_size=10,
                min_time_between_sounds=4.0,
                max_time_between_sounds=16.0,
            ),
        ], p=0.5,),
        AA.OneOf([
            AA.AddBackgroundNoise(
                sounds_path="../input/train_soundscapes/nocall",
                min_snr_in_db=0,
                max_snr_in_db=3,
                p=1.0,
                lru_cache_size=3,),
            AA.AddBackgroundNoise(
                sounds_path="../input/aicrowd2020_noise_30sec/noise_30sec",
                min_snr_in_db=0,
                max_snr_in_db=3,
                p=1.0,
                lru_cache_size=450,),
        ], p=0.5,),
        AA.LowPassFilter(p=0.5),
    ]
)

taa_augmentation = TAA.Compose(
    transforms=[
        TAA.PitchShift(
            sample_rate=CFG.sample_rate,
            mode="per_example",
            p=0.2,
            ),
    ]
)


class BirdClef2023Dataset(torchdata.Dataset):
    def __init__(
        self,
        data_path: str = 'DATA_PATH',
        period: float = 5.0,
        secondary_coef: float = 1.0,
        smooth_label: float = 0.05,
        df: pd.DataFrame = 'DATAFRAME',
        train: bool = True,
    ):

        self.df = df
        self.data_path = data_path
        self.filenames = df["filename"]

        self.primary_label = df["primary_label"]
        self.secondary_labels = (
            df["secondary_labels"]
            .map(
                lambda s: s.replace("[", "")
                .replace("]", "")
                .replace(",", "")
                .replace("'", "")
                .split(" ")
            ).values
        )

        self.secondary_coef = secondary_coef
        self.type = df["type"]
        self.teacher_preds = df["teacher_preds"]
        self.rating = df["rating"]
        self.period = period
        self.smooth_label = smooth_label + 1e-6
        self.wave_transforms = tr_transforms
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        filename = os.path.join(self.data_path, self.filenames[idx])

        if self.train:
            waveform, waveform_seg, sample_rate, start = load_wave_and_crop(
                filename, self.period
            )
            waveform_seg = self.wave_transforms(
                samples=waveform_seg, sample_rate=sample_rate
                )
        else:
            waveform, waveform_seg, sample_rate, start = load_wave_and_crop(
                filename, self.period, 0
            )

        waveform_seg = torch.from_numpy(np.nan_to_num(waveform_seg)).float()

        rating = self.rating[idx]

        teacher_preds = torch.from_numpy(np.nan_to_num(self.teacher_preds[idx])).float()

        target = np.zeros(CFG.num_classes, dtype=np.float32)
        if self.primary_label[idx] != 'nocall':
            primary_label = CFG.bird2id[self.primary_label[idx]]
            target[primary_label] = 1.0

            if self.train:
                for s in self.secondary_labels[idx]:
                    if s != "" and s in CFG.bird2id.keys():
                        target[CFG.bird2id[s]] = self.secondary_coef

        target = torch.from_numpy(target).float()

        return {
            "wave": waveform_seg,
            "rating": rating,
            "primary_targets": (target > 0.5).float(),
            "loss_target": target * (1-self.smooth_label) + self.smooth_label / target.size(-1),
            "teacher_preds": teacher_preds,
        }


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)


def gem_freq(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1.0 / p)


class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem_freq(x, p=self.p, eps=self.eps)


class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        norm_min, norm_max = \
            Xstd.min(-1)[0].min(-1)[0], Xstd.max(-1)[0].max(-1)[0]
        fix_ind = (norm_max - norm_min) > self.eps * torch.ones_like(
            (norm_max - norm_min)
        )
        V = torch.zeros_like(Xstd)
        if fix_ind.sum():
            V_fix = Xstd[fix_ind]
            norm_max_fix = norm_max[fix_ind, None, None]
            norm_min_fix = norm_min[fix_ind, None, None]
            V_fix = torch.max(
                torch.min(V_fix, norm_max_fix),
                norm_min_fix,
            )
            V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
            V[fix_ind] = V_fix
        return V


class Mixup(nn.Module):
    def __init__(self, mix_beta):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None, teacher_preds=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            teacher_preds = coeffs.view(-1, 1) * teacher_preds + (1 - coeffs.view(-1, 1)) * teacher_preds[perm]
            return X, Y, weight, teacher_preds


class AttModel(nn.Module):
    def __init__(
        self,
        backbone="resnet34",
        num_class=397,
        train_period=15.0,
        infer_period=5.0,
        in_chans=1,
    ):
        super().__init__()

        if CFG.use_pcen:
            self.pcen = StreamingPCENTransform(
                n_mels=CFG.n_mels,
                n_fft=CFG.n_fft,
                hop_length=CFG.hop_length,
                trainable=True,
                use_cuda_kernel=False
                )
        else:
            self.logmelspec_extractor = nn.Sequential(
                MelSpectrogram(
                    sample_rate=CFG.sample_rate,
                    n_mels=CFG.n_mels,
                    f_min=CFG.fmin,
                    f_max=CFG.fmax,
                    n_fft=CFG.n_fft,
                    hop_length=CFG.hop_length,
                    normalized=True,
                ),
                AmplitudeToDB(top_db=80.0),
                NormalizeMelSpec(),
            )

        base_model = timm.create_model(
            backbone,
            features_only=False,
            pretrained=CFG.pretrained,
            in_chans=CFG.in_channels
        )

        layers = list(base_model.children())[:-2]
        self.backbone = nn.Sequential(*layers)
        if "efficientnet" in CFG.backbone:
            dense_input = base_model.num_features
        elif hasattr(base_model, "fc"):
            dense_input = base_model.fc.in_features
        else:
            dense_input = base_model.feature_info[-1]["num_chs"]

        self.train_period = train_period
        self.infer_period = infer_period
        self.factor = int(self.train_period / self.infer_period)
        self.mixup = Mixup(mix_beta=1)
        self.global_pool = GeM()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])
        self.head = nn.Linear(dense_input, num_class)

    def forward(self, input):

        if self.training:
            x = input['wave']
            bs, time = x.shape
            x = x.reshape(bs * self.factor, time // self.factor)
            y = input["loss_target"]
            weight = input["rating"]
            teacher_preds = input["teacher_preds"]
        else:
            x = input['wave']
            y = input["loss_target"]
            weight = input["rating"]
            teacher_preds = input["teacher_preds"]

        if CFG.use_pcen:
            x = self.pcen(x).unsqueeze(1)
            self.pcen.reset()
        else:
            x = self.logmelspec_extractor(x)[:, None]

        if self.training:
            if np.random.random() <= 0.5:
                y2 = torch.repeat_interleave(y, self.factor, dim=0)
                weight2 = torch.repeat_interleave(weight, self.factor, dim=0)
                teacher_preds2 = torch.repeat_interleave(teacher_preds, self.factor, dim=0)

                for i in range(0, x.shape[0], self.factor):
                    x[i: i + self.factor], _, _, _ = self.mixup(
                        x[i: i + self.factor],
                        y2[i: i + self.factor],
                        weight2[i: i + self.factor],
                        teacher_preds2[i: i + self.factor],
                    )

            b, c, f, t = x.shape
            x = x.permute(0, 3, 1, 2)
            x = x.reshape(b // self.factor, self.factor * t, c, f)

            if np.random.random() <= 1.0:  # 0.5
                x, y, weight, teacher_preds = self.mixup(x, y, weight, teacher_preds)

            x = x.reshape(b, t, c, f)
            x = x.permute(0, 2, 3, 1)

        x = self.backbone(x)

        if self.training:
            b, c, f, t = x.shape
            x = x.permute(0, 3, 1, 2)
            x = x.reshape(b // self.factor, self.factor * t, c, f)
            x = x.permute(0, 2, 3, 1)

        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        logit = sum([self.head(dropout(x)) for dropout in self.dropouts]) / 5

        return {"logit": logit, "target": y, "rating": weight, "teacher_preds": teacher_preds}


class BCEKDLoss(nn.Module):
    def __init__(self, weights=[0.1, 0.9], class_weights=None):
        super().__init__()

        self.weights = weights
        self.T = 20

    def forward(self, output):
        input_ = output["logit"]
        target = output["target"].float()
        rating = output["rating"]
        teacher_preds = output["teacher_preds"]

        rating = rating.unsqueeze(1).repeat(1, CFG.num_classes)
        loss = nn.BCEWithLogitsLoss(
            weight=rating,
            reduction='mean',
        )(input_, target)

        KD_loss = nn.KLDivLoss()(
            F.log_softmax(input_ / self.T, dim=1),
            F.softmax(teacher_preds / self.T, dim=1)
            ) * (self.weights[1] * self.T * self.T)

        return self.weights[0] * loss + KD_loss


def padded_cmap(solution, submission, padding_factor=5):
    solution = solution
    submission = submission
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat(
        [solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat(
        [submission, new_rows]).reset_index(drop=True).copy()
    score = metrics.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average='macro',
    )
    return score


def map_score(solution, submission):
    solution = solution
    submission = submission
    score = metrics.average_precision_score(
        solution.values,
        submission.values,
        average='micro',  # 'macro'
    )
    return score


def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


def train_fn(data_loader, model, criterion, optimizer, scheduler, epoch):

    model.train()
    losses = AverageMeter()
    optimizer.zero_grad(set_to_none=True)
    scaler = GradScaler(enabled=CFG.apex)
    iters = len(data_loader)
    gt = []
    preds = []

    with tqdm(enumerate(data_loader), total=len(data_loader)) as t:
        for i, (data) in t:
            inputs = batch_to_device(data, device)
            targets = data['primary_targets'].to(device)

            inputs['wave'] = taa_augmentation(inputs['wave'].unsqueeze(1))
            inputs['wave'] = inputs['wave'].squeeze(1)

            with autocast(enabled=CFG.apex):
                outputs = model(inputs)
                loss = criterion(outputs)

            losses.update(loss.item(), inputs['wave'].size(0))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=CFG.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step(epoch + i / iters)
            t.set_postfix(
                loss=losses.avg,
                grad=grad_norm.item(),
                lr=optimizer.param_groups[0]["lr"]
                )

            gt.append(targets.cpu().detach().numpy())
            preds.append(outputs["logit"].sigmoid().cpu().detach().numpy())

        val_df = pd.DataFrame(
            np.concatenate(gt), columns=CFG.target_columns)
        pred_df = pd.DataFrame(
            np.concatenate(preds), columns=CFG.target_columns)
        cmAP_1 = padded_cmap(val_df, pred_df, padding_factor=1)
        cmAP_5 = padded_cmap(val_df, pred_df, padding_factor=5)
        mAP = map_score(val_df, pred_df)

    return losses.avg, cmAP_1, cmAP_5, mAP


def valid_fn(data_loader, model, criterion, epoch):
    model.eval()
    losses = AverageMeter()
    gt = []
    preds = []

    with tqdm(enumerate(data_loader), total=len(data_loader)) as t:
        for i, (data) in t:
            inputs = batch_to_device(data, device)
            targets = data['primary_targets'].to(device)

            with autocast(enabled=CFG.apex):
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs)

            losses.update(loss.item(), inputs['wave'].size(0))
            t.set_postfix(loss=losses.avg)

            gt.append(targets.cpu().detach().numpy())
            preds.append(outputs["logit"].sigmoid().cpu().detach().numpy())

    val_df = pd.DataFrame(np.concatenate(gt), columns=CFG.target_columns)
    pred_df = pd.DataFrame(np.concatenate(preds), columns=CFG.target_columns)
    cmAP_1 = padded_cmap(val_df, pred_df, padding_factor=1)
    cmAP_5 = padded_cmap(val_df, pred_df, padding_factor=5)
    mAP = map_score(val_df, pred_df)

    return losses.avg, cmAP_1, cmAP_5, mAP


def train_loop(
    train_loader,
    val_loader,
    model,
    optimizer,
    scheduler,
    criterion,
    epochs=10,
    fold=None,
):

    best_score = 0.0
    patience = CFG.early_stopping
    n_patience = 0

    for epoch in range(epochs):

        # train for one epoch
        train_loss, train_score, train_cmAP5, train_mAP = train_fn(
            train_loader, model, criterion, optimizer, scheduler, epoch)

        # evaluate on validation set
        val_loss, val_score, val_cmAP5, val_mAP = valid_fn(
            val_loader, model, criterion, epoch,)

        logger.info(f"Epoch {epoch} - Train loss: {train_loss:.4f}, Train cmAP1: {train_score:.4f}, Train cmAP5: {train_cmAP5:.4f}, Train mAP: {train_mAP:.4f}, Valid loss: {val_loss:.4f}, Valid cmAP1: {val_score:.4f}, Valid cmAP5: {val_cmAP5:.4f}, Valid mAP: {val_mAP:.4f}")

        is_better = val_score > best_score
        best_score = max(val_score, best_score)

        # Save the best model
        if is_better:
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_loss": best_score,
                "optimizer": optimizer.state_dict(),
            }
            logger.info(
                f"Epoch {epoch} - Save Best Score: {best_score:.4f} Model\n")
            torch.save(
                state,
                os.path.join(CFG.model_output_path, f"fold_{fold}_model.bin")
                )
            n_patience = 0
        else:
            n_patience += 1
            logger.info(
                f"Valid loss didn't improve last {n_patience} epochs.\n")

        if n_patience >= patience:
            logger.info(
                "Early stop, Training End.\n")
            break

    return


logger = init_logger(log_file=f"../log/train_{CFG.exp_name}.log")
device = get_device()
set_seed(CFG.seed)
os.makedirs(os.path.join(CFG.model_output_path), exist_ok=True)


def main():

    with open(TRAIN_DATA_PATH + 'pretrain_metadata_10fold_pseudo.pickle', 'rb') as f:
        train = pickle.load(f)
        del_files = [
            '../input/birdclef-2021/train_short_audio/ovenbi1/XC165471.ogg',
            '../input/birdclef-2021/train_short_audio/rewbla/XC313157.ogg'
            ]  # broken files
        train = train[~train.filename.isin(del_files)].reset_index(drop=True)

    train["rating"] = np.clip(train["rating"] / train["rating"].max(), 0.1, 1.0)

    logger.info(train.shape)
    train.head()

    # main loop
    for fold in range(5):

        if fold not in CFG.folds:
            continue
        logger.info("=" * 90)
        logger.info(f"Fold {fold} Training")
        logger.info("=" * 90)

        trn_df = train[train['fold'] != fold].reset_index(drop=True)
        val_df = train[train['fold'] == fold].reset_index(drop=True)

        sampler = None
        if CFG.use_sampler:
            one_hot_target = np.zeros(
                (trn_df.shape[0], len(CFG.target_columns)), dtype=np.float32
                )

            for i, label in enumerate(trn_df.primary_label):
                primary_label = CFG.bird2id[label]
                one_hot_target[i, primary_label] = 1.0

            sampler = MultilabelBalancedRandomSampler(
                one_hot_target,
                trn_df.index,
                class_choice="least_sampled"
                )

        logger.info(trn_df.shape)
        logger.info(trn_df['primary_label'].value_counts())
        logger.info(val_df.shape)
        logger.info(val_df['primary_label'].value_counts())

        loaders = {}
        trn_dataset = BirdClef2023Dataset(
                data_path=CFG.train_datadir,
                period=CFG.period,
                secondary_coef=CFG.secondary_coef,
                train=True,
                df=trn_df,
        )
        loaders['train'] = torchdata.DataLoader(
            trn_dataset,
            sampler=sampler,
            **CFG.loader_params['train']
        )
        val_dataset = BirdClef2023Dataset(
                data_path=CFG.train_datadir,
                period=5,
                secondary_coef=CFG.secondary_coef,
                train=False,
                df=val_df,
        )
        loaders['valid'] = torchdata.DataLoader(
            val_dataset,
            **CFG.loader_params['valid']
        )

        model = AttModel(
            backbone=CFG.backbone,
            num_class=CFG.num_classes,
            train_period=CFG.period,
            infer_period=5,
        )
        model = model.to(device)

        criterion = BCEKDLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CFG.lr_max,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=CFG.weight_decay,
            amsgrad=False,
            )
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=10,
            warmup_t=1,
            cycle_limit=40,
            cycle_decay=1.0,
            lr_min=CFG.lr_min,
            t_in_epochs=True,
        )

        # start training
        train_loop(
            loaders['train'],
            loaders['valid'],
            model,
            optimizer,
            scheduler,
            criterion,
            epochs=CFG.epochs,
            fold=fold,
            )

    logger.info('training done!')


if __name__ == "__main__":
    main()
