import logging
import os
import sys
import warnings

# Настройка окружения
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"

# Настройка логирования и подавление предупреждений
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

import argparse
import datetime
import json
from distutils.util import strtobool
from random import randint
from time import time as ttime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.getcwd()))
from rvc.lib.algorithm.commons import grad_norm, slice_segments
from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from rvc.train.mel_processing import (
    MultiScaleMelSpectrogramLoss,
    mel_spectrogram_similarity,
    mel_spectrogram_torch,
    plot_spectrogram_to_numpy,
    spec_to_mel_torch,
)
from rvc.train.utils.data_utils import DistributedBucketSampler, TextAudioCollateMultiNSFsid, TextAudioLoaderMultiNSFsid
from rvc.train.utils.train_utils import HParams, attempt_load_checkpoint_pair, extract_model, save_checkpoint

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

global_step = 0


def generate_config(config_save_path, sample_rate, vocoder):
    config_path = os.path.join("rvc", "train", "configs", f"{sample_rate}.json")
    import pathlib

    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            with open(config_path, "r", encoding="utf-8") as config_file:
                config_data = json.load(config_file)
                config_data["model"]["vocoder"] = vocoder
                json.dump(config_data, f, ensure_ascii=False, indent=2)


def get_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Model")
    parser.add_argument("--total_epoch", type=int, default=300)
    parser.add_argument("--save_every_epoch", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sample_rate", type=int, default=40000)
    parser.add_argument("--vocoder", type=str, default="HiFi-GAN")
    parser.add_argument("--pretrain_g", type=str, default="")
    parser.add_argument("--pretrain_d", type=str, default="")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--save_to_zip", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--save_backup", type=lambda x: bool(strtobool(x)), default=False)
    args = parser.parse_args()

    experiment_dir = os.path.join(args.experiment_dir, args.model_name)
    config_save_path = os.path.join(experiment_dir, "data", "config.json")

    # Генерация файла конфигурации
    if not os.path.exists(config_save_path):
        generate_config(config_save_path, args.sample_rate, args.vocoder)

    # Загрузка файла конфигурации
    with open(config_save_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    hparams = HParams(**config)
    hparams.model_dir = experiment_dir
    hparams.model_name = args.model_name
    hparams.total_epoch = args.total_epoch
    hparams.save_every_epoch = args.save_every_epoch
    hparams.batch_size = args.batch_size
    hparams.pretrain_g = args.pretrain_g
    hparams.pretrain_d = args.pretrain_d
    hparams.gpus = args.gpus
    hparams.save_to_zip = args.save_to_zip
    hparams.save_backup = args.save_backup
    hparams.data.training_files = f"{experiment_dir}/data/filelist.txt"
    return hparams


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = round(now_time - self.last_time, 1)
        self.last_time = now_time
        return f"[{str(datetime.timedelta(seconds=int(elapsed_time)))}]"


def main():
    hps = get_hparams()
    global_step = 0

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    gpus = [int(item) for item in hps.gpus.split("-")] if device.type == "cuda" else [0]
    n_gpus = len(gpus)
    if device.type == "cpu":
        print("Обучение с использованием процессора займёт много времени.", flush=True)

    children = []
    for rank, device_id in enumerate(gpus):
        subproc = mp.Process(
            target=run,
            args=(hps, rank, n_gpus, device, device_id, global_step),
        )
        children.append(subproc)
        subproc.start()

    for subproc in children:
        subproc.join()

    sys.exit(0)


def run(hps, rank, n_gpus, device, device_id, global_step):
    global global_step
    try:
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval")) if rank == 0 else None
        fn_mel_loss = MultiScaleMelSpectrogramLoss(sample_rate=hps.data.sample_rate)

        dist.init_process_group(
            backend="gloo" if sys.platform == "win32" or device.type != "cuda" else "nccl",
            init_method="env://",
            world_size=n_gpus if device.type == "cuda" else 1,
            rank=rank if device.type == "cuda" else 0,
        )

        torch.manual_seed(hps.train.seed)
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)

        collate_fn = TextAudioCollateMultiNSFsid()
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data)
        train_sampler = DistributedBucketSampler(
            train_dataset,
            hps.batch_size * n_gpus,
            [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_dataset,
            num_workers=2,  # 4
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
            persistent_workers=True,
            prefetch_factor=8,
        )

        net_g = Synthesizer(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            use_f0=True,
            sr=hps.data.sample_rate,
            checkpointing=False,
            randomized=True,
        )
        net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm, checkpointing=False)

        if device.type == "cuda":
            net_g = net_g.cuda(device_id)
            net_d = net_d.cuda(device_id)
        else:
            net_g = net_g.to(device)
            net_d = net_d.to(device)

        optim_g = torch.optim.AdamW(
            net_g.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
        optim_d = torch.optim.AdamW(
            net_d.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )

        if n_gpus > 1 and device.type == "cuda":
            net_g = DDP(net_g, device_ids=[device_id])
            net_d = DDP(net_d, device_ids=[device_id])

        # Загрузка чекпоинтов
        checkpoint_paths = [("G_checkpoint.pth", "D_checkpoint.pth"), ("G_checkpoint_backup.pth", "D_checkpoint_backup.pth")]

        loaded = False
        for g_file, d_file in checkpoint_paths:
            g_path = os.path.join(hps.model_dir, g_file)
            d_path = os.path.join(hps.model_dir, d_file)
            if os.path.exists(g_path) and os.path.exists(d_path):
                try:
                    epoch_str = attempt_load_checkpoint_pair(net_g, optim_g, g_path, net_d, optim_d, d_path)
                    epoch_str += 1
                    global_step = (epoch_str - 1) * len(train_loader)
                    loaded = True
                    break
                except:
                    continue

        if not loaded:
            epoch_str = 1
            global_step = 0

            # Если чекпоинты не загрузились, пробуем загрузить претрейны
            if hps.pretrain_g not in ("", "None", None):
                if rank == 0:
                    print(f"Загрузка претрейна '{hps.pretrain_g}'", flush=True)
                g_model = net_g.module if hasattr(net_g, "module") else net_g
                g_model.load_state_dict(torch.load(hps.pretrain_g, map_location="cpu", weights_only=True)["model"])

            if hps.pretrain_d not in ("", "None", None):
                if rank == 0:
                    print(f"Загрузка претрейна '{hps.pretrain_d}'", flush=True)
                d_model = net_d.module if hasattr(net_d, "module") else net_d
                d_model.load_state_dict(torch.load(hps.pretrain_d, map_location="cpu", weights_only=True)["model"])

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

        print("\nЗапуск процесса обучения модели...", flush=True)
        for epoch in range(epoch_str, hps.total_epoch + 1):
            train_and_evaluate(
                hps,
                rank,
                epoch,
                global_step,
                [net_g, net_d],
                [optim_g, optim_d],
                train_loader,
                writer_eval,
                fn_mel_loss,
                device,
            )
            scheduler_g.step()
            scheduler_d.step()
    finally:
        # Уничтожение группы процессов для корректного закрытия программы
        if dist.is_initialized():
            dist.destroy_process_group()


def train_and_evaluate(hps, rank, epoch, global_step, nets, optims, train_loader, writer_eval, fn_mel_loss, device):
    global global_step

    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader.batch_sampler.set_epoch(epoch)

    net_g.train()
    net_d.train()

    epoch_recorder = EpochRecorder()
    for _, info in enumerate(train_loader):
        info = [tensor.to(device, non_blocking=device.type == "cuda") for tensor in info]
        phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, _, sid = info

        model_output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
        y_hat, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = model_output
        wave = slice_segments(wave, ids_slice * hps.data.hop_length, hps.train.segment_size, dim=3)

        # Discriminator loss
        for _ in range(1):  # default x1
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
            optim_d.zero_grad()
            loss_disc.backward()
            grad_norm_d = grad_norm(net_d.parameters())
            optim_d.step()

        # Generator loss
        for _ in range(1):  # default x1
            _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            loss_mel = fn_mel_loss(wave, y_hat) * hps.train.c_mel / 3.0
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _ = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
            optim_g.zero_grad()
            loss_gen_all.backward()
            grad_norm_g = grad_norm(net_g.parameters())
            optim_g.step()

        # learning rates
        current_lr_d = optim_d.param_groups[0]["lr"]
        current_lr_g = optim_g.param_groups[0]["lr"]

        global_step += 1

    if rank == 0 and epoch % hps.train.log_interval == 0:
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sample_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
        y_mel = slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length, dim=3)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.float().squeeze(1),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sample_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
        mel_similarity = mel_spectrogram_similarity(y_hat_mel, y_mel)

        scalar_dict = {
            "grad/norm_d": grad_norm_d,
            "grad/norm_g": grad_norm_g,
            "learning_rate/d": current_lr_d,
            "learning_rate/g": current_lr_g,
            "loss/avg/d": loss_disc,
            "loss/avg/g": loss_gen,
            "loss/g/fm": loss_fm,
            "loss/g/mel": loss_mel,
            "loss/g/kl": loss_kl,
            "loss/g/total": loss_gen_all,
            "metrics/mel_sim": mel_similarity,
            "metrics/mse_wave": F.mse_loss(y_hat, wave),
            "metrics/mse_pitch": F.mse_loss(pitchf, pitch),
        }
        image_dict = {
            "mel/slice/real": plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "mel/slice/fake": plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
        }
        for k, v in scalar_dict.items():
            writer_eval.add_scalar(k, v, epoch)
        for k, v in image_dict.items():
            writer_eval.add_image(k, v, epoch, dataformats="HWC")

    if rank == 0:
        print(
            f"{epoch_recorder.record()} - {hps.model_name} | "
            f"Эпоха: {epoch}/{hps.total_epoch} | "
            f"Шаг: {global_step} | "
            f"Сходство mel (G/R): {mel_similarity:.2f}%",
            flush=True,
        )

        save_final = epoch >= hps.total_epoch
        save_checkpoint_cond = (epoch % hps.save_every_epoch == 0) or save_final

        if save_checkpoint_cond:
            g_path = os.path.join(hps.model_dir, "G_checkpoint.pth")
            d_path = os.path.join(hps.model_dir, "D_checkpoint.pth")

            # Создание бэкапов
            if hps.save_backup and os.path.exists(g_path) and os.path.exists(d_path):
                print("Создание бэкапа предыдущего чекпоинта...", flush=True)
                try:
                    os.replace(g_path, g_path.replace("checkpoint", "checkpoint_backup"))
                    os.replace(d_path, d_path.replace("checkpoint", "checkpoint_backup"))
                except Exception as e:
                    print(f"Не удалось создать бэкап чекпоинта: {e}", flush=True)

            save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, g_path)
            save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, d_path)

            checkpoint = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
            print(extract_model(hps, checkpoint, epoch, global_step, final_save=save_final), flush=True)

        if save_final:
            if hps.save_to_zip:
                import zipfile

                zip_filename = os.path.join(hps.model_dir, f"{hps.model_name}.zip")
                with zipfile.ZipFile(zip_filename, "w") as zipf:
                    for ext in (".pth", ".index"):
                        file_path = os.path.join(hps.model_dir, f"{hps.model_name}{ext}")
                        zipf.write(file_path, os.path.basename(file_path))
                print(f"Файлы модели заархивированы в `{zip_filename}`", flush=True)

            print("\nОбучение успешно завершено!", flush=True)
            return


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
