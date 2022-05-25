import os
import itertools
import logging
from time import time
from pprint import pformat
from shutil import copy
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np

import hparam as h
from models import *
from data import Dataset
from audio import get_mel

os.environ['LIBROSA_CACHE_LEVEL'] = '50'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.enable    = True
torch.backends.cudnn.benchmark = True
torch.     manual_seed(h.randseed)
torch.cuda.manual_seed(h.randseed)

ZERO = torch.Tensor([0]).to(device)
torch.autograd.set_detect_anomaly(hp.debug)


def train(a):
    '''LOGGING'''
    os.makedirs(a.log_path, exist_ok=True)
    copy('hparam.py', a.log_path)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(os.path.join(a.log_path, 'rtg.log'), 'a', encoding='utf-8'))
    logger.addHandler(logging.StreamHandler())
    logger.info('Hyperparams:')
    logger.info(pformat({k: getattr(hp, k) for k in dir(h) if not k.startswith('__')}, indent=2))
    logger.info(f'log path: {a.log_path}')
    sw = SummaryWriter(a.log_path)

    '''MODEL'''
    Generator = globals().get(f'Generator_{h.generator_ver}')
    generator = Generator()               .to(device)
    msd       = MultiScaleDiscriminator() .to(device)
    mpd       = MultiPeriodDiscriminator().to(device)
    mtd       = MultiStftDiscriminator()  .to(device)
    logger.info(generator)
    logger.info(mpd)
    logger.info(msd)
    logger.info(mtd)
    logger.info('model parameter count:')
    logger.info(f'  gen: {get_param_cnt(generator)}')
    logger.info(f'  msd: {get_param_cnt(msd)}')
    logger.info(f'  mpd: {get_param_cnt(mpd)}')
    logger.info(f'  mtd: {get_param_cnt(mtd)}')

    cp_g  = scan_checkpoint(a.log_path, 'g_')
    cp_do = scan_checkpoint(a.log_path, 'do_')
    if cp_g is None or cp_do is None:
        state_dict_do = None
        steps = 0
        last_epoch = -1             # scheduler consider -1 as fresh train 
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        generator.load_state_dict(state_dict_g['generator'])
        state_dict_do = load_checkpoint(cp_do, device)
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        mtd.load_state_dict(state_dict_do['mtd'])
        steps      = state_dict_do['steps']
        last_epoch = state_dict_do['epoch']
    
    discriminator_parameters = itertools.chain(msd.parameters(), mpd.parameters(), mtd.parameters())
    optim_g = AdamW(generator.parameters(),   h.learning_rate_g, betas=[h.adam_b1, h.adam_b2])
    optim_d = AdamW(discriminator_parameters, h.learning_rate_d, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    '''DATA'''
    trainset     = Dataset('train', data_dp=a.data_dp, finetune=a.finetune)
    train_loader = DataLoader(trainset, num_workers=0 if h.debug else h.num_workers, shuffle=True, batch_size=h.batch_size, pin_memory=True, drop_last=True)
    validset     = Dataset('test',  data_dp=a.data_dp, finetune=a.finetune, limit=h.valid_limit)
    valid_loader = DataLoader(validset, num_workers=0 if h.debug else 1, shuffle=False, batch_size=1, pin_memory=False, drop_last=False)
    logger.info(f'dataset: {len(trainset)} for train, {len(validset)} for valid')

    '''RAIN'''
    generator.train()
    mpd.train()
    msd.train()
    mtd.train()
    for epoch in range(max(0, last_epoch), a.epochs):
        start_e = time()
        logger.info(f">> Epoch: {epoch + 1}")

        for _, batch in enumerate(train_loader):
            start_b = time()

            '''Data'''
            if hp.split_cv:
                x_c, x_v, y_tmpl_c, y_tmpl_v, y, uv_ex = batch
                x_c      = x_c     .to(device, non_blocking=True)
                x_v      = x_v     .to(device, non_blocking=True)
                y_tmpl_c = y_tmpl_c.to(device, non_blocking=True).unsqueeze(1)   # add depth => [B, 1, T]
                y_tmpl_v = y_tmpl_v.to(device, non_blocking=True).unsqueeze(1)
                y        = y       .to(device, non_blocking=True).unsqueeze(1)
                uv_ex    = uv_ex   .to(device, non_blocking=True).unsqueeze(1)
                
                y_g_hat = generator(x_c, x_v, y_tmpl_c, y_tmpl_v, uv_ex)    # [B, 1, T]
            else:
                x, y_tmpl, y = batch
                x      = x     .to(device, non_blocking=True)
                y_tmpl = y_tmpl.to(device, non_blocking=True).unsqueeze(1)   # add depth => [B, 1, T]
                y      = y     .to(device, non_blocking=True).unsqueeze(1)

                y_g_hat = generator(x, y_tmpl)    # [B, 1, T]
            
            assert y.shape[-1] == y_g_hat.shape[-1] == hp.segment_size     # assure the model kept seq-len

            '''Discriminator'''
            y_g_hat_detach = y_g_hat.detach()     # .detach() stops grad to Generator (aka. freeze the Generator) 
            for _ in range(hp.d_train_times):
                optim_d.zero_grad()

                # discriminators score true wav `y` with genarated wav `y_g_hat`
                # [16, 2, 1025,  35]
                # [16, 2,  513,  69]
                # [16, 2,  257, 137]
                S, S_g_hat_detach = multi_stft_loss(y, y_g_hat_detach, ret_specs=True)
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat_detach)
                y_dp_hat_r, y_dp_hat_g, _, _ = mpd(y, y_g_hat_detach)
                y_dt_hat_r, y_dt_hat_g, _, _ = mtd(S, S_g_hat_detach)
                loss_disc_s = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
                loss_disc_p = discriminator_loss(y_dp_hat_r, y_dp_hat_g)
                loss_disc_t = discriminator_loss(y_dt_hat_r, y_dt_hat_g)
                with torch.no_grad():
                    # [B, 128/65/33]
                    sc_s_r = sum(torch.mean(d) for d in y_ds_hat_r) / len(y_ds_hat_r)
                    sc_s_g = sum(torch.mean(d) for d in y_ds_hat_g) / len(y_ds_hat_g)
                    # [B, 2731/1639/1171]
                    sc_p_r = sum(torch.mean(d) for d in y_dp_hat_r) / len(y_dp_hat_r)
                    sc_p_g = sum(torch.mean(d) for d in y_dp_hat_g) / len(y_dp_hat_g)
                    # [B, 430/387/396]
                    sc_t_r = sum(torch.mean(d) for d in y_dt_hat_r) / len(y_dt_hat_r)
                    sc_t_g = sum(torch.mean(d) for d in y_dt_hat_g) / len(y_dt_hat_g)

                loss_disc_all = loss_disc_s + loss_disc_p + loss_disc_t
                if not torch.isnan(loss_disc_all):
                    loss_disc_all.backward()
                optim_d.step()

            '''Generator'''
            optim_g.zero_grad()

            loss_mstft, (S, S_g_hat) = multi_stft_loss(y, y_g_hat, ret_loss=True, ret_specs=True)
            loss_env                 = envelope_loss  (y, y_g_hat) if hp.envelope_loss     else ZERO
            loss_dyn                 = dynamic_loss   (y, y_g_hat) if hp.dynamic_loss      else ZERO
            loss_sm                  = strip_mirror_loss (y_g_hat) if hp.strip_mirror_loss else ZERO
            
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)    # real, generated
            y_dp_hat_r, y_dp_hat_g, fmap_p_r, fmap_p_g = mpd(y, y_g_hat)
            y_dt_hat_r, y_dt_hat_g, fmap_t_r, fmap_t_g = mtd(S, S_g_hat)
            loss_fm_s  = feature_loss(fmap_s_r, fmap_s_g)
            loss_fm_p  = feature_loss(fmap_p_r, fmap_p_g)
            loss_fm_t  = feature_loss(fmap_t_r, fmap_t_g)
            loss_gen_s = generator_loss(y_ds_hat_g, y_ds_hat_r)
            loss_gen_p = generator_loss(y_dp_hat_g, y_dp_hat_r)
            loss_gen_t = generator_loss(y_dt_hat_g, y_dt_hat_r)

            loss_gen_all = (
                loss_gen_s + 
                loss_gen_p + 
                loss_gen_t + 
                loss_fm_s  * hp.w_loss_fm +
                loss_fm_p  * hp.w_loss_fm +
                loss_fm_t  * hp.w_loss_fm +
                loss_mstft * hp.w_loss_mstft + 
                loss_env   * hp.w_loss_env +
                loss_dyn   * hp.w_loss_dyn +
                loss_sm    * hp.w_loss_sm)
            if not torch.isnan(loss_gen_all):
                loss_gen_all.backward()
            optim_g.step()

            '''Tasks'''
            if steps % a.stdout_interval == 0:
                with torch.no_grad():
                    t_g_l   = loss_gen_all.item()
                    g_s_l   = loss_gen_s.item()
                    g_p_l   = loss_gen_p.item()
                    g_t_l   = loss_gen_t.item()
                    fm_s_l  = loss_fm_s.item()  * hp.w_loss_fm
                    fm_p_l  = loss_fm_p.item()  * hp.w_loss_fm
                    fm_t_l  = loss_fm_t.item()  * hp.w_loss_fm
                    mstft_l = loss_mstft.item() * hp.w_loss_mstft
                    dyn_l   = loss_dyn.item()   * hp.w_loss_dyn
                    env_l   = loss_env.item()   * hp.w_loss_env
                    sm_l    = loss_sm.item()    * hp.w_loss_sm

                    t_d_l   = loss_disc_all.item()
                    d_s_l   = loss_disc_s.item()
                    d_p_l   = loss_disc_p.item()
                    d_t_l   = loss_disc_t.item()

                logger.info(f'[{steps:d} ({time() - start_b:.2f} s/b)]' + 
                            f'\n' + 
                            f'  total gen: {t_g_l:.3f}, ' +
                            f'g_s: {g_s_l:.3f}, ' +
                            f'g_p: {g_p_l:.3f}, ' +
                            f'g_t: {g_t_l:.3f}, ' +
                            f'fm_s: {fm_s_l:.3f}, ' +
                            f'fm_p: {fm_p_l:.3f}, ' +
                            f'fm_t: {fm_t_l:.3f}, ' +
                            f'mstft: {mstft_l:.3f}, ' +
                            f'dyn: {dyn_l:.3f}, ' +
                            f'env: {env_l:.3f}, ' +
                            f'sm: {sm_l:.3f}, ' +
                            f'\n' + 
                            f'  total disc: {t_d_l:.3f}, ' +
                            f'd_s: {d_s_l:.3f}, ' +
                            f'd_p: {d_p_l:.3f}, ' +
                            f'd_t: {d_t_l:.3f}, ') 
                logger.info(f'  d-scores >> ' + 
                            f'sc_s_r: {sc_s_r:.3f}, ' +
                            f'sc_s_g: {sc_s_g:.3f}, ' +
                            f'sc_p_r: {sc_p_r:.3f}, ' +
                            f'sc_p_g: {sc_p_g:.3f}, ' +
                            f'sc_t_r: {sc_t_r:.3f}, ' +
                            f'sc_t_g: {sc_t_g:.3f}')
                
                if np.isnan(t_g_l) or t_g_l > 1e5:
                    print('Oh dude, your loss has exploded!')
                    exit(0)

            if steps % a.summary_interval == 0:
                sw.add_scalar("train/lr_g",         scheduler_g.get_last_lr(), steps)
                sw.add_scalar("train/lr_d",         scheduler_d.get_last_lr(), steps)
                sw.add_scalar("train/loss_disc_s",  loss_disc_s,  steps)
                sw.add_scalar("train/loss_disc_p",  loss_disc_p,  steps)
                sw.add_scalar("train/loss_disc_t",  loss_disc_t,  steps)
                sw.add_scalar("train/loss_gen_all", loss_gen_all, steps)
                sw.add_scalar("train/loss_gen_s",   loss_gen_s,   steps)
                sw.add_scalar("train/loss_gen_p",   loss_gen_p,   steps)
                sw.add_scalar("train/loss_gen_t",   loss_gen_t,   steps)
                sw.add_scalar("train/loss_fm_s",    loss_fm_s,    steps)
                sw.add_scalar("train/loss_fm_p",    loss_fm_p,    steps)
                sw.add_scalar("train/loss_fm_t",    loss_fm_t,    steps)
                sw.add_scalar("train/loss_mstft",   loss_mstft,   steps)
                sw.add_scalar("train/loss_dyn",     loss_dyn,     steps)
                sw.add_scalar("train/loss_env",     loss_env,     steps)
                sw.add_scalar("train/loss_sm",      loss_sm,      steps)

            if steps % a.checkpoint_interval == 0:
                save_checkpoint(os.path.join(a.log_path, f"g_{steps:08d}"),
                                {'generator': generator.state_dict()})
                save_checkpoint(os.path.join(a.log_path, f"do_{steps:08d}"), 
                                {'msd':     msd.state_dict(),
                                 'mpd':     mpd.state_dict(),
                                 'mtd':     mtd.state_dict(),
                                 'optim_g': optim_g.state_dict(), 
                                 'optim_d': optim_d.state_dict(), 
                                 'steps':   steps,
                                 'epoch':   epoch})

            if steps % a.validation_interval == 0:
                generator.eval()
                #torch.cuda.empty_cache()

                loss_mstft, loss_env, loss_dyn, loss_sm = 0, 0, 0, 0
                with torch.no_grad():
                    for j, batch in enumerate(valid_loader):
                        if hp.split_cv:
                            x_c, x_v, y_tmpl_c, y_tmpl_v, y, uv_ex = batch
                            x_c      = x_c     .to(device, non_blocking=True)
                            x_v      = x_v     .to(device, non_blocking=True)
                            y_tmpl_c = y_tmpl_c.to(device, non_blocking=True).unsqueeze(1)   # add depth => [B, 1, T]
                            y_tmpl_v = y_tmpl_v.to(device, non_blocking=True).unsqueeze(1)
                            y        = y       .to(device, non_blocking=True).unsqueeze(1)
                            uv_ex    = uv_ex   .to(device, non_blocking=True).unsqueeze(1)
                            
                            y_g_hat = generator(x_c, x_v, y_tmpl_c, y_tmpl_v, uv_ex)    # [B, 1, T]
                        else:
                            x, y_tmpl, y = batch
                            x      = x     .to(device, non_blocking=True)
                            y_tmpl = y_tmpl.to(device, non_blocking=True).unsqueeze(1)   # add depth => [B, 1, T]
                            y      = y     .to(device, non_blocking=True).unsqueeze(1)
                            
                            y_g_hat = generator(x, y_tmpl)    # [B, 1, T]

                        loss_mstft += multi_stft_loss(y, y_g_hat, ret_loss=True).item()
                        loss_dyn   += dynamic_loss   (y, y_g_hat).item()
                        loss_env   += envelope_loss  (y, y_g_hat).item()
                        loss_sm    += strip_mirror_loss (y_g_hat).item()

                        if j < 4:       # 前k个用于谱展示
                            if steps == 0:
                                y_s = y.squeeze(1)
                                mel = get_mel(y_s[0].cpu().numpy())
                                sw.add_figure(f'raw/y_spec_{j}', plot_spectrogram(mel), steps)
                                sw.add_audio(f'raw/y_{j}', y_s[0], steps, h.sample_rate)

                            y_g_hat_s = y_g_hat.squeeze(1)
                            mel = get_mel(y_g_hat_s[0].cpu().numpy())
                            sw.add_figure(f'gen/y_hat_spec_{j}', plot_spectrogram(mel), steps)
                            sw.add_audio(f'gen/y_hat_{j}', y_g_hat_s[0], steps, h.sample_rate)

                    sw.add_scalar("valid/loss_mstft", loss_mstft / j, steps)
                    sw.add_scalar("valid/loss_env",   loss_env   / j, steps)
                    sw.add_scalar("valid/loss_dyn",   loss_dyn   / j, steps)
                    sw.add_scalar("valid/loss_sm",    loss_sm    / j, steps)
                
                generator.train()
            
            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        logger.info(f'<< Epoch {epoch + 1} took {time() - start_e:.2f}s\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--data_dp',             default=None)
    parser.add_argument('--log_path',            default='rtg-logs')
    parser.add_argument('--epochs',              default=100,  type=int)
    parser.add_argument('--stdout_interval',     default=10,   type=int)
    parser.add_argument('--summary_interval',    default=250,  type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--checkpoint_interval', default=1000, type=int)
    args = parser.parse_args()

    train(args)
