import collections
import numpy as np
import torch
import torch.nn.functional as F
import lpips
from pathlib import Path
from torchvision.utils import save_image
# local modules
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import utils.loss as loss_utils
from utils.myutil import mean
from utils.training_utils import make_flow_movie, select_evenly_spaced_elements, make_tc_vis, make_vw_vis
from utils.data import data_sources


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, loss_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = max(len(data_loader) // 100, 1)
        self.val_log_step = max(len(valid_data_loader) // 100, 1)

        self.raw_metric_keys = ('raw_ssim', 'raw_ssim_global', 'raw_mse', 'raw_lpips', 'raw_tc')
        self.tc_l0 = 1
        for loss_ftn in self.loss_ftns:
            if loss_ftn.__class__.__name__ == 'temporal_consistency_loss':
                self.tc_l0 = loss_ftn.L0
                break

        self.lpips_metric = lpips.LPIPS(net='vgg')
        self.lpips_metric = self.lpips_metric.to(self.device)
        self.lpips_metric.eval()

        mt_keys = ['loss'] + list(self.raw_metric_keys)
        for data_source in data_sources:
            mt_keys.append(f'loss/{data_source}')
            for l in self.loss_ftns:
                mt_keys.append(f'{l.__class__.__name__}/{data_source}')
            for metric_key in self.raw_metric_keys:
                mt_keys.append(f'{metric_key}/{data_source}')
        self.train_metrics = MetricTracker(*mt_keys, writer=self.writer)
        self.valid_metrics = MetricTracker(*mt_keys, writer=self.writer)

        self.num_previews = config['trainer']['num_previews']
        self.val_num_previews = config['trainer'].get('val_num_previews', self.num_previews)
        self.val_preview_indices = select_evenly_spaced_elements(self.val_num_previews, len(self.valid_data_loader))
        self.valid_only = config['trainer'].get('valid_only', False)
        self.true_once = True  # True at init, turns False at end of _train_epoch

    def to_device(self, item):
        events = item['events'].float().to(self.device)
        image = item['frame'].float().to(self.device)
        flow = None if item['flow'] is None else item['flow'].float().to(self.device)
        return events, image, flow

    def forward_sequence(self, sequence, all_losses=False):
        losses = collections.defaultdict(list)
        raw_metrics = collections.defaultdict(list)
        self.model.reset_states()
        prev_image = None
        prev_pred_image = None
        for i, item in enumerate(sequence):
            events, image, flow = self.to_device(item)
            pred = self.model(events)
            pred_image = pred['image']

            with torch.no_grad():
                pred_image_detached = pred_image.detach()
                image_detached = image.detach()
                raw_metrics['raw_mse'].append(F.mse_loss(pred_image_detached, image_detached))
                raw_metrics['raw_ssim'].append(self.compute_ssim(pred_image_detached, image_detached))
                raw_metrics['raw_ssim_global'].append(self.compute_ssim_global(pred_image_detached, image_detached))
                raw_metrics['raw_lpips'].append(self.compute_lpips(pred_image_detached, image_detached))

                if prev_image is not None and flow is not None and i >= self.tc_l0:
                    tc_raw = loss_utils.temporal_consistency_loss(
                        prev_image,
                        image_detached,
                        prev_pred_image,
                        pred_image_detached,
                        -flow.detach(),
                    )
                    raw_metrics['raw_tc'].append(tc_raw)

                prev_image = image_detached
                prev_pred_image = pred_image_detached

            for loss_ftn in self.loss_ftns:
                loss_name = loss_ftn.__class__.__name__
                tmp_weight = loss_ftn.weight 
                if all_losses:
                    loss_ftn.weight = 1.0
                if loss_name == 'perceptual_loss':
                    losses[loss_name].append(loss_ftn(pred_image, image, normalize=True))
                if loss_name == 'l2_loss':
                    losses[loss_name].append(loss_ftn(pred_image, image))
                if loss_name == 'temporal_consistency_loss':
                    l = loss_ftn(i, image, pred_image, flow)
                    if l is not None:
                        losses[loss_name].append(l)
                if loss_name in ['flow_loss', 'flow_l1_loss'] and flow is not None:
                    losses[loss_name].append(loss_ftn(pred['flow'], flow))
                if loss_name == 'warping_flow_loss':
                    l = loss_ftn(i, image, pred['flow'])
                    if l is not None:
                        losses[loss_name].append(l)
                if loss_name == 'voxel_warp_flow_loss' and flow is not None:
                    losses[loss_name].append(loss_ftn(events, pred['flow']))
                if loss_name == 'flow_perceptual_loss':
                    losses[loss_name].append(loss_ftn(pred['flow'], flow))
                if loss_name == 'combined_perceptual_loss':
                    losses[loss_name].append(loss_ftn(pred_image, pred['flow'], image, flow))
                loss_ftn.weight = tmp_weight
        idx = int(item['data_source_idx'].mode().values.item())
        data_source = data_sources[idx]
        losses = {f'{k}/{data_source}': mean(v) for k, v in losses.items()}
        losses['loss'] = sum(losses.values())
        losses[f'loss/{data_source}'] = losses['loss']

        raw_metrics = {k: mean(v) for k, v in raw_metrics.items()}
        for metric_key in self.raw_metric_keys:
            metric_val = raw_metrics.get(metric_key, torch.tensor(0.0, device=self.device))
            losses[metric_key] = metric_val
            losses[f'{metric_key}/{data_source}'] = metric_val

        return losses

    def compute_lpips(self, pred, target):
        if pred.shape[1] == 1:
            pred = torch.cat([pred, pred, pred], dim=1)
        if target.shape[1] == 1:
            target = torch.cat([target, target, target], dim=1)
        return self.lpips_metric.forward(pred, target, normalize=True).mean()

    def compute_ssim(self, pred, target):
        """Compute window-based SSIM matching skimage implementation.
        
        Uses 11x11 Gaussian window with sigma=1.5, matching skimage.metrics.structural_similarity.
        This is the primary SSIM metric used for comparison with paper results.
        """
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        
        # Constants matching skimage defaults
        window_size = 11
        sigma = 1.5
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Create 2D Gaussian window matching skimage
        # Coordinates from -5 to +5 (for window_size=11)
        coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        # Normalize to sum to 1
        g = g / g.sum()
        # Outer product to get 2D kernel
        window_2d = g.unsqueeze(0) * g.unsqueeze(1)  # [11, 11]
        # Expand to [C, 1, 11, 11] for grouped convolution
        channel = pred.size(1)
        window = window_2d.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size).contiguous()
        
        # Valid padding (no padding) like skimage - only compute where full window fits
        pad = window_size // 2
        
        # Compute local means using convolution
        mu_pred = F.conv2d(pred, window, padding=pad, groups=channel)
        mu_target = F.conv2d(target, window, padding=pad, groups=channel)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        # Compute local variances and covariance
        sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad, groups=channel) - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, window, padding=pad, groups=channel) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, window, padding=pad, groups=channel) - mu_pred_target
        
        # SSIM formula
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        # Clamp to valid range like skimage
        ssim_map = torch.clamp(ssim_map, -1.0, 1.0)
        
        return ssim_map.mean()

    def compute_ssim_global(self, pred, target):
        """Compute global SSIM using mean over entire image (no sliding window).
        
        This computes SSIM using global statistics (single mean/variance per image).
        Useful for understanding overall image similarity without local structure consideration.
        """
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        reduce_dims = (1, 2, 3)
        mu_pred = pred.mean(dim=reduce_dims)
        mu_target = target.mean(dim=reduce_dims)

        pred_centered = pred - mu_pred.view(-1, 1, 1, 1)
        target_centered = target - mu_target.view(-1, 1, 1, 1)
        var_pred = (pred_centered ** 2).mean(dim=reduce_dims)
        var_target = (target_centered ** 2).mean(dim=reduce_dims)
        cov = (pred_centered * target_centered).mean(dim=reduce_dims)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        numerator = (2.0 * mu_pred * mu_target + C1) * (2.0 * cov + C2)
        denominator = (mu_pred ** 2 + mu_target ** 2 + C1) * (var_pred + var_target + C2)
        ssim = numerator / (denominator + 1e-8)
        return torch.clamp(ssim, -1.0, 1.0).mean()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        if self.valid_only:
            with torch.no_grad():
                val_log = self._valid_epoch(epoch)
                return {'val_' + k : v for k, v in val_log.items()}
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, sequence in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            losses = self.forward_sequence(sequence)
            loss = losses['loss']
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            for k, v in losses.items():
                self.train_metrics.update(k, v.item())

            if batch_idx % self.log_step == 0:
                msg = 'Train Epoch: {} {}'.format(epoch, self._progress(batch_idx, self.data_loader))
                for k, v in losses.items():
                    msg += ' {}: {:.4f}'.format(k, v.item())
                self.logger.debug(msg)

            if batch_idx < self.num_previews and (epoch - 1) % self.save_period == 0:
                with torch.no_grad():
                    self.preview(sequence, epoch, tag_prefix=f'train_{batch_idx}')

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        print("validation")
        if self.do_validation and epoch%10==0:
            with torch.no_grad():
                val_log = self._valid_epoch(epoch)
                log.update(**{'val_' + k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.true_once = False
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        i = 0
        for batch_idx, sequence in enumerate(self.valid_data_loader):
            self.optimizer.zero_grad()
            losses = self.forward_sequence(sequence, all_losses=True)
            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            for k, v in losses.items():
                self.valid_metrics.update(k, v.item())

            if batch_idx % self.val_log_step == 0:
                msg = 'Valid Epoch: {} {}'.format(epoch, self._progress(batch_idx, self.valid_data_loader))
                for k, v in losses.items():
                    msg += ' {}: {:.4f}'.format(k, v.item())
                self.logger.debug(msg)

            if batch_idx in self.val_preview_indices:
                if (epoch - 1) % self.save_period == 0:
                    self.preview(sequence, epoch, tag_prefix=f'val_{i}')
                i += 1
            
            if epoch == self.epochs:
                self.save_validation_images(sequence, epoch, batch_idx)

        return self.valid_metrics.result()

    def _progress(self, batch_idx, data_loader):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(data_loader, 'n_samples'):
            current = batch_idx * data_loader.batch_size
            total = data_loader.n_samples
        else:
            current = batch_idx
            total = len(data_loader)
        return base.format(current, total, 100.0 * current / total)

    def preview(self, sequence, epoch, tag_prefix=''):
        """
        Plot visualisation to tensorboard.
        Plots input, output, groundtruth histograms and movies
        """
        print(f'Making preview {tag_prefix}')
        event_previews, pred_flows, pred_images, flows, images, voxels = [], [], [], [], [], []
        self.model.reset_states()
        for i, item in enumerate(sequence):
            item = {k: v[0:1, ...] for k, v in item.items()}  # set batch size to 1
            events, image, flow = self.to_device(item)
            pred = self.model(events)
            event_previews.append(torch.sum(events, dim=1, keepdim=True))
            pred_flows.append(pred.get('flow', 0 * flow))
            pred_images.append(pred['image'])
            flows.append(flow)
            images.append(image)
            voxels.append(events)

        tc_loss_ftn = self.get_loss_ftn('temporal_consistency_loss')
        if self.true_once and tc_loss_ftn is not None:
            for i, image in enumerate(images):
                output = tc_loss_ftn(i, image, pred_images[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_tc_vis(output[1])
                    self.writer.writer.add_video(f'warp_vis/tc_{tag_prefix}',
                            video_tensor, global_step=epoch, fps=2)
                    break

        vw_loss_ftn = self.get_loss_ftn('voxel_warp_flow_loss')
        if self.true_once and vw_loss_ftn is not None:
            for i, image in enumerate(images):
                output = vw_loss_ftn(voxels[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_vw_vis(output[1])
                    self.writer.writer.add_video(f'warp_vox/tc_{tag_prefix}',
                            video_tensor, global_step=epoch, fps=1)
                    break
        
        non_zero_voxel = torch.stack([s['events'] for s in sequence])
        non_zero_voxel = non_zero_voxel[non_zero_voxel != 0]
        if torch.numel(non_zero_voxel) == 0:
            non_zero_voxel = 0
        self.writer.add_histogram(f'{tag_prefix}_flow/groundtruth',
                                  torch.stack(flows))
        self.writer.add_histogram(f'{tag_prefix}_image/groundtruth',
                                  torch.stack(images))
        self.writer.add_histogram(f'{tag_prefix}_input',
                                  non_zero_voxel)
        self.writer.add_histogram(f'{tag_prefix}_flow/prediction',
                                  torch.stack(pred_flows))
        self.writer.add_histogram(f'{tag_prefix}_image/prediction',
                                  torch.stack(pred_images))
        video_tensor = make_flow_movie(event_previews, pred_images, images, pred_flows, flows)
        self.writer.writer.add_video(f'{tag_prefix}', video_tensor, global_step=epoch, fps=20)

    def get_loss_ftn(self, loss_name):
        for loss_ftn in self.loss_ftns:
            if loss_ftn.__class__.__name__ == loss_name:
                return loss_ftn
        return None

    def save_validation_images(self, sequence, epoch, batch_idx):
        """Save original and reconstructed images as PNG during validation.
        
        Reuses the same model forward pass as validation (no reset_states).
        Processes the sequence frame by frame, accumulating recurrent states
        naturally, then saves the results.
        """
        save_dir = Path(self.checkpoint_dir) / 'validation_images' / f'epoch_{epoch}'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.reset_states()
        with torch.no_grad():
            for i, item in enumerate(sequence):
                item_single = {k: v[0:1, ...] for k, v in item.items()}
                events, image, flow = self.to_device(item_single)
                pred = self.model(events)
                pred_image = pred['image']
                
                # Clamp to [0,1] before saving (model output has no final activation)
                pred_image = torch.clamp(pred_image, 0.0, 1.0)
                image = torch.clamp(image, 0.0, 1.0)
                
                # Save original and reconstructed images
                save_image(image, save_dir / f'batch{batch_idx}_frame{i:03d}_original.png')
                save_image(pred_image, save_dir / f'batch{batch_idx}_frame{i:03d}_reconstructed.png')
