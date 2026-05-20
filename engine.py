# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from contextlib import contextmanager
from typing import Iterable, Optional
import contextlib
import numpy as np

from util.utils import slprint, to_device

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

# Prefer the extended ranking loss implementation (supports `ccm_feature` argument)
try:
    from models.loss_density_ranking import DensityMapLossWithRanking, DensityMapLoss
    try:
        from models.loss_density_with_dfl import DensityMapLossWithDFL
    except ImportError:  # pragma: no cover - optional
        DensityMapLossWithDFL = None
    DENSITY_LOSS_AVAILABLE = True
except ImportError:
    # Fallback to legacy module
    try:
        from models.loss_density import DensityMapLoss, DensityMapLossWithRanking
        try:
            from models.loss_density_with_dfl import DensityMapLossWithDFL
        except ImportError:  # pragma: no cover - optional
            DensityMapLossWithDFL = None
        DENSITY_LOSS_AVAILABLE = True
    except ImportError:
        DensityMapLossWithDFL = None
        DensityMapLossWithRanking = None
        DENSITY_LOSS_AVAILABLE = False

print_freq = 5000
CCM_LOSS = torch.nn.CrossEntropyLoss()
ccm_coeff = 1

def _temporarily_disable_bn_updates(module: torch.nn.Module):
    """Temporarily switch BatchNorm layers to eval mode to freeze running stats."""
    bn_layers = []
    for layer in module.modules():
        if isinstance(layer, _BatchNorm):
            bn_layers.append((layer, layer.training))
            layer.eval()
    try:
        yield
    finally:
        for layer, was_training in bn_layers:
            layer.train(was_training)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    # Enable anomaly detection when adversarial branch is active to catch in-place ops early
    if getattr(args, 'use_adv_training', False):
        torch.autograd.set_detect_anomaly(True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Get the actual model (unwrap DDP if needed)
    model_unwrapped = model.module if hasattr(model, 'module') else model

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    # 检测是否使用密度图监督
    use_density_supervision = getattr(args, 'use_density_supervision', False)
    density_loss_coeff = getattr(args, 'density_loss_coeff', 1.0)
    
    if use_density_supervision and DENSITY_LOSS_AVAILABLE:
        # 优先使用Ranking Loss（若ranking_weight > 0）
        density_ranking_weight = getattr(args, 'density_ranking_weight', 0.2)
        
        if density_ranking_weight > 0 and DensityMapLossWithRanking is not None:
            # 使用新的 Ranking Loss 方案（解决二维分布问题）
            density_loss_fn = DensityMapLossWithRanking(
                weight_pixel=getattr(args, 'density_pixel_weight', 1.0),
                weight_integral=getattr(args, 'density_integral_weight', 0.1),
                integral_lambda_low=getattr(args, 'density_integral_lambda_low', 5.0),
                integral_lambda_high=getattr(args, 'density_integral_lambda_high', 0.5),
                pixel_over_weight=getattr(args, 'density_pixel_over_weight', 0.5),
                pixel_under_weight=getattr(args, 'density_pixel_under_weight', 1.0),
                weight_ranking=density_ranking_weight,
                weight_support=getattr(args, 'density_weight_support', 0.5),
                weight_distribution=getattr(args, 'density_weight_distribution', 0.1),
                ranking_grid_size=getattr(args, 'density_ranking_grid_size', 8),
                ranking_margin=getattr(args, 'density_ranking_margin', 0.1),
                adaptive_margin=getattr(args, 'density_ranking_adaptive_margin', True),
                density_scale=getattr(args, 'density_scale', 1.0),
            )
            print(f"[INFO] Using DensityMapLossWithRanking (weight_ranking={density_ranking_weight})")
        elif getattr(args, 'use_dfl', False) and DensityMapLossWithDFL is not None:
            # 回退到DFL方案（可选）
            density_loss_fn = DensityMapLossWithDFL(
                loss_type=getattr(args, 'density_loss_type', 'l2'),
                weight_pixel=getattr(args, 'density_pixel_weight', 1.0),
                weight_smooth=getattr(args, 'density_smooth_weight', 0.1),
                weight_integral=getattr(args, 'density_integral_weight', 0.5),
                integral_lambda_low=getattr(args, 'density_integral_lambda_low', 5.0),
                integral_lambda_high=getattr(args, 'density_integral_lambda_high', 0.5),
                pixel_over_weight=getattr(args, 'density_pixel_over_weight', 0.5),
                pixel_under_weight=getattr(args, 'density_pixel_under_weight', 1.0),
                weight_edge=getattr(args, 'density_edge_weight', 0.0),
                use_dfl=getattr(args, 'use_dfl', False),
                dfl_weight=getattr(args, 'dfl_weight', 1.0),
                dfl_gamma=getattr(args, 'dfl_gamma', 2.0),
            )
            print(f"[INFO] Using DensityMapLossWithDFL")
        else:
            # 回退到原始Loss（仅pixel + integral）
            density_loss_fn = DensityMapLoss(
                loss_type=getattr(args, 'density_loss_type', 'l2'),
                weight_smooth=getattr(args, 'density_smooth_weight', 0.0),
                weight_integral=getattr(args, 'density_integral_weight', 0.5),
                integral_lambda_low=getattr(args, 'density_integral_lambda_low', 5.0),
                integral_lambda_high=getattr(args, 'density_integral_lambda_high', 0.5),
                pixel_over_weight=getattr(args, 'density_pixel_over_weight', 0.5),
                pixel_under_weight=getattr(args, 'density_pixel_under_weight', 1.0),
                weight_edge=getattr(args, 'density_edge_weight', 0.0),
                count_weighting=getattr(args, 'density_count_weighting', 'none'),
                count_weight_alpha=getattr(args, 'density_count_weight_alpha', 0.0),
            )
            print(f"[INFO] Using DensityMapLoss (basic)")
        print(f"[INFO] Density Supervision enabled with coefficient {density_loss_coeff}")
    else:
        density_loss_fn = None
        if use_density_supervision:
            print(f"[WARNING] use_density_supervision=True but DensityMapLoss not available")

    # Whether density supervision (integral-based) is actually active for this run
    density_supervision_active = density_loss_fn is not None

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(
            window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    ccm_params = args.ccm_params

    _cnt = 0
    # Accumulators for per-epoch statistics
    epoch_gt_sum = 0.0
    epoch_pred_sum = 0.0
    epoch_query_sum = 0.0
    epoch_image_count = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        samples = samples.to(device)
        
        # 处理CCM targets（分类方案）和GT密度图（密度方案）
        if use_density_supervision and density_loss_fn is not None:
            # 密度图监督模式：从targets中加载GT密度图
            gt_density_batch = None
            if 'gt_density_map' in targets[0]:
                # 如果targets中已有GT密度图，直接使用
                batch_size = len(targets)
                # 获取第一张图的大小以初始化批次张量
                first_density = targets[0]['gt_density_map']
                if isinstance(first_density, np.ndarray):
                    first_density = torch.from_numpy(first_density)
                
                density_shape = first_density.shape if len(first_density.shape) == 3 else (1,) + first_density.shape
                gt_density_batch = torch.zeros(batch_size, *density_shape).to(device)
                
                for i, target in enumerate(targets):
                    if 'gt_density_map' in target:
                        density = target['gt_density_map']
                        if isinstance(density, np.ndarray):
                            density = torch.from_numpy(density).to(device)
                        else:
                            density = density.to(device)
                        
                        if len(density.shape) == 2:
                            density = density.unsqueeze(0)
                        gt_density_batch[i] = density
                # accumulate GT integral for this batch (sum over all images)
                try:
                    batch_gt_sum = float(gt_density_batch.sum().detach().item())
                except Exception:
                    batch_gt_sum = 0.0
                epoch_gt_sum += batch_gt_sum
                epoch_image_count += batch_size
            else:
                # 如果没有GT密度图，警告并降级到分类方案
                print(f"[WARNING] gt_density_map not found in targets, using classification mode")
                use_density_supervision = False
        else:
            # 分类监督模式：计算CCM targets
            ccm_targets = []
            for i in range(len(targets)):
                tgt_num = targets[i]['labels'].shape[0]
                t = 0
                for j in range(len(ccm_params)):
                    if tgt_num >= ccm_params[j]:
                        t = j + 1
                ccm_targets.append(t)
            ccm_targets = torch.tensor(ccm_targets, dtype=torch.int64).to(device)
            gt_density_batch = None
            # accumulate GT counts by number of boxes when density GT not available
            try:
                batch_size = len(targets)
                batch_gt_sum = 0.0
                for i in range(len(targets)):
                    batch_gt_sum += int(targets[i]['labels'].shape[0])
                epoch_gt_sum += batch_gt_sum
                epoch_image_count += batch_size
            except Exception:
                pass

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        adv_loss = None
        adv_loss_weight = getattr(model_unwrapped.transformer, 'adv_loss_weight', 0.0)
        adv_info = None

        # First clean forward pass (used both for adversarial gradients and logging baseline)
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs_clean = model(samples, targets)
            else:
                outputs_clean = model(samples)

            adv_info = outputs_clean.pop('adv_info', None)
            loss_dict_clean = criterion(outputs_clean, targets)
            weight_dict = criterion.weight_dict
            detection_loss_clean = sum(loss_dict_clean[k] * weight_dict[k]
                                       for k in loss_dict_clean.keys() if k in weight_dict)
            
            # 根据监督方案计算CCM/密度损失
            if use_density_supervision and gt_density_batch is not None:
                # 密度图监督
                pred_density = outputs_clean.get('pred_density_map')
                if pred_density is not None:
                    # 如果尺寸不一致，调整GT到预测尺寸并保持积分不变
                    if pred_density.shape[-2:] != gt_density_batch.shape[-2:]:
                        tgt_h, tgt_w = gt_density_batch.shape[-2:]
                        pred_h, pred_w = pred_density.shape[-2:]
                        gt_density_batch = F.interpolate(
                            gt_density_batch, size=(pred_h, pred_w),
                            mode='bilinear', align_corners=False)
                        # 保持总和（积分）一致：面积比例
                        area_scale = (tgt_h * tgt_w) / float(pred_h * pred_w)
                        gt_density_batch = gt_density_batch * area_scale
                    # accumulate predicted integral for this batch (sum over images)
                    try:
                        batch_pred_sum = float(pred_density.sum().detach().item())
                        epoch_pred_sum += batch_pred_sum
                    except Exception:
                        pass
                    # accumulate predicted query count (num_select) if available
                    try:
                        candidate_num_select = outputs_clean.get('num_select', None)
                        if candidate_num_select is None and 'pred_boxes' in outputs_clean:
                            candidate_num_select = outputs_clean['pred_boxes'].shape[1]
                        if candidate_num_select is not None:
                            if torch.is_tensor(candidate_num_select):
                                qval = float(candidate_num_select.item())
                            else:
                                qval = float(candidate_num_select)
                            epoch_query_sum += qval * float(batch_size)
                    except Exception:
                        pass

                    # Pass ccm_feature to density loss so a feature-level ranking term can be computed
                    ccm_feat = outputs_clean.get('ccm_feature', None)
                    # provide single_object_integral to density loss so it can normalize units
                    single_int = getattr(model_unwrapped.transformer, 'single_object_integral', 1.0)
                    try:
                        single_int = float(single_int) if single_int is not None and float(single_int) > 0.0 else 1.0
                    except Exception:
                        single_int = 1.0
                    density_loss_clean, density_loss_dict = density_loss_fn(pred_density, gt_density_batch, ccm_feature=ccm_feat, single_object_integral=single_int)
                    ccm_loss_clean = density_loss_clean
                    # 记录density loss分量
                    for key, val in density_loss_dict.items():
                        loss_dict_clean[f'density_{key}'] = torch.tensor(val, device=device)
                        # 训练早期打印密度监督的积分和损失，验证是否生效
                        if utils.is_main_process() and _cnt < 2:
                            try:
                                gt_sum_dbg = float(gt_density_batch.sum().detach().item())
                                pred_sum_dbg = float(pred_density.sum().detach().item())
                                loss_dbg = float(density_loss_clean.detach().item())
                                # normalize by single_object_integral for readable counts
                                single_int = getattr(model_unwrapped.transformer, 'single_object_integral', 1.0)
                                try:
                                    single_int = float(single_int) if single_int is not None and float(single_int) > 0.0 else 1.0
                                except Exception:
                                    single_int = 1.0
                                gt_count_dbg = gt_sum_dbg / single_int
                                pred_count_dbg = pred_sum_dbg / single_int
                                print(f"[DEBUG][density] epoch {epoch} iter {_cnt} gt_sum={gt_sum_dbg:.2f} pred_sum={pred_sum_dbg:.2f} gt_count={gt_count_dbg:.2f} pred_count={pred_count_dbg:.2f} loss={loss_dbg:.4f}")
                            except Exception:
                                pass
                else:
                    print(f"[WARNING] pred_density_map not found in outputs")
                    ccm_loss_clean = torch.tensor(0.0, device=device)
                    # try to collect num_select even when pred_density missing
                    try:
                        candidate_num_select = outputs_clean.get('num_select', None)
                        if candidate_num_select is None and 'pred_boxes' in outputs_clean:
                            candidate_num_select = outputs_clean['pred_boxes'].shape[1]
                        if candidate_num_select is not None:
                            if torch.is_tensor(candidate_num_select):
                                qval = float(candidate_num_select.item())
                            else:
                                qval = float(candidate_num_select)
                            epoch_query_sum += qval * float(batch_size)
                    except Exception:
                        pass
            else:
                # 分类监督（原始方案）
                ccm_loss_clean = CCM_LOSS(outputs_clean['pred_bbox_number'], ccm_targets)

            ccm_weight_local = density_loss_coeff if use_density_supervision else ccm_coeff
            clean_loss_for_adv = detection_loss_clean + ccm_weight_local * ccm_loss_clean

        adv_requested = (
            model.training and
            getattr(model_unwrapped.transformer, 'use_adv_training', False) and
            adv_loss_weight > 0.0 and
            adv_info is not None)

        feature_delta = None
        query_delta = None

        if adv_requested:
            feature_tensor = adv_info.get('encoder_memory')
            query_tensor = adv_info.get('decoder_input')

            feature_grad = None
            query_grad = None

            # Optimization: Use autograd.grad to avoid double backward (speed up)
            # We accept higher memory usage (retain_graph=True) for faster training.
            
            inputs_to_grad = []
            if feature_tensor is not None and feature_tensor.requires_grad:
                inputs_to_grad.append(feature_tensor)
            if query_tensor is not None and query_tensor.requires_grad:
                inputs_to_grad.append(query_tensor)

            if inputs_to_grad:
                # Scale loss if using AMP to avoid underflow
                loss_to_grad = scaler.scale(clean_loss_for_adv) if args.amp else clean_loss_for_adv
                
                grads = torch.autograd.grad(
                    loss_to_grad,
                    inputs_to_grad,
                    retain_graph=True,
                    allow_unused=True)
                
                grad_idx = 0
                if feature_tensor is not None and feature_tensor.requires_grad:
                    feature_grad = grads[grad_idx]
                    if feature_grad is not None:
                        feature_grad = feature_grad.detach()
                        # If scaled, we technically should unscale, but normalization makes it invariant
                        # feature_grad = feature_grad / scaler.get_scale() 
                    grad_idx += 1
                
                if query_tensor is not None and query_tensor.requires_grad:
                    query_grad = grads[grad_idx]
                    if query_grad is not None:
                        query_grad = query_grad.detach()
                    grad_idx += 1

            # Optimization: Backward clean loss immediately to free graph
            ccm_weight_local = density_loss_coeff if use_density_supervision else ccm_coeff
            clean_loss = detection_loss_clean + ccm_weight_local * ccm_loss_clean
            if args.amp:
                scaler.scale(clean_loss).backward()
            else:
                clean_loss.backward()

            def _build_perturbation(grad_tensor: Optional[torch.Tensor], epsilon: float) -> Optional[torch.Tensor]:
                if grad_tensor is None or epsilon <= 0.0:
                    return None
                grad_detached = grad_tensor.detach()
                grad_shape = grad_detached.shape
                flat = grad_detached.reshape(grad_shape[0], -1)
                norm = flat.norm(p=2, dim=1, keepdim=True)
                norm = torch.clamp(norm, min=1e-6)
                norm = norm.view([grad_shape[0]] + [1] * (len(grad_shape) - 1))
                normalized = grad_detached / norm  # S7: Δ = ε * g / ||g||₂
                return (epsilon * normalized).detach()

            feature_eps = getattr(model_unwrapped.transformer, 'feature_adv_epsilon', getattr(model_unwrapped.transformer, 'adv_epsilon', 0.0))
            query_eps = getattr(model_unwrapped.transformer, 'query_adv_epsilon', getattr(model_unwrapped.transformer, 'adv_epsilon', 0.0))

            feature_delta = _build_perturbation(feature_grad, feature_eps)
            query_delta = _build_perturbation(query_grad, query_eps)
        else:
            # If adv not requested, just backward clean loss
            ccm_weight_local = density_loss_coeff if use_density_supervision else ccm_coeff
            clean_loss = detection_loss_clean + ccm_weight_local * ccm_loss_clean
            if args.amp:
                scaler.scale(clean_loss).backward()
            else:
                clean_loss.backward()

        adv_performed = adv_requested and (feature_delta is not None or query_delta is not None)
        adv_loss = None

        if adv_performed:
            with _temporarily_disable_bn_updates(model_unwrapped):
                with torch.cuda.amp.autocast(enabled=args.amp):
                    if need_tgt_for_training:
                        adv_outputs = model(
                            samples, targets,
                            feature_perturbation=feature_delta,
                            query_perturbation=query_delta)
                    else:
                        adv_outputs = model(
                            samples,
                            feature_perturbation=feature_delta,
                            query_perturbation=query_delta)

                    adv_outputs.pop('adv_info', None)
                    adv_loss_dict = criterion(adv_outputs, targets)
                    adv_loss = sum(adv_loss_dict[k] * weight_dict[k]
                                   for k in adv_loss_dict.keys() if k in weight_dict)
            
            # Backward adv loss
            if args.amp:
                scaler.scale(adv_loss * adv_loss_weight).backward()
            else:
                (adv_loss * adv_loss_weight).backward()

        loss_dict = loss_dict_clean
        detection_loss = detection_loss_clean
        ccm_loss = ccm_loss_clean

        adv_info = None

        adv_info = None

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}

        # 添加CCM/密度损失到日志
        ccm_weight_local = density_loss_coeff if use_density_supervision else ccm_coeff
        if use_density_supervision:
            loss_dict_reduced_unscaled['density_loss_unscaled'] = ccm_loss
            loss_dict_reduced_scaled['density_loss'] = ccm_loss * ccm_weight_local
        else:
            loss_dict_reduced_unscaled['ccm_loss_unscaled'] = ccm_loss
            loss_dict_reduced_scaled['ccm_loss'] = ccm_loss * ccm_weight_local
        
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        if adv_loss is not None:
            adv_loss_reduced = utils.reduce_dict({'adv_loss': adv_loss.detach()})['adv_loss']
            loss_dict_reduced_unscaled['adv_loss_unscaled'] = adv_loss_reduced
            loss_dict_reduced_scaled['adv_loss'] = adv_loss_reduced * adv_loss_weight
            losses_reduced_scaled += adv_loss_reduced * adv_loss_weight

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if args.amp:
            # scaler.scale(losses).backward() # Already backwarded
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # losses.backward() # Already backwarded
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        
        # Explicitly clear large tensors to help memory
        adv_info = None
        feature_tensor = None
        query_tensor = None
        feature_grad = None
        query_grad = None
        feature_delta = None
        query_delta = None
        outputs_clean = None
        adv_outputs = None
        loss_dict_clean = None
        adv_loss_dict = None
        clean_loss_for_adv = None
        adv_loss = None
        
        # Optional: Empty cache if memory is tight (can slow down training)
        # torch.cuda.empty_cache()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    # Reduce epoch accumulators across processes and print means
    try:
        t = torch.tensor([epoch_gt_sum, epoch_pred_sum, epoch_query_sum, float(epoch_image_count)], device=device, dtype=torch.float64)
        if utils.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(t)
        total_images = float(t[3].item())
        # If density supervision was active, epoch_gt_sum/epoch_pred_sum are integrals
        # and should be converted to object counts by dividing by single_object_integral.
        if density_supervision_active:
            single_integral = getattr(model_unwrapped.transformer, 'single_object_integral', 1.0)
            try:
                single_integral = float(single_integral) if single_integral is not None and float(single_integral) > 0.0 else 1.0
            except Exception:
                single_integral = 1.0
            gt_mean = float((t[0].item() / total_images) / single_integral) if total_images > 0 else 0.0
            pred_mean = float((t[1].item() / total_images) / single_integral) if total_images > 0 else 0.0
        else:
            gt_mean = float(t[0].item() / total_images) if total_images > 0 else 0.0
            pred_mean = float(t[1].item() / total_images) if total_images > 0 else 0.0
        query_mean = float(t[2].item() / total_images) if total_images > 0 else 0.0
        if utils.is_main_process():
            print(f"GT count mean: {gt_mean:.6f}")
            print(f"Pred count mean: {pred_mean:.6f}")
            print(f"Query count mean: {query_mean:.6f}")
    except Exception:
        pass

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg
               for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update(
            {f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(
            window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(
        k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {}  # for debug only

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        samples = samples.to(device)
        targets = [{k: to_device(v, device)
                    for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}

        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, outputs['num_select'])

        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output
               for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](
                outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']

            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """ pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat(
                    (_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp

        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(
            args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k,
             meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator


# @torch.no_grad()
# def infer(model, postprocessors, data_loader, device, args=None):
#     model.eval()
#
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'
#
#     all_results = []
#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
#         samples = samples.to(device)
#         targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
#
#         with torch.cuda.amp.autocast(enabled=args.amp):
#             outputs = model(samples)
#
#         orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#         results = postprocessors['bbox'](outputs, orig_target_sizes, outputs['num_select'])
#
#         print("\n#################################33")
#         print(samples)
#         print("\n#################################33")
#
#         for i in range(len(results)):
#             image_result = {
#                 'image_id': samples[i]['image_id'].item(),
#                 'result': results[i]
#             }
#             all_results.append(image_result)
#
#     return all_results
