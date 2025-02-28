<<<<<<< HEAD
import argparse
from lib2to3.pgen2.pgen import generate_grammar
import os
import random
import shutil
import time
from typing import Optional, Union
import warnings
from pathlib import Path
from loss.chamfer_distance_loss import ChamferDistanceLoss
from loss.prototypes import get_target_prototypes
from loss.pseudo import (
    get_pseudo_label,
    get_pseudo_label_msp,
    get_pseudo_mask,
    get_pseudo_mask_from_prototypes,
)
from loss.unknown_loss import InfoNCELoss

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter

from util.common_util import get_aupr_auroc

# from thop import profile
# from util.tsne_visualization import (
#     balanced_downampling,
#     tsne,
#     visual_mean_prototypes_oneItem,
# )

warnings.filterwarnings("ignore")


def get_parser():
    from util import config

    parser = argparse.ArgumentParser(
        description="PyTorch Point Cloud Semantic Segmentation"
    )
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     default="config/s3dis/s3dis_stratified_transformer_debug.yaml",
    #     help="config file",
    # )
    parser.add_argument(
        "--config",
        type=str,
        default="config/scannetv2/scannetv2_stratified_transformer_debug.yaml",
        help="config file",
    )
    parser.add_argument(
        "opts",
        help="see config/s3dis/s3dis_stratified_transformer.yaml for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    save_path = Path(cfg.save_path)
    if save_path.name != "debug":
        save_path.mkdir(exist_ok=False)
        shutil.copy(args.config, f'{save_path / args.config.split("/")[-1]}')
        shutil.copy(__file__, f"{save_path / os.path.basename(__file__)}")
        shutil.copytree("model", f"{save_path / 'model'}")
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0
    )


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.train_gpu)
    from util.common_util import find_free_port

    # import torch.backends.mkldnn
    # ackends.mkldnn.enabled = False
    # os.environ["LRU_CACHE_CAPACITY"] = "1"
    # cudnn.deterministic = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(
            main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args)
        )
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    from util.logger import get_logger
    from util.lr import MultiStepWithWarmup, PolyLR

    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    model = get_model(args)

    # set loss func
    # criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    criterion = {
        "CE": nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(),
        "kl_div": nn.SmoothL1Loss().cuda(),
        "cd": ChamferDistanceLoss().cuda(),
        "BCE": nn.BCEWithLogitsLoss().cuda(),
    }

    # set optimizer
    optimizer = None
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "AdamW":
        transformer_lr_scale = args.get("transformer_lr_scale", 0.1)
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "blocks" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "blocks" in n and p.requires_grad
                ],
                "lr": args.base_lr * transformer_lr_scale,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.base_lr, weight_decay=args.weight_decay
        )
    assert isinstance(optimizer, torch.optim.Optimizer)

    if main_process():
        global logger, writer
        logger = get_logger(args.save_path)
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
        logger.info(
            "#Model parameters: {}".format(
                sum([x.nelement() for x in model.parameters()])
            )
        )
        if args.get("max_grad_norm", None):
            logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))

    if args.distributed:
        torch.cuda.set_device(gpu)
        # args.batch_size = int(args.batch_size / ngpus_per_node)
        # args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], find_unused_parameters=True
        )
    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint["state_dict"])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    scheduler_state_dict = None
    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda()
            )
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler_state_dict = checkpoint["scheduler"]
            best_iou = checkpoint["best_iou"]
            if main_process():
                logger.info(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        args.resume, checkpoint["epoch"]
                    )
                )
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    train_loader, train_sampler = get_train_dataset(args)
    val_loader = get_val_dataset(args)

    # set scheduler
    if args.scheduler == "MultiStepWithWarmup":
        assert args.scheduler_update == "step"
        if main_process():
            logger.info(
                "scheduler: MultiStepWithWarmup. scheduler_update: {}".format(
                    args.scheduler_update
                )
            )
        iter_per_epoch = len(train_loader)
        milestones = [
            int(args.epochs * 0.6) * iter_per_epoch,
            int(args.epochs * 0.8) * iter_per_epoch,
        ]
        scheduler = MultiStepWithWarmup(
            optimizer,
            milestones=milestones,
            gamma=0.1,
            warmup=args.warmup,
            warmup_iters=args.warmup_iters,
            warmup_ratio=args.warmup_ratio,
        )
    elif args.scheduler == "MultiStep":
        assert args.scheduler_update == "epoch"
        milestones = (
            [int(x) for x in args.milestones.split(",")]
            if hasattr(args, "milestones")
            else [int(args.epochs * 0.6), int(args.epochs * 0.8)]
        )
        gamma = args.gamma if hasattr(args, "gamma") else 0.1
        if main_process():
            logger.info(
                "scheduler: MultiStep. scheduler_update: {}. milestones: {}, gamma: {}".format(
                    args.scheduler_update, milestones, gamma
                )
            )
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    elif args.scheduler == "Poly":
        if main_process():
            logger.info(
                "scheduler: Poly. scheduler_update: {}".format(args.scheduler_update)
            )
        if args.scheduler_update == "epoch":
            scheduler = PolyLR(optimizer, max_iter=args.epochs, power=args.power)
        elif args.scheduler_update == "step":
            iter_per_epoch = len(train_loader)
            scheduler = PolyLR(
                optimizer, max_iter=args.epochs * iter_per_epoch, power=args.power
            )
        else:
            raise ValueError(
                "No such scheduler update {}".format(args.scheduler_update)
            )
    else:
        raise ValueError("No such scheduler {}".format(args.scheduler))

    if args.resume and os.path.isfile(args.resume):
        assert isinstance(scheduler_state_dict, dict)
        scheduler.load_state_dict(scheduler_state_dict)
        print("resume scheduler")

    ###################
    # start training #
    ###################

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if main_process():
            logger.info("lr: {}".format(scheduler.get_last_lr()))

        loss_train, mIoU_train, mAcc_train, allAcc_train, others_train = train(
            train_loader, model, criterion, optimizer, epoch, scaler, scheduler
        )
        if args.scheduler_update == "epoch":
            scheduler.step()
        epoch_log = epoch + 1

        if main_process():
            writer.add_scalar("loss_train", loss_train, epoch_log)
            writer.add_scalar("mIoU_train", mIoU_train, epoch_log)
            writer.add_scalar("mAcc_train", mAcc_train, epoch_log)
            writer.add_scalar("allAcc_train", allAcc_train, epoch_log)
            writer.add_scalar("loss2_train", others_train["loss2"], epoch_log)
            writer.add_scalar("loss3_train", others_train["loss3"], epoch_log)
            writer.add_scalar("aupr_batch_mean_train", others_train["aupr"], epoch_log)
            writer.add_scalar(
                "auroc_batch_mean_train", others_train["auroc"], epoch_log
            )
            # writer.add_scalar("aupr_train", others_train["aupr_epoch"], epoch_log)
            # writer.add_scalar("auroc_train", others_train["auroc_epoch"], epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val, val_meters = validate(
                val_loader, model, criterion, epoch
            )

            if main_process():
                writer.add_scalar("loss_val", loss_val, epoch_log)
                writer.add_scalar("mIoU_val", mIoU_val, epoch_log)
                writer.add_scalar("mAcc_val", mAcc_val, epoch_log)
                writer.add_scalar("allAcc_val", allAcc_val, epoch_log)
                writer.add_scalar("loss2_val", val_meters["loss2"], epoch_log)
                writer.add_scalar("loss3_val", val_meters["loss3"], epoch_log)
                writer.add_scalar("aupr_batch_mean_val", val_meters["aupr"], epoch_log)
                writer.add_scalar(
                    "auroc_batch_mean_val", val_meters["auroc"], epoch_log
                )
                # writer.add_scalar("aupr_val", val_meters["aupr_epoch"], epoch_log)
                # writer.add_scalar("auroc_ val", val_meters["auroc_epoch"], epoch_log)
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            if not os.path.exists(args.save_path + "/model/"):
                os.makedirs(args.save_path + "/model/")
            filename = args.save_path + "/model/model_last.pth"
            logger.info("Saving checkpoint to: " + filename)
            torch.save(
                {
                    "epoch": epoch_log,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_iou": best_iou,
                    "is_best": is_best,
                },
                filename,
            )
            if is_best:
                shutil.copyfile(filename, args.save_path + "/model/model_best.pth")
            if epoch_log % args.save_freq2 == 0:
                shutil.copyfile(
                    filename,
                    args.save_path + "/model/model_epoch_{}.pth".format(epoch_log),
                )

    if main_process():
        writer.close()
        logger.info("==>Training done!\nBest Iou: %.3f" % (best_iou))


def train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler):
    import torch.distributed as dist
    import torch_points_kernels as tp

    from util.common_util import AverageMeter, intersectionAndUnionGPU

    criterion, criterion_2, criterion_3, criterion_4 = (
        criterion["CE"],
        criterion["kl_div"],
        criterion["cd"],
        criterion["BCE"],
    )
    milestones = (
        [int(x) for x in args.milestones.split(",")]
        if hasattr(args, "milestones")
        else [int(args.epochs * 0.6), int(args.epochs * 0.8)]
    )
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss2_meter = AverageMeter()
    loss3_meter = AverageMeter()
    loss4_meter = AverageMeter()
    loss5_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_id_meter = AverageMeter()
    # target_list = []
    unknown_logits_list = []
    aupr_meter = AverageMeter()
    auroc_meter = AverageMeter()
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (coord, feat, target, offset) in enumerate(
        train_loader
    ):  # (n, 3), (n, c), (n), (b)
        data_time.update(time.time() - end)

        # coord, feat, target, offset = new_coord.clone(), new_feat.clone(), new_label_ori.clone(), new_offset.clone()
        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat(
            [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0
        ).long()

        sigma = 1.0
        radius = 2.5 * args.grid_size * sigma
        neighbor_idx = tp.ball_query(
            radius,
            args.max_num_neighbors,
            coord,
            coord,
            mode="partial_dense",
            batch_x=batch,
            batch_y=batch,
        )[0]

        coord, feat, target, offset = (
            coord.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            target.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
        )
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]
        # new_label_ori = new_label_ori.cuda(non_blocking=True)

        # unknown_mask = (target == 20).float().cuda()[..., None]
        target_id = target.clone()
        target_id[
            torch.isin(
                target_id, torch.tensor(args.unknown_label, device=target_id.device)
            )
        ] = -100
        assert not torch.isin(
            target_id, torch.tensor(args.unknown_label, device=target_id.device)
        ).any()

        if args.concat_xyz:
            feat = torch.cat([feat, coord], 1)

        use_amp = args.use_amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            output, others = model(feat, coord, offset, batch, neighbor_idx)
            unknown_pred = others["unknown"]
            # unknown_dist = others["unknown_dist"].clone().detach()
            # unknown_logits = unknown_pred
            # unknown_logits_list.append(unknown_logits.clone().detach())
            # last_feats = others["last_feats"]
            # pred_prototypes = others["prototypes"]
            # pred_mean_proto = others["mean_prototypes"]
            # pred_proto_dist = others["proto_dist"]
            assert output.shape[1] == args.classes
            if target.shape[-1] == 1:
                assert True, "unexpected situation"
                # target = target[:, 0]  # for cls
                # target_pseudo = target_pseudo[:, 0]  # for cls
            pseudo_mask = get_pseudo_label_msp(output)
            target_loss = target_id.clone()
            target_loss[pseudo_mask] = 20
            # pseudo_mask = get_pseudo_mask_from_prototypes(
            #     last_feats,
            #     target,
            #     offset,
            #     pred_prototypes,
            #     args.classes,
            #     args.ignore_label,
            # ) # pseudo label
            # (
            #     target_proto,
            #     target_mean_proto,
            #     # target_proto_dist,
            #     unique_mask,
            # ) = get_target_prototypes(
            #     last_feats,
            #     pred_prototypes,
            #     target_id,
            #     offset,
            #     args.classes,
            #     ignore_index=args.ignore_label,
            # )

            output = torch.cat([output, unknown_pred], -1)  # pseudo label
            # output = output.softmax(-1)
            loss = criterion(output, target_loss)
            # loss2 = (
            #     criterion_3(
            #         pred_prototypes[unique_mask],
            #         target_proto[unique_mask],
            #     )
            #     * 1
            # )
            # loss3 = (
            #     criterion_2(
            #         pred_mean_proto[unique_mask],
            #         target_mean_proto[unique_mask],
            #     )
            #     * 100
            # )
            # loss4 = (
            #     criterion_2(
            #         pred_proto_dist[unique_mask],
            #         target_proto_dist[unique_mask],
            #     )
            #     * 100
            # )
            # assert not torch.isnan(loss4)
            # loss5 = criterion_4(unknown_pred, unknown_mask)
            loss2 = torch.tensor(0.0).cuda()
            loss3 = torch.tensor(0.0).cuda()
            loss4 = torch.tensor(0.0).cuda()
            loss5 = torch.tensor(0.0).cuda()
            loss_all = loss + loss2 + loss3 + loss4 + loss5
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss_all).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_all.backward()
            optimizer.step()

        if args.scheduler_update == "step":
            scheduler.step()

        # unknown_pred = output[..., [-1]]
        output = output[:, :20]  # split pseudo label
        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            loss2 *= n
            loss3 *= n
            loss4 *= n
            loss5 *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss)
            dist.all_reduce(count)
            dist.all_reduce(loss2)
            dist.all_reduce(loss3)
            dist.all_reduce(loss4)
            dist.all_reduce(loss5)
            n = count.item()
            loss /= n
            loss2 /= n
            loss3 /= n
            loss4 /= n
            loss5 /= n

        aupr, auroc = get_aupr_auroc(
            unknown_pred,
            target,
            args.unknown_label,
            args.ignore_label,
            logger if main_process() else None,
            distributed=args.distributed,
            main_process=main_process(),
        )

        if aupr > 0 and auroc > 0:
            aupr_meter.update(aupr)
            auroc_meter.update(auroc)
        elif main_process():
            logger.warning("This batch does not contain any OOD pixels or is only OOD.")

        intersection, union, target_id = intersectionAndUnionGPU(
            output, target_id, args.classes, args.ignore_label
        )
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection)
            dist.all_reduce(union)
            dist.all_reduce(target_id)
        intersection, union, target_id = (
            intersection.cpu().numpy(),
            union.cpu().numpy(),
            target_id.cpu().numpy(),
        )
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_id_meter.update(target_id)

        accuracy = sum(intersection_meter.val) / (sum(target_id_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        loss2_meter.update(loss2.item(), n)
        loss3_meter.update(loss3.item(), n)
        loss4_meter.update(loss4.item(), n)
        loss5_meter.update(loss5.item(), n)
        batch_time.update(time.time() - end)

        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            lr = scheduler.get_last_lr()
            if isinstance(lr, list):
                lr = [round(x, 8) for x in lr]
            elif isinstance(lr, float):
                lr = round(lr, 8)
            logger.info(
                "Epoch: [{}/{}][{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Remain {remain_time} "
                "Loss {loss_meter.val:.4f} "
                "Loss2 {loss2_meter.val:.4f} "
                "Loss3 {loss3_meter.val:.4f} "
                "Loss4 {loss4_meter.val:.4f} "
                "Loss5 {loss5_meter.val:.4f} "
                "Lr: {lr} "
                "Accuracy {accuracy:.4f}. "
                "aupr {aupr:.4f} ({aupr_meter.avg:.4f}) "
                "auroc {auroc:.4f} ({auroc_meter.avg:.4f})".format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    remain_time=remain_time,
                    loss_meter=loss_meter,
                    loss2_meter=loss2_meter,
                    loss3_meter=loss3_meter,
                    loss4_meter=loss4_meter,
                    loss5_meter=loss5_meter,
                    lr=lr,
                    accuracy=accuracy,
                    aupr=aupr,
                    aupr_meter=aupr_meter,
                    auroc=auroc,
                    auroc_meter=auroc_meter,
                )
            )
        if main_process():
            writer.add_scalar("loss_train_batch", loss_meter.val, current_iter)
            writer.add_scalar("loss2_train_batch", loss2_meter.val, current_iter)
            writer.add_scalar("loss3_train_batch", loss3_meter.val, current_iter)
            # writer.add_scalar("loss4_train_batch", loss4_meter.val, current_iter)
            writer.add_scalar(
                "mIoU_train_batch",
                np.mean(intersection / (union + 1e-10)),
                current_iter,
            )
            writer.add_scalar(
                "mAcc_train_batch",
                np.mean(intersection / (target_id + 1e-10)),
                current_iter,
            )
            writer.add_scalar("allAcc_train_batch", accuracy, current_iter)

            # writer.add_histogram(
            #     "unknown_label_train_batch", unknown_logits.float(), current_iter
            # )
            # writer.add_histogram("unknown_dist_train_batch", unknown_dist, current_iter)
            # if aupr > 0 and auroc > 0:
            #     writer.add_scalar("aupr_train_batch", aupr, current_iter)
            #     writer.add_scalar("auroc_train_batch", auroc, current_iter)
        # visual_prototypes(
        #     epoch,
        #     i,
        #     batch,
        #     last_feats,
        #     target_ori,
        #     pred_prototypes,
        #     "train",
        #     f"{args.save_path}/prototypes_tsne",
        # )
        # visual_mean_prototypes()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_id_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_id_meter.sum) + 1e-10)

    # unknown_logits_all = torch.cat(unknown_logits_list, 0)
    # target_all = torch.cat(target_ori_list, 0)
    # aupr_epoch, auroc_epoch = get_aupr_auroc(
    #     unknown_logits_all,
    #     target_all,
    #     args.unknown_label,
    #     args.ignore_label,
    #     logger if main_process() else None,
    #     distributed=args.distributed,
    #     main_process=main_process(),
    # )

    if main_process():
        logger.info(
            "Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                epoch + 1, args.epochs, mIoU, mAcc, allAcc
            )
        )
        logger.info(
            f"Train AUPR/AUROC batch mean: {aupr_meter.avg:.4f}/{auroc_meter.avg:.4f}"
        )
        # logger.info(f"Train AUPR/AUROC epoch: {aupr_epoch:.4f}/{auroc_epoch:.4f}")

    return (
        loss_meter.avg,
        mIoU,
        mAcc,
        allAcc,
        {
            "loss2": loss2_meter.avg,
            "loss3": loss3_meter.avg,
            "aupr": aupr_meter.avg,
            "auroc": auroc_meter.avg,
            # "aupr_epoch": aupr_epoch,
            # "auroc_epoch": auroc_epoch,
        },
    )


def validate(val_loader, model, criterion, epoch):
    import torch.distributed as dist
    import torch_points_kernels as tp

    from util.common_util import AverageMeter, intersectionAndUnionGPU

    criterion, criterion_2, criterion_3 = (
        criterion["CE"],
        criterion["kl_div"],
        criterion["cd"],
    )

    milestones = (
        [int(x) for x in args.milestones.split(",")]
        if hasattr(args, "milestones")
        else [int(args.epochs * 0.6), int(args.epochs * 0.8)]
    )
    if main_process():
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss2_meter = AverageMeter()
    loss3_meter = AverageMeter()
    loss4_meter = AverageMeter()
    loss5_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    target_ori_list = []
    unknown_logits_list = []
    aupr_meter = AverageMeter()
    auroc_meter = AverageMeter()
    torch.cuda.empty_cache()

    model.eval()
    end = time.time()
    for i, (coord, feat, target, offset) in enumerate(val_loader):
        data_time.update(time.time() - end)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat(
            [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0
        ).long()

        sigma = 1.0
        radius = 2.5 * args.grid_size * sigma
        neighbor_idx = tp.ball_query(
            radius,
            args.max_num_neighbors,
            coord,
            coord,
            mode="partial_dense",
            batch_x=batch,
            batch_y=batch,
        )[0]

        coord, feat, target, offset = (
            coord.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            target.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
        )
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]

        target_id = target.clone().detach()
        # target_ori_list.append(target_id)
        target_id[
            torch.isin(
                target, torch.tensor(args.unknown_label, device=target_id.device)
            )
        ] = args.ignore_label
        assert not torch.isin(
            target_id, torch.tensor(args.unknown_label, device=target_id.device)
        ).any()

        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls

        if args.concat_xyz:
            feat = torch.cat([feat, coord], 1)

        with torch.no_grad():
            output, others = model(feat, coord, offset, batch, neighbor_idx)
            unknown_pred = others["unknown"]
            # unknown_dist = others["unknown_dist"]
            # unknown_logits_list.append(unknown_logits.clone().detach())
            # last_feats = others["last_feats"]
            # pred_prototypes = others["prototypes"]
            # pred_mean_proto = others["mean_prototypes"]
            # pred_proto_dist = others["proto_dist"]
            # pseudo_mask = get_pseudo_mask(
            #     output, target, args.classes, epoch, milestones, args.ignore_label
            # ) # pseudo label
            # (
            #     target_proto,
            #     target_mean_proto,
            #     # target_proto_dist,
            #     unique_mask,
            # ) = get_target_prototypes(
            #     last_feats,
            #     pred_prototypes,
            #     target_id,
            #     offset,
            #     args.classes,
            #     ignore_index=args.ignore_label,
            # )
            # output = torch.cat([output, unknown_pred], -1) # pseudo label
            # target_pseudo[pseudo_mask] = args.classes # pseudo label
            # output = output.softmax(-1)
            loss = criterion(output, target)
            # loss2 = (
            #     criterion_3(
            #         pred_prototypes[unique_mask],
            #         target_proto[unique_mask],
            #     )
            #     * 1
            # )
            # loss3 = (
            #     criterion_2(
            #         pred_mean_proto[unique_mask],
            #         target_mean_proto[unique_mask],
            #     )
            #     * 100
            # )
            # loss4 = (
            #     criterion_2(
            #         pred_proto_dist[unique_mask],
            #         target_proto_dist[unique_mask],
            #     )
            #     * 100
            # )
            loss2 = torch.tensor(0.0).cuda()
            loss3 = torch.tensor(0.0).cuda()
            loss4 = torch.tensor(0.0).cuda()

        # unknown_logits = output[:, -1]
        output = output[:, :20]  # pseudo label
        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            loss2 *= n
            loss3 *= n
            loss4 *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss)
            dist.all_reduce(count)
            dist.all_reduce(loss2)
            dist.all_reduce(loss3)
            dist.all_reduce(loss4)
            n = count.item()
            loss /= n
            loss2 /= n
            loss3 /= n
            loss4 /= n

        aupr, auroc = get_aupr_auroc(
            unknown_pred,
            target,
            args.unknown_label,
            args.ignore_label,
            logger if main_process() else None,
            distributed=args.distributed,
            main_process=main_process(),
        )
        if aupr > 0 and auroc > 0:
            aupr_meter.update(aupr)
            auroc_meter.update(auroc)
        elif main_process():
            logger.warning("This batch does not contain any OOD pixels or is only OOD.")

        intersection, union, target = intersectionAndUnionGPU(
            output, target, args.classes, args.ignore_label
        )
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection)
            dist.all_reduce(union)
            dist.all_reduce(target)
        intersection, union, target = (
            intersection.cpu().numpy(),
            union.cpu().numpy(),
            target.cpu().numpy(),
        )
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        loss2_meter.update(loss2.item(), n)
        loss3_meter.update(loss3.item(), n)
        loss4_meter.update(loss4.item(), n)
        batch_time.update(time.time() - end)

        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info(
                "Test: [{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                "Loss2 {loss2_meter.val:.4f} ({loss2_meter.avg:.4f}) "
                "Loss3 {loss3_meter.val:.4f} ({loss2_meter.avg:.4f}) "
                "Loss4 {loss4_meter.val:.4f} ({loss4_meter.avg:.4f}) "
                "Accuracy {accuracy:.4f}. "
                "aupr {aupr:.4f} ({aupr_meter.avg:.4f}) "
                "auroc {auroc:.4f} ({auroc_meter.avg:.4f})".format(
                    i + 1,
                    len(val_loader),
                    data_time=data_time,
                    batch_time=batch_time,
                    loss_meter=loss_meter,
                    loss2_meter=loss2_meter,
                    loss3_meter=loss3_meter,
                    loss4_meter=loss4_meter,
                    accuracy=accuracy,
                    aupr=aupr,
                    aupr_meter=aupr_meter,
                    auroc=auroc,
                    auroc_meter=auroc_meter,
                )
            )
        current_iter = epoch * len(val_loader) + i + 1
        # if main_process():
        #     writer.add_histogram(
        #         "unknown_label_val_batch", unknown_logits.float(), current_iter
        #     )
        #     writer.add_histogram("unknown_dist_val_batch", unknown_dist, current_iter)
        # visual_prototypes(
        #     epoch,
        #     i,
        #     batch,
        #     last_feats,
        #     target_ori,
        #     pred_prototypes,
        #     "validate",
        #     f"{args.save_path}/prototypes_tsne",
        # )
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # unknown_logits_all = torch.cat(unknown_logits_list, 0)
    # target_all = torch.cat(target_ori_list, 0)
    # aupr_epoch, auroc_epoch = get_aupr_auroc(
    #     unknown_logits_all,
    #     target_all,
    #     args.unknown_label,
    #     args.ignore_label,
    #     logger if main_process() else None,
    #     distributed=args.distributed,
    #     main_process=main_process(),
    # )

    if main_process():
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )
        for i in range(args.classes):
            logger.info(
                "Class_{} Result: iou/accuracy {:.4f}/{:.4f}.".format(
                    i, iou_class[i], accuracy_class[i]
                )
            )

        logger.info(
            f"Val AUPR/AUROC batch mean: {aupr_meter.avg:.4f}/{auroc_meter.avg:.4f}"
        )
        # logger.info(f"Val AUPR/AUROC epoch: {aupr_epoch:.4f}/{auroc_epoch:.4f}")

        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    return (
        loss_meter.avg,
        mIoU,
        mAcc,
        allAcc,
        {
            "loss2": loss2_meter.avg,
            "loss3": loss3_meter.avg,
            "aupr": aupr_meter.avg,
            "auroc": auroc_meter.avg,
            # "aupr_epoch": aupr_epoch,
            # "auroc_epoch": auroc_epoch,
        },
    )


def get_model(args):
    # get model
    if args.arch == "stratified_transformer":
        from model.stratified_transformer import Stratified

        args.patch_size = args.grid_size * args.patch_size
        args.window_size = [
            args.patch_size * args.window_size * (2**i)
            for i in range(args.num_layers)
        ]
        args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

        model = Stratified(
            args.downsample_scale,
            args.depths,
            args.channels,
            args.num_heads,
            args.window_size,
            args.up_k,
            args.grid_sizes,
            args.quant_sizes,
            rel_query=args.rel_query,
            rel_key=args.rel_key,
            rel_value=args.rel_value,
            drop_path_rate=args.drop_path_rate,
            concat_xyz=args.concat_xyz,
            num_classes=args.classes,
            ratio=args.ratio,
            k=args.k,
            prev_grid_size=args.grid_size,
            sigma=1.0,
            num_layers=args.num_layers,
            stem_transformer=args.stem_transformer,
        )
    elif (
        args.arch == "stratified_transformer_v1b1_gridfeat"
        or args.arch == "stratified_transformer_opwd"
        or args.arch == "stratified_transformer_v1_proj"
        or args.arch == "stratified_transformer_v2_learnproto"
        or args.arch == "stratified_transformer_v2b1_improve"
        or args.arch == "stratified_transformer_v2b2_meanproto"
        or args.arch == "stratified_transformer_v2b3_lastfeats"
        or args.arch == "stratified_transformer_v2b4_dist"
        or args.arch == "stratified_transformer_v2b5_unknown"
    ):
        import importlib

        StratifiedOpwd = importlib.import_module("model." + args.arch).StratifiedOpwd

        args.patch_size = args.grid_size * args.patch_size
        args.window_size = [
            args.patch_size * args.window_size * (2**i)
            for i in range(args.num_layers)
        ]
        args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

        model = StratifiedOpwd(
            args.downsample_scale,
            args.depths,
            args.channels,
            args.num_heads,
            args.window_size,
            args.up_k,
            args.grid_sizes,
            args.quant_sizes,
            rel_query=args.rel_query,
            rel_key=args.rel_key,
            rel_value=args.rel_value,
            drop_path_rate=args.drop_path_rate,
            concat_xyz=args.concat_xyz,
            num_classes=args.classes,
            ratio=args.ratio,
            k=args.k,
            prev_grid_size=args.grid_size,
            sigma=1.0,
            num_layers=args.num_layers,
            stem_transformer=args.stem_transformer,
        )
    elif args.arch == "swin3d_transformer":
        from model.swin3d_transformer import Swin

        args.patch_size = args.grid_size * args.patch_size
        args.window_sizes = [
            args.patch_size * args.window_size * (2**i)
            for i in range(args.num_layers)
        ]
        args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

        model = Swin(
            args.depths,
            args.channels,
            args.num_heads,
            args.window_sizes,
            args.up_k,
            args.grid_sizes,
            args.quant_sizes,
            rel_query=args.rel_query,
            rel_key=args.rel_key,
            rel_value=args.rel_value,
            drop_path_rate=args.drop_path_rate,
            concat_xyz=args.concat_xyz,
            num_classes=args.classes,
            ratio=args.ratio,
            k=args.k,
            prev_grid_size=args.grid_size,
            sigma=1.0,
            num_layers=args.num_layers,
            stem_transformer=args.stem_transformer,
        )

    else:
        raise Exception("architecture {} not supported yet".format(args.arch))

    return model


def get_train_dataset(args):
    from functools import partial

    from util import transform
    from util.data_util import collate_fn_limit
    from util.s3dis import S3DIS
    from util.scannet_v2_ori import Scannetv2

    if args.data_name == "s3dis":
        train_transform = None
        if args.aug:
            jitter_sigma = args.get("jitter_sigma", 0.01)
            jitter_clip = args.get("jitter_clip", 0.05)
            if main_process():
                logger.info("augmentation all")
                logger.info(
                    "jitter_sigma: {}, jitter_clip: {}".format(
                        jitter_sigma, jitter_clip
                    )
                )
            train_transform = transform.Compose(
                [
                    transform.RandomRotate(along_z=args.get("rotate_along_z", True)),
                    transform.RandomScale(
                        scale_low=args.get("scale_low", 0.8),
                        scale_high=args.get("scale_high", 1.2),
                    ),
                    transform.RandomJitter(sigma=jitter_sigma, clip=jitter_clip),
                    transform.RandomDropColor(
                        color_augment=args.get("color_augment", 0.0)
                    ),
                ]
            )
        train_data = S3DIS(
            split="train",
            data_root=args.data_root,
            test_area=args.test_area,
            voxel_size=args.voxel_size,
            voxel_max=args.voxel_max,
            transform=train_transform,
            shuffle_index=True,
            loop=args.loop,
        )
    elif args.data_name == "scannetv2":
        train_transform = None
        if args.aug:
            if main_process():
                logger.info("use Augmentation")
            train_transform = transform.Compose(
                [
                    transform.RandomRotate(along_z=args.get("rotate_along_z", True)),
                    transform.RandomScale(
                        scale_low=args.get("scale_low", 0.8),
                        scale_high=args.get("scale_high", 1.2),
                    ),
                    transform.RandomDropColor(
                        color_augment=args.get("color_augment", 0.0)
                    ),
                ]
            )

        train_split = args.get("train_split", "train")
        if main_process():
            logger.info("scannet. train_split: {}".format(train_split))

        train_data = Scannetv2(
            split=train_split,
            data_root=args.data_root,
            voxel_size=args.voxel_size,
            voxel_max=args.voxel_max,
            transform=train_transform,
            shuffle_index=True,
            loop=args.loop,
        )
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if main_process():
        logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=partial(
            collate_fn_limit,
            max_batch_points=args.max_batch_points,
            logger=logger if main_process() else None,
        ),
    )

    return train_loader, train_sampler


def get_val_dataset(args):
    from util.data_util import collate_fn
    from util.s3dis import S3DIS
    from util.scannet_v2 import Scannetv2

    val_transform = None
    if args.data_name == "s3dis":
        val_data = S3DIS(
            split="val",
            data_root=args.data_root,
            test_area=args.test_area,
            voxel_size=args.voxel_size,
            voxel_max=800000,
            transform=val_transform,
            loop=args.val_loop,
        )
    elif args.data_name == "scannetv2":
        val_data = Scannetv2(
            split="val",
            data_root=args.data_root,
            voxel_size=args.voxel_size,
            voxel_max=800000,
            transform=val_transform,
            # loop=args.val_loop,
        )
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=collate_fn,
    )
    return val_loader


if __name__ == "__main__":
    import gc

    gc.collect()
    main()
=======
import argparse
from lib2to3.pgen2.pgen import generate_grammar
import os
import random
import shutil
import time
from typing import Optional, Union
import warnings
from pathlib import Path
from loss.chamfer_distance_loss import ChamferDistanceLoss
from loss.prototypes import get_target_prototypes
from loss.pseudo import (
    get_pseudo_label,
    get_pseudo_label_msp,
    get_pseudo_mask,
    get_pseudo_mask_from_prototypes,
)
from loss.unknown_loss import InfoNCELoss

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter

from util.common_util import get_aupr_auroc

# from thop import profile
# from util.tsne_visualization import (
#     balanced_downampling,
#     tsne,
#     visual_mean_prototypes_oneItem,
# )

warnings.filterwarnings("ignore")


def get_parser():
    from util import config

    parser = argparse.ArgumentParser(
        description="PyTorch Point Cloud Semantic Segmentation"
    )
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     default="config/s3dis/s3dis_stratified_transformer_debug.yaml",
    #     help="config file",
    # )
    parser.add_argument(
        "--config",
        type=str,
        default="config/scannetv2/scannetv2_stratified_transformer_debug.yaml",
        help="config file",
    )
    parser.add_argument(
        "opts",
        help="see config/s3dis/s3dis_stratified_transformer.yaml for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    save_path = Path(cfg.save_path)
    if save_path.name != "debug":
        save_path.mkdir(exist_ok=False)
        shutil.copy(args.config, f'{save_path / args.config.split("/")[-1]}')
        shutil.copy(__file__, f"{save_path / os.path.basename(__file__)}")
        shutil.copytree("model", f"{save_path / 'model'}")
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0
    )


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.train_gpu)
    from util.common_util import find_free_port

    # import torch.backends.mkldnn
    # ackends.mkldnn.enabled = False
    # os.environ["LRU_CACHE_CAPACITY"] = "1"
    # cudnn.deterministic = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(
            main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args)
        )
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    from util.logger import get_logger
    from util.lr import MultiStepWithWarmup, PolyLR

    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    model = get_model(args)

    # set loss func
    # criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    criterion = {
        "CE": nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(),
        "kl_div": nn.SmoothL1Loss().cuda(),
        "cd": ChamferDistanceLoss().cuda(),
        "BCE": nn.BCEWithLogitsLoss().cuda(),
    }

    # set optimizer
    optimizer = None
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "AdamW":
        transformer_lr_scale = args.get("transformer_lr_scale", 0.1)
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "blocks" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "blocks" in n and p.requires_grad
                ],
                "lr": args.base_lr * transformer_lr_scale,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.base_lr, weight_decay=args.weight_decay
        )
    assert isinstance(optimizer, torch.optim.Optimizer)

    if main_process():
        global logger, writer
        logger = get_logger(args.save_path)
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
        logger.info(
            "#Model parameters: {}".format(
                sum([x.nelement() for x in model.parameters()])
            )
        )
        if args.get("max_grad_norm", None):
            logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))

    if args.distributed:
        torch.cuda.set_device(gpu)
        # args.batch_size = int(args.batch_size / ngpus_per_node)
        # args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], find_unused_parameters=True
        )
    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint["state_dict"])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    scheduler_state_dict = None
    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda()
            )
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler_state_dict = checkpoint["scheduler"]
            best_iou = checkpoint["best_iou"]
            if main_process():
                logger.info(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        args.resume, checkpoint["epoch"]
                    )
                )
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    train_loader, train_sampler = get_train_dataset(args)
    val_loader = get_val_dataset(args)

    # set scheduler
    if args.scheduler == "MultiStepWithWarmup":
        assert args.scheduler_update == "step"
        if main_process():
            logger.info(
                "scheduler: MultiStepWithWarmup. scheduler_update: {}".format(
                    args.scheduler_update
                )
            )
        iter_per_epoch = len(train_loader)
        milestones = [
            int(args.epochs * 0.6) * iter_per_epoch,
            int(args.epochs * 0.8) * iter_per_epoch,
        ]
        scheduler = MultiStepWithWarmup(
            optimizer,
            milestones=milestones,
            gamma=0.1,
            warmup=args.warmup,
            warmup_iters=args.warmup_iters,
            warmup_ratio=args.warmup_ratio,
        )
    elif args.scheduler == "MultiStep":
        assert args.scheduler_update == "epoch"
        milestones = (
            [int(x) for x in args.milestones.split(",")]
            if hasattr(args, "milestones")
            else [int(args.epochs * 0.6), int(args.epochs * 0.8)]
        )
        gamma = args.gamma if hasattr(args, "gamma") else 0.1
        if main_process():
            logger.info(
                "scheduler: MultiStep. scheduler_update: {}. milestones: {}, gamma: {}".format(
                    args.scheduler_update, milestones, gamma
                )
            )
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    elif args.scheduler == "Poly":
        if main_process():
            logger.info(
                "scheduler: Poly. scheduler_update: {}".format(args.scheduler_update)
            )
        if args.scheduler_update == "epoch":
            scheduler = PolyLR(optimizer, max_iter=args.epochs, power=args.power)
        elif args.scheduler_update == "step":
            iter_per_epoch = len(train_loader)
            scheduler = PolyLR(
                optimizer, max_iter=args.epochs * iter_per_epoch, power=args.power
            )
        else:
            raise ValueError(
                "No such scheduler update {}".format(args.scheduler_update)
            )
    else:
        raise ValueError("No such scheduler {}".format(args.scheduler))

    if args.resume and os.path.isfile(args.resume):
        assert isinstance(scheduler_state_dict, dict)
        scheduler.load_state_dict(scheduler_state_dict)
        print("resume scheduler")

    ###################
    # start training #
    ###################

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if main_process():
            logger.info("lr: {}".format(scheduler.get_last_lr()))

        loss_train, mIoU_train, mAcc_train, allAcc_train, others_train = train(
            train_loader, model, criterion, optimizer, epoch, scaler, scheduler
        )
        if args.scheduler_update == "epoch":
            scheduler.step()
        epoch_log = epoch + 1

        if main_process():
            writer.add_scalar("loss_train", loss_train, epoch_log)
            writer.add_scalar("mIoU_train", mIoU_train, epoch_log)
            writer.add_scalar("mAcc_train", mAcc_train, epoch_log)
            writer.add_scalar("allAcc_train", allAcc_train, epoch_log)
            writer.add_scalar("loss2_train", others_train["loss2"], epoch_log)
            writer.add_scalar("loss3_train", others_train["loss3"], epoch_log)
            writer.add_scalar("aupr_batch_mean_train", others_train["aupr"], epoch_log)
            writer.add_scalar(
                "auroc_batch_mean_train", others_train["auroc"], epoch_log
            )
            # writer.add_scalar("aupr_train", others_train["aupr_epoch"], epoch_log)
            # writer.add_scalar("auroc_train", others_train["auroc_epoch"], epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val, val_meters = validate(
                val_loader, model, criterion, epoch
            )

            if main_process():
                writer.add_scalar("loss_val", loss_val, epoch_log)
                writer.add_scalar("mIoU_val", mIoU_val, epoch_log)
                writer.add_scalar("mAcc_val", mAcc_val, epoch_log)
                writer.add_scalar("allAcc_val", allAcc_val, epoch_log)
                writer.add_scalar("loss2_val", val_meters["loss2"], epoch_log)
                writer.add_scalar("loss3_val", val_meters["loss3"], epoch_log)
                writer.add_scalar("aupr_batch_mean_val", val_meters["aupr"], epoch_log)
                writer.add_scalar(
                    "auroc_batch_mean_val", val_meters["auroc"], epoch_log
                )
                # writer.add_scalar("aupr_val", val_meters["aupr_epoch"], epoch_log)
                # writer.add_scalar("auroc_ val", val_meters["auroc_epoch"], epoch_log)
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            if not os.path.exists(args.save_path + "/model/"):
                os.makedirs(args.save_path + "/model/")
            filename = args.save_path + "/model/model_last.pth"
            logger.info("Saving checkpoint to: " + filename)
            torch.save(
                {
                    "epoch": epoch_log,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_iou": best_iou,
                    "is_best": is_best,
                },
                filename,
            )
            if is_best:
                shutil.copyfile(filename, args.save_path + "/model/model_best.pth")
            if epoch_log % args.save_freq2 == 0:
                shutil.copyfile(
                    filename,
                    args.save_path + "/model/model_epoch_{}.pth".format(epoch_log),
                )

    if main_process():
        writer.close()
        logger.info("==>Training done!\nBest Iou: %.3f" % (best_iou))


def train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler):
    import torch.distributed as dist
    import torch_points_kernels as tp

    from util.common_util import AverageMeter, intersectionAndUnionGPU

    criterion, criterion_2, criterion_3, criterion_4 = (
        criterion["CE"],
        criterion["kl_div"],
        criterion["cd"],
        criterion["BCE"],
    )
    milestones = (
        [int(x) for x in args.milestones.split(",")]
        if hasattr(args, "milestones")
        else [int(args.epochs * 0.6), int(args.epochs * 0.8)]
    )
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss2_meter = AverageMeter()
    loss3_meter = AverageMeter()
    loss4_meter = AverageMeter()
    loss5_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_id_meter = AverageMeter()
    # target_list = []
    unknown_logits_list = []
    aupr_meter = AverageMeter()
    auroc_meter = AverageMeter()
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (coord, feat, target, offset) in enumerate(
        train_loader
    ):  # (n, 3), (n, c), (n), (b)
        data_time.update(time.time() - end)

        # coord, feat, target, offset = new_coord.clone(), new_feat.clone(), new_label_ori.clone(), new_offset.clone()
        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat(
            [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0
        ).long()

        sigma = 1.0
        radius = 2.5 * args.grid_size * sigma
        neighbor_idx = tp.ball_query(
            radius,
            args.max_num_neighbors,
            coord,
            coord,
            mode="partial_dense",
            batch_x=batch,
            batch_y=batch,
        )[0]

        coord, feat, target, offset = (
            coord.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            target.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
        )
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]
        # new_label_ori = new_label_ori.cuda(non_blocking=True)

        # unknown_mask = (target == 20).float().cuda()[..., None]
        target_id = target.clone()
        target_id[
            torch.isin(
                target_id, torch.tensor(args.unknown_label, device=target_id.device)
            )
        ] = -100
        assert not torch.isin(
            target_id, torch.tensor(args.unknown_label, device=target_id.device)
        ).any()

        if args.concat_xyz:
            feat = torch.cat([feat, coord], 1)

        use_amp = args.use_amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            output, others = model(feat, coord, offset, batch, neighbor_idx)
            unknown_pred = others["unknown"]
            # unknown_dist = others["unknown_dist"].clone().detach()
            # unknown_logits = unknown_pred
            # unknown_logits_list.append(unknown_logits.clone().detach())
            # last_feats = others["last_feats"]
            # pred_prototypes = others["prototypes"]
            # pred_mean_proto = others["mean_prototypes"]
            # pred_proto_dist = others["proto_dist"]
            assert output.shape[1] == args.classes
            if target.shape[-1] == 1:
                assert True, "unexpected situation"
                # target = target[:, 0]  # for cls
                # target_pseudo = target_pseudo[:, 0]  # for cls
            pseudo_mask = get_pseudo_label_msp(output)
            target_loss = target_id.clone()
            target_loss[pseudo_mask] = 20
            # pseudo_mask = get_pseudo_mask_from_prototypes(
            #     last_feats,
            #     target,
            #     offset,
            #     pred_prototypes,
            #     args.classes,
            #     args.ignore_label,
            # ) # pseudo label
            # (
            #     target_proto,
            #     target_mean_proto,
            #     # target_proto_dist,
            #     unique_mask,
            # ) = get_target_prototypes(
            #     last_feats,
            #     pred_prototypes,
            #     target_id,
            #     offset,
            #     args.classes,
            #     ignore_index=args.ignore_label,
            # )

            output = torch.cat([output, unknown_pred], -1)  # pseudo label
            # output = output.softmax(-1)
            loss = criterion(output, target_loss)
            # loss2 = (
            #     criterion_3(
            #         pred_prototypes[unique_mask],
            #         target_proto[unique_mask],
            #     )
            #     * 1
            # )
            # loss3 = (
            #     criterion_2(
            #         pred_mean_proto[unique_mask],
            #         target_mean_proto[unique_mask],
            #     )
            #     * 100
            # )
            # loss4 = (
            #     criterion_2(
            #         pred_proto_dist[unique_mask],
            #         target_proto_dist[unique_mask],
            #     )
            #     * 100
            # )
            # assert not torch.isnan(loss4)
            # loss5 = criterion_4(unknown_pred, unknown_mask)
            loss2 = torch.tensor(0.0).cuda()
            loss3 = torch.tensor(0.0).cuda()
            loss4 = torch.tensor(0.0).cuda()
            loss5 = torch.tensor(0.0).cuda()
            loss_all = loss + loss2 + loss3 + loss4 + loss5
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss_all).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_all.backward()
            optimizer.step()

        if args.scheduler_update == "step":
            scheduler.step()

        # unknown_pred = output[..., [-1]]
        output = output[:, :20]  # split pseudo label
        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            loss2 *= n
            loss3 *= n
            loss4 *= n
            loss5 *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss)
            dist.all_reduce(count)
            dist.all_reduce(loss2)
            dist.all_reduce(loss3)
            dist.all_reduce(loss4)
            dist.all_reduce(loss5)
            n = count.item()
            loss /= n
            loss2 /= n
            loss3 /= n
            loss4 /= n
            loss5 /= n

        aupr, auroc = get_aupr_auroc(
            unknown_pred,
            target,
            args.unknown_label,
            args.ignore_label,
            logger if main_process() else None,
            distributed=args.distributed,
            main_process=main_process(),
        )

        if aupr > 0 and auroc > 0:
            aupr_meter.update(aupr)
            auroc_meter.update(auroc)
        elif main_process():
            logger.warning("This batch does not contain any OOD pixels or is only OOD.")

        intersection, union, target_id = intersectionAndUnionGPU(
            output, target_id, args.classes, args.ignore_label
        )
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection)
            dist.all_reduce(union)
            dist.all_reduce(target_id)
        intersection, union, target_id = (
            intersection.cpu().numpy(),
            union.cpu().numpy(),
            target_id.cpu().numpy(),
        )
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_id_meter.update(target_id)

        accuracy = sum(intersection_meter.val) / (sum(target_id_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        loss2_meter.update(loss2.item(), n)
        loss3_meter.update(loss3.item(), n)
        loss4_meter.update(loss4.item(), n)
        loss5_meter.update(loss5.item(), n)
        batch_time.update(time.time() - end)

        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            lr = scheduler.get_last_lr()
            if isinstance(lr, list):
                lr = [round(x, 8) for x in lr]
            elif isinstance(lr, float):
                lr = round(lr, 8)
            logger.info(
                "Epoch: [{}/{}][{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Remain {remain_time} "
                "Loss {loss_meter.val:.4f} "
                "Loss2 {loss2_meter.val:.4f} "
                "Loss3 {loss3_meter.val:.4f} "
                "Loss4 {loss4_meter.val:.4f} "
                "Loss5 {loss5_meter.val:.4f} "
                "Lr: {lr} "
                "Accuracy {accuracy:.4f}. "
                "aupr {aupr:.4f} ({aupr_meter.avg:.4f}) "
                "auroc {auroc:.4f} ({auroc_meter.avg:.4f})".format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    remain_time=remain_time,
                    loss_meter=loss_meter,
                    loss2_meter=loss2_meter,
                    loss3_meter=loss3_meter,
                    loss4_meter=loss4_meter,
                    loss5_meter=loss5_meter,
                    lr=lr,
                    accuracy=accuracy,
                    aupr=aupr,
                    aupr_meter=aupr_meter,
                    auroc=auroc,
                    auroc_meter=auroc_meter,
                )
            )
        if main_process():
            writer.add_scalar("loss_train_batch", loss_meter.val, current_iter)
            writer.add_scalar("loss2_train_batch", loss2_meter.val, current_iter)
            writer.add_scalar("loss3_train_batch", loss3_meter.val, current_iter)
            # writer.add_scalar("loss4_train_batch", loss4_meter.val, current_iter)
            writer.add_scalar(
                "mIoU_train_batch",
                np.mean(intersection / (union + 1e-10)),
                current_iter,
            )
            writer.add_scalar(
                "mAcc_train_batch",
                np.mean(intersection / (target_id + 1e-10)),
                current_iter,
            )
            writer.add_scalar("allAcc_train_batch", accuracy, current_iter)

            # writer.add_histogram(
            #     "unknown_label_train_batch", unknown_logits.float(), current_iter
            # )
            # writer.add_histogram("unknown_dist_train_batch", unknown_dist, current_iter)
            # if aupr > 0 and auroc > 0:
            #     writer.add_scalar("aupr_train_batch", aupr, current_iter)
            #     writer.add_scalar("auroc_train_batch", auroc, current_iter)
        # visual_prototypes(
        #     epoch,
        #     i,
        #     batch,
        #     last_feats,
        #     target_ori,
        #     pred_prototypes,
        #     "train",
        #     f"{args.save_path}/prototypes_tsne",
        # )
        # visual_mean_prototypes()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_id_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_id_meter.sum) + 1e-10)

    # unknown_logits_all = torch.cat(unknown_logits_list, 0)
    # target_all = torch.cat(target_ori_list, 0)
    # aupr_epoch, auroc_epoch = get_aupr_auroc(
    #     unknown_logits_all,
    #     target_all,
    #     args.unknown_label,
    #     args.ignore_label,
    #     logger if main_process() else None,
    #     distributed=args.distributed,
    #     main_process=main_process(),
    # )

    if main_process():
        logger.info(
            "Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                epoch + 1, args.epochs, mIoU, mAcc, allAcc
            )
        )
        logger.info(
            f"Train AUPR/AUROC batch mean: {aupr_meter.avg:.4f}/{auroc_meter.avg:.4f}"
        )
        # logger.info(f"Train AUPR/AUROC epoch: {aupr_epoch:.4f}/{auroc_epoch:.4f}")

    return (
        loss_meter.avg,
        mIoU,
        mAcc,
        allAcc,
        {
            "loss2": loss2_meter.avg,
            "loss3": loss3_meter.avg,
            "aupr": aupr_meter.avg,
            "auroc": auroc_meter.avg,
            # "aupr_epoch": aupr_epoch,
            # "auroc_epoch": auroc_epoch,
        },
    )


def validate(val_loader, model, criterion, epoch):
    import torch.distributed as dist
    import torch_points_kernels as tp

    from util.common_util import AverageMeter, intersectionAndUnionGPU

    criterion, criterion_2, criterion_3 = (
        criterion["CE"],
        criterion["kl_div"],
        criterion["cd"],
    )

    milestones = (
        [int(x) for x in args.milestones.split(",")]
        if hasattr(args, "milestones")
        else [int(args.epochs * 0.6), int(args.epochs * 0.8)]
    )
    if main_process():
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss2_meter = AverageMeter()
    loss3_meter = AverageMeter()
    loss4_meter = AverageMeter()
    loss5_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    target_ori_list = []
    unknown_logits_list = []
    aupr_meter = AverageMeter()
    auroc_meter = AverageMeter()
    torch.cuda.empty_cache()

    model.eval()
    end = time.time()
    for i, (coord, feat, target, offset) in enumerate(val_loader):
        data_time.update(time.time() - end)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat(
            [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0
        ).long()

        sigma = 1.0
        radius = 2.5 * args.grid_size * sigma
        neighbor_idx = tp.ball_query(
            radius,
            args.max_num_neighbors,
            coord,
            coord,
            mode="partial_dense",
            batch_x=batch,
            batch_y=batch,
        )[0]

        coord, feat, target, offset = (
            coord.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            target.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
        )
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]

        target_id = target.clone().detach()
        # target_ori_list.append(target_id)
        target_id[
            torch.isin(
                target, torch.tensor(args.unknown_label, device=target_id.device)
            )
        ] = args.ignore_label
        assert not torch.isin(
            target_id, torch.tensor(args.unknown_label, device=target_id.device)
        ).any()

        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls

        if args.concat_xyz:
            feat = torch.cat([feat, coord], 1)

        with torch.no_grad():
            output, others = model(feat, coord, offset, batch, neighbor_idx)
            unknown_pred = others["unknown"]
            # unknown_dist = others["unknown_dist"]
            # unknown_logits_list.append(unknown_logits.clone().detach())
            # last_feats = others["last_feats"]
            # pred_prototypes = others["prototypes"]
            # pred_mean_proto = others["mean_prototypes"]
            # pred_proto_dist = others["proto_dist"]
            # pseudo_mask = get_pseudo_mask(
            #     output, target, args.classes, epoch, milestones, args.ignore_label
            # ) # pseudo label
            # (
            #     target_proto,
            #     target_mean_proto,
            #     # target_proto_dist,
            #     unique_mask,
            # ) = get_target_prototypes(
            #     last_feats,
            #     pred_prototypes,
            #     target_id,
            #     offset,
            #     args.classes,
            #     ignore_index=args.ignore_label,
            # )
            # output = torch.cat([output, unknown_pred], -1) # pseudo label
            # target_pseudo[pseudo_mask] = args.classes # pseudo label
            # output = output.softmax(-1)
            loss = criterion(output, target)
            # loss2 = (
            #     criterion_3(
            #         pred_prototypes[unique_mask],
            #         target_proto[unique_mask],
            #     )
            #     * 1
            # )
            # loss3 = (
            #     criterion_2(
            #         pred_mean_proto[unique_mask],
            #         target_mean_proto[unique_mask],
            #     )
            #     * 100
            # )
            # loss4 = (
            #     criterion_2(
            #         pred_proto_dist[unique_mask],
            #         target_proto_dist[unique_mask],
            #     )
            #     * 100
            # )
            loss2 = torch.tensor(0.0).cuda()
            loss3 = torch.tensor(0.0).cuda()
            loss4 = torch.tensor(0.0).cuda()

        # unknown_logits = output[:, -1]
        output = output[:, :20]  # pseudo label
        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            loss2 *= n
            loss3 *= n
            loss4 *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss)
            dist.all_reduce(count)
            dist.all_reduce(loss2)
            dist.all_reduce(loss3)
            dist.all_reduce(loss4)
            n = count.item()
            loss /= n
            loss2 /= n
            loss3 /= n
            loss4 /= n

        aupr, auroc = get_aupr_auroc(
            unknown_pred,
            target,
            args.unknown_label,
            args.ignore_label,
            logger if main_process() else None,
            distributed=args.distributed,
            main_process=main_process(),
        )
        if aupr > 0 and auroc > 0:
            aupr_meter.update(aupr)
            auroc_meter.update(auroc)
        elif main_process():
            logger.warning("This batch does not contain any OOD pixels or is only OOD.")

        intersection, union, target = intersectionAndUnionGPU(
            output, target, args.classes, args.ignore_label
        )
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection)
            dist.all_reduce(union)
            dist.all_reduce(target)
        intersection, union, target = (
            intersection.cpu().numpy(),
            union.cpu().numpy(),
            target.cpu().numpy(),
        )
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        loss2_meter.update(loss2.item(), n)
        loss3_meter.update(loss3.item(), n)
        loss4_meter.update(loss4.item(), n)
        batch_time.update(time.time() - end)

        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info(
                "Test: [{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                "Loss2 {loss2_meter.val:.4f} ({loss2_meter.avg:.4f}) "
                "Loss3 {loss3_meter.val:.4f} ({loss2_meter.avg:.4f}) "
                "Loss4 {loss4_meter.val:.4f} ({loss4_meter.avg:.4f}) "
                "Accuracy {accuracy:.4f}. "
                "aupr {aupr:.4f} ({aupr_meter.avg:.4f}) "
                "auroc {auroc:.4f} ({auroc_meter.avg:.4f})".format(
                    i + 1,
                    len(val_loader),
                    data_time=data_time,
                    batch_time=batch_time,
                    loss_meter=loss_meter,
                    loss2_meter=loss2_meter,
                    loss3_meter=loss3_meter,
                    loss4_meter=loss4_meter,
                    accuracy=accuracy,
                    aupr=aupr,
                    aupr_meter=aupr_meter,
                    auroc=auroc,
                    auroc_meter=auroc_meter,
                )
            )
        current_iter = epoch * len(val_loader) + i + 1
        # if main_process():
        #     writer.add_histogram(
        #         "unknown_label_val_batch", unknown_logits.float(), current_iter
        #     )
        #     writer.add_histogram("unknown_dist_val_batch", unknown_dist, current_iter)
        # visual_prototypes(
        #     epoch,
        #     i,
        #     batch,
        #     last_feats,
        #     target_ori,
        #     pred_prototypes,
        #     "validate",
        #     f"{args.save_path}/prototypes_tsne",
        # )
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # unknown_logits_all = torch.cat(unknown_logits_list, 0)
    # target_all = torch.cat(target_ori_list, 0)
    # aupr_epoch, auroc_epoch = get_aupr_auroc(
    #     unknown_logits_all,
    #     target_all,
    #     args.unknown_label,
    #     args.ignore_label,
    #     logger if main_process() else None,
    #     distributed=args.distributed,
    #     main_process=main_process(),
    # )

    if main_process():
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )
        for i in range(args.classes):
            logger.info(
                "Class_{} Result: iou/accuracy {:.4f}/{:.4f}.".format(
                    i, iou_class[i], accuracy_class[i]
                )
            )

        logger.info(
            f"Val AUPR/AUROC batch mean: {aupr_meter.avg:.4f}/{auroc_meter.avg:.4f}"
        )
        # logger.info(f"Val AUPR/AUROC epoch: {aupr_epoch:.4f}/{auroc_epoch:.4f}")

        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    return (
        loss_meter.avg,
        mIoU,
        mAcc,
        allAcc,
        {
            "loss2": loss2_meter.avg,
            "loss3": loss3_meter.avg,
            "aupr": aupr_meter.avg,
            "auroc": auroc_meter.avg,
            # "aupr_epoch": aupr_epoch,
            # "auroc_epoch": auroc_epoch,
        },
    )


def get_model(args):
    # get model
    if args.arch == "stratified_transformer":
        from model.stratified_transformer import Stratified

        args.patch_size = args.grid_size * args.patch_size
        args.window_size = [
            args.patch_size * args.window_size * (2**i)
            for i in range(args.num_layers)
        ]
        args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

        model = Stratified(
            args.downsample_scale,
            args.depths,
            args.channels,
            args.num_heads,
            args.window_size,
            args.up_k,
            args.grid_sizes,
            args.quant_sizes,
            rel_query=args.rel_query,
            rel_key=args.rel_key,
            rel_value=args.rel_value,
            drop_path_rate=args.drop_path_rate,
            concat_xyz=args.concat_xyz,
            num_classes=args.classes,
            ratio=args.ratio,
            k=args.k,
            prev_grid_size=args.grid_size,
            sigma=1.0,
            num_layers=args.num_layers,
            stem_transformer=args.stem_transformer,
        )
    elif (
        args.arch == "stratified_transformer_v1b1_gridfeat"
        or args.arch == "stratified_transformer_opwd"
        or args.arch == "stratified_transformer_v1_proj"
        or args.arch == "stratified_transformer_v2_learnproto"
        or args.arch == "stratified_transformer_v2b1_improve"
        or args.arch == "stratified_transformer_v2b2_meanproto"
        or args.arch == "stratified_transformer_v2b3_lastfeats"
        or args.arch == "stratified_transformer_v2b4_dist"
        or args.arch == "stratified_transformer_v2b5_unknown"
    ):
        import importlib

        StratifiedOpwd = importlib.import_module("model." + args.arch).StratifiedOpwd

        args.patch_size = args.grid_size * args.patch_size
        args.window_size = [
            args.patch_size * args.window_size * (2**i)
            for i in range(args.num_layers)
        ]
        args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

        model = StratifiedOpwd(
            args.downsample_scale,
            args.depths,
            args.channels,
            args.num_heads,
            args.window_size,
            args.up_k,
            args.grid_sizes,
            args.quant_sizes,
            rel_query=args.rel_query,
            rel_key=args.rel_key,
            rel_value=args.rel_value,
            drop_path_rate=args.drop_path_rate,
            concat_xyz=args.concat_xyz,
            num_classes=args.classes,
            ratio=args.ratio,
            k=args.k,
            prev_grid_size=args.grid_size,
            sigma=1.0,
            num_layers=args.num_layers,
            stem_transformer=args.stem_transformer,
        )
    elif args.arch == "swin3d_transformer":
        from model.swin3d_transformer import Swin

        args.patch_size = args.grid_size * args.patch_size
        args.window_sizes = [
            args.patch_size * args.window_size * (2**i)
            for i in range(args.num_layers)
        ]
        args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

        model = Swin(
            args.depths,
            args.channels,
            args.num_heads,
            args.window_sizes,
            args.up_k,
            args.grid_sizes,
            args.quant_sizes,
            rel_query=args.rel_query,
            rel_key=args.rel_key,
            rel_value=args.rel_value,
            drop_path_rate=args.drop_path_rate,
            concat_xyz=args.concat_xyz,
            num_classes=args.classes,
            ratio=args.ratio,
            k=args.k,
            prev_grid_size=args.grid_size,
            sigma=1.0,
            num_layers=args.num_layers,
            stem_transformer=args.stem_transformer,
        )

    else:
        raise Exception("architecture {} not supported yet".format(args.arch))

    return model


def get_train_dataset(args):
    from functools import partial

    from util import transform
    from util.data_util import collate_fn_limit
    from util.s3dis import S3DIS
    from util.scannet_v2_ori import Scannetv2

    if args.data_name == "s3dis":
        train_transform = None
        if args.aug:
            jitter_sigma = args.get("jitter_sigma", 0.01)
            jitter_clip = args.get("jitter_clip", 0.05)
            if main_process():
                logger.info("augmentation all")
                logger.info(
                    "jitter_sigma: {}, jitter_clip: {}".format(
                        jitter_sigma, jitter_clip
                    )
                )
            train_transform = transform.Compose(
                [
                    transform.RandomRotate(along_z=args.get("rotate_along_z", True)),
                    transform.RandomScale(
                        scale_low=args.get("scale_low", 0.8),
                        scale_high=args.get("scale_high", 1.2),
                    ),
                    transform.RandomJitter(sigma=jitter_sigma, clip=jitter_clip),
                    transform.RandomDropColor(
                        color_augment=args.get("color_augment", 0.0)
                    ),
                ]
            )
        train_data = S3DIS(
            split="train",
            data_root=args.data_root,
            test_area=args.test_area,
            voxel_size=args.voxel_size,
            voxel_max=args.voxel_max,
            transform=train_transform,
            shuffle_index=True,
            loop=args.loop,
        )
    elif args.data_name == "scannetv2":
        train_transform = None
        if args.aug:
            if main_process():
                logger.info("use Augmentation")
            train_transform = transform.Compose(
                [
                    transform.RandomRotate(along_z=args.get("rotate_along_z", True)),
                    transform.RandomScale(
                        scale_low=args.get("scale_low", 0.8),
                        scale_high=args.get("scale_high", 1.2),
                    ),
                    transform.RandomDropColor(
                        color_augment=args.get("color_augment", 0.0)
                    ),
                ]
            )

        train_split = args.get("train_split", "train")
        if main_process():
            logger.info("scannet. train_split: {}".format(train_split))

        train_data = Scannetv2(
            split=train_split,
            data_root=args.data_root,
            voxel_size=args.voxel_size,
            voxel_max=args.voxel_max,
            transform=train_transform,
            shuffle_index=True,
            loop=args.loop,
        )
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if main_process():
        logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=partial(
            collate_fn_limit,
            max_batch_points=args.max_batch_points,
            logger=logger if main_process() else None,
        ),
    )

    return train_loader, train_sampler


def get_val_dataset(args):
    from util.data_util import collate_fn
    from util.s3dis import S3DIS
    from util.scannet_v2 import Scannetv2

    val_transform = None
    if args.data_name == "s3dis":
        val_data = S3DIS(
            split="val",
            data_root=args.data_root,
            test_area=args.test_area,
            voxel_size=args.voxel_size,
            voxel_max=800000,
            transform=val_transform,
            loop=args.val_loop,
        )
    elif args.data_name == "scannetv2":
        val_data = Scannetv2(
            split="val",
            data_root=args.data_root,
            voxel_size=args.voxel_size,
            voxel_max=800000,
            transform=val_transform,
            # loop=args.val_loop,
        )
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=collate_fn,
    )
    return val_loader


if __name__ == "__main__":
    import gc

    gc.collect()
    main()
>>>>>>> e2bd8910591550a2151466bb4320b8c82300dc4e
