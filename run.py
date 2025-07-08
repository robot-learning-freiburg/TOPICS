import utils
import argparser
import builtins
import wandb
import numpy as np
import random
import torch
import geoopt
import os

from functools import reduce
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.logger import Logger
from models import make_model, init_model_step
from train import Trainer
from dataset import get_dataset
from utils import StreamSegMetrics, save_ckpt
from utils.visualization import PredictionVis


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"


def main(opts):
    if opts.debug or torch.cuda.device_count() == 1:
        world_size = 1
        rank = 0
        opts.local_rank = 0
        print("One GPU")
    else:

        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        opts.local_rank = os.environ['LOCAL_RANK']
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        if opts.local_rank != 0:
            def print_pass(*args, **kwargs):
                pass

            builtins.print = print_pass

    device_id, device = int(opts.local_rank), torch.device(int(opts.local_rank))
    torch.cuda.set_device(device_id)

    opts.batch_size = int(opts.batch_size / world_size)
    opts.num_workers = int((opts.num_workers + world_size - 1) / world_size)
    opts.ngpus_per_node = world_size

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Initialize logging
    task_name = f"{opts.dataset}_{opts.task.split('/')[-1]}"
    if opts.overlap:
        task_name += "-ov"
    opts.logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/step{opts.step}"

    if rank == 0:
        if not os.path.exists(opts.logdir_full):
            os.makedirs(opts.logdir_full, exist_ok=True)
        logger = Logger(opts.logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step)
        if opts.wandb:
            if opts.wb_version == 0.0:
                project_name = f"HCIL-{opts.dataset}"
            else:
                project_name = f"HCIL-{opts.dataset}-{opts.wb_version}"
            wandb.init(project=project_name, name=opts.name + "_step" + str(opts.step),
                       config=opts, sync_tensorboard=True, settings=wandb.Settings(start_method="fork"))
    else:
        logger = Logger(opts.logdir_full, rank=rank, debug=opts.debug, summary=False, step=opts.step)

    logger.print(f"Device: {device}")

    # xxx Set up dataloader
    train_loader, val_loader, tst_loader, task = get_dataset(opts, logger, world_size, rank)

    # xxx Set up model
    logger.info(f"Backbone: {opts.backbone}")
    classes = task.classes(opts.step)
    if opts.freeze_last or opts.wbce:
        classes = [[0]] + [task.classes(opts.step)[0][1:]] + task.classes(opts.step)[1:]
    model, checkpoint, _ = make_model(opts, classes=classes, model_new=True, logger=logger) # adj_dict=None,

    model_old, step_checkpoint = None, None
    if opts.step > 0:  # if step 0, we don't need to instance the model_old
        opts.path_old = f"{opts.logdir}/{task_name}/{opts.name}/step{opts.step - 1}/model_final.pth"
        if opts.save_best:
            opts.path_old = f"{opts.logdir}/{task_name}/{opts.name}/step{opts.step - 1}/model_best.pth"
        if opts.step_prev_ckpt is None:
            opts.step_prev_ckpt = opts.path_old
        if opts.method != 'FT' or (opts.pseudo is not None) or opts.distance_sim_weight or (opts.dkd_kd != 0):
            old_classes = task.classes(opts.step - 1)
            if opts.freeze_last or opts.wbce:
                old_classes = [[0]] + [task.classes(opts.step - 1)[0][1:]] + task.classes(opts.step - 1)[1:]
                logger.info(old_classes)
            model_old, step_checkpoint, state_dict_filt = make_model(opts, classes=old_classes,
                                                                     model_new=False, logger=logger)
            # freeze old model
            model_old.eval()
        else:
            # load weights of old model and use as init for new model
            step_checkpoint = torch.load(opts.step_prev_ckpt, map_location=torch.device('cpu'))
            state_dict_filt = {k.replace('module.', ''): v for k, v in step_checkpoint['model_state'].items() if
                               ('module.' in k) and (not 'fc' in k)}
            del step_checkpoint['model_state']
        if opts.resume or opts.ckpt or opts.test:
            logger.info(f"[!] New model already initialized from prior checkpoint.")
        else:
            model = init_model_step(opts, model, state_dict_filt)
            logger.info(f"[!] New model initialized from {opts.step_prev_ckpt}.")

    logger.debug(model)

    # move old model to gpu
    if model_old:
        model_old = model_old.cuda(device)
    model = model.cuda(device)

    if not opts.debug and world_size > 1:
        # Convert model to DDP
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print("Convert model to DDP!")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id],
                                                          output_device=device_id, find_unused_parameters=(
                        opts.method == "MICROSEG" and opts.step > 0))

    # xxx Set up optimizer, scaler and scheduler
    cur_epoch = 0
    if opts.poincare:
        optimizer = geoopt.optim.RiemannianSGD(model.parameters(), lr=opts.lr, momentum=0.9, nesterov=True,
                                                   weight_decay=opts.weight_decay)
        scaler = None
    else:
        if opts.diff_lr:
            mod = model.module if world_size > 1 else model
            if opts.method == "DKD":
                if opts.step > 0:
                    params = [{"params": mod.get_backbone_params(), "weight_decay": 0},
                              {"params": mod.get_head_params(), "lr": opts.lr * 10, "weight_decay": 0},
                              {"params": mod.get_old_classifer_params(), "lr": opts.lr * 10, "weight_decay": 0},
                              {"params": mod.get_new_classifer_params(), "lr": opts.lr * 10}]
                else:
                    params = [{"params": mod.get_backbone_params()},
                              {"params": mod.get_head_params(), "lr": opts.lr * 10},
                              {"params": mod.get_classifer_params(), "lr": opts.lr * 10}]
            elif opts.method == "MICROSEG":
                if opts.step > 0 and opts.freeze:
                    for param in mod.parameters():
                        param.requires_grad = False

                    for i in [0, 1, -1]:
                        for lvl in [mod.cls[i].parameters(), mod.head2[i].parameters(),
                                    mod.proposal_head[i].parameters()]:
                            for param in lvl:
                                param.requires_grad = True
                    params = [{"params": mod.get_bkg_params(), "lr": opts.lr * 1e-4},
                              {"params": mod.get_unknown_params(), "lr": opts.lr},
                              {"params": mod.get_old_proposal_params(), "lr": opts.lr},
                              {"params": mod.get_new_classifer_params(), "lr": opts.lr}]
                else:
                    params = [{'params': mod.get_backbone_params(), 'lr': opts.lr * 0.1},
                              {"params": mod.get_head_params(), "lr": opts.lr},
                              {'params': mod.get_classifer_params(), 'lr': opts.lr}]
        else:
            params = model.parameters()

        optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True,
                                    weight_decay=opts.weight_decay)
        
        if opts.autocast:
            scaler = torch.cuda.amp.GradScaler()
            logger.info("Running with autocast!")
        else:
            scaler = None

    logger.debug("Trainer len:\n%s" % len(train_loader))

    if opts.lr_policy == 'warmpoly':
        warmup_iters = 0.1 if opts.method == "MICROSEG" else 0.2
        scheduler = utils.WarmupPolyLR(optimizer, max_iters=opts.epochs * len(train_loader),
                                       power=opts.lr_power, warmup_iters=warmup_iters)
    elif opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, max_iters=opts.epochs * len(train_loader), power=opts.lr_power)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[opts.epochs * len(train_loader) * 0.6,
                                                                                opts.epochs * len(train_loader) * 0.8],
                                                         gamma=opts.lr_decay_factor)
    elif opts.lr_policy == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    else:
        raise NotImplementedError

    logger.debug("Optimizer:\n%s" % optimizer)

    if checkpoint:  # load training settings if available
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if ("grad_scaler" in checkpoint) and opts.autocast:
            scaler.load_state_dict(checkpoint["grad_scaler"])
        cur_epoch = checkpoint["epoch"] + 1
        logger.info(f"[!] Initialized optimizer, scheduler and grad scaler from {opts.ckpt}.")

    del checkpoint, step_checkpoint

    # instance trainer (model must have already the previous step weights)
    adj_dict = None
    hier_matrices = None
    if opts.hyp_hier:
        adj_dict = task.entail_matrix(opts.step, level=-1)
        hier_matrices = task.sibling_ancestor_matrix(opts.step)
        if opts.pseudo is not None and opts.step > 0:
            hier_matrices_old = task.sibling_ancestor_matrix(opts.step - 1)
            hier_matrices += hier_matrices_old
    trainer = Trainer(model_old=model_old, device=device, opts=opts, classes=task.classes(opts.step),
                      adj_dict=adj_dict,
                      distributed=(world_size > 1), masking_value=task.masking_value,
                      internal_masking_value=task.internal_masking_value,
                      hier_matrices=hier_matrices)

    # xxx Train procedure
    # print opts before starting training to log all parameters
    logger.add_table("Opts", vars(opts))

    TRAIN = not opts.test
    tot_classes = len(reduce(lambda a, b: a + b, task.classes(opts.step)))
    val_metrics = StreamSegMetrics(tot_classes, task, opts.logdir_full, opts.step, opts.no_bg_class)
    results = {}
    autocast = False if scaler is None else True
    best_score, current_score = -1, -1

    if trainer.pseudo_flag:
        trainer.pseudo_before(train_loader, logger, model_old, opts)

    if opts.att > 0 or opts.compute_att:
        result = trainer.awt_before(train_loader, logger, model_old, opts)
        if result is not None:
            if world_size > 1:
                print(model.module.cls[-1])
            mod = model.module if world_size > 1 else model
            mod.init_classifier_awt(result)
            if world_size > 1:
                print(model.module.cls[-1])

    # check if random is equal here.
    logger.print(torch.randint(0, 100, (1, 1)))
    # train/val here)
    while (cur_epoch < opts.epochs) and TRAIN:
        # =====  Train  =====

        scheduler_train = scheduler if opts.lr_policy != 'plateau' else None

        epoch_loss = trainer.train(model, model_old, cur_epoch=cur_epoch, optim=optimizer, scaler=scaler,
                                   train_loader=train_loader, scheduler=scheduler_train, logger=logger)

        logger.info(f"End of Epoch {cur_epoch}/{opts.epochs}, Average Loss={epoch_loss[0] + epoch_loss[1]},"
                    f" Class Loss={epoch_loss[0]}, Reg Loss={epoch_loss[1]}")

        # =====  Log metrics on Tensorboard =====
        logger.add_scalar("E-Loss", epoch_loss[0] + epoch_loss[1], cur_epoch)
        logger.add_scalar("E-Loss-reg", epoch_loss[1], cur_epoch)
        logger.add_scalar("E-Loss-cls", epoch_loss[0], cur_epoch)
        if opts.lr_policy == 'plateau' and not opts.plateau_base:
            scheduler.step(epoch_loss[0], epoch=cur_epoch)
        if opts.wandb and rank == 0:
            wandb.log({"E-Loss": epoch_loss[0] + epoch_loss[1], "E-Loss-reg": epoch_loss[1],
                       "E-Loss-cls": epoch_loss[0]})

        # =====  Save Model  =====
        if rank == 0 and cur_epoch % opts.ckpt_interval == 0:  # save best model at the last iteration
            # best model to build incremental steps
            save_ckpt(f"{opts.logdir_full}/model_tmp.pth",
                      model, scaler, optimizer, scheduler, cur_epoch)
            logger.info("[!] Checkpoint saved.")

        # =====  Validation  =====
        if ((cur_epoch + 1) % opts.val_interval == 0) or (cur_epoch + 1 == opts.epochs):
            logger.info("validate on val set...")
            model.eval()
            val_loss, val_score = trainer.validate(model, model_old, loader=val_loader,
                                                                metrics=val_metrics,
                                                                ret_samples_ids=None, logger=logger,
                                                                autocast=autocast)
            
            logger.info(f"End of Validation {cur_epoch}/{opts.epochs}, Validation Loss={val_loss[0] + val_loss[1]},"
                        f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]}")

            if opts.lr_policy == 'plateau' and opts.plateau_base:
                scheduler.step(val_score['Base IoU'], epoch=cur_epoch)
            current_lr = scheduler.optimizer.param_groups[0]['lr']

            logger.info(
                val_metrics.to_str(val_score))
            if world_size > 1:
                torch.distributed.barrier()
            # =====  Log metrics on Tensorboard =====
            # visualize validation score and samples
            logger.add_scalar("V-Loss", val_loss[0] + val_loss[1], cur_epoch)
            logger.add_scalar("V-Loss-reg", val_loss[1], cur_epoch)
            logger.add_scalar("V-Loss-cls", val_loss[0], cur_epoch)
            logger.add_scalar("Val_Overall_Acc", val_score['Overall Acc'], cur_epoch)
            logger.add_scalar("Val_MeanIoU", val_score['Mean IoU'], cur_epoch)
            logger.add_table("Val_Class_IoU", val_score['Class IoU'], cur_epoch, cls_transform=True)
            logger.add_table("Val_Acc_IoU", val_score['Class Acc'], cur_epoch, cls_transform=True)
            if opts.wandb and rank == 0:
                wandb.log({"V-Loss": val_loss[0] + epoch_loss[1], "V-Loss-reg": val_loss[1], "V-Loss-cls": val_loss[0],
                           "Val_Overall_Acc": val_score['Overall Acc'], "Val_MeanIoU": val_score['Mean IoU'],
                           "Val_MeanAcc": val_score['Mean Acc'], "Val_BaseIoU": val_score['Base IoU'],
                           "Val_NovelIoU": val_score['Novel IoU'], "Val_OldIoU": val_score['Old IoU'],
                           "Val_NewIoU": val_score['New IoU'], "lr": current_lr})

            if rank == 0 and opts.save_best:
                current_score = (sum([val_score['Class IoU'][x] for x in classes[-1]])) / len(classes[-1])
                if current_score > best_score:
                    best_score = current_score
                    save_ckpt(f"{opts.logdir_full}/model_best.pth", model, scaler, optimizer, scheduler, 
                              cur_epoch)
                    logger.info(f"Saving new best model!")

        cur_epoch += 1

    # =====  Save Best Model at the end of training =====
    if rank == 0 and TRAIN and (not opts.save_best):  # save best model at the last iteration
        # best model to build incremental steps
        save_ckpt(f"{opts.logdir_full}/model_final.pth",
                  model, scaler, optimizer, scheduler, cur_epoch)
        logger.info("[!] Final checkpoint saved.")

    if opts.method == "AWT" and opts.compute_att:
        return
    # testing
    if opts.save_best and (current_score != best_score):
        logger.info(f"Loading best model!")
        checkpoint = torch.load(f"{opts.logdir_full}/model_best.pth", map_location=torch.device('cpu'))
        if 'module.' in next(iter(checkpoint['model_state'])):
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state'].items() if
                               'module.' in k}
        else:
            state_dict = checkpoint['model_state']
        mod = model.module if isinstance(model, (torch.nn.DataParallel, DDP)) else model
        mod.load_state_dict(state_dict, strict=True)
    model.eval()

    if (rank == 0) and (opts.sample_num > 0):
        sample_ids = list(range(opts.sample_num))
        logger.info(f"The samples id are {sample_ids}. Used during testing only.")
        trainer.prediction_vis = PredictionVis(logger, dataset=opts.dataset, task=task, opts=opts)
    else:
        sample_ids = None
        print("No sample ids!")

    if opts.method =="MICROSEG":
        model_old = None
        torch.cuda.empty_cache()

    window_stitch_size = opts.crop_size if opts.window_stitching else None

    if opts.save_pred is not None and ("demoVideo" in opts.save_pred):
        trainer.test(model, loader=tst_loader)

    else:
        _, val_score = trainer.validate(model, model_old, loader=tst_loader,
                                                     metrics=val_metrics, autocast=autocast,
                                                     ret_samples_ids=sample_ids, logger=logger,
                                                     test=True, window_stitching=window_stitch_size)
        if world_size > 1:
            torch.distributed.barrier()

        logger.info("*** Final test...")
        logger.info(
            val_metrics.to_str(val_score))
        logger.add_table("Test_Class_IoU", val_score['Class IoU'], cls_transform=True)
        logger.add_table("Test_Class_Acc", val_score['Class Acc'], cls_transform=True)
        # logger.add_figure("Test_Confusion_Matrix", val_score['Confusion Matrix'])
        results["T-IoU"] = val_score['Class IoU']
        results["T-Acc"] = val_score['Class Acc']
        logger.add_results(results)

        logger.add_scalar("T_Overall_Acc", val_score['Overall Acc'], opts.step)
        logger.add_scalar("T_MeanIoU", val_score['Mean IoU'], opts.step)
        logger.add_scalar("T_MeanAcc", val_score['Mean Acc'], opts.step)
        logger.add_scalar("T_BaseIoU", val_score['Base IoU'], opts.step)
        logger.add_scalar("T_NovelIoU", val_score['Novel IoU'], opts.step)
        if opts.wandb and rank == 0:
            # img = wandb.Image(val_score['Confusion Matrix'], caption="Confusion matrix")
            wandb.log(
                {"MeanIoU": val_score['Mean IoU'], "MeanAcc": val_score['Mean Acc'], "Base IoU": val_score['Base IoU'],
                 "Novel IoU": val_score['Novel IoU'], "Old IoU": val_score['Old IoU'],
                 "New IoU": val_score['New IoU'], "time": val_score['time']})  # , "conf_m": img})

    logger.close()


if __name__ == '__main__':
    parser = argparser.get_argparser()
    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)
    print(opts.dataset)

    try:
        main(opts)
    except Exception as e:
        import traceback
        print("Caught exception:", e)
        traceback.print_exc()
        raise
