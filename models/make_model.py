import torch
import os
import torch.nn as nn
import torch.nn.functional as functional
from functools import reduce, partial
from inplace_abn import InPlaceABNSync, InPlaceABN
import torch.distributed as distributed

from .sem_head import DeeplabV3
from .backbones import ResNet
from .sem_head import ASPP
from .incremental_seg import IncrementalSegmentationModule
from .hyperbolic_classifier import IncrementalHyperbolicSegmentationModule



def make_model(opts, classes=None, model_new=True, logger=None): # adj_dict=None, 
    
    plop = ((opts.pod is not None) and (opts.step > 0))
    dkd = opts.method == "DKD"
    microseg = opts.method == "MICROSEG"

    if 'resnet101' in opts.backbone:
        logger.info("[!] Inplace-ABN activated.")
        norm_act = "iabn_sync"
        if opts.ngpus_per_node > 1:
            norm = partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01,
                               group=distributed.group.WORLD)
        else:
            norm = partial(InPlaceABN, activation="leaky_relu", activation_param=.01)

        body = ResNet(norm_act=norm, plop=plop)
        if opts.pretrained:
            pretrained_path = f'pretrained/{opts.backbone}.pth.tar'
            pre_dict = torch.load(pretrained_path, map_location='cpu')
            state_dict_filt = {k.replace('module.', ''): v for k, v in pre_dict['state_dict'].items() if
                               ('module.' in k) and (not 'fc' in k)}
            missing_weights = set(body.state_dict()) - set(state_dict_filt)
            missing_weights = [x for x in missing_weights if 'num_batches_tracked' not in x]
            logger.info(f"Following weights are not found in the model: {missing_weights}")
            body.load_state_dict(state_dict_filt, strict=not opts.debug)
            del pre_dict  # free memory
            logger.info("[!] Model init from ImageNet pretrained weights.")

        if opts.freeze:
            for param in body.parameters():
                param.requires_grad = False

        head = DeeplabV3(in_channels=body.out_channels, out_channels=opts.head_channels,
                         hidden_channels=opts.head_channels, norm_act=norm, pooling_size=32)

        if microseg:
            head = ASPP(in_channels=body.out_channels, atrous_rates=[12, 24, 36])
            print("ASPP version!")

        if opts.poincare:
            model = IncrementalHyperbolicSegmentationModule(body=body, classes=classes,
                                                      embed_dim=opts.hyp_dim,
                                                      input_dim=opts.head_channels, curv_init=opts.curv_init,
                                                      clipping=opts.hyp_clipping,
                                                      head=head, 
                                                      logger=logger, norm_act=norm_act)
        else:
            model = IncrementalSegmentationModule(body, head, opts.head_channels, classes=classes, plop=plop,
                                                  dkd=dkd, microseg=microseg,
                                                  embed_dim=opts.head_channels, norm_act=norm_act,
                                                  unseen_cluster=opts.unseen_cluster)
            
    else:
        model = None

    if model_new:
        if opts.resume or opts.ckpt or opts.test:
            model, checkpoint = init_model(model, opts, logger)
        else:
            logger.info("[!] Train from epoch 0.")
            checkpoint = None
        state_dict_filt = None
    else:
        model, checkpoint, state_dict_filt = init_model_old(model, opts, logger)

    return model, checkpoint, state_dict_filt


def init_model(model, opts, logger):
    # Load weights if resuming run
    strict_load = False if opts.debug else True
    if not opts.ckpt:
        opts.ckpt = f"{opts.logdir_full}/model_tmp.pth"
        if opts.test:
            opts.ckpt = f"{opts.logdir_full}/model_final.pth"
            if opts.save_best:
                opts.ckpt = f"{opts.logdir_full}/model_best.pth"
    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    if 'module.' in next(iter(checkpoint['model_state'])):
        state_dict_filt_new = {k.replace('module.', ''): v for k, v in checkpoint['model_state'].items() if
                               'module.' in k}
    else:
        state_dict_filt_new = checkpoint['model_state']
    if opts.debug:
         return model, None
    model.load_state_dict(state_dict_filt_new, strict=strict_load)
    logger.info(f"[!] New model initialized from {opts.ckpt}.")

    del checkpoint['model_state']
    return model, checkpoint


def init_model_old(model, opts, logger):
    # Load old model from prior step checkpoint
    strict_load = False if opts.debug else True
    # get model path
    if opts.step_prev_ckpt:
        path_old = opts.step_prev_ckpt
    else:
        path_old = opts.path_old
    logger.info(f"Loading model from {path_old}")
    step_checkpoint = torch.load(path_old, map_location=torch.device('cpu'))
    if 'module.' in next(iter(step_checkpoint['model_state'])):
        state_dict_filt = {k.replace('module.', ''): v for k, v in step_checkpoint['model_state'].items() if
                           'module.' in k}
    else:
        state_dict_filt = step_checkpoint['model_state']
    model.load_state_dict(state_dict_filt, strict=strict_load)  # Load also here old parameters
    logger.info(f"[!] Old model loaded from {path_old}")
    del step_checkpoint['model_state']

    return model, step_checkpoint, state_dict_filt


def init_model_step(opts, model, state_dict_filt):
    # init new model at t from step t-1
    if opts.debug:
       return model
    model.load_state_dict(state_dict_filt, strict=False)  # False because of incr. classifiers
    print("Following weights are not found in the model:", set(model.state_dict()) - set(state_dict_filt))
    # init classifier according to method
    if opts.init_balanced:
        model.init_new_classifier()
    if opts.init_novel:
        model.init_novel_classifier()
    # AWT intialization
    if opts.att > 0:
        imp_path = f"{opts.logdir_full}/att.pt"
        if os.path.exists(imp_path):
            imp_c = torch.load(imp_path)
            model.init_classifier_awt(imp_c)
        elif not opts.compute_att:
            print('AWT init needs to be computed before run!')
            exit()
    return model