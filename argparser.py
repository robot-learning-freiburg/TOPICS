import argparse
import json
import os


def modify_command_options(opts):
    if not opts.visualize:
        opts.sample_num = 0
    
    if opts.dataset == "map":
        opts.version = "config_v2.0.json"
    elif opts.dataset == "city":
        opts.version = "city_labels.json"
    elif opts.dataset == "mmor":
        opts.version = "ImageSets/meta.json"
    elif opts.dataset == "synmedi":
        opts.version='semantic_class_mapping.json'
    elif opts.dataset == "endovis":
        opts.version='labels.json'
    elif opts.dataset == "voc":
        opts.version='pascal.txt'
    else:
        raise NotImplementedError

    if opts.method is not None:
        if opts.method == 'FT':
            pass
        if opts.method == 'JT':
            pass
        if opts.method == 'LWF':
            opts.loss_kd = 100
        if opts.method == 'LWF-MC':
            opts.icarl = True
            opts.icarl_importance = 10
        if opts.method == 'ILT':
            opts.loss_kd = 100
            opts.loss_de = 100
        if opts.method == 'MiB':
            opts.loss_kd = 10
            opts.unce = True
            opts.unkd = True
            opts.init_balanced = True
        if opts.method == 'PLOP':
            opts.pod = "local"
            opts.pod_logits = True
            opts.pod_options = {"switch": {"after": {"extra_channels": "sum", "factor": 0.0005, "type": "local"}}}
            if not opts.pseudo == "entropybkg":
                opts.pseudo = "entropy"  #
            print(opts.pseudo, "is set")
            if opts.overlap:
                opts.threshold = 0.001
                opts.pod_factor = 0.01
            else:
                opts.threshold = 0.5
                opts.pod_factor = 0.0001
            opts.classif_adaptive_factor = True
            opts.init_balanced = True
        if opts.method == "DKD":
            opts.wbce = True
            if opts.dataset == "voc" and opts.step == 0:
                opts.pos_weight = 2  # init for voc is 2
            elif opts.dataset != "voc":
                opts.pos_weight = 35  # ADE20k and MAP
            else:
                opts.pos_weight = 1
            opts.dkd_kd = 5
            opts.dkd_pos = 5
            opts.dkd_neg = 5
            opts.freeze_bn = True
            opts.init_novel = True
            opts.diff_lr = True
            opts.lr_policy = 'warmpoly'
        if opts.method == "AWT":
            # mib
            opts.loss_kd = 10
            opts.unce = True
            opts.unkd = True
            opts.init_balanced = False  # no MiB init!
            # att
            opts.att = 25
            opts.mask_att = True
        if opts.method == "MICROSEG":
            opts.proposal = True
            if opts.step > 0:
                opts.freeze_bn = True
                opts.freeze = True
            opts.save_best = True
            opts.unknown = True
            opts.loss_tred = True
            opts.unseen_loss = True
            opts.proposal_channel = 100
            opts.pseudo = "microseg"
            opts.diff_lr = True
            opts.lr_policy = 'poly'
        if opts.method == "TOPICS":
            if opts.pseudo is None:
                opts.pseudo = "sigmoid_5"
            if opts.triplet_loss_weight == 0.0:
                opts.triplet_loss_weight = 10
            opts.triplet_loss_top = 3
            opts.trip_abs = True
            opts.trip_sel = "infonce"
            opts.all_sim = "dist0"
            if opts.distance_sim_weight == 0.0:
                opts.distance_sim_weight = 0.01
            opts.poincare = True
            if opts.curv_init == 1.0:
                opts.curv_init = 2.0
            opts.hyp_hier = True
            opts.hiera = True
    opts.no_overlap = not opts.overlap

    return opts


def get_argparser():
    parser = argparse.ArgumentParser()

    # Performance Options
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help='number of workers (default: 1)')

    # Datset Options
    parser.add_argument("--root", type=str, default='',
                        help="path to Dataset")
    parser.add_argument("--version", type=str, default='',
                        help="name of dataset label file.")
    parser.add_argument("--dataset", type=str, default='map',
                        choices=['voc', 'ade', 'city', 'map', 'mmor', 'synmedi'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Method Options
    # BE CAREFUL USING THIS, THEY WILL OVERRIDE ALL THE OTHER PARAMETERS.
    parser.add_argument("--method", type=str, default='FT',
                        choices=['FT', 'JT', 'LWF', 'LWF-MC', 'ILT', 'MiB', 'PLOP',
                                 'DKD', 'AWT', 'MICROSEG', 'TOPICS'],
                        help="The method you want to use. BE CAREFUL USING THIS, IT MAY OVERRIDE OTHER PARAMETERS.")

    # Train Options
    parser.add_argument("--epochs", type=int, default=30,
                        help="epoch number (default: 30)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 4)')
    parser.add_argument("--crop_size", type=int, default=1024,
                        help="crop size (default: 513)")
    parser.add_argument("--skip_step", type=int, default=1,
                        help="How many training samples to use.")

    parser.add_argument("--lr", type=float, default=0.007,
                        help="learning rate (default: 0.007)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum for SGD (default: 0.9)')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--lr_policy", type=str, default='poly',
                        choices=['poly', 'step', 'warmpoly', 'plateau'], help="lr schedule policy (default: poly)")
    parser.add_argument("--plateau_base", default=False, action='store_true',
                        help="Whether to plateau base or mean.")
    parser.add_argument("--lr_decay_step", type=int, default=5000,
                        help="decay step for stepLR (default: 5000)")
    parser.add_argument("--lr_decay_factor", type=float, default=0.1,
                        help="decay factor for stepLR (default: 0.1)")
    parser.add_argument("--lr_power", type=float, default=0.9,
                        help="power for polyLR (default: 0.9)")
    parser.add_argument("--bce", default=False, action='store_true',
                        help="Whether to use BCE or not (default: no)")
    parser.add_argument("--autocast", default=False, action='store_true',
                        help="Whether to use torch autocast.")
    parser.add_argument("--ckpt", default=None, type=str,
                        help="path to trained model. Leave it None if you want to use def path")
    parser.add_argument("--step_prev_ckpt", default=None, type=str,
                        help="path to trained model at previous step. Leave it None if you want to use def path")
    parser.add_argument("--resume", action='store_true', default=False,  # resume training
                        help="Use this to resume from checkpoint.")

    # Validation Options
    parser.add_argument("--crop_val", action='store_false', default=True,
                        help='do crop for validation (default: True)')
    parser.add_argument("--val_interval", type=int, default=10,
                        help="epoch interval for eval (default: 5)")

    # Logging Options
    parser.add_argument("--logdir", type=str, default='output',
                        help="path to Log directory (default: output)")
    parser.add_argument("--name", type=str, default='experiment',
                        help="name of the experiment - to append to log directory (default: Experiment)")
    parser.add_argument("--sample_num", type=int, default=0,
                        help='number of samples for visualization (default: 0)')
    parser.add_argument("--debug", action='store_true', default=False,
                        help="verbose option")
    parser.add_argument("--visualize", action='store_false', default=True,
                        help="visualization on tensorboard (def: Yes)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--ckpt_interval", type=int, default=5,
                        help="epoch interval for saving model (default: 5)")
    parser.add_argument("--wandb", action='store_false', default=True,
                        help="Push to wandb.")

    # Model Options
    parser.add_argument("--backbone", type=str, default='resnet101',
                        choices=['resnet101'],
                        help='backbone for the body (def: resnet50)')
    parser.add_argument("--output_stride", type=int, default=16,
                        choices=[8, 16], help='stride for the backbone (def: 16)')
    parser.add_argument("--head_channels", type=int, default=256,
                        choices=[16, 32, 64, 128, 256, 512], help='number of channels before classification layer.')
    parser.add_argument("--pretrained", action='store_false', default=True,
                        help='Wheather to use pretrained or not (def: True)')
    parser.add_argument("--temperature", type=float, default=0.07,
                        help='temperature for contrastive loss')
    
    # Test and Checkpoint options
    parser.add_argument("--test", action='store_true', default=False,
                        help="Whether to train or test only (def: train and test)")
    parser.add_argument("--save_pred", type=str, default=None,
                        help="Whether to save predictions for test.")
    parser.add_argument("--window_stitching", action='store_true', default=False,
                        help="Whether to stitch together crops for testing.")

    # Parameters for Knowledge Distillation of ILTSS (https://arxiv.org/abs/1907.13372)
    parser.add_argument("--freeze", action='store_true', default=False,
                        help="Use this to freeze the feature extractor in incremental steps.")
    parser.add_argument("--freeze_bn", action='store_true', default=False,
                        help="Use this to freeze the bn and dropout in incremental steps.")
    parser.add_argument("--diff_lr", action='store_true', default=False,
                        help="Use this to trigger different lr for different parts of the network.")
    parser.add_argument("--loss_de", type=float, default=0.,  # Distillation on Encoder
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable distillation on Encoder (L2)")
    parser.add_argument("--loss_kd", type=float, default=0.,  # Distillation on Output
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable Knowledge Distillation (Soft-CrossEntropy)")
    parser.add_argument("--unkd", default=False, action='store_true',
                        help="Enable Unbiased Knowledge Distillation instead of Knowledge Distillation")
    parser.add_argument("--alpha", default=1., type=float,
                        help="The parameter to hard-ify the soft-labels. Def is 1.")
    parser.add_argument("--init_balanced", default=False, action='store_true',
                        help="Enable Background-based initialization for new classes")

    # Arguments for ICaRL (from https://arxiv.org/abs/1611.07725)
    parser.add_argument("--icarl", default=False, action='store_true',
                        help="If enable ICaRL or not (def is not)")
    parser.add_argument("--icarl_importance", type=float, default=1.,
                        help="the regularization importance in ICaRL (def is 1.)")
    parser.add_argument("--icarl_disjoint", action='store_true', default=False,
                        help="Which version of icarl is to use (def: combined)")
    parser.add_argument("--icarl_bkg", action='store_true', default=False,
                        help="If use background from GT (def: No)")
    # AWT
    parser.add_argument("--att", default=0.0, type=float, help="threshold for selecting most significant channels")
    parser.add_argument("--compute_att", default=False, action="store_true", help="att compute")
    parser.add_argument("--mask_att", default=False, action="store_true", help="enable masking of attribution maps")

    # Arguments for PLOP
    parser.add_argument(
        "--pod",
        default=None,
        type=str,
        choices=[
            "spatial", "local", "global"
        ]
    )
    parser.add_argument("--pod_factor", default=5., type=float)
    parser.add_argument("--pod_options", default=None, type=json.loads)
    parser.add_argument("--pod_prepro", default="pow", type=str)
    parser.add_argument("--no_pod_schedule", default=False, action="store_true")
    parser.add_argument(
        "--pod_apply", default="all", type=str, choices=["all", "backbone", "deeplab"]
    )
    parser.add_argument("--pod_deeplab_mask", default=False, action="store_true")
    parser.add_argument(
        "--pod_deeplab_mask_factor", default=None, type=float, help="By default as the POD factor"
    )
    parser.add_argument("--deeplab_mask_downscale", action="store_true", default=False)
    parser.add_argument("--pod_interpolate_last", default=False, action="store_true")
    parser.add_argument(
        "--pod_logits", default=False, action="store_true", help="Also apply POD to logits."
    )
    parser.add_argument(
        "--pod_large_logits", default=False, action="store_true", help="Also apply POD to large logits."
    )
    parser.add_argument(
        "--pseudo",
        type=str,
        default=None,
        help="Pseudo-labeling method." +
             ", ".join(["naive", "confidence", "threshold_5", "threshold_9", "median", "entropy", "entropybkg",
                        "softplusentropy", "microseg"])
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--pseudo_nb_bins", default=None, type=int)
    parser.add_argument("--classif_adaptive_factor", default=False, action="store_true")
    parser.add_argument("--classif_adaptive_min_factor", default=0.0, type=float)

    # DKD
    parser.add_argument("--wbce", default=False, action='store_true', help="Enables BCE with logits.")
    parser.add_argument("--dkd_kd", default=0.0, type=float)
    parser.add_argument("--pos_weight", default=1, type=int)
    parser.add_argument("--init_novel", default=False, action='store_true',
                        help="Enable Background-based initialization for new classes according to DKD")

    # MicroSEG
    parser.add_argument("--proposal", default=False, action='store_true', help="Enables Proposal Loading.")
    parser.add_argument("--loss_tred", default=False, action='store_true', help="Enables Proposal Loading.")
    parser.add_argument("--unseen_multi", action='store_true', help='use multi-unseen class')
    parser.add_argument("--unseen_cluster", type=int, default=5, help='number of unseen cluster')
    parser.add_argument("--save_best", action='store_true', help='save best model.')

    # MiB/AWT
    parser.add_argument("--unce", default=False, action='store_true',
                        help="Enable Unbiased Cross Entropy instead of CrossEntropy")

    # METHODS TOPICS
    parser.add_argument("--curv_init", type=float, default=1.0,
                        help="Init for curvature (default:0.1).")
    parser.add_argument("--poincare", default=False, action='store_true',
                        help="If activating poincare model.")
    parser.add_argument("--hyp_hier", default=False, action='store_true',
                        help="If activated, load hyperbolic hierarchy.")
    parser.add_argument("--hiera", default=False, action='store_true',
                        help="If activated, use hyperbolic SOTA bce loss.")
    parser.add_argument("--hiera_feat", default=False, action='store_true',
                        help="If activated, use tree triplet loss with feat0 within hiera.")
    parser.add_argument("--dice", type=float, default=0.0,
                        help="Whether to activate dice loss.")
    parser.add_argument("--hiera_factor", type=float, default=5.0,
                        help="Weight of tree min loss in hiera.")
    parser.add_argument("--hbce", default=False, action='store_true',
                        help="If activated, use hyperbolic hierarchy bce loss. Not hiera")
    parser.add_argument("--freeze_last", default=False, action='store_true',
                        help="If activated, freeze hyperbolic layer.")
    parser.add_argument("--distance_sim_weight", type=float, default=0.0,
                        help="If activated, use distance loss on features of old and new model.")
    parser.add_argument("--all_sim", type=str, choices=["l1", "fr", "distm", "dist0", "wmid", "wmid0"],
                        help="Measure of similarity.")
    parser.add_argument("--hyp_dim", type=int, default=128,
                        help="Dimension of hyperbolic space in last layer.")
    parser.add_argument("--hyp_clipping", default=False, action='store_true',
                        help="If true, stop gradient clipping.")
    parser.add_argument("--triplet_loss_weight", type=float, default=0.0,
                        help="Weight for triplet loss.")
    parser.add_argument("--triplet_loss_top", type=float, default=5.0,
                        help="How many samples to use as positives in label creation.")
    parser.add_argument("--hier_trip", default=False, action='store_true',
                        help="Only use triplet loss on same hierarchy samples.")
    parser.add_argument("--trip_increment_only", default=False, action='store_true',
                        help="Anchors are only novel classes.")
    parser.add_argument("--trip_remove_hier", default=False, action='store_true',
                        help="Remove ancestors from triplet computation.")
    parser.add_argument("--trip_abs", default=False, action='store_true',
                        help="Use absolut distances (the smaller, the better).")
    parser.add_argument("--no_margin", default=False, action='store_true',
                        help="Don't use margin.")
    parser.add_argument("--trip_norm", default=False, action='store_true',
                        help="Normalize distances, no exp().")
    parser.add_argument("--margin_factor", default=1., type=float,
                        help="Factor of x * std of distances.")
    parser.add_argument("--trip_sel", type=str, default="all", choices=["random", "hard", "all", "infonce", "infonce2"],
                        help="Triplet Selector.")
    parser.add_argument("--tau", type=float, default=0.07,
                        help="TAU for InfoNCE")

    # Incremental parameters
    parser.add_argument("--task", type=str, default="",
                        help="Task to be executed (csv file).")
    parser.add_argument("--wb_version", type=float, default=0.0,
                        help="Version in wandb for tracking.")
    parser.add_argument("--step", type=int, default=0,
                        help="The incremental step in execution (default: 0)")
    parser.add_argument("--bg_shift", action='store_false', default=True,  # background shift
                        help="Use this to not mask the old classes in new training set")
    parser.add_argument("--no_bg_class", action='store_true', default=False,  # don't learn void class
                        help="Remove 0 class.")
    parser.add_argument("--other_class", action='store_true', default=False,  # other class
                        help="Use this to mask old classes to other class.")
    parser.add_argument("--overlap", action='store_false', default=True,
                        help="Use this to not use the new classes in the old training set")
  
    # multi GPU parameters and new
    parser.add_argument("--MASTER_PORT", type=str, default='29501',
                        help="port")
    parser.add_argument("--MASTER_ADDRESS", type=str, default='0',
                        help="GPU adress")
    return parser
