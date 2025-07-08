import random
import torch

from .map import MapillarySegmentationIncremental
from .voc import PascalVOCIncremental
from .city import CityscapesSegmentationIncremental
from .mmor import MMORIncremental
from .synmedi import SynMediIncremental
from .endovis import EndovisIncremental
from .task import Task
from .transform import get_transforms
from  . import dataset_meta


def get_dataset(opts, logger, world_size, rank):
    """ Dataset And Augmentation
    """

    if opts.dataset == "map":
        dataset = MapillarySegmentationIncremental
        meta = dataset_meta.get_mapillary_vistas_meta(opts.root, opts.version)
    elif opts.dataset == "voc":
        dataset = PascalVOCIncremental
        meta = dataset_meta.pascal_voc_meta(opts.root, opts.version)
    elif opts.dataset == "city":
        dataset = CityscapesSegmentationIncremental
        meta = dataset_meta.city_meta(opts.root, opts.version)
    elif opts.dataset == "mmor":
        dataset = MMORIncremental
        meta = dataset_meta.mmor_meta(opts.root, opts.version)
    elif opts.dataset == "synmedi":
        dataset = SynMediIncremental
        meta = dataset_meta.synmedi_meta(opts.root, opts.version)
    elif opts.dataset == "endovis":
        dataset = EndovisIncremental
        meta = dataset_meta.endovis_meta(opts.root, opts.version)
    else:
        raise NotImplementedError

    incremental_level = 0 if opts.method == 'JT' else -1

    task = Task(opts.task, opts.step, bg_shift=opts.bg_shift, overlap=opts.overlap,
                incremental_level=incremental_level, other_class=opts.other_class, no_bg_class=opts.no_bg_class,
                hyp_hier=opts.hyp_hier, meta=meta)
    
    train_transform, val_transform, test_transform, opts = get_transforms(opts, task, logger)
    
    train_dst = dataset(mode='train', transform=train_transform,
                        task=task, _root=opts.root, _version=opts.version, proposal=opts.proposal, skip_step=opts.skip_step)
    val_dst = dataset(mode='val', transform=val_transform,
                      task=task, _root=opts.root, _version=opts.version, proposal=opts.proposal)
    test_dst = dataset(mode='test', transform=test_transform,
                       task=task, _root=opts.root, _version=opts.version, proposal=opts.proposal, save_pred=opts.save_pred)

    logger.classes = task.class_list()
    # reset the seed, this revert changes in random seed
    random.seed(opts.random_seed)
    
    train_batch_size = 1 if (opts.test or opts.compute_att) else opts.batch_size
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=train_batch_size, num_workers=opts.num_workers,
                                               sampler=torch.utils.data.distributed.DistributedSampler(train_dst,
                                                                                                       num_replicas=world_size,
                                                                                                       rank=rank),
                                               drop_last=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=opts.batch_size if opts.crop_val else 1,
                                             num_workers=opts.num_workers,
                                             sampler=torch.utils.data.distributed.DistributedSampler(val_dst,
                                                                                                     num_replicas=world_size,
                                                                                                     rank=rank), pin_memory=True,
                                             drop_last=False)
    tst_loader = torch.utils.data.DataLoader(test_dst, batch_size=1,
                                             num_workers=opts.num_workers,
                                             sampler=torch.utils.data.distributed.DistributedSampler(test_dst,
                                                                                                     num_replicas=world_size,
                                                                                                     rank=rank), pin_memory=True,
                                             drop_last=False)
    logger.info(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)},"
                f" Test set: {len(test_dst)}, n_classes {task.classes(opts.step)}. Train batch size {train_batch_size}")
    logger.info(f"Total batch size is {opts.batch_size * world_size}")


    return train_loader, val_loader, tst_loader, task
