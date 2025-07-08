import torch
from functools import reduce
import wandb
import os
import time
import numpy as np
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.loss import KnowledgeDistillationLoss, BCEWithLogitsLossWithIgnoreIndex, \
    UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy, IcarlLoss, \
    HierarchicalLoss, features_distillation, DistanceSim, \
    TripletLoss, ACLoss, KDLoss, WBCELoss, HBCELoss, UnseenAugLoss, HIERALoss

from utils.pseudo_label import ComputePseudoLabel


class Trainer:
    def __init__(self, model_old, device, opts, masking_value=255, classes=None, adj_dict=None, distributed=False,
                 internal_masking_value=255, hier_matrices=None):

        self.device = device
        self.step = opts.step
        self.debug = opts.debug
        self.dkdloss = False
        self.microsegloss = False
        self.hiera = False
        self.proposal = opts.proposal
        self.hyp_hier = opts.hyp_hier
        self.hbce = opts.hbce
        self.freeze_bn = opts.freeze_bn
        self.freeze = opts.freeze

        if classes:
            self.classes = classes
            new_classes = len(classes[-1])
            self.tot_classes = len(reduce(lambda a, b: a + b, classes))
            self.old_classes = self.tot_classes - new_classes  # includes bkg
            print(f"new classes {new_classes}, tot_classes {self.tot_classes}, old classes {self.old_classes}")
        else:
            self.old_classes = 0

        # Set main loss type
        reduction = 'none'
        self.bce = opts.bce or opts.icarl
        if self.bce:
            self.criterion = BCEWithLogitsLossWithIgnoreIndex(reduction=reduction, ignore_index=masking_value)
        elif opts.unce and self.old_classes != 0 and not opts.hyp_hier:
            self.criterion = UnbiasedCrossEntropy(old_cl=self.old_classes, ignore_index=masking_value,
                                                  reduction=reduction)
        elif opts.hyp_hier:
            # Hierarchical losses
            if opts.hbce:
                # hierarchical binary cross entropy loss
                self.criterion = HBCELoss(hier_matrices=hier_matrices, device=device, ignore_label=masking_value,
                                          no_classes=self.tot_classes)
            elif opts.hiera:
                # TOPICS loss
                self.hiera = True
                self.criterion = HIERALoss(hier_matrices=hier_matrices, device=device, ignore_label=masking_value,
                                           no_classes=self.tot_classes, adj_dict=adj_dict,
                                           factor=opts.hiera_factor, hiera_feat=opts.hiera_feat, 
                                           dice=opts.dice)
            else:
                # Softmax cross entropy loss with hierarchical matrices
                self.criterion = HierarchicalLoss(hier_matrices=hier_matrices, device=device,
                                                  ignore_label=masking_value, no_classes=self.tot_classes)
        elif opts.wbce:
            # DKD
            if self.step == 0:
                new_classes -= 1  # subtract bkg
                self.old_classes += 1  # old_class = 1 (bkg)
            pos_weight = torch.ones([new_classes], device=self.device) * opts.pos_weight
            self.criterion = WBCELoss(pos_weight=pos_weight, n_old_classes=self.old_classes,
                                      n_new_classes=new_classes)
            self.ACLoss = ACLoss()
            self.dkdloss = True
        elif opts.loss_tred:
            # MICROSEG
            self.microsegloss = True
            self.ualoss = UnseenAugLoss()
            self.proposal_channel = opts.proposal_channel
            self.criterion = BCEWithLogitsLossWithIgnoreIndex(ignore_index=masking_value, reduction='mean')
            self.tot_classes += 1
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=masking_value, reduction=reduction)

        # init losses for steps > 0
        if model_old:
            self.model_old_flag = True
            self.lde_flag = opts.loss_de > 0.
            self.lkd_flag = opts.loss_kd > 0.
            self.icarl_flag = opts.icarl
            self.pod_flag = False if opts.pod is None else True
            self.pseudo_flag = False if opts.pseudo is None else True
            self.dkd_flag = True if opts.method == "DKD" else False

            # ILTSS
            if self.lde_flag:
                self.lde_w = opts.loss_de
                self.lde_loss = torch.nn.MSELoss()

            if self.lkd_flag:
                self.lkd_w = opts.loss_kd
                if opts.unkd:
                    self.lkd_loss = UnbiasedKnowledgeDistillationLoss(alpha=opts.alpha)

                else:
                    self.lkd_loss = KnowledgeDistillationLoss(alpha=opts.alpha)

            # ICARL
            if self.icarl_flag:
                self.icarl_combined_flag = not opts.icarl_disjoint
                if opts.icarl_disjoint:
                    self.licarl = IcarlLoss(reduction='mean', bkg=opts.icarl_bkg)
                    self.icarl_combined_flag = False
                else:
                    self.licarl = torch.nn.BCEWithLogitsLoss(reduction='mean')
                    self.icarl_w = opts.icarl_importance
                    self.icarl_combined_flag = False

            if self.pod_flag:
                self.pod_internal_layers = False
                self.pod_final_layer = False
                self.index_new_class = self.old_classes
                self.nb_current_classes = self.tot_classes
                self.nb_new_classes = new_classes
                self.pod = "local"
                self.pod_factor = opts.pod_factor
                self.pod_logits = True
                self.pod_apply = 'all'
                self.pod_deeplab_mask = False
                self.pod_deeplab_mask_factor = None
                self.pod_options = opts.pod_options
                self.use_pod_schedule = True
                self.pod_interpolate_last = False
                self.pod_large_logits = False
                self.pod_prepro = 'pow'
                self.deeplab_mask_downscale = False
                self.spp_scales = [1, 2, 4]

            if self.pseudo_flag:
                self.compute_pseudo = ComputePseudoLabel(opts.pseudo, masking_value, device=self.device,
                                                         threshold=opts.threshold,
                                                         tot_classes=self.tot_classes, thresholds=None,
                                                         compute_classif_adaptive_factor=opts.classif_adaptive_factor,
                                                         internal_masking_value=internal_masking_value,
                                                         hyp_hier=opts.hyp_hier,
                                                         hier_matrices=hier_matrices)

            if self.dkd_flag:
                self.kdloss = KDLoss(pos_weight=None, reduction='none')
                self.dkd_pos = opts.dkd_pos
                self.dkd_neg = opts.dkd_neg
                self.dkd_kd = opts.dkd_kd

        else:
            self.model_old_flag, self.lde_flag, self.lkd_flag, self.icarl_flag, self.pseudo_flag, self.pod_flag, self.dkd_flag = False, False, False, False, False, False, False

        self.distributed = distributed
        self.masking_value = masking_value

        self.hyp_dist_sim = (opts.distance_sim_weight > 0) and (opts.step > 0)
        self.triplet = (opts.triplet_loss_weight > 0) and (opts.step > 0)

        if self.triplet:
            matrix = hier_matrices[4] if opts.hier_trip or opts.trip_remove_hier else None
            self.triplet_loss = TripletLoss(None, device, top=opts.triplet_loss_top, hier_matrices=matrix,
                                                trip_increment_only=opts.trip_increment_only, hier_trip=opts.hier_trip,
                                                remove_hier=opts.trip_remove_hier, abs=opts.trip_abs,
                                                no_margin=opts.no_margin,
                                                margin_factor=opts.margin_factor, norm=opts.trip_norm,
                                                trip_sel=opts.trip_sel, tau=opts.tau)
            self.triplet_loss_weight = opts.triplet_loss_weight
            self.init_triplet_loss(model_old)
        if self.hyp_dist_sim:
            self.hyp_dist_sim_weight = opts.distance_sim_weight
            self.dist_loss = DistanceSim(old_classes=self.old_classes, internal_masking_value=internal_masking_value,
                                         masking_value=masking_value,
                                         all_sim=opts.all_sim)

        self.prediction_vis = None

        if self.lde_flag or self.pod_flag or self.hyp_dist_sim or self.dkd_flag or self.microsegloss or self.hiera:
            self.feature_flag = True
        else:
            self.feature_flag = False

    def init_triplet_loss(self, model):
        with torch.no_grad():
            offset = model.offset
            normal = model.normal
            manifold = model.manifold
            self.triplet_loss.update_distance(offset=offset[:self.step], normal=normal[:self.step], manifold=manifold)

    def pseudo_before(self, train_loader, logger, model_old, opts):
        # compute median thresholds before training (used for PLOP)
        if os.path.exists(f'{opts.logdir_full}/max_entropy.pt'):
            self.compute_pseudo.max_entropy = torch.load(f'{opts.logdir_full}/max_entropy.pt',
                                                         map_location=torch.device('cpu'))
            self.compute_pseudo.thresholds = torch.load(f'{opts.logdir_full}/thresholds.pt',
                                                        map_location=torch.device('cpu')).to(self.device)
            logger.info("Loaded median compute!")
            return

        if self.pseudo_flag is not None:
            logger.info("Find median score.")
            result = self.compute_pseudo.find_median(train_loader, logger, model_old)
            if result is not None:
                torch.save(result[1], f'{opts.logdir_full}/max_entropy.pt')
                torch.save(result[0], f'{opts.logdir_full}/thresholds.pt')
                logger.info("Saved median compute!")

    def awt_before(self, train_loader, logger, model_old, opts):
        imp_path = f"{opts.logdir_full}/att.pt"
        if os.path.exists(imp_path):
            print('AWT already computed and loaded!')
            return
        print('compute AWT!')
        from utils import compute_attribute
        imp_c = compute_attribute(train_loader, logger, model_old, opts, self.classes[-1], self.device)

        if imp_c is not None:
            print('important channels ', imp_c.shape, imp_c.sum())
            imp_name = f"{opts.logdir_full}/att.pt"
            torch.save(imp_c, imp_name)
        else:
            print("imp_c is None")
        exit()

    def train(self, model, model_old, cur_epoch, optim, scaler, train_loader, scheduler=None, print_int=10,
              logger=None):
        """Train and return epoch loss"""
        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        model.train()

        # freeze for MICROSEG
        if self.freeze_bn and (self.step > 0):
            mod = model.module if isinstance(model, (nn.DataParallel, DDP)) else model
            affine_freeze = True if self.microsegloss else False
            mod.freeze_bn_dropout(affine_freeze=affine_freeze)

        if self.model_old_flag:
            model_old.eval()
        
        epoch_loss = 0.0
        reg_loss = 0.0
        interval_loss = 0.0
        autocast = True if scaler else False

        # loaded proposals for MICRO
        proposal = None

        train_loader.sampler.set_epoch(cur_epoch)
        len_data = len(train_loader)   

        for cur_step, (images, labels) in enumerate(train_loader):

            if isinstance(self.criterion, HIERALoss):
                self.criterion.update_param(cur_epoch, cur_step, len_data)

            if self.proposal:
                proposal = labels[1]
                labels = labels[0]

            loss, l_reg, l_kd, loss_tot = self.run_batch(model, model_old, images, labels, optim, scheduler,
                                                           autocast, scaler=scaler, proposal=proposal, logger=logger)

            epoch_loss += loss.item()
            reg_loss += l_reg.item() if l_reg != 0 else 0.
            reg_loss += l_kd.item() if l_kd != 0 else 0.

            interval_loss += loss_tot.item()
            interval_loss += l_reg.item() if l_reg != 0 else 0

            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.info(f"Epoch {cur_epoch}, Batch {cur_step + 1}/{len(train_loader)},"
                            f" Loss={interval_loss}")
                # visualization
                if logger is not None:
                    x = cur_epoch * len(train_loader) + cur_step + 1
                    logger.add_scalar('Loss', interval_loss, x)
                interval_loss = 0.0

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)

        if self.distributed:
            torch.distributed.reduce(epoch_loss, dst=0)
            torch.distributed.reduce(reg_loss, dst=0)

            if torch.distributed.get_rank() == 0:
                epoch_loss = epoch_loss / torch.distributed.get_world_size() / len(train_loader)
                reg_loss = reg_loss / torch.distributed.get_world_size() / len(train_loader)
                return epoch_loss.item(), reg_loss.item()
            else:
                return -1, -1
        else:
            epoch_loss = epoch_loss / 1 / len(train_loader)
            reg_loss = reg_loss / 1 / len(train_loader)
            return epoch_loss.item(), reg_loss.item()

    def run_batch(self, model, model_old, images, labels, optim, scheduler, autocast, scaler=None,
                    val=False, test=False, proposal=None, logger=None):
        """Run a single batch and return loss (train, val) or outputs (test)"""

        l_kd, l_reg = 0, 0

        images = images.to(self.device, dtype=torch.float32)
        labels = labels.to(self.device, dtype=torch.long)

        with torch.cuda.amp.autocast(enabled=autocast):

            if proposal is not None:
                # MICROSEG
                n_cl = torch.tensor(self.proposal_channel).to(proposal.device)
                proposals_n = torch.where(proposal != self.masking_value, proposal, n_cl)
                proposals_n = torch.where(proposals_n <= self.proposal_channel, proposal, n_cl)
                proposals_1hot = torch.nn.functional.one_hot(proposals_n.long(), self.proposal_channel + 1).permute(0,3,1,2)
                proposal = proposals_1hot[:, :-1]
                proposal = proposal.to(self.device, dtype=torch.float)
                labels = torch.where(labels != 255, torch.add(labels, 1), labels)  # add plus one
                labels = torch.where(labels == 1, torch.zeros_like(labels), labels)  # set bkg to 0.

            if self.model_old_flag and model_old is not None:
                # compute old models outputs and features
                with torch.no_grad():
                    outputs_old, features_old = model_old(images, features=self.feature_flag, proposal=proposal)

            if self.pseudo_flag and model_old is not None:
                # compute pseudo labels
                if self.hyp_hier:
                    outputs_old, _ = self.criterion.decide(outputs_old, old_model=True, keep_parent_logit=self.compute_pseudo.type.startswith("hiersigmoid"))
                labels, classif_adaptive_factor, mask_valid_pseudo, sample_weights = self.compute_pseudo(labels,outputs_old)

            if optim:
                optim.zero_grad()

            outputs, features = model(images, features=self.feature_flag, proposal=proposal)  

            if test:
                return 0, 0, 0, 0, outputs, features, labels

            if self.icarl_flag and not self.icarl_combined_flag:
                loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old)).mean()
            elif self.pseudo_flag and not self.microsegloss:
                loss = self.criterion(outputs, labels)
                loss = classif_adaptive_factor * loss
                if sample_weights is not None:
                    print(sample_weights)
                    loss = loss * sample_weights
                loss = loss.mean()
            elif self.dkdloss:
                # DKD
                loss = self.criterion(outputs, labels).mean(dim=[0, 2, 3]).sum() + self.ACLoss(outputs[:, 0:1]).mean(
                    dim=[0, 2, 3]).sum()  # ACLoss on bkg only
            elif self.microsegloss:
                # MICROSEG
                loss = self.criterion(outputs, labels)
                l_kd = self.ualoss(features[0])
                if self.step == 0:
                    loss += self.criterion(features[1], labels)  # was outputs_pixel
            elif self.hiera:
                # Hierarchical TOPICS loss with features
                loss = self.criterion(outputs, labels, feats=features[0]).mean()
            else:
                loss = self.criterion(outputs, labels).mean()
                if torch.isnan(loss).any():
                    logger.info("Loss is nan!")

            if self.triplet:
                # triplet loss
                offset = model.module.offset if self.distributed else model.offset
                normal = model.module.normal if self.distributed else model.normal
                manifold = model.module.manifold if self.distributed else model.manifold
                triplet_loss = self.triplet_loss(offset=offset, normal=normal, curv=manifold)
                if triplet_loss is not None:
                    l_kd += self.triplet_loss_weight * triplet_loss

            if self.hyp_dist_sim:
                # distance similarity loss
                manifold = model.module.manifold if self.distributed else model.manifold
                dist_loss = self.dist_loss(features=features[1], labels=labels, features_old=features_old[1],
                                           manifold=manifold)
                if dist_loss is not None:
                    l_kd += self.hyp_dist_sim_weight * dist_loss

            if self.icarl_flag and self.icarl_combined_flag:
                # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                n_cl_old = outputs_old.shape[1]
                # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                l_kd += self.icarl_w * n_cl_old * self.licarl(outputs.narrow(1, 0, n_cl_old),
                                                              torch.sigmoid(outputs_old))

            # xxx ILTSS (distillation on features or logits)
            if self.lde_flag:
                l_kd += self.lde_w * (self.lde_loss(features[0], features_old[0]) + self.lde_loss(
                    features[1], features_old[1]))

            if self.lkd_flag:
                l_kd += self.lkd_w * self.lkd_loss(outputs, outputs_old)

            if self.dkd_flag:
                # DKD
                loss_kd = self.kdloss(outputs[:, 1:self.old_classes], outputs_old[:, 1:].sigmoid()).mean(
                    dim=[0, 2, 3]).sum()  # [|C0:t-1|]
                loss_dkd_pos = self.kdloss(features[0][:, :self.old_classes - 1],
                                           features_old[0].sigmoid()).mean(dim=[0, 2, 3]).sum()  # [|C0:t-1|]
                loss_dkd_neg = self.kdloss(features[1][:, :self.old_classes - 1], features_old[1].sigmoid()).mean(
                    dim=[0, 2, 3]).sum()  # [|C0:t-1|]
                l_kd += self.dkd_pos * loss_dkd_pos + self.dkd_neg * loss_dkd_neg + self.dkd_kd * loss_kd

            if self.pod_flag:
                # PLOP
                if not type(features[0][0]) == list:
                    attentions_old = [xs for xs in features_old[0]]
                    attentions_new = [xs for xs in features[0]]
                else:  # hrnet gives more dimensions to be split
                    attentions_old = [x for xs in features_old[0] for x in xs]
                    attentions_new = [x for xs in features[0] for x in xs]
                attentions_old.append(features_old[2])
                attentions_new.append(features[2])

                l_kd += features_distillation(
                    attentions_old,
                    attentions_new,
                    collapse_channels=self.pod,
                    labels=labels,
                    index_new_class=self.old_classes,
                    pod_apply=self.pod_apply,
                    pod_deeplab_mask=self.pod_deeplab_mask,
                    pod_deeplab_mask_factor=self.pod_deeplab_mask_factor,
                    interpolate_last=self.pod_interpolate_last,
                    pod_factor=self.pod_factor,
                    prepro=self.pod_prepro,
                    deeplabmask_upscale=not self.deeplab_mask_downscale,
                    spp_scales=self.spp_scales,
                    pod_options=self.pod_options,
                    outputs_old=outputs_old,
                    use_pod_schedule=self.use_pod_schedule,
                    nb_current_classes=self.nb_current_classes,
                    nb_new_classes=self.nb_new_classes,
                    internal_layers_only=self.pod_internal_layers,
                    final_layer_only=self.pod_final_layer
                )

            loss_tot = loss + l_kd

        if val:
            return loss, l_reg, l_kd, loss_tot, outputs, features, labels

        if scaler:
            scaler.scale(loss_tot).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
        elif optim:
            if loss_tot > 0:
                loss_tot.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)  # .1
                optim.step()
                optim.zero_grad()

        if scheduler is not None:
            scheduler.step()

        return loss, l_reg, l_kd, loss_tot

    def validate(self, model, model_old, loader, metrics, ret_samples_ids=None, logger=None, 
                 autocast=False, test=False, window_stitching=None):
        """Do validation and return specified samples"""
        metrics.reset()
        model.eval()

        class_loss = 0.0
        reg_loss = 0.0

        proposal = None
        time_measure = []

        for i, (images, labels) in enumerate(loader):
            
            if test:
                time_measure.append(time.time())

            if isinstance(labels, list) or isinstance(labels, tuple):
                idx = -1
                if self.proposal:
                    proposal = labels[-1]
                    idx = -2
                labels_val = labels[idx].clone().cpu().numpy()
                labels = labels[0]
            else:
                labels_val = labels.clone().cpu().numpy()

            if (window_stitching is not None) and test:
                # window stitching for large images
                # ref: https://github.com/lingorX/HieraSeg/blob/main/Pytorch/mmseg/models/segmentors/encoder_decoder.py
                crop_proposal = None
                h_crop, w_crop = window_stitching  # same as stride
                h_stride, w_stride = window_stitching
                batch_size, _, h_img, w_img = images.size()
                h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
                w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
                outputs = images.new_zeros((batch_size, self.tot_classes, h_img, w_img)).to(self.device)
                pseudo_label = torch.zeros_like(labels).to(self.device)
                count_mat = torch.zeros_like(labels)
                for h_idx in range(h_grids):
                    for w_idx in range(w_grids):
                        y1 = h_idx * h_stride
                        x1 = w_idx * w_stride
                        y2 = min(y1 + h_crop, h_img)
                        x2 = min(x1 + w_crop, w_img)
                        y1 = max(y2 - h_crop, 0)
                        x1 = max(x2 - w_crop, 0)
                        crop_img = images[:, :, y1:y2, x1:x2]
                        crop_label = labels[:, y1:y2, x1:x2]
                        if self.proposal:
                            crop_proposal = proposal[:, y1:y2, x1:x2]
                        with torch.no_grad():
                            _, _, _, _, crop_outputs, _, crop_pseudo_label = self.run_batch(model, model_old,
                                                                                              crop_img,
                                                                                              crop_label,
                                                                                              None,
                                                                                              None,
                                                                                              autocast=autocast,
                                                                                              scaler=None,
                                                                                              val=True,
                                                                                              test=test,
                                                                                              proposal=crop_proposal)
                        outputs += torch.nn.functional.pad(crop_outputs, (
                        int(x1), int(outputs.shape[3] - x2), int(y1), int(outputs.shape[2] - y2)))
                        pseudo_label += torch.nn.functional.pad(crop_pseudo_label, (
                            int(x1), int(outputs.shape[3] - x2), int(y1), int(outputs.shape[2] - y2)))
                        count_mat[:, y1:y2, x1:x2] += 1
                loss, l_reg, l_kd, features = None, None, None, None
                assert (count_mat == 0).sum() == 0
            else:
                # normal validation
                with torch.no_grad():
                    loss, l_reg, l_kd, _, outputs, features, pseudo_label = self.run_batch(model, model_old,
                                                                                                    images, labels,
                                                                                                    None,
                                                                                                    None,
                                                                                                    autocast=autocast,
                                                                                                    scaler=None,
                                                                                                    val=True,
                                                                                                    test=test,
                                                                                                    proposal=proposal)
                if not test:
                    class_loss += loss.item()
                    reg_loss += l_reg.item() if l_reg != 0. else 0.
                    reg_loss += l_kd.item() if l_kd != 0. else 0.

            if self.hyp_hier:
                # decide outputs with hierarchical loss
                _, prediction = self.criterion.decide(outputs)
            elif self.dkdloss or self.bce:
                # DKD
                logit = torch.sigmoid(outputs)
                prediction = logit[:, 1:].argmax(dim=1) + 1  # pred: [N. H, W]
                idx = (logit[:, 1:] > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]
                prediction[idx == 0] = 0  # set background (non-target class)
            elif self.microsegloss:
                # MICROSEG
                logit = torch.sigmoid(outputs)
                logit[:, 1] += logit[:, 0]
                prediction = logit[:, 1:].argmax(dim=1)
                pseudo_label = torch.where(pseudo_label == 0, pseudo_label, pseudo_label - 1)
            else:
                # max logit
                _, prediction = outputs.max(dim=1)

            if test:
                time_measure[i]= time.time() - time_measure[i]

            pseudo_label = pseudo_label.cpu().numpy()
            labels = labels_val
            prediction = prediction.cpu().numpy()
            metrics.update(labels, prediction)
            outputs = outputs.float().cpu().numpy()

            if (ret_samples_ids is not None) and (i in ret_samples_ids):
                img = self.prediction_vis(images, labels, pseudo_label, prediction, i, file_names=None)
                # log samples on wandb
                if not self.prediction_vis.save_pred:
                    wandb.log({"sample": img})

            del loss, l_reg, l_kd, outputs, features, pseudo_label, labels_val, labels

            # collect statistics from multiple processes
            if self.distributed:
                metrics.synch(self.device)
            score = metrics.get_results()

        if test:
            time_measure = np.asarray(time_measure).mean()
            score["time"] = time_measure
            return None, score

        class_loss = torch.tensor(class_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)

        if self.distributed:
            torch.distributed.reduce(class_loss, dst=0)
            torch.distributed.reduce(reg_loss, dst=0)
            if torch.distributed.get_rank() == 0:
                class_loss = class_loss / torch.distributed.get_world_size() / len(loader)
                reg_loss = reg_loss / torch.distributed.get_world_size() / len(loader)
                return (class_loss, reg_loss), score
        else:
            class_loss = class_loss / 1 / len(loader)
            reg_loss = reg_loss / 1 / len(loader)

        if logger is not None:
            logger.info(f"Validation, Class Loss={class_loss}, Reg Loss={reg_loss} (without scaling)")

        return (class_loss, reg_loss), score

    def test(self, model, loader):
        """Do test and return all output"""
        model.eval()

        with torch.no_grad():
            for i, (images, file_name) in enumerate(loader):
                images = images.to(self.device, dtype=torch.float32)
                outputs, _ = model(images)

                if self.hyp_hier:
                    # decide outputs with hierarchical loss
                    _, prediction = self.criterion.decide(outputs)
                elif self.dkdloss or self.bce:
                    # DKD
                    logit = torch.sigmoid(outputs)
                    prediction = logit[:, 1:].argmax(dim=1) + 1  # pred: [N. H, W]
                    idx = (logit[:, 1:] > 0.5).float()  # logit: [N, C, H, W]
                    idx = idx.sum(dim=1)  # logit: [N, H, W]
                    prediction[idx == 0] = 0  # set background (non-target class)
                elif self.microsegloss:
                    # MICROSEG
                    logit = torch.sigmoid(outputs)
                    logit[:, 1] += logit[:, 0]
                    prediction = logit[:, 1:].argmax(dim=1)
                    pseudo_label = torch.where(pseudo_label == 0, pseudo_label, pseudo_label - 1)
                else:
                    _, prediction = outputs.max(dim=1)
                prediction = prediction.cpu().numpy()
                self.prediction_vis(images, labels=None, pseudo_label=None, prediction=prediction, i=i, file_names=file_name)
