import torch
import math
import random

def compute_entropy(probs):
    """Computes the entropy per pixel.

        # References:
            * ESL: Entropy-guided Self-supervised Learning for Domain Adaptation in Semantic Segmentation
              Saporta et al.
              CVPR Workshop 2020

        :param probabilities: Tensor of shape (b, c, w, h).
        :return: One entropy per pixel, shape (b, w, h)
        """
    factor = 1 / math.log(probs.shape[1] + 1e-8)
    return -factor * torch.mean(probs * torch.log(probs + 1e-8), dim=1)

class ComputePseudoLabel:
    "Adopted from: https://github.com/arthurdouillard/CVPR2021_PLOP"
    def __init__(self, pseudo_type, masking_value, tot_classes, threshold, device=None, thresholds=None,
                 max_entropy=None, 
                 compute_classif_adaptive_factor=False,
                 internal_masking_value=255, hyp_hier=False, hier_matrices=None):
        self.type = pseudo_type
        self.masking_value = masking_value
        self.internal_masking_value = internal_masking_value
        self.device = device
        self.thresholds = thresholds
        self.max_entropy = max_entropy
        self.compute_classif_adaptive_factor = compute_classif_adaptive_factor
        self.tot_classes = tot_classes
        self.threshold = threshold
        self.hyp_hier = hyp_hier
        if self.type.startswith("hiersigmoid_"):
            self.ancestor_matrix = hier_matrices[4].double().to(device)
            self.helper_tree = torch.eye(hier_matrices[3].shape[1]).double().to(device)
            for i in hier_matrices[5]:
                if i < self.helper_tree.shape[1]:
                    self.helper_tree[i, i] = 0

    def __call__(self, labels, outputs_old, val=True):
        masked_area = labels == self.internal_masking_value
        classif_adaptive_factor = 1.0
        mask_valid_pseudo = None
        sample_weights = None
        if not val:
            valid_class = [x for x in torch.unique(labels) if (x != self.internal_masking_value and x != self.masking_value)]
            if len(valid_class) > 0:
                labels[masked_area] = self.masking_value
                return labels, classif_adaptive_factor, mask_valid_pseudo, sample_weights
        if self.type == "naive":
            labels[masked_area] = outputs_old.argmax(dim=1)[masked_area]
        elif self.type.startswith("threshold_"):
            threshold = float(self.type.split("_")[1]) * 0.1
            probs = torch.softmax(outputs_old, dim=1)
            pseudo_labels = probs.argmax(dim=1)
            pseudo_labels[probs.max(dim=1)[0] < threshold] = self.masking_value
            labels[masked_area] = pseudo_labels[masked_area]
        elif self.type == "confidence":
            probs_old = torch.softmax(outputs_old, dim=1)
            labels[masked_area] = probs_old.argmax(dim=1)[masked_area]
            sample_weights = torch.ones_like(labels).to(self.device, dtype=torch.float32)
            sample_weights[masked_area] = probs_old.max(dim=1)[0][masked_area]
        elif self.type == "median":
            probs = torch.softmax(outputs_old, dim=1)
            max_probs, pseudo_labels = probs.max(dim=1)
            pseudo_labels[max_probs < self.thresholds[pseudo_labels]] = self.masking_value
            labels[masked_area] = pseudo_labels[masked_area]
        elif (self.type == "entropy") or (self.type == "entropybkg"):
            if self.hyp_hier:
                probs = torch.clamp(outputs_old, min=1e-15)
            else:
                probs = torch.softmax(outputs_old, dim=1)

            max_probs, pseudo_labels = probs.max(dim=1)

            mask_valid_pseudo = (compute_entropy(probs) / self.max_entropy) < self.thresholds[pseudo_labels]

            # All old labels that are NOT confident enough to be used as pseudo labels:
            if self.type == "entropybkg":  # bkg = 0 if not confident rather than ignored.
                mask_valid_pseudo[pseudo_labels == self.internal_masking_value] = True
            labels[~mask_valid_pseudo & masked_area] = self.masking_value
            # All old labels that are confident enough to be used as pseudo labels:
            labels[mask_valid_pseudo & masked_area] = pseudo_labels[mask_valid_pseudo & masked_area]

            if self.compute_classif_adaptive_factor:
                # Number of old/bg pixels that are certain
                num = (mask_valid_pseudo & masked_area).float().sum(dim=(1, 2))
                # Number of old/bg pixels
                den = masked_area.float().sum(dim=(1, 2))
                # If all old/bg pixels are certain the factor is 1 (loss not changed)
                # Else the factor is < 1, i.e. the loss is reduced to avoid
                # giving too much importance to new pixels
                classif_adaptive_factor = num / (den + 1e-6)
                classif_adaptive_factor = classif_adaptive_factor[:, None, None]
                classif_adaptive_factor = classif_adaptive_factor.clamp(min=0.0)
        elif self.type.startswith("sigmoid_"):
            pred_prob = torch.sigmoid(outputs_old).detach()
            threshold = float(self.type.split("_")[1]) * 0.1
            if threshold > 1:
                threshold *= 0.1
            pseudo_labels = outputs_old.argmax(dim=1)  # pred: [N. H, W]
            idx = (pred_prob > threshold).float()  # logit: [N, C, H, W], all classes are below threshold!
            idx = idx.sum(dim=1)  # logit: [N, H, W]
            pseudo_labels[
                idx == 0] = self.masking_value  # set background (non-target class), if no other class is set to true
            labels[masked_area] = pseudo_labels[masked_area]
        elif self.type.startswith("hiersigmoid_"):
            pred_prob = torch.sigmoid(outputs_old).detach()
            begin_threshold = float(self.type.split("_")[1]) * 0.1
            if begin_threshold > 1:
                begin_threshold *= 0.1

            logits_wo_ancestors = torch.tensordot(outputs_old, self.helper_tree, dims=([1], [1])).permute(0, 3, 1, 2)
            pred_prob_wo_ancestors = torch.tensordot(pred_prob, self.helper_tree, dims=([1], [1])).permute(0, 3, 1, 2)
            pseudo_labels = logits_wo_ancestors.argmax(dim=1)  
            idx = (pred_prob_wo_ancestors > begin_threshold).float()  # logit: [N, C, H, W], all classes are below threshold!
            idx = idx.sum(dim=1)  # logit: [N, H, W]
            pseudo_labels[idx == 0] = self.masking_value 

            for class_idx in range(C):
                ancestor_ids = torch.where(self.ancestor_matrix[class_idx] == 1)[0]
                full_hierarchy = [i.item() for i in reversed(ancestor_ids)]
                for level, cls_id in enumerate(full_hierarchy):
                    if level == 0:
                        threshold = begin_threshold
                    elif level == 1:
                        threshold = begin_threshold + 0.2
                    elif level == 2:
                        threshold = begin_threshold + 0.2
                    else:
                        continue  

                    mask = (pred_prob[:, cls_id, :, :] >= threshold) & (pseudo_labels == self.masking_value)
                    pseudo_labels[mask] = cls_id

            labels[masked_area] = pseudo_labels[masked_area]


        elif self.type.startswith("microseg"):
            pred_prob = torch.sigmoid(outputs_old).detach()
            pred_scores, pred_labels = torch.max(pred_prob, dim=1)
            labels = torch.where((labels <= 1) & (pred_labels > 1) & (pred_scores >= 0.7),
                                 pred_labels, labels)
        else:
            raise ValueError(f"Unknown type of pseudo_labeling={self.type}")

        return labels, classif_adaptive_factor, mask_valid_pseudo, sample_weights
    
    
    def find_median(self, train_loader, logger, model_old):
    
        """Find the median prediction score per class with the old model.

            Computing the median naively uses a lot of memory, to allievate it, instead
            we put the prediction scores into a histogram bins and approximate the median.

            https://math.stackexchange.com/questions/2591946/how-to-find-median-from-a-histogram
            """
        if (self.type == "entropy") or (self.type == "entropybkg"):
            mode = "entropy"
            max_value = torch.log(torch.tensor(self.tot_classes).float().to(self.device))
            nb_bins = 100
        elif self.type == "median":
            mode = "probability"
            max_value = 1.0
            nb_bins = 20  # Bins of 0.05 on a range [0, 1]
        else:
            return

        histograms = torch.zeros(self.tot_classes, nb_bins).long().to(self.device)

        for cur_step, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)

            outputs_old, features_old = model_old(images)

            mask_bg = labels == self.internal_masking_value
            probas = torch.softmax(outputs_old, dim=1)

            max_probas, pseudo_labels = probas.max(dim=1)
            if mode == "entropy":
                values_to_bins = compute_entropy(probas)[mask_bg].view(-1) / max_value
            else:
                values_to_bins = max_probas[mask_bg].view(-1)

            x_coords = pseudo_labels[mask_bg].view(-1)
            y_coords = torch.clamp((values_to_bins * nb_bins).long(), max=nb_bins - 1)

            histograms.index_put_(
                (x_coords, y_coords),
                torch.LongTensor([1]).expand_as(x_coords).to(histograms.device),
                accumulate=True
            )

            if cur_step % 10 == 0:
                logger.info(f"Median computing {cur_step}/{len(train_loader)}.")

        thresholds = torch.zeros(self.tot_classes, dtype=torch.float32).to(self.device)
        # zeros or ones? If old_model never predict a class it may be important

        logger.info("Approximating median")
        for c in range(self.tot_classes):
            total = histograms[c].sum()
            if total <= 0.:
                continue

            half = total / 2
            running_sum = 0.
            for lower_border in range(nb_bins):
                lower_border = lower_border / nb_bins
                bin_index = int(lower_border * nb_bins)
                if half >= running_sum and half <= (running_sum + histograms[c, bin_index]):
                    break
                running_sum += lower_border * nb_bins

            median = lower_border + ((half - running_sum) /
                                     histograms[c, bin_index].sum()) * (1 / nb_bins)

            thresholds[c] = median

        base_threshold = self.threshold
        if "_" in mode:
            mode, base_threshold = mode.split("_")
            base_threshold = float(base_threshold)

        if (mode == "entropy"):
            for c in range(len(thresholds)):
                thresholds[c] = max(thresholds[c], base_threshold)
        else:
            for c in range(len(thresholds)):
                thresholds[c] = min(thresholds[c], base_threshold)
        logger.info(f"Finished computing median {thresholds}, {max_value}")
        self.thresholds = thresholds.to(self.device)
        self.max_entropy = max_value

        return [self.thresholds, self.max_entropy]