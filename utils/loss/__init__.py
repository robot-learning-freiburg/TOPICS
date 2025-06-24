from .distill_loss import MaskKnowledgeDistillationLoss, KnowledgeDistillationLoss, UnbiasedKnowledgeDistillationLoss, IcarlLoss
from .topics_reg import TripletLoss, DistanceSim
from .topics_hier import HIERALoss
from .dice import DiceCrossEntropyLoss
from .other_hier import HierarchicalLoss, HBCELoss
from .plop import entropy, features_distillation
from .loss import FocalLoss, MaskCrossEntropy,  UnbiasedCrossEntropy, BCEWithLogitsLossWithIgnoreIndex, get_loss, SoftCrossEntropy
from .dkd_loss import ACLoss, KDLoss, WBCELoss
from .microseg import UnseenAugLoss
