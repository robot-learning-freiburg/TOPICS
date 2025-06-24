import torch
import math
from functools import reduce
import torch.nn as nn
import geoopt
from inplace_abn import ABN


def compute_poincare_distance_matrix(features, offset, k, bias):
    dim = -1
    temp = 1e-3
    eps = torch.tensor(1e-15)

    def mobius_add_batch(p, z, w):
        z2 = z.pow(2).sum(dim=dim, keepdim=True)
        p2 = p.pow(2).sum(dim=dim, keepdim=True).permute(1, 0)  # keepdim=False
        w2 = torch.linalg.norm(w, axis=1)
        pw = (p * torch.nn.functional.normalize(w, p=2, dim=1)).sum(dim=dim)[None, None, None, :]

        pz = torch.einsum("kj,imnj->imnk", (p, z))  # dot product
        zw = torch.einsum("imnj,kj->imnk", (z, torch.nn.functional.normalize(w, p=2, dim=1)))  # dot product

        denom1 = 1 + 2 * k * pz + k ** 2 * z2 * p2
        a = (1 + 2 * k * pz + k * z2) / denom1
        b = (1 - k * p2) / denom1
        squared_norm = (a ** 2 * p2) + (2 * a * b * pz) + (b ** 2 * z2)

        maxnorm = (1.0 - temp) / k ** 0.5

        norm_projected = torch.where(squared_norm ** 0.5 < maxnorm, squared_norm,
                                     torch.ones_like(squared_norm) * maxnorm ** 2)

        project_normalized = torch.where(squared_norm ** 0.5 > maxnorm,
                                         (maxnorm / torch.maximum(squared_norm ** 0.5, eps)),
                                         torch.ones_like(squared_norm))
        nom = 2 * (k ** 0.5) * ((a * pw + b * zw) * project_normalized)
        denorm = (torch.maximum(1 - k * norm_projected, eps))
        res = (2.0 * w2 / k ** 0.5) * torch.asinh(nom / denorm)

        return res

    features = features.permute(0, 2, 3, 1)
    dist_mat = mobius_add_batch(-offset, features.double(), bias)
    dist_mat = dist_mat.permute(0, 3, 1, 2)
    return dist_mat


class IncrementalHyperbolicSegmentationModule(nn.Module):

    def __init__(
            self,
            body,
            classes,
            logger,
            embed_dim: int,
            input_dim: int,
            norm_act: str = "bn_sync",
            curv_init: float = 2.0,
            clipping: bool = False,
            head: object = None,
            adj_dict: object = None,
    ):

        super(IncrementalHyperbolicSegmentationModule, self).__init__()

        self.body = body
        self.head = head
        self.classes = classes

        self.embed_dim = embed_dim
        self.clipping = clipping
        self.offset = []
        self.normal = []
        self.norm_act = norm_act

        self._curv_minmax = {
            "max": curv_init * 10,
            "min": curv_init / 10,
        }
        if not self.clipping:
            self.visual_alpha = nn.Parameter(torch.tensor(embed_dim ** -0.5).log())
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

        if adj_dict:
            self.adj_dict = adj_dict
            hier_values = len(reduce(lambda xs, ys: xs + ys, list(self.adj_dict.values()))) if len(
                self.adj_dict) > 0 else 0
            tot_classes = len(reduce(lambda a, b: a + b, classes))
            no_hier = tot_classes - hier_values
            self.class_translate = torch.tensor(list(range(tot_classes)))
            for key, values in self.adj_dict.items():
                for value in values:
                    self.class_translate[value] = key
            print(self.class_translate)

        self.manifold = geoopt.PoincareBall(c=curv_init, learnable=False)
        for i, c in enumerate(classes):
            object_type_embds = torch.empty((len(c), embed_dim), dtype=torch.float64)
            torch.nn.init.constant_(object_type_embds, 0.)
            offset = geoopt.ManifoldParameter(object_type_embds, requires_grad=True, manifold=self.manifold)
            self.offset.append(offset)
            emds_normal = torch.empty((len(c), embed_dim), dtype=torch.float64).normal_(mean=0,
                                                                                            std=embed_dim ** -0.5 / math.sqrt(
                                                                                                curv_init))
            normal_vector = torch.nn.Parameter(emds_normal, requires_grad=True)
            logger.info(f"Curv init {curv_init}, {self.curv}")
            self.normal.append(normal_vector)

        self.offset = nn.ParameterList(self.offset)
        self.normal = nn.ParameterList(self.normal)

        if input_dim != embed_dim:
            self.input_proj = torch.nn.Conv2d(input_dim, embed_dim, kernel_size=1, bias=False)
            torch.nn.init.xavier_uniform_(self.input_proj.weight)
            self.proj = True
        else:
            self.proj = False
            logger.info("No input proj, head channels = embed dim!")

        self.delta = 0.
        self.counter = 0

    @property
    def curv_pointer(self):
        return self.manifold.c

    @property
    def curv(self):
        return self.manifold.c.data.detach().clone()

    def init_new_classifier(self):
        return

    def get_backbone_params(self):
        modules = [self.body]
        return self.get_module(modules)

    def get_head_params(self):
        if self.head is not None:
            modules = [self.head]
            return self.get_module(modules)
        return []

    def get_classifer_params(self):
        modules = [self.offset] + [self.normal]
        return modules

    def get_old_classifer_params(self):
        modules = [self.offset[i] for i in range(0, len(self.offset) - 1)] + [self.normal[i] for i in
                                                                              range(0, len(
                                                                                  self.normal) - 1)]
        return modules

    def get_new_classifer_params(self):
        modules = [self.offset[len(self.offset) - 1]] + [self.normal[len(self.normal) - 1]]
        return modules

    def get_module(self, modules, check_bn=True):
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
                elif check_bn:
                    if isinstance(m[1], (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                    elif self.norm_act == 'iabn_sync':
                        if isinstance(m[1], ABN):
                            for p in m[1].parameters():
                                if p.requires_grad:
                                    yield p

    def forward(self, img, features=False, proposal=None):

        out_size = img.shape[-2:]

        if self.head is not None:
            b_features = self.body(img)
            h_features = self.head(b_features)
        else:
            b_features, h_features = self.body(img)

        if not torch.isfinite(h_features).all():
            assert torch.isfinite(img).all()
            assert torch.isfinite(b_features).all()
            assert torch.isfinite(h_features).all()

        if self.proj:
            features_proj = self.input_proj(h_features)
        else:
            features_proj = h_features

        if not self.clipping:
            self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
            features_proj = features_proj * self.visual_alpha.exp()

        assert torch.isfinite(features_proj).all()

        with torch.cuda.amp.autocast(enabled=False):
            features_proj = self.manifold.expmap0(features_proj, project=True, dim=1)
            assert torch.isfinite(features_proj).all()
            out = []
            for i, (lvl_offset, lvl_norm) in enumerate(zip(self.offset, self.normal)):
                conformal_factor = 1 - self.curv_pointer * lvl_offset.pow(2).sum(dim=1, keepdim=True)
                lvl_norm = lvl_norm * conformal_factor
                out.append(compute_poincare_distance_matrix(features_proj, lvl_offset, self.manifold.c, lvl_norm))
            dist_mat = torch.cat(out, dim=1)

        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)

        cls_scores = dist_mat * self.logit_scale.exp()
        sem_logits = nn.functional.interpolate(cls_scores, size=out_size, mode="bilinear", align_corners=False)

        if features:
            return sem_logits, (b_features, features_proj, dist_mat)
        else:
            return sem_logits, None
