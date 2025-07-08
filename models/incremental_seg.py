import torch
import torch.nn as nn
import torch.nn.functional as functional
from inplace_abn import ABN

class IncrementalSegmentationModule(nn.Module):

    def __init__(self, body, head, head_channels, classes, plop=False, dkd=False, microseg=False,
                 unseen_cluster=0, embed_dim=128,
                 norm_act="bn_sync"):
        super(IncrementalSegmentationModule, self).__init__()
        self.body = body
        self.head = head
        self.plop = plop
        self.dkd = dkd
        self.microseg = microseg

        self.classes = classes
        self.freeze_all_bn = False
        # classes must be a list where [n_class_task[i] for i in tasks]
        assert isinstance(classes, list), \
            "Classes must be a list where to every index correspond the num of classes for that task"
        self.cls = nn.ModuleList(
            [nn.Conv2d(embed_dim, len(c), kernel_size=1, stride=1, padding=0, bias=True) for c in classes])


        if microseg:
            self.unseen_cluster = unseen_cluster
            classes.insert(0, list(range(unseen_cluster)))
            classes.insert(1, [0])
            classes[2] = classes[2][1:] # remove bkg
            print(f"Updated classes, {classes}")
            self.cls = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
                        nn.BatchNorm2d(embed_dim),
                        nn.ReLU(inplace=True),
                    ) for _ in classes]
            )

            self.head2 = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(embed_dim, len(c), 1, bias=True)  # True
                ) for c in classes]
            )
            self.proposal_head = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim, len(c), 1, bias=True)
                ) for c in classes]
            )
            self.set_bn_momentum()

        self.norm_act = norm_act
        self.proj = False
        if embed_dim != head_channels:
            print("Channels are projected!")
            self.proj = True
            self.input_proj = torch.nn.Conv2d(head_channels, embed_dim, kernel_size=1, bias=False)
            torch.nn.init.xavier_uniform_(self.input_proj.weight)

    def init_new_classifier(self):
        cls = self.cls[-1]
        imprinting_w = self.cls[0].weight[0]
        bkg_bias = self.cls[0].bias[0]

        bias_diff = torch.log(torch.FloatTensor([len(self.classes[-1]) + 1]))
        new_bias = (bkg_bias - bias_diff)

        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)
        self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def string(self):
        """
        """
        string = str(self.body) + '\n' + str(self.head) + '\n' + str(self.cls)
        return string

    def forward(self, x, features=False, proposal=None):

        out_size = x.shape[-2:]
       
        if self.head is not None:
             # backbone
            features_body = self.body(x)
            if self.plop:
                features_head = self.head(features_body[0])
                features_body = features_body[1]
            else:
                features_head = self.head(features_body)
        else:
            features_body, features_head = self.body(x)

        assert torch.isfinite(features_head).all()

        if self.proj:
            features_head = self.input_proj(features_head)

        # final classification layer (increments)
        out = []

        for i, mod in enumerate(self.cls):
            if i == 0 and self.dkd:
                out.append(mod(features_head.detach()))
            else:
                out.append(mod(features_head))

        if self.microseg:
            x, sem_logits_raw = self.mix_features_proposals(out, features_head, proposal)
            x = torch.einsum('bnc,bnhw->bchw', x, proposal)
            x = functional.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        else:
            sem_logits_raw = torch.cat(out, dim=1)
        
        # interpolate final logits
        sem_logits = functional.interpolate(sem_logits_raw, size=out_size, mode="bilinear", align_corners=False)

        if features:
            if self.plop:
                if self.head is None:
                    features_head = [features_head]
                return sem_logits, (features_body + [features_head], None, sem_logits_raw)
            elif self.dkd:
                sem_neg_logits_small = self.forward_class_prediction_negative(features_head)
                sem_pos_logits_small = self.forward_class_prediction_positive(features_head)
                return sem_logits, (sem_pos_logits_small, sem_neg_logits_small)
            elif self.microseg:
                unseen_clust = x[:, :self.unseen_cluster]
                x = torch.cat([torch.sum(unseen_clust, dim=1, keepdim=True),
                               x[:, self.unseen_cluster:, ]], dim=1)

                unseen_clust_pixel = sem_logits[:, :self.unseen_cluster]
                sem_logits = torch.cat([torch.sum(unseen_clust_pixel, dim=1, keepdim=True),
                                        sem_logits[:, self.unseen_cluster:, ]], dim=1)
                return x, (unseen_clust, sem_logits)

            return sem_logits, (features_body, features_head, sem_logits_raw)

        return sem_logits, None
    
    def att_map(self, x):
        # sptial attention, from AWT + MIB https://github.com/dfki-av/AWT-for-CISS
        a = torch.sum(x ** 2, dim=1).detach()
        # channel attention
        for i in range(a.shape[0]):
            a[i] = a[i] / torch.norm(a[i])
        a = torch.unsqueeze(a, 1)
        x = a * x
        return x

    def set_bn_momentum(self):
        for m in self.body.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.01

    def freeze_bn_dropout(self, affine_freeze=False):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if affine_freeze:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
            elif self.norm_act == 'bn_sync':
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
            elif self.norm_act == 'iabn_sync':
                if isinstance(m, (ABN, nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()

        if not affine_freeze:
            self.freeze_all_bn = True

    def init_novel_classifier(self):
        # Initialize novel classifiers using an auxiliary classifier
        cls = self.cls[-1]  # New class classifier
        for i in range(len(self.classes[-1])):
            cls.weight[i:i + 1].data.copy_(self.cls[0].weight)
            cls.bias[i:i + 1].data.copy_(self.cls[0].bias)

    def init_classifier_awt(self, imp_c):
        cls = self.cls[-1]
        imprinting_w = self.cls[0].weight[0]  # bkg
        cls.weight.data += imprinting_w * (imp_c.detach().cpu()).float()
        print('Selected bkg classifier weights added to new classifier weights', imp_c.sum())

    def forward_class_prediction_negative(self, x_pl):
        # x_pl: [N, C, H, W], DKD from https://github.com/cvlab-yonsei/DKD
        out = []
        for i, mod in enumerate(self.cls):
            if i == 0:
                continue
            w = mod.weight  # [|C|, c]
            w = w.where(w < 0, torch.zeros_like(w, device=w.device))
            out.append(torch.matmul(x_pl.permute(0, 2, 3, 1), w.T).permute(0, 3, 1, 2))  # [N, |C|, H, W]
        x_o = torch.cat(out, dim=1)  # [N, |Ct|, H, W]
        return x_o

    def forward_class_prediction_positive(self, x_pl):
        # x_pl: [N, C, H, W], DKD from https://github.com/cvlab-yonsei/DKD
        out = []
        for i, mod in enumerate(self.cls):
            if i == 0:
                continue
            w = mod.weight  # [|C|, c]
            w = w.where(w > 0, torch.zeros_like(w, device=w.device))
            out.append(torch.matmul(x_pl.permute(0, 2, 3, 1), w.T).permute(0, 3, 1, 2))  # [N, |C|, H, W]
        x_o = torch.cat(out, dim=1)  # [N, |Ct|, H, W]
        return x_o

    def mix_features_proposals(self, out, features_head, proposal):
        # MicroSeg from https://github.com/zkzhang98/MicroSeg
        heads = []
        for i, h in enumerate(self.head2):
            heads.append(h(out[i]))
        heads = torch.cat(heads, dim=1)

        proposal = (functional.interpolate(
            input=proposal.float(), size=(features_head.shape[2], features_head.shape[3]),
            mode='nearest')).float()
        PPs = []
        for i in range(proposal.shape[1]):
            PF = features_head * (proposal[:, i].unsqueeze(dim=1))  # B C H W
            PP = torch.sum(PF, dim=[2, 3]) / (
                    torch.sum(proposal[:, i], dim=[1, 2]).unsqueeze(dim=1) + 1e-7)
            PPs.append(PP)
        del PF, PP

        PPs = torch.stack(PPs, dim=1)
        B_, N_, C_ = PPs.shape
        PPs = PPs.view(B_ * N_, -1)
        PPs = PPs.unsqueeze(-1).unsqueeze(-1)

        cl = [ph(PPs) for ph in self.proposal_head]

        cl = torch.cat(cl, dim=1)
        cl = cl.view(B_, N_, cl.shape[1])

        return cl, heads

    def get_backbone_params(self):
        modules = [self.body]
        return self.get_module(modules)

    def get_head_params(self):
        if self.head is not None:
            modules = [self.head]
            return self.get_module(modules)
        return []

    def get_classifer_params(self):
        modules = [self.cls]
        if self.microseg:
            modules = [self.cls, self.head2, self.proposal_head]
            return self.get_module(modules, check_bn=True)
        return self.get_module(modules, check_bn=False)

    def get_old_classifer_params(self):
        modules = [self.cls[i] for i in range(0, len(self.cls) - 1)]
        return self.get_module(modules, check_bn=False)

    def get_old_proposal_params(self):
        modules = [self.proposal_head[0], self.proposal_head[1]]
        return self.get_module(modules, check_bn=True)

    def get_unknown_params(self):
        modules = [self.cls[0], self.head2[0]]
        return self.get_module(modules, check_bn=True)

    def get_bkg_params(self):
        modules = [self.cls[1], self.head2[1]]
        return self.get_module(modules, check_bn=True)

    def get_new_classifer_params(self):
        modules = [self.cls[len(self.cls) - 1]]
        if self.microseg:
            modules = [self.cls[-1], self.head2[-1], self.proposal_head[-1]]
            return self.get_module(modules, check_bn=True)
        return self.get_module(modules, check_bn=False)

    def get_module(self, modules, check_bn=True):
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
                elif check_bn and (not self.freeze_all_bn):
                    if isinstance(m[1], (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                    elif self.norm_act == 'iabn_sync':
                        if isinstance(m[1], ABN):
                            for p in m[1].parameters():
                                if p.requires_grad:
                                    yield p
