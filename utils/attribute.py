import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import LayerIntegratedGradients



def compute_attribute(train_loader, logger, model_old, opts, new_cls, device):

    def prev(inp):
        out = model_old(inp)[0]
        return out.sum(dim=(2, 3))


    torch.cuda.empty_cache()
    lig = LayerIntegratedGradients(forward_func=prev, layer=model_old.cls[0]) # TODO multi-GPU solution
    print('bkg cls ', model_old.cls[0].weight[0].data.shape)

    gc.collect()
    attr = []
    miss = 0
    imp_c = None
    if opts.dataset == 'ade':
        n_step = 30
    else:
        n_step = 50

    break_map = True if ((len(new_cls) > 50) and (opts.dataset == "map")) else False

    for cur_step, (images, labels) in enumerate(train_loader):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        for id, (label, image) in enumerate(zip(labels, images)):

            # add batch dim
            image = image[None, :]
            label = label[None, :]

            if opts.mask_att:
                masks = []
                for m in new_cls:
                    masks.append(label == m)
                mask = masks[0]
                for it in range(1, len(masks)):
                    mask = mask | masks[it]

                mask = F.interpolate(mask[None, :].float(), size=([32, 32]), mode="nearest")

                mask = mask.expand([1, 256, 32, 32])

                if mask[0][0].sum() < 1:
                    miss += 1
                    continue

            torch.cuda.empty_cache()
            del masks
            del label
            with torch.no_grad():
                attribution = lig.attribute(image, target=0, n_steps=n_step, attribute_to_layer_input=True)
            attr.append(attribution[0] * mask)

            del image
            del mask
            del attribution

            torch.cuda.empty_cache()
            gc.collect()
        if cur_step % 100 == 0:
            logger.info(f"Atrribute compute for {cur_step}/{len(train_loader)} compute!")
        if cur_step > 1999 and opts.dataset == 'ade' and opts.task != "50":
            break
        if opts.task == "50" and cur_step > 4999:
            break
        if cur_step > 4999 and break_map:
            break

    logger.info(f'Number of missed images {miss}.')
    att = torch.cat(attr, dim=0)
    logger.info(f'Att {att.shape}.')

    att = torch.mean(att, dim=0)
    logger.info(f'Avg Att {att.shape}')
    att = nn.MaxPool2d(32)(att)
    logger.info(f'Avg Max Pool {att.shape}')

    top = int(opts.att)
    logger.info(f'Top weights {top}')
    if top != 0:
        if top == 10:
            high = 26
        elif top == 25:
            high = 64
        elif top == 50:
            high = 128
        elif top == 75:
            high = 192
        else:
            print("top not configured!")
            exit()
        top_att = torch.topk(att.squeeze(), high)[1]

        imp_c = torch.zeros_like(att)
        imp_c[top_att] = 1
        imp_c = imp_c > 0

    return imp_c



