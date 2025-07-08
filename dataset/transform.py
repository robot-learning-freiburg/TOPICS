import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(opts, task, logger):
    """ Dataset And Augmentation
    """
    if opts.debug:
            opts.crop_size = (128, 128)
    elif opts.compute_att:
        opts.crop_size = (opts.crop_size, opts.crop_size)
    elif opts.dataset == "city":
            opts.crop_size = (512, opts.crop_size)
    elif opts.dataset == "mmor":
        if opts.crop_size == 512:
            opts.crop_size = (512, opts.crop_size)
        else:
            opts.crop_size = (768, opts.crop_size)
    elif opts.dataset == "synmedi":
        opts.crop_size = (512, opts.crop_size)
    elif opts.dataset == "endovis":
        opts.crop_size = (640, opts.crop_size)
    else:
        opts.crop_size = (opts.crop_size, opts.crop_size)

    logger.info(f"Using crop_size {opts.crop_size}.")

    if opts.debug:
        test_transform = A.Compose([
                A.Resize(*opts.crop_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    else:
            test_transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    if opts.dataset == "map":

        train_transform = A.Compose([
                A.LongestMaxSize(max_size=2177),
                A.CropNonEmptyMaskIfExists(*opts.crop_size, ignore_values=[task.internal_masking_value]),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

        if opts.crop_val:
                val_transform = A.Compose([
                    A.LongestMaxSize(max_size=2177),
                    A.CenterCrop(*opts.crop_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),

                ])
        else:
                # no crop, batch size = 1
                val_transform = A.Compose([
                    A.LongestMaxSize(max_size=2177),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])


    elif opts.dataset == "voc":
        train_transform = A.Compose([
                A.RandomResizedCrop(*opts.crop_size, scale=(0.5, 2.0)),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

        if opts.crop_val:
                val_transform = A.Compose([
                    A.Resize(*opts.crop_size),
                    A.CenterCrop(*opts.crop_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),

                ])
        else:
                val_transform = A.Compose([
                    A.Resize(*opts.crop_size),
                    A.CenterCrop(*opts.crop_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])

        test_transform = A.Compose([
                A.Resize(*opts.crop_size),
                A.CenterCrop(*opts.crop_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    elif opts.dataset == "city":
        train_transform = A.Compose([
                A.RandomResizedCrop(2*opts.crop_size[0], 2*opts.crop_size[1], scale=(0.5, 4.0)),
                A.CropNonEmptyMaskIfExists(*opts.crop_size, ignore_values=[task.internal_masking_value]),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

        if opts.crop_val:
                val_transform = A.Compose([
                    A.CenterCrop(*opts.crop_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),

                ])
        else:
                # no crop, batch size = 1
                val_transform = A.Compose([
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])

    elif opts.dataset == "mmor" or opts.dataset == "synmedi":
        train_transform = A.Compose([
                A.CropNonEmptyMaskIfExists(*opts.crop_size, ignore_values=[task.internal_masking_value]),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

        if opts.crop_val:
                val_transform = A.Compose([
                    A.CenterCrop(*opts.crop_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),

                ])
        else:
                val_transform = A.Compose([
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])

    elif opts.dataset == "endovis":
        
        train_transform = A.Compose([
            A.RandomCrop(*opts.crop_size, p=1.0),
            A.Affine(
                # Tuple for scale (will be used for both x and y)
                scale=(0.8, 1.2),
                # Dictionary with tuples for different x/y translations
                translate_percent={"x": (-0.2, 0.2), "y": (-0.1, 0.1)},
                # Tuple for rotation range
                rotate=(-30, 30),
                # Interpolation methods
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                # Other parameters
                fit_output=False,
                keep_ratio=True,
                rotate_method="largest_box",
                balanced_scale=True,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.7),
            A.ElasticTransform(alpha=1, sigma=50, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomGamma(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


        if opts.crop_val:
                val_transform = A.Compose([
                    #A.CenterCrop(*opts.crop_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),

                ])
        else:
                # no crop, batch size = 1
                val_transform = A.Compose([
                    A.Normalize(mean=[0.485, 0.456, 0.406],  
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])

    return train_transform, val_transform, test_transform, opts