import cv2
import random
import torch


def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


def paired_random_crop(img_ir, img_vis, gt_size):
    """Paired random crop for IR and Visible images.

    It crops IR and visible images with corresponding locations.

    Args:
        img_ir (list[ndarray] | ndarray | list[Tensor] | Tensor): Infrared images.
        img_vis (list[ndarray] | ndarray | list[Tensor] | Tensor): Visible images.
        gt_size (int): Crop patch size.

    Returns:
        tuple: Cropped IR and Visible images. If input is single image, return single image.
    """

    if not isinstance(img_ir, list):
        img_ir = [img_ir]
    if not isinstance(img_vis, list):
        img_vis = [img_vis]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_ir[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_ir, w_ir = img_ir[0].size()[-2:]
        h_vis, w_vis = img_vis[0].size()[-2:]
    else:
        h_ir, w_ir = img_ir[0].shape[0:2]
        h_vis, w_vis = img_vis[0].shape[0:2]

    # check if IR and Visible images have the same size
    if h_ir != h_vis or w_ir != w_vis:
        raise ValueError(f'Size mismatches. IR ({h_ir}, {w_ir}) and Visible ({h_vis}, {w_vis}) '
                         f'should have the same size.')
    
    if h_ir < gt_size or w_ir < gt_size:
        raise ValueError(f'Image size ({h_ir}, {w_ir}) is smaller than patch size '
                         f'({gt_size}, {gt_size}).')

    # randomly choose top and left coordinates for patch
    top = random.randint(0, h_ir - gt_size)
    left = random.randint(0, w_ir - gt_size)

    # crop patches
    if input_type == 'Tensor':
        img_ir = [v[:, :, top:top + gt_size, left:left + gt_size] for v in img_ir]
        img_vis = [v[:, :, top:top + gt_size, left:left + gt_size] for v in img_vis]
    else:
        img_ir = [v[top:top + gt_size, left:left + gt_size, ...] for v in img_ir]
        img_vis = [v[top:top + gt_size, left:left + gt_size, ...] for v in img_vis]
    
    # if only one image, return as single image instead of list
    if len(img_ir) == 1:
        img_ir = img_ir[0]
    if len(img_vis) == 1:
        img_vis = img_vis[0]
    
    return img_ir, img_vis


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img
