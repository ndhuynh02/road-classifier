import torch

def denormalize_batch(batch, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> torch.Tensor:
    # batch.shape = (Batch_size, channel, height, width)
    # 3, H, W, B
    ten = batch.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> torch.Tensor:
    return denormalize_batch(image.unsqueeze(0), mean, std).squeeze(0)