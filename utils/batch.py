import torch


class BatchUtil():

    @staticmethod
    def cat_list(images, fill_value=0):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        batch_shape = (len(images),) + max_size
        batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
        return batched_imgs

    @staticmethod
    def segmentation_collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = BatchUtil.cat_list(images, fill_value=0)
        batched_targets = BatchUtil.cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

    @staticmethod
    def collate_fn(batch):
        result = [(b, c) for (b, c), a in batch]
        files = [(a, d) for (b, c), (a, d) in batch]

        if isinstance(result[0][1], torch.Tensor):
            b = BatchUtil.segmentation_collate_fn(result), files
            return b
        else:
            return tuple(zip(*result)), files
