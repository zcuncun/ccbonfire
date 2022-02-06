from torch.utils.data import DataLoader
import torch.distributed as dist

# class BatchLoader(DataLoader):
#     def __init__(self, dataset,shuffle=False, sampler=None,
#            batch_sampler=None, num_workers=0, collate_fn=None,
#            pin_memory=False, drop_last=False, timeout=0,
#            worker_init_fn=None, *, prefetch_factor=2,
#            persistent_workers=False):
#         super(BatchLoader, self).__init__(dataset, batch_size=1, shuffle=False, sampler=None,
#            batch_sampler=None, num_workers=0, collate_fn=None,
#            pin_memory=False, drop_last=False, timeout=0,
#            worker_init_fn=None, prefetch_factor=2,
#            persistent_workers=False)


class Loader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        prefetch_factor=2,
        persistent_workers=False,
    ):
    if dist.is_initialized():

        super(Loader, self).__init__(
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            collate_fn=None,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=2,
            persistent_workers=False,
        )


def test():
    pass


if __name__ == "__main__":
    test()
