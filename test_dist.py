import os
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

def worker(rank, world_size):
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:29500',
                            rank=rank, world_size=world_size)

    print(f"[Rank {rank}] init done")
    dist.barrier()
    if rank == 0: print("Barrier passed")

    # 广播一个小张量测试
    x = torch.tensor([rank], device=rank)
    dist.broadcast(x, src=0)
    print(f"[Rank {rank}] broadcast x = {x.item()}")

if __name__ == '__main__':
    world_size = 2
    spawn(worker, args=(world_size,), nprocs=world_size)

