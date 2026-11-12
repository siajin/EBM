from pathlib import Path
import sys
root = Path(sys.argv[1])
sys.path.insert(0, str(root/'code/tests'))
from gpu_positive_smoke import worker
if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(str(root), 'checkpoint_shared'), nprocs=2, join=True)
