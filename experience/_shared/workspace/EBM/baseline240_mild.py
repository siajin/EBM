"""Replace only the audio source of the archived240 TrainSet; preserve its views."""
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate


@dataclass
class TrainRow:
    long: np.ndarray
    short: np.ndarray
    label: int
    metadata: dict


def original_views(original_dataset, index, wave):
    """Run the byte-preserved original __getitem__, including both RawBoost draws."""
    sf_module=original_dataset.__getitem__.__func__.__globals__['sf']
    reader=sf_module.read
    calls=[]
    expected=Path(original_dataset.items[index].path)
    def cached_read(path,*args,**kwargs):
        if Path(path)!=expected or args or kwargs:
            raise ValueError('Unexpected original dataset read')
        calls.append(str(path))
        return wave.copy(),16000
    try:
        sf_module.read=cached_read
        result=original_dataset[index]
    finally:
        sf_module.read=reader
    if len(calls)!=2:
        raise ValueError('Original240 must read the same source for long and short')
    if len(result[0])!=64600 or len(result[1])!=48440:
        raise ValueError('Original240 fixed long/short sizes changed')
    if not np.isfinite(result[0]).all() or not np.isfinite(result[1]).all():
        raise ValueError('Original augmentation produced nonfinite audio')
    return result


class MildOriginal240TrainSet(Dataset):
    def __init__(self,args,items,original_class,mild_dataset,root):
        self.items=list(items)
        self.original=original_class(args,self.items)
        self.mild=mild_dataset
        self.root=str(root);self.args=args;self.epoch=0;self.visits={}
        mapping={Path(item.path).stem:i for i,item in enumerate(mild_dataset.items)}
        self.indices=[mapping[Path(item.path).stem] for item in self.items]

    def __len__(self):return len(self.items)

    def __getitem__(self,index):
        global_index=self.indices[index]
        visit=self.visits.get(global_index,0);self.visits[global_index]=visit+1
        worker_seed=torch.initial_seed()
        spec=self.mild.crop_spec(global_index,[self.args['rand_seed'],self.args['rank'],self.epoch,
            global_index,visit,worker_seed%(2**32),worker_seed//(2**32)])
        wave=self.mild.read_crop(spec,augment=False).numpy().astype(np.float64)
        long,short,label=original_views(self.original,index,wave)
        v=self.mild.records[global_index]['variants'][spec.variant]
        metadata={'epoch':self.epoch,'rank':self.args['rank'],'index':global_index,
            'uid':Path(self.items[index].path).stem,'label':int(label),'variant':spec.variant,
            'selected_samples':len(wave),'speed_rate':v['rate'],'source_start':v['start'],
            'source_kept_samples':v['kept_samples'],'long_repeated':int(len(wave)<64600),
            'short_repeated':int(len(wave)<48440)}
        return TrainRow(long,short,label,metadata)


def collate_rows(rows):
    return (*default_collate([(r.long,r.short,r.label) for r in rows]),[r.metadata for r in rows])


class ConsumedIterator:
    def __init__(self,iterator,dataset):self.iterator=iterator;self.dataset=dataset
    def __iter__(self):return self
    def __len__(self):return len(self.iterator)
    def __next__(self):
        long,short,label,metadata=next(self.iterator)
        # Record consumed batches only: the original small-loader recycling can
        # prefetch batches which are never used for an optimizer update.
        path=Path(self.dataset.root)/f'consumed_epoch{self.dataset.epoch}_rank{self.dataset.args["rank"]}.csv'
        exists=path.exists()
        with path.open('a',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(metadata[0]))
            if not exists:writer.writeheader()
            writer.writerows(metadata)
        return long,short,label


class ConsumedDataLoader(DataLoader):
    def __init__(self,dataset,*args,**kwargs):
        if isinstance(dataset,MildOriginal240TrainSet):kwargs['collate_fn']=collate_rows
        super().__init__(dataset,*args,**kwargs)
    def __iter__(self):
        iterator=super().__iter__()
        return ConsumedIterator(iterator,self.dataset) if isinstance(self.dataset,MildOriginal240TrainSet) else iterator
