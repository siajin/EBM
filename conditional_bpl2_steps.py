"""Exact embedding-gradient replay for original OC/BPL plus conditional paired BPL2."""
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from native_steps import features,unwrap,_freeze

def objective(model,z,batch,teacher5):
    labels=batch['labels'].to(z['long'].device);n=batch['bona']
    assert int(labels.sum())==n and z['student'].shape[0]==n
    oc=model.loss(z['long'],labels,logical_batch_size=len(labels))[0]
    legacy=model.loss_bpl(z['long'][:n],z['short'][:n])
    reference=z['long'][:n].detach().clone()
    if batch['teacher5_indices']:
        reference[batch['teacher5_indices']]=teacher5.detach()
    similarity=F.cosine_similarity(reference,z['student'],dim=1)
    paired=(F.softplus(model.loss_bpl.alpha*(1-similarity))-.3).mean()
    total=oc+.5*legacy+.5*paired
    return total,{'oc':float(oc.detach()),'bpl':float(legacy.detach()),'bpl2':float(paired.detach()),
                  'loss1':float(total.detach()),'legacy_pairs':n*n,'BPL2_pairs':n,
                  'pair54':len(batch['teacher5_indices']),'pair43':n-len(batch['teacher5_indices'])}

def train_step(trainer,batch):
    classifier=trainer.classifier;model=unwrap(classifier);device=trainer.args['device'];micro=8
    cuda=torch.device(device).type=='cuda';devices=[torch.device(device).index] if cuda else []
    _freeze(model,False);trainer.optimizer.zero_grad(set_to_none=True)
    teacher=[]
    for start in range(0,len(batch['teacher5']),micro):
        x,mask=features(trainer.plm,batch['teacher5'][start:start+micro],device)
        with torch.no_grad():teacher.append(model(x,mask=mask,return_embedding=True).detach())
        del x,mask
    teacher5=torch.cat(teacher) if teacher else None
    collected={};records=[]
    for name in ['long','short','student']:
        chunks=[]
        for start in range(0,len(batch[name]),micro):
            x,mask=features(trainer.plm,batch[name][start:start+micro],device)
            cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state(device) if cuda else None
            with torch.no_grad():chunks.append(model(x,mask=mask,return_embedding=True).detach())
            records.append((name,start,cpu,gpu));del x,mask
        collected[name]=torch.cat(chunks).requires_grad_(True)
    loss,stats=objective(model,collected,batch,teacher5)
    deriv=torch.autograd.grad(loss,tuple(collected.values()));grad=dict(zip(collected,deriv));del loss,collected
    for number,(name,start,cpu,gpu) in enumerate(records):
        sync=classifier.no_sync() if hasattr(classifier,'no_sync') and number+1<len(records) else nullcontext()
        with sync,torch.random.fork_rng(devices=devices):
            x,mask=features(trainer.plm,batch[name][start:start+micro],device)
            torch.set_rng_state(cpu)
            if cuda:torch.cuda.set_rng_state(gpu,device)
            z=classifier(x,mask=mask,return_embedding=True);z.backward(grad[name][start:start+micro])
        del x,mask,z
    trainer.optimizer.step();del grad
    _freeze(model,True);trainer.optimizer.zero_grad(set_to_none=True);loss2=0.
    n=batch['bona']
    for start in range(0,n,micro):
        stop=min(start+micro,n)
        sync=classifier.no_sync() if hasattr(classifier,'no_sync') and stop<n else nullcontext()
        with sync:
            x,mask=features(trainer.plm,batch['long'][start:stop],device)
            value,_=classifier(x,mask=mask,label=torch.ones(stop-start,device=device,dtype=torch.long),logical_batch_size=n)
            value=value*((stop-start)/n);value.backward();loss2+=float(value.detach())
        del x,mask,value
    trainer.optimizer.step();stats.update(loss2=loss2,optimizer_steps=2,center_updates=1)
    assert all(p.grad is None for p in trainer.plm.parameters())
    return stats
