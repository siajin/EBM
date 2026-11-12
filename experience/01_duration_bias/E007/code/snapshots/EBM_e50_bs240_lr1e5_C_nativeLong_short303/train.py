import torch
import torch.distributed as dist
from tqdm import tqdm
from sklearn import metrics
import numpy as np
from native_steps import features, train_batch, unwrap


class ModelTrainer:
    logger = None
    da = None

    def run(self):
        for epoch in range(1,self.args['epoch']+1):
            self.train(epoch)
            if epoch % 5 == 0 or epoch == self.args['epoch']:
                self.evaluation(epoch)
                if self.logger is not None:
                    self.logger.save_model(f'CLS_{epoch}',self.classifier.state_dict())
                    self.logger.save_model(f'TrainState_{epoch}',{
                        'epoch':epoch,'classifier':self.classifier.state_dict(),
                        'optimizer':self.optimizer.state_dict(),'scheduler':self.lr_scheduler.state_dict(),
                        'args':{k:v for k,v in self.args.items() if not any(t in k.lower() for t in ['token','password','secret'])}})
            self._synchronize()

    def train(self,epoch):
        self.plm.eval()
        self.plm.requires_grad_(False)
        self.classifier.train()
        self.train_sampler.set_epoch(epoch)
        with tqdm(self.train_loader,ncols=90,disable=not self.args['flag_parent']) as progress:
            for batch in progress:
                values = train_batch(self,batch)
                if self.logger is not None:
                    for name,value in values.items():
                        self.logger.log_metric(name,value)
                progress.set_description(f"epoch{epoch}: loss1={values['loss1']:.3f} loss2={values['loss2']:.3f}")
        self.lr_scheduler.step()

    def evaluation(self,epoch,final=False,avg=False,eval_only=False):
        # Target sets are available for explicit test(), never checkpoint selection.
        return self.test('V',epoch)

    def test(self,phase,epoch):
        self.plm.eval()
        self.classifier.eval()
        loader = self.val_loader if phase == 'V' else self.eval_loader[phase]
        model = unwrap(self.classifier)
        rows = []
        with torch.no_grad():
            for batch in loader:
                x,mask = features(self.plm,batch['waves'],self.args['device'])
                _,scores = model(x,mask=mask)
                rows.extend((int(i),float(s),int(y)) for i,s,y in
                            zip(batch['indices'],scores.flatten().cpu(),batch['labels']))
        if dist.is_initialized():
            gathered = [None]*dist.get_world_size()
            dist.all_gather_object(gathered,rows)
            rows = [row for part in gathered for row in part]
        rows.sort()
        if [r[0] for r in rows] != list(range(len(loader.dataset))):
            raise RuntimeError('Evaluation indices are missing or duplicated')
        self.last_eval_rows = rows
        eer = self.calculate_EER([r[1] for r in rows],[r[2] for r in rows])
        if self.logger is not None:
            self.logger.log_metric(f'{phase}/EER',eer,step=epoch)
        return eer,0,0,0

    def calculate_EER(self,scores,labels):
        if len(scores)!=len(labels) or not len(scores) or len(set(labels))!=2:
            raise ValueError('EER needs matching nonempty scores with both classes')
        fpr,tpr,_ = metrics.roc_curve(labels,scores,pos_label=1,drop_intermediate=False)
        difference = fpr-(1-tpr)
        k = int(np.flatnonzero(difference>=0)[0])
        if k==0:
            return float(100*fpr[0])
        fraction = -difference[k-1]/(difference[k]-difference[k-1])
        return float(100*(fpr[k-1]+fraction*(fpr[k]-fpr[k-1])))

    def _synchronize(self):
        if dist.is_initialized():
            dist.barrier()
