"""Original EBM architecture with explicit backend, center and scoring phases."""
import torch
from model.b_linear import B_Linear
from model.loss import BPL
from role_objectives import RoleConfig, RoleOCSoftmax, checked_labels


class RoleBLinear(B_Linear):
    def __init__(self, role, num_layer=25, hidden_size=1024, output_size=160, agg_size=128):
        # Original main initializes OC before constructing the backend.
        loss = RoleOCSoftmax(output_size, role)
        super().__init__(num_layer, hidden_size, output_size, agg_size,
                         loss=loss, loss_bpl=BPL(role.alpha_pair))
        self.role = role
        self.architecture = dict(num_layer=num_layer, hidden_size=hidden_size,
                                 output_size=output_size, agg_size=agg_size)

    def encode(self, x, mask=None):
        return super().forward(x, mask=mask, return_embedding=True)

    def forward(self, x, x_short=None, label=None, *, phase='score', mask=None, mask_short=None):
        if phase not in ('backend', 'center', 'score', 'embedding'):
            raise ValueError('Unknown role model phase')
        if phase in ('score', 'embedding'):
            if label is not None or x_short is not None:
                raise ValueError('Scoring/embedding does not accept labels or short views')
            z = self.encode(x, mask)
            return z if phase == 'embedding' else self.loss.score(z)
        labels = checked_labels(label, len(x), x.device)
        compact = labels == self.role.compact_label
        if phase == 'center':
            if x_short is not None or not bool(compact.all()):
                raise ValueError('Center phase needs only compact-class long rows')
            z = self.encode(x, mask)
            return self.loss.center_loss(z, labels), {'center_rows': len(labels)}
        if x_short is None or len(x_short) != len(x) or not bool(compact.any()) or bool(compact.all()):
            raise ValueError('Backend phase requires aligned long/short views and both classes')
        z, short = self.encode(x, mask), self.encode(x_short, mask_short)
        self.loss.validate_embeddings(short)
        oc = self.loss.backend_loss(z, labels)
        pair = self.loss_bpl(z[compact], short[compact])
        total = oc + pair
        if not bool(torch.isfinite(total)):
            raise ValueError('Nonfinite role objective')
        stats = dict(oc=oc.detach(), pair=pair.detach(), compact_rows=int(compact.sum()),
                     pair_count=int(compact.sum()) ** 2)
        with torch.no_grad():
            unit = torch.nn.functional.normalize(z.detach(), dim=1)
            center = torch.nn.functional.normalize(self.loss.weight.detach(), dim=1)
            canonical = self.role.score_sign * (unit @ center.T).flatten()
            for name, y in [('bona',1), ('spoof',0)]:
                group = unit[labels == y]
                scores = canonical[labels == y]
                stats[name+'_score_mean'] = scores.mean()
                stats[name+'_score_std'] = scores.std(unbiased=False)
                # Batch geometry only, not an estimate of full-Train effective rank.
                stats[name+'_unit_variance_trace'] = ((group-group.mean(0))**2).sum(1).mean()
        return total, stats


def config_from_model(model):
    base = model.module if hasattr(model, 'module') else model
    if not isinstance(base, RoleBLinear) or base.role.metadata() != base.loss.role.metadata():
        raise ValueError('Expected an explicitly role-aware EBM head')
    return base.role


def validate_model_metadata(metadata):
    if not isinstance(metadata, dict) or set(metadata) != {'role', 'architecture'}:
        raise ValueError('Missing model role/architecture metadata')
    role = RoleConfig.from_metadata(metadata['role'])
    shape = metadata['architecture']
    if (not isinstance(shape, dict) or set(shape) != {'num_layer', 'hidden_size', 'output_size', 'agg_size'}
            or any(type(v) is not int or v < 1 for v in shape.values())):
        raise ValueError('Invalid model architecture')
    return role, shape
