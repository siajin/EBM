"""Role-aware OC objectives. Dataset labels remain spoof=0, bonafide=1."""
from dataclasses import dataclass, fields
import math

import torch
import torch.nn.functional as F
from model.loss import OCSoftmax


@dataclass(frozen=True)
class RoleConfig:
    compact_label: int = 0
    r_pull: float = .9
    r_push: float = .2
    alpha_oc: float = 20.
    alpha_pair: float = 30.
    center_loss_multiplier: float = 5.

    def __post_init__(self):
        if type(self.compact_label) is not int or self.compact_label not in (0, 1):
            raise ValueError('compact_label must be 0 (spoof) or 1 (bonafide)')
        values = [getattr(self, f.name) for f in fields(self) if f.name != 'compact_label']
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError('Role coefficients must be finite numbers')
        if not -1 < self.r_push < self.r_pull < 1:
            raise ValueError('Require -1 < r_push < r_pull < 1')
        if min(self.alpha_oc, self.alpha_pair, self.center_loss_multiplier) <= 0:
            raise ValueError('Loss scales must be positive')

    @property
    def score_sign(self):
        return 1 if self.compact_label == 1 else -1

    def metadata(self):
        return dict(schema_version='ebm_role_v1',
                    **{f.name: getattr(self, f.name) for f in fields(self)},
                    label_semantics={'0': 'spoof', '1': 'bonafide'}, pair_class=self.compact_label,
                    backend_class_weights=[5/9, 5.], pair_reduction='mean_all_local_cross_pairs',
                    score_bona_transform=('center_cosine' if self.score_sign == 1 else 'negative_center_cosine'))

    @classmethod
    def from_metadata(cls, metadata):
        if not isinstance(metadata, dict):
            raise ValueError('A role manifest is required; tensor shapes do not identify the center class')
        names = {f.name for f in fields(cls)}
        if not names.issubset(metadata):
            raise ValueError('Incomplete role metadata')
        result = cls(**{k: metadata[k] for k in names})
        if metadata != result.metadata():
            raise ValueError('Inconsistent or unsupported role metadata')
        return result


def checked_labels(labels, n, device):
    if (not torch.is_tensor(labels) or labels.ndim != 1 or len(labels) != n or labels.device != device
            or labels.dtype != torch.long or not bool(((labels == 0) | (labels == 1)).all())):
        raise ValueError('Expected one int64 protocol label per embedding on the same device')
    return labels


class RoleOCSoftmax(OCSoftmax):
    """Keep original weight initialization/state key, make class roles explicit."""
    def __init__(self, embedding_size, role):
        if not isinstance(role, RoleConfig):
            raise TypeError('role must be RoleConfig')
        super().__init__(embedding_size, r_real=role.r_pull, r_fake=role.r_push,
                         alpha=role.alpha_oc, class_weight=[5/9, 5.])
        self.role = role

    def validate_embeddings(self, embeddings):
        if (not torch.is_tensor(embeddings) or embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_size
                or not len(embeddings) or not bool(torch.isfinite(embeddings).all())
                or not bool(torch.isfinite(self.weight).all())):
            raise ValueError('Nonempty, finite embeddings and center are required')
        if (bool((embeddings.norm(dim=1) <= 1e-12).any())
                or bool((self.weight.norm(dim=1) <= 1e-12).any())):
            raise ValueError('Zero-norm embeddings or center have no angular meaning')

    def cosine(self, embeddings):
        self.validate_embeddings(embeddings)
        return F.normalize(embeddings, dim=1) @ F.normalize(self.weight, dim=1).T

    def score(self, embeddings):
        raw = self.cosine(embeddings)
        return {'raw_center_cosine': raw, 'score_bona': self.role.score_sign * raw}

    def backend_loss(self, embeddings, labels):
        labels = checked_labels(labels, len(embeddings), embeddings.device)
        if len(labels) <= 4:
            raise ValueError('Original240 backend batches must exceed four rows')
        q = self.cosine(embeddings)
        penalty = torch.where((labels == self.role.compact_label)[:, None],
                              self.role.r_pull - q, q - self.role.r_push)
        # Weights stay attached to protocol classes, never to the compact/push role.
        weights = torch.where((labels == 1)[:, None], q.new_tensor(5.), q.new_tensor(5/9))
        return (F.softplus(self.role.alpha_oc * penalty) * weights).mean()

    def center_loss(self, embeddings, labels):
        labels = checked_labels(labels, len(embeddings), embeddings.device)
        if not bool((labels == self.role.compact_label).all()):
            raise ValueError('Center pass contains the wrong protocol class')
        # Multiply before mean, matching original OC's floating-point operation
        # order as well as its formula and Adam moment history in the bona arm.
        return (self.role.center_loss_multiplier * F.softplus(
            self.role.alpha_oc * (self.role.r_pull - self.cosine(embeddings)))).mean()

    def forward(self, x, label=None, logical_batch_size=None):
        if label is None:
            # No training indexing or score mutation during inference.
            return x.new_zeros(()), self.score(x)['score_bona']
        if logical_batch_size is not None and logical_batch_size != len(x):
            raise ValueError('Role training uses full graphs, not microbatch loss scaling')
        return self.backend_loss(x, label), None
