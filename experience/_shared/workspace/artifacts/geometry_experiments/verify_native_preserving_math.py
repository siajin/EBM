"""Synthetic checks for corrected controls and objective-invariance claims."""

import json
import numpy as np

from validate_native_preserving import bounded_score, oc_loss, rotate_azimuth, fit_monotonic, fit_residual, probability, eer


rng = np.random.default_rng(20261105)
x = rng.normal(size=(128, 160))
x /= np.linalg.norm(x, axis=1, keepdims=True)
c = rng.normal(size=160)
c /= np.linalg.norm(c)
y = np.r_[np.zeros(64), np.ones(64)].astype(int)
checks = []
for theta in [.1, 1, 2.8]:
    moved, v = rotate_azimuth(x, c, theta)
    assert np.max(np.abs(moved@c-x@c)) < 1e-12
    assert np.max(np.abs(np.linalg.norm(moved, axis=1)-1)) < 1e-12
    assert np.max(np.abs(oc_loss(moved, c, y)-oc_loss(x, c, y))) < 1e-12
    assert np.max(np.abs(v@c)) < 1e-12
checks.append("three azimuth rotations preserve cosine, norms and OC loss")
s = rng.normal(size=1000)
g = np.tanh(rng.normal(size=1000))
for epsilon in [0,.01,.03,.1]:
    actual = bounded_score(s, g, epsilon)
    assert np.max(np.abs(actual-s)) <= epsilon+1e-12
    distant = np.abs(s[:500]-s[500:]) > 2*epsilon
    assert np.array_equal(np.sign((actual[:500]-actual[500:])[distant]), np.sign((s[:500]-s[500:])[distant]))
    assert np.array_equal(bounded_score(s, g, 0), s)
checks.append("bounded correction and zero-residual identity; sufficiently separated pairs retain order")
fixed_g = np.full(100, .4)
ordered = np.linspace(-1,1,100)
assert np.all(np.diff(bounded_score(ordered,fixed_g,.1)) > 0)
checks.append("native coefficient 1 is monotonic for fixed direction correction")
train_s = np.r_[rng.normal(-.8,.2,1000),rng.normal(.8,.2,1000)]
labels = np.r_[np.zeros(1000),np.ones(1000)].astype(int)
cal = fit_monotonic(train_s, labels)
assert cal['slope'] > 0
assert eer(labels,train_s) == eer(labels,cal['slope']*train_s+cal['intercept'])
checks.append("positive scalar calibration preserves ranking EER")
features = np.zeros((len(labels),160))
model, selected, panel = fit_residual(features,features,labels,labels,train_s,cal)
assert selected['epsilon'] == 0 and selected['C'] == .01
checks.append("uninformative constant directions choose exact native fallback on ties")
print(json.dumps({'status':'pass','checks':checks},indent=2))
