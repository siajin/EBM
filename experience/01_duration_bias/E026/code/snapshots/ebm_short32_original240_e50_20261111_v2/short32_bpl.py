"""Opt-in original OC/BPL43 plus symmetric BPL32 for native-short bona fide speech.

No launcher, optimizer, GPU initialization, or training side effects at import.
"""
RATE = 16000
LONG = 64600
SHORT = 48440
TWO = 32000


def validate_options(args):
    if args.get('epoch', 50) > args.get('T_0', 150):
        raise ValueError('Short32 retains the original frozen-frontend epoch regime')
    if args.get('bpl_pairing', 'legacy_long_short') != 'legacy_long_short':
        raise ValueError('short32 retains original BPL43; matched pairing is a separate experiment')
    if args.get('duration_policy_path') or args.get('activity_policy_path'):
        raise ValueError('short32 does not apply other crop/activity policies')
    if args.get('short_oc_weight', 0.) or args.get('padding_loss_weight', 0.):
        raise ValueError('short32 retains only original OC plus conditional BPL')
    if args.get('num_train_frames', 404) != 404 or args.get('num_train_frames_short', 303) != 303:
        raise ValueError('Retain fixed64600/48440 original training views')


def configure(args):
    validate_options(args)
    from short32_original240 import validate_runtime
    validate_runtime()
    args.setdefault('distributed_timeout_seconds', 600)
    args.update(short32_bpl=True, short32_train_input_policy='fixed_wrap_64600_48440',
                native_length=True, cap_long=True)
    # native_length configures evaluation only. Frozen original240 TrainSet
    # always uses its original wrap/crop logic for both classes.
    return args

