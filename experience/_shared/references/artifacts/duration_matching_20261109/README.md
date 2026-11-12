> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

# Constrained duration-distribution matching · 2026-11-09

Metadata-only feasibility study on the same 803,565 utterances used in the preceding histograms. No SSH, audio processing, inference, training, checkpoint/source edits, or sampling-weight changes.

## Constraints and interpretation

Both classes may transform each utterance independently. Playback/tempo rate s is in [0.8,1.2], with s>1 making audio faster. Crop fraction c is in [0,0.2]. Final duration is d' = d(1-c)/s, so d' is in [2d/3,1.25d]. Cropping is a fraction of original duration; applying speed first produces the same length interval.

Within each dataset, every original utterance keeps its uniform within-class sampling probability. The two classes are each normalized to 100%. Files are not filtered or reweighted. Whole original utterance duration, including silence, is the only measurement.

## Maximum common distribution

Two utterances can produce the same duration iff their permitted duration intervals intersect. With both classes allowed to transform this means max(d1,d2)/min(d1,d2) <= 15/8 = 1.875. We maximize common probability mass with an ordered capacitated bipartite matching. Integer capacities use the LCM of class counts, avoiding floating-point flow error. The common-overlap metric is 1 minus total variation, expressed as a percentage.

Optimality is independently certified via residual-graph reachability and a cut whose capacity equals the achieved flow. See validation.json. These exact upper limits permit a file to randomize among multiple feasible durations on successive uses. They refer to continuous durations, with sub-sample output rounding not modeled.

| Dataset | Crop only maximum | Speed only maximum | Combined maximum |
|---|---:|---:|---:|
| 19LA train | 97.6748% | 99.7181% | 99.9737% |
| 19LA dev | 97.8663% | 99.6558% | 100.0000% |
| 19LA eval | 89.4448% | 96.5661% | 99.6121% |
| 21LA eval | 79.0896% | 92.7466% | 99.2847% |
| 21DF eval | 98.0612% | 99.9849% | 99.9983% |

## One transformed version per utterance

For a concrete deterministic plan, choose the conditional weighted-median output length of each file in the optimal randomized plan. This is a feasible construction, not a proof of the global optimum over single-version histogram plans. All 803,565 planned rates, crop fractions, and reconstructed target durations were checked. The plan is in single_transform_plan.parquet.

Histogram overlap is sum over bins of min(class-normalized probabilities), using common bins starting at zero and containing all observations. This coarsened metric depends on bin width; it must not be confused with exact common probability mass. At 0.05-second bins:

| Dataset | Before | One-version plan |
|---|---:|---:|
| 19LA train | 78.4198% | 99.2916% |
| 19LA dev | 82.4476% | 99.1810% |
| 19LA eval | 71.1172% | 99.3421% |
| 21LA eval | 58.6843% | 99.1634% |
| 21DF eval | 79.7623% | 99.8501% |

Summary.csv also gives 0.1/0.25-second bin sensitivity, KS, and Wasserstein distance. Operations.csv gives speed/crop use and under-2s shares. Each randomized_segments.parquet stores source indices in duration/UID order, target duration and conditional probabilities; analyze.py reconstructs that source order from hash-verified caches.

## Interpretation limits

This optimizes each dataset independently with access to its class labels. It is an oracle duration-only feasibility result; it does not establish that a single train-fitted policy transfers to evaluation sets. No waveform was generated and no pitch/artifact preservation, linguistic content retention, spoof-cue preservation, or detection performance was measured. Uniformly drawing random speed/crop factors is a different policy and does not automatically achieve these values.

## Minimum crop-only cap for 100% exact common distribution

Continuation requested on 2026-11-09. Speed is fixed at 1 and the same randomized-augmentation/equal-exposure definition above is retained. For each exact quantile interval, reachable duration intervals must intersect. The necessary and sufficient cap is max_q[1-min(Q_b(q),Q_s(q))/max(Q_b(q),Q_s(q))]. Quantile breakpoints are swept using integer LCM units, and source durations remain integer sample counts. At the exact rational threshold, the independently implemented maximum-matching routine reaches 100%; lowering the cap by 0.0001 percentage point fails in all five cases. Rounded-up caps also pass.

| Dataset | Exact minimum (decimal %) | Cap rounded up to 0.1 percentage point |
|---|---:|---:|
| 19LA train | 52.1102853473% | 52.2% |
| 19LA dev | 45.5790967363% | 45.6% |
| 19LA eval | 53.5831004594% | 53.6% |
| 21LA eval | 55.0000000000% | 55.0% |
| 21DF eval | 54.1484716157% | 54.2% |

A shared maximum crop cap of 55% suffices for each dataset's two classes separately. This does not mean that every file is cropped by 55%, that a single class-blind policy has been fitted, or that all five datasets have been aligned to one shared distribution. The 21LA bottleneck pairs a 1.6-second bona fide quantile with a 0.72-second spoof quantile, requiring 55% removal. The DF bottleneck is in the upper tail, 29.312-second spoof versus 13.44-second bona fide, requiring 54.14847% removal. Exact rational values, quantile witnesses, and independent matching checks are in minimum_crop_100pct.csv and minimum_crop_validation.json. No audio was processed.
