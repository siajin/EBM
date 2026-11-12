> 표시 명칭·링크 이름은 설명형으로 정리했다. 원래 코드·경로·수치는 그대로 보존한다.

> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

# 방향·채널 결과 문서 안내

최신 통합 분석은 [고정 EBM 전체 결과와 연구 가설](../ebm_frozen_all_results_20261105.md)이다. 아래 문서는 기존 DF 캐시 진단의 방법·수치를 보존하면서 새 T/V 전이·matching·조건 분석에 맞춰 해석을 수정했다. 실험 수행 완료와 가설 지지를 구분한다.

| 문서 | 현재 읽어야 할 결론 |
| --- | --- |
| [① 잔여 방향](01_direction.md) | DF 내부 접근성 이득과 T/V 전이의 EER/LL 결과를 분리한다. 새 strict matching은 표본 부족이다. |
| [② 채널 부분집합](02_channel_ablation.md) | 특정 좌표의 낮은 EER와 학습 gate의 전이 개선은 다른 주장이다. 전체 성능 동등성은 입증되지 않았다. |
| [80→10개](02_channel_reduction_to10.md) | 하위10개 DF EER2.9444%는 해당 표본·점수의 관찰이다. 후속1개까지 이미 시험했다. |
| [5·3·2개](02_channel_reduction_to2.md) | 하위2개 EER2.9222%지만 AUROC·V와 일부 ROC 구간은 악화됐다. |
| [1개와 두 좌표 설명](02_channel_reduction_to1_and_explanation.md) | 하위1개 EER4.345939%;1D cosine은부호±1이며두좌표가모델전체차원은아니다. |
| [ROC 교차](02_why_more_channels_can_worsen_eer.md) | EER·AUROC·운용점·확률 log-loss가 서로 다른 질문임을 구분한다. |
| [③ LayerNorm](03_layernorm.md) | h가 옛 동일 probe에서 더 잘 읽혔으나 정보 손실의 인과 증명은 아니다. 새 증분 P4는 미완료다. |
| [④ 데이터 조건](04_conditions.md) | K=4 NMI와 전체 u의 eta²를 구분하며 source/codec/padding 연관을 인과 효과로 읽지 않는다. |
| [관련 문헌과 우리 결과](02_related_work_fulltext_review_20261105.md) | 문헌의 보고 결과와 우리 후속 결과를 분리하고 미실행 목록을 정정했다. |
| [상세 설계·실험 기록](figures_clusters/captions.md) | 그림 표본과평가표본을구분하고,투영분리로성능·군집수·차원을확정하지않는다. |
| [저자 설명용 그림 캡션](figures_author/captions.md) | 기존DF부분집합그림이며새21LA/gate실험결과가아니다. |

새 P1/P2는 실제 T/V12fit을 완료했다. dynamic의 DF EER 평균 개선0.007407pp의99% CI는0을 포함했고21LA는0.80%로 같았다. static/metric은 V에서epoch0 identity를 선택했다. 새 P3 선형 전이는 LL을 작게 줄였으나 EER는 개선하지 못했다.

기존 DF 내부1.20%/0.60%는 새 probe를 DF에서 학습한 결과다. T에서만 학습한 검출기의 일반화 성능이나 새 EBM의 성능으로 사용하지 않는다. 모든 문서의 scalar baseline은 원래 EBM과 다를 수 있다. 원래 native는cos(z,c)만 쓰며,크기는 별도 판별 입력이 아니다.

진행 중인 F1 전체·P3-EBM-NoRepeat/P4·전체 데이터 추출·R2/R3는 [실행 체크표](../ebm_frozen_model_experiments_20261105.md)를 따른다. 각 문서의 새 해석은 작성 시점의 검수 완료 자료에 한정되며,이후 완료 결과는 최신 통합 분석에 반영된다.

기존 [통합 결과와 검수 이력](../ebm_cached_core_results_20261105.md), [기존 독립 검수](../../../artifacts/geometry_cache/local_analysis/cached_core_v1/output_audit.json), [새 실험 검수](../../../artifacts/geometry_cache/local_analysis/frozen_all_v1/verification.json), [P3 전이 검수](../../../artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/A/verification.json)를 함께 확인할 수 있다.
