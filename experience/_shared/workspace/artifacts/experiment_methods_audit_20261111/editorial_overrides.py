"""Human-reviewed additions to the experiment-method registry."""
from copy import deepcopy
from method_records import RECORDS as BASE

RECORDS = deepcopy(BASE)
BY_ID = {r['id']: r for r in RECORDS}

def append(identifier, field, sentence):
    r = BY_ID[identifier]
    if isinstance(r[field], list):
        r[field].append(sentence)
    else:
        r[field] += ' ' + sentence

append('E013', 'sources', 'artifacts/padding_analysis_20261030/analyze_padding.py')
append('E013', 'sources', 'artifacts/spoof_error_audit_20261108/analyze.py')
append('E013', 'procedure', '초기 전체 길이 조사의 반복량은 목표 길이 M과 원음 길이 n에 대해 부족 비율 max(0,M−n)/M, 접합 횟수 floor((M−1)/n)로 계산했다. 긴 입력과 짧은 입력을 각각 조사했다. 발화 파일을 읽어 반복 신호를 탐지한 값이 아니라, 입력 생성 규칙에서 계산한 반복량이다.')
append('E027', 'sources', 'artifacts/geometry_cache/local_analysis/native_short303_E1_20261106_v1/verification.json')
append('E027', 'interpretation_and_limits', '초기 short303 설정은 짧은 입력 303프레임, 추가 짧은 분류 손실 0, 반복 관련 추가 손실 0으로 arguments.py만 바꾼 준비 단계였다. 해당 검증 표식은 full_training_started=false다. 이후 실제 50epoch 학습은 E006에서 별도로 기록한다.')
append('E042', 'sources', 'artifacts/geometry_cache/local_analysis/df_interpretation_v1/interpretation.json')
append('E042', 'measurement', '후속 사후 해석에서는 6개 거리 후보 각각에 대해 출처·화자 군집 2,000회 대응 bootstrap의 점별 95% 구간을 계산했다. 6개 후보의 다중 비교 보정은 하지 않았다. 이 재분석은 독립적인 새 검증 표본이 아니다.')
append('E069', 'procedure', '정상 참조 수 N_B, 위조 참조 수 N_S와 반경 안의 개수 n_B,n_S에 대해 b=n_B/N_B, f=n_S/(N_S−2)로 클래스별 밀도를 보정한다. N_B(b+f)<5는 희소 구간, 관측이 전혀 없으면 빈 구간으로 구분하고, 그 밖에서는 2배 우세 기준으로 정상 우세·위조 우세·혼합을 나눈다. 두 위조 끝점은 참조에서 제외한다.')
append('E070', 'sources', 'artifacts/geometry_cache/local_analysis/perturbation_candidate_audit_20261106_v1/summary.json')
append('E080', 'procedure', '변형 반응을 읽는 별도 미실행안도 있었다. 내부 층 가중치를 학습값과 균등값 사이에서 0/0.5/1 비율로 섞는 조건, 긴 원음에서 1.5/2/3초 앞·뒤 구간의 반복/비반복 점수 차이를 구하는 조건을 제안했다. 학습·개발 자료에서 원점수와 반응을 읽는 작은 분류기를 맞추고 독립 평가로 옮기는 안이다. 확대 추론과 추가 분류기 학습은 실행하지 않았다.')
append('E080', 'sources', 'docs/experiments/ebm_perturbation_candidate_audit_and_plan_20261106.md')
append('E083', 'sources', 'artifacts/geometry_cache/local_analysis/author_distribution_v1/provenance.json')
append('E083', 'procedure', '선택 좌표 예시는 113번·95번 두 좌표를 사용했다. 전체 160차원 그림은 DF 진단 자료에서 정상·위조 각 2,000개, seed 20261105로 표시 표본을 고르고 그 4,000개에 PCA 50축을 맞춘 뒤 t-SNE를 적용했다. t-SNE는 perplexity 40, seed 20261105, 1,000회, PCA 초기화, 자동 학습률이다. 투영 계산에 라벨을 넣지는 않았지만 표시 표본의 클래스 균형에는 라벨을 사용했다.')

RECORDS.append(dict(
    id='E085',
    title='기존 변형 결과에서 정상 안정성과 위조의 정상 방향 이동을 따로 검토',
    status='기존 반응 점수의 탐색적 재분석 완료',
    purpose='두 클래스의 평균 이동 차이만 보고 정상은 유지되고 위조만 정상 쪽으로 이동한다고 결론 내릴 수 있는지 점검한다.',
    model_and_data='배치 160의 고정 모델. DF·21LA 진단 표본 각 20,000개에 저장한 층/시간 가중치 개입과 잡음·Opus 결과, 별도로 고정한 DF 471개와 그중 긴 원음 144개의 반복·절단 결과를 재사용했다. 새 음성 추론·분류기 학습은 없다.',
    procedure=[
        '발화별 변형 전 점수 s와 변형 후 점수 s_after로 d=s_after−s를 만든다. 클래스별 평균 d, 평균 |d|, 중앙값, 절대 변화의 90백분위, |d|≤0.02 비율과 d>0.05 비율을 함께 계산한다.',
        '층 가중치 균등, 시간 가중치 균등, 둘 다 균등, 백색잡음 20dB·10dB, Opus 32kbps의 6개 조건을 비교한다. 2,000개 확인용 표본은 20,000개 안에 있으므로 독립 반복으로 합치지 않는다.',
        '긴 원음 절단은 앞·뒤 점수를 원음별로 먼저 평균한다. 4.0375초에서 비반복 1.5초 또는 2초로 줄이는 비교와, 같은 짧은 내용의 반복/비반복 비교를 나누어 내용 감소와 반복 문맥을 구별한다.',
        '출처 안에서 화자 군집을 5,000회 복원 추출해 대응 점별 95% 구간을 계산한다. 여러 조건의 다중 비교 보정은 하지 않았으며, 이미 관찰한 결과를 다시 본 탐색 분석이다.',
        '원점수와 메타데이터가 비슷한 정상·위조의 반응 차이도 조사했다. 초기 단순 매칭의 한계를 확인한 뒤 전체 자료의 최적 매칭은 E070으로 확장했다. 확대 반응 추론은 실행하지 않았다.',
    ],
    conditions='원점수가 높은 정상과 낮은 위조의 전체 평균을 직접 비교하면 점수 상한·출발점 차이가 섞인다. 정상의 평균 이동이 0에 가깝다는 것과 개별 정상의 변화가 작다는 것도 구별한다.',
    measurement='클래스별 부호 있는 이동과 절대 이동, 출처·길이별 안정성, 같은 출발 점수의 반응 차이. 원행 156개 통계의 독립 산술 대조 최대 오차는 1.51e-14였으나 bootstrap 구간을 독립 재실행한 것은 아니다.',
    observed_result='층 가중치 균등화에서 DF 정상 평균 이동 +0.304742, 위조 +1.314300이고 정상 평균 절대 이동은 0.308168이었다. DF 정상 중앙값 −0.000925만 보면 놓치는 큰 변화가 있었다. 1.5초 비반복 절단에서도 정상 +0.124023, 위조 +1.245396이지만 출발 점수 평균은 정상 0.871198, 위조 −0.950127로 달랐다. 정상 유지·위조만 이동하는 방법을 확정하지 못했다.',
    interpretation_and_limits='전체 평균 격차와 조건부 판별 정보, 변형 후 점수 하나의 EER과 원점수+반응을 읽는 새 분류기의 성능은 각각 다른 질문이다. 후자의 전이 성능은 미실행이다. 후보 실패를 모든 반응 정보의 부재로 해석하지 않는다.',
    sources=[
        'docs/experiments/ebm_perturbation_candidate_audit_and_plan_20261106.md',
        'artifacts/geometry_experiments/audit_perturbation_candidates_20261106.py',
        'artifacts/geometry_cache/local_analysis/perturbation_candidate_audit_20261106_v1/audit_scope.json',
        'artifacts/geometry_cache/local_analysis/perturbation_candidate_audit_20261106_v1/summary.json',
        'artifacts/geometry_cache/local_analysis/perturbation_candidate_audit_20261106_v1/independent_verification.json',
    ],
))
RECORDS.sort(key=lambda r: r['id'])
