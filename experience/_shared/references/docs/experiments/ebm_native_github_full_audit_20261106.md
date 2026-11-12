> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

# EBM GitHub 전체 비교와 원길이 학습 재점검

**이후 기본값은 B로 변경:** Long4.0375초·Short3.0275초 상한 유지, 반복 없음, CPU wrapped 생성 없음, 보조 손실0이다. 아래는 이전 시점의 전체 비교 기록이다. [B 변경·검증 기록](ebm_control_B_20261106.md)을 현재 설정 기준으로 사용한다. 기존240은 보관 소스에서 활성 codec 호출을 확인했으나 실제 실행의 codec 적용 여부는 확인하지 못했으므로 codec-on 실행으로 단정하지 않는다.

**후속 20:58 설정:** 아래전체비교와속도측정은20:43의E3코드기준이다. 현재기본값은E1(`short_oc_weight=0`, `padding_loss_weight=0`)로전환해모델에는반복W를넣지않는다. 전체batch240·short303·codec-off는유지한다. 아래측정중현재설정에대응하는조건은‘원길이두view’다. 변경기록은 [E1 검증](../../artifacts/geometry_cache/local_analysis/native_short303_E1_20261106_v1/verification.json)에있다.

2026-11-06 GitHub main의 commit [`16f9cba7ced226fbe31fb3f217cfafbc6e801075`](https://github.com/kimho1wq/EBM/tree/16f9cba7ced226fbe31fb3f217cfafbc6e801075)를 고정해 서버 `/home/mentee/code/EBM`과 비교했다. GitHub의 전체34개 파일과 실제 tree의 파일명·내용hash를 대조하고, 공개 archive의 Git blob hash도 검산했다. **25개 동일,9개 변경,누락0개**다. 실행 코드 추가는 `native_steps.py` 하나이며, 그 밖의 추가파일2개는 기존 arguments backup이다.

현재 데이터 길이 범위에서는 원길이 long의 학습·평가에 shape/gradient/메모리 오류를 확인하지 못했다. 다만 새로운 분할 재계산과 보조 branch는 계산 비용이 있고, 판별 성능 개선은 재학습 비교 전에는 확인할 수 없다. 이번에는 전체epoch 학습을 시작하지 않았다.

## 1 GitHub와 다른 파일만 정리

| 파일 | GitHub 대비 차이 | 구분 |
|---|---|---|
| `arguments.py` | epoch150→50,전체batch40→240,lr와lr_min1e-6→1e-5,실험명변경. native_length=True,physical/eval batch8,short_OC=.25,padding_loss=.10 추가 | 학습 하이퍼파라미터는 기존 사용자 변경 보존,새 native 설정만 추가 |
| `dataset/ASVspoof2019_la_train.py` | 스푸프마다 추가하던 `vocoder/hifi_gan/flac` item 생성·append 블록이 주석 | 기존 사용자 변경 유지. 기본19LA train25,380개만 사용 |
| `dataset/ASVspoof2021_df_eval.py` | DF 음성 확장자 `.flac`→`.wav` | 기존 사용자 데이터 경로 변경 유지 |
| `main.py` | 기존ACN waveform augmentation 주석·trainer.da=None,DDP timeout12시간. 새 loader 연결,현재CUDA device지정,frozen XLS-R의DDP wrapper제거,새설정/기존결과경로검증 | ACN/timeout은 기존 사용자 변경. 나머지는 새 입력 연결 |
| `data_loaders.py` | 두 클래스별 고정길이 loader→클래스 비율을 맞춘 길이별 joint batch. Long/eval 전체원길이,short 최대48,440sample,보조wrap생성,길이/index반환,중복없는분산eval. 실시간codec2블록 주석 | 새 입력 처리 및 이번 codec 재점검 수정 |
| `model/b_linear.py` | waveform 이후frame mask를bridge의25개ASP와최종ASP에전달. mask된sample std,embedding반환경로추가 | 가중치shape·160차원·모델계층수 유지 |
| `model/loss.py` | OC에 논리적batch크기 인자 추가. physical microbatch가작아도원래class weight유지 | **BPL클래스의수식은GitHub와동일**,embedding전역gather없음 |
| `train.py` | joint loader·step helper연결. 고정epoch학습/dev검증·정기checkpoint저장. target별best모델선택제거. 길이mask추론·index집계/누락/중복검사·전체ROC EER | 새 연결 및 평가 정책 |
| `logger/local.py` | 같은 결과폴더의재귀삭제→FileExistsError | 기존결과보존 |
| **추가 `native_steps.py`** | frozen XLS-R mask/feature추출,세branch OC+기존교차BPL+정상1:1반복정렬,embedding-gradient cache/replay,encoder→centroid 두단계update | 새 계산 helper131줄 |

Dockerfile·README·각 augmentation 구현·scheduler·나머지dataset/model/logger/utils 등 나머지25개 파일은 byte hash가 동일하다. 인증 관련 값은 비교 출력이나 보고서에 노출하지 않았다. 파일별hash·변경symbol·설정차이는 [전체 비교JSON](../../artifacts/geometry_cache/local_analysis/ebm_full_github_audit_20261106_v1/repo_comparison.json)에 있다.

## 2 코덱 비활성화 확인과 정정

서로 다른 경로가 있었다.

1. **추가 HiFi-GAN 데이터:** 사용자가 주석으로 막은 dataset 생성 블록은 계속 주석 상태였다. 실제 reader에서 정상2,580/스푸프22,800,합계25,380개를 확인했다.
2. **ACN 증강:** main의 `WaveformAugmetation(['ACN'],...)` 호출도 주석이며 `trainer.da=None`이다. ACN은 별도의 noise augmentation으로 실시간 codec과 같은 경로가 아니다.
3. **실시간 MP3/OGG/AAC/a-law/u-law codec:** TrainSet의 long·short `self.codec_aug.convert(...)` 두 호출은 이번 점검 직전까지 활성 상태였다. 앞선 수정에서는 설정과호출이 그대로인지 확인했으나,그것이codec-off라는확인과동일하지않았다.

이번에3번의 두 conditional block 전체를 주석 처리했다. **확률값·CodecAugmentationLight 구현·codec 품질목록은 수정하지 않았다.** 전체block을주석처리했으므로codec을위한사전crop과확률추첨도실행하지않는다. Long 전체원길이·short crop규칙은그대로다. 기존RawBoost0~8 random augmentation은계속유지한다.

변경 전 활성convert AST call은2개였고,변경 후0개다. 실제TrainSet에서 codec 변환함수를 호출되면실패하는함수로교체하고,확률조건을강제로참이되게만든뒤,서로다른길이의원음3개×RawBoost종류9개를각GPU에서검사했다. **양GPU모두codec실제호출0회**,각long/short길이와유한값검사를통과했다. 따라서현재학습경로에서실시간codec도꺼져있다.

이번추가수정은 `data_loaders.py`뿐이며다른코드는그대로다. 직전파일은서버 `/home/mentee/analysis/ebm_full_audit_20261106_2031/before_codec_fix/data_loaders.py`에보존했다.

## 3 크롭 없는 long의 실제 실행 확인

전체 현재Python29개파일의syntax를검사했다. 실제사용경로는 reader→TrainSet→길이sampler/collate→frozenXLS-R→mask변환→25개layer요약→최종pooling→160차원→OC/BPL→두단계optimizer→평가순서로검토했다. 앞선반영검사에서는기존checkpointstrict-load,mask의단독/배치동일성,세연속두단계DDP update와full-graph/replay일치,빈정렬쌍,평가rank길이불일치와중복제거도통과했다.

이번에는codec을끈실제코드로다음을추가검사했다.

| 검사 | 결과 |
|---|---|
| 원음길이별실제RawBoost0~8 | 최단스푸프·최단정상·최장학습음성에서long원길이유지,short최대48,440sample,NaN/Inf없음 |
| 길이분위수로선택한실제240개배치 | 양GPU의forward/backward·Adam update·frozenfrontend통과 |
| 최장정상11.129초와최장스푸프13.188초로채운240개stress배치 | label에맞는최장원음을반복샘플링. 크롭없이갱신통과,peak allocated약3,315MiB=3.24GiB |
| 평가최장29.312초원음8개/GPU | 원길이전체forward·유한출력통과,peak allocated약5,747MiB=5.61GiB |

GPU는각24GB급TITAN RTX이며physicalmicrobatch8을사용했다. 마지막두검사는현재데이터의극단길이를넣은stress검사다. 데이터전체의50epoch성공이나미래의더긴음성까지무제한처리한다는보장은아니다. Short의3.0275초crop은여전히유지한다.

## 4 속도와 계산량

같은실제240개길이분위수배치,같은physicalmicrobatch8,같은gradient-cache방식으로비교했다. 조건별3회중첫회는warmup,다음2회에서양rank중느린시간을한논리적step시간으로삼아평균했다. 디스크읽기·파형증강·검증데이터추론시간은포함하지않았다.

| 입력 및 손실 조건 | 평균step시간 | 최대allocated |
|---|---:|---:|
| 고정4.0375/3.0275초,기존두view손실 | 9.65초 | 1,940MiB |
| 원길이long/최대3.0275초short,기존두view손실 | 10.18초 | 3,315MiB |
| 현재기본:원길이+short+wrap보조branch | 14.61초 | 3,315MiB |

이한배치에서원길이변경은동일cache방식의고정길이조건보다약5.5%느렸고,보조branch추가는원길이두view보다약43.5%느렸다. 현재세view는같은cache방식의고정두view보다약51.3%느렸다. 최장학습음성만채운stressstep은단1회23.82초였다.

**9.65초는원래GitHub의큰full-graph배치를직접측정한값이아니다.** 네이티브구현과동일한분할재계산방식으로고정길이입력을처리한대조다. 따라서이를근거로원래코드전체학습보다정확히51.3%느리다고말하면안된다.

전체T의검사용108step schedule(epoch0)을입력길이로계산하면현재세view의XLS-R forward용padded waveform sample 처리량은기존두view단일forward기준의약2.62~2.68배다. 실제학습epoch별shuffle에따라값은달라진다. 이는FLOPs나wall-clock비율이아니며attention에는길이제곱항도있다. 메모리를줄이려고frontend를재계산하고W를추가한비용이남기때문에,**패딩량이감소했다고계산량도같은비율로줄지는않는다.**

전체학습시간은별도다. 새코드는학습중대규모target평가를반복하지않고공식dev만검증하므로평가시간은줄어든다. 한step측정만으로50epoch전체소요시간을확정하지않았다. 비치명적인DDP gradient-stride성능경고가있었으며기존전체graph와update수치일치검사는통과했다.

## 5 판별 성능에 대한 결론

가변길이는batch내time차원을mask로처리하고시간pooling후같은160차원으로변환하므로,**모델구조때문에long을4초로잘라야하는제약은없다.** 현재데이터의길이범위와microbatch8에서는실제gradient·메모리검사를통과했다.

다만판별성능이유지되거나향상된다는결론은아직없다. 반복제거는XLS-R문맥과pooling입력분포를바꾸고,길이별batch는기존BPL이만나는정상쌍을바꾸며,현재기본값은short/wrap분류와새정렬항도추가한다. 기존고정모델에서는반복제거가정상오류를줄여도스푸프통과를늘린사례가있었다. 코드를정상실행할수있다는것과재학습성능을구별해야한다.

현재부가가중치(.25/.10)가최적이라는검증도없다. 설계한E1/E2/E3비교에서EER와정상FRR·스푸프FAR를함께확인해야한다. GitHub와의성능차이는이미달라진epoch·batch·lr·추가HiFi-GAN·ACN구성의영향도포함하므로원길이하나의효과로해석하지않는다.

## 6 검수 기록

파생자료는 `artifacts/geometry_cache/local_analysis/ebm_full_github_audit_20261106_v1/`의 [전체파일비교](../../artifacts/geometry_cache/local_analysis/ebm_full_github_audit_20261106_v1/repo_comparison.json), [속도·메모리요약](../../artifacts/geometry_cache/local_analysis/ebm_full_github_audit_20261106_v1/performance_summary.json), [GPU0stress](../../artifacts/geometry_cache/local_analysis/ebm_full_github_audit_20261106_v1/rank0_stress.json), [GPU1stress](../../artifacts/geometry_cache/local_analysis/ebm_full_github_audit_20261106_v1/rank1_stress.json), [최종무결성](../../artifacts/geometry_cache/local_analysis/ebm_full_github_audit_20261106_v1/final_verification.json)에있다.

서버audit루트는 `/home/mentee/analysis/ebm_full_audit_20261106_2031`이며 `stress.exit=0`이다. 원본소스·음성·checkpoint·SSLhidden을로컬로복사하지않았다. 공개GitHub비교자료만서버에추가하고파생검수JSON만로컬에보관했다. Baselinecheckpoint는불변이다. 최종20:43:43KST확인에서기존셸은유휴,지정GPU6/7은각6MiB·0%였다.
