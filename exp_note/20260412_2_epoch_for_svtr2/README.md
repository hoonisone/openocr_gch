# Epoch은 얼마나 해야 하나?

## Intro
* 빠른 실험 필요.
* 성능 차이가 크지 않으면서, 적은 epoch이 빠르고 정확한 실험에 유리함

## Experiment
* SVTRv2 (GCH) - G, C
* epoch 5 vs 10 비교

## Result

### C
| Dataset | Split | Model | 5     | 10    | Delta |
|---------|-------|-------|-------|-------|-------|
| AIHub   | Eval  | Main  | 97.41 | 97.62 | 0.21  |
| AIHub   | Test  | Main  | 98.91 | 99.11 | 0.20  |
| GIST    | Test  | Main  | 93.05 | 93.56 | 0.51  |
| KAIST   | Test  | Main  | 79.98 | 80.52 | 0.54  |

### G
| Dataset | Split | Model | 5     | 10    | Delta |
|---------|-------|-------|-------|-------|-------|
| AIHub   | Eval  | Main  | 97.26 | 97.43 | 0.17  |
| AIHub   | Test  | Main  | 98.7  | 98.95 | 0.25  |
| GIST    | Test  | Main  | 92.02 | 92.59 | 0.57  |
| KAIST   | Test  | Main  | 78.08 | 77.81 | -0.27 |

### G+C (0.5) C
| Dataset | Split | Model | 5     | 10    | Delta |
|---------|-------|-------|-------|-------|-------|
| AIHub   | Eval  | Main  | 97.54 | 97.7  | 0.16  |
| AIHub   | Test  | Main  | 99.01 | 99.16 | 0.15  |
| GIST    | Test  | Main  | 93.09 | 94.06 | 0.97  |
| KAIST   | Test  | Main  | 80.33 | 79.68 | -0.65 |

### G+C (0.5) G
| Dataset | Split | Model | 5     | 10    | Delta |
|---------|-------|-------|-------|-------|-------|
| AIHub   | Eval  | Main  | 97.28 | 97.48 | 0.20  |
| AIHub   | Test  | Main  | 98.74 | 98.94 | 0.20  |
| GIST    | Test  | Main  | 91.98 | 92.69 | 0.71  |
| KAIST   | Test  | Main  | 77.93 | 77.66 | -0.27 |

* epoch 10에서 평균적으로 성능이 더 높음
* KAIST는 성능 저하가 있는데,  문제는 KAIST가 테스트 셋임에도 불구하고 레이블 오류가 꽤 있다는 것
* 또는 KAIST가 학습 셋과 차이가 커서, AIHub에 대한 과적합이 KAIST에 불리했을 수 있음

### 결론
* 일단 epoch은 10 정도로 하고
* KAIST에 대해서는 떨어지니 10에서 더 늘리지는 말고
* KAIST에 대해서는 완전 검수를 해봐야 할 듯
* KAIST가 총 6000장 정도의 데이터 중 한글 세로는 2000장 정도라서 그리 어렵지 않을 듯

