# LR Warmup 조절에 따른 성능 차이는?

## Intro
* Goal: LR Warmup 조절에 따른 성능 차이는?
* Why? epoch을 50에서 5로 줄이는 과정에서 Warmup도 같이 조절해야 할 것 같아서, 
민감도 확인 필요

## Experiment
- SVTRv2 (C, G 버전)
- epoch = 5
- warmup만 0.5와 1.5 비교

## Result


### C
| Dataset | Split | Model | 0.5   | 1.5   | Delta |
|---------|-------|-------|-------|-------|-------|
| AIHub   | Eval  | Main  | 97.41 | 97.43 | -0.02 |
| AIHub   | Test  | Main  | 98.91 | 98.92 | -0.01 |
| GIST    | Test  | Main  | 93.05 | 92.8  | 0.25  |
| KAIST   | Test  | Main  | 79.98 | 80.48 | -0.5  |


### G
| Dataset | Split | Model | 0.5   | 1.5   | Delta |
|---------|-------|-------|-------|-------|-------|
| AIHub   | Eval  | Main  | 97.26 | 97.2  | 0.06  |
| AIHub   | Test  | Main  | 98.7  | 98.63 | 0.07  |
| GIST    | Test  | Main  | 92.02 | 91.76 | 0.26  |
| KAIST   | Test  | Main  | 78.08 | 78.31 | -0.23 |

warmup에 따른 성능 변화가 큰 차이 없어보임

## Discussion

## 결론
- warmup은 크게 민감하지 않은 것으로 보임
- warmup은 0.5를 사용
- 어짜피 성능 차이가 크게 없다면
- epoch에 따라 warmup이 줄어드는 것이 자연스럽고
- 0.5가 나름 G에서 미미한 우위를 보여서 (C에서의 감소량 보다)
