# krowd_pp_stanleyff_control — PP + Stanley 블렌딩 제어기 (곡률 feedforward 있음)

`krowd_pp_stanley_control`과 **Stanley 항 하나만 다르다**:

```
δ_st = feedforward_gain · atan(L·κ)  +  0.6 · heading_err  +  atan(0.3 · e / (3.0 + v))
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 이 패키지에만 있음
```

`κ`는 base_link 투영점의 경로 곡률(yaw-rate 감쇠가 쓰는 그 값), `L` 축거 1.34 m. 코너에서 필요한 조향각을
지도에서 미리 넣어 주므로 Stanley가 **오차가 쌓이기를 기다리지 않고** 코너를 돈다. Pure Pursuit는
원호 위에서 정확히 `atan(L/R)`을 내므로(`test_blended_steering.py`), 이 항을 넣으면 두 법칙이 같은
기하 항을 갖게 되고 가중치는 "오차를 어떤 방식으로 지울지"만 고른다 — 그래서 여기서는
`blend.max_stanley_weight`를 0.5 위로 올려도 코너 기하가 사라지지 않는다.

- `blend.feedforward_gain: 1.0` — 기본. `0.0`이면 자매 패키지와 **출력이 같아야** 한다
  (두 패키지가 갈라졌는지 확인하는 방법이기도 하다).
- 노드 이름 `pp_stanleyff_control_node`. arm 서비스 없음 (2026-10-03 제거: 준비되면 스스로 출발). 토픽은 동일.
- **두 패키지를 동시에 띄우지 말 것** — 둘 다 `/target_angle` `/target_speed` `/brake`를 낸다.
- `avoidance_stanley.enabled: true`인 통합 실행에서는
  `/integration_control/avoiding=true` 동안에만 `cone_avoidance_0924_3`의
  지연보상 Stanley 법칙을 같은 노드 안에서 사용한다. 기본값은 `false`라서 일반 주행에는 영향이 없다.

실행·맵 프로파일·시뮬레이션·진단·튜닝·차량 확인 항목은 [krowd_pp_stanley_control/README.md](../krowd_pp_stanley_control/README.md)와 같다
(패키지 이름과 노드 이름만 바꿔 읽는다).

## 실행 요약

```bash
# 실차: 터미널 1 = 센서 스택 + localization.launch.py (RViz 는 여기서), 터미널 2 =
ros2 launch krowd_pp_stanleyff_control control.launch.py

# 시뮬: 터미널 1 = 차량 모델 + 맵 좌표 + RViz, 터미널 2 = 같은 제어 launch
ros2 launch krowd_pp_stanleyff_control sim.launch.py map_dir:=~/krowd_maps/20260913_103732
ros2 launch krowd_pp_stanleyff_control control.launch.py sim:=true      # + force_route_only:=true (분석 PC 576줄 맵)
```

## A/B 시험

같은 맵, 같은 출발점, 같은 `speed_limit_kmph`로 두 패키지를 번갈아 돌리고 비교한다:

| 볼 것 | 어디서 |
|---|---|
| 코너 안 정상상태 횡오차 (바깥 밀림) | `/control/cross_track_error` — FF 있으면 0 근처, 없으면 바깥(코너 방향 반대 부호)으로 밀림 |
| 코너 진입 조향 시점 | `/control/steering_debug` 의 `delta_stanley_deg` vs `delta_pp_deg`, `feedforward_deg` |
| 후보 조향 개입 | `/control/corridor_status` 의 `candidate_offset_deg` — FF 없는 쪽이 벽에 더 자주 걸린다 |

시뮬레이터에서 먼저 보고 실차로 간다. 조향모터 지연·속도 부족은 시뮬에 있고 노면 미끄러짐은 없다.

```bash
cd ~/krowd2026_ws/src/krowd_pp_stanleyff_control && python3 -m pytest -q test
```
