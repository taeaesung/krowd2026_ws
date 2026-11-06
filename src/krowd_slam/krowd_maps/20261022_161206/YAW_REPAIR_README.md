# 예선 적용 완료 — 2026-10-26

적용 경로: `/home/krowd/krowd_maps/20261022_161206/global_map_ready.mm`

백업: `/home/krowd/krowd_maps/20261022_161206/backup_before_yaw_apply_20261026_161050`

기존 예선 map_dir / map_filename 명령으로 보정본을 읽는다. native MM, export PLY, 초기 정합용 PLY를 함께 교체했다.
waypoint·corridor·미션/정지/학교/감속 구간과 센서/제어 설정은 유지했다. 본선 파일은 변경하지 않았다.
현재 localization 프로세스는 발견되지 않아 재시작하지 않았다. 다음 localization 실행부터 반영된다.
구조 검사 PASS, native MM 재로딩 성공, 후보와 실제 적용 파일 SHA256 일치. 실차 전체 localization 검증은 아직 남아 있다.

```bash
ros2 launch mola_slam_mapping localization.launch.py \
  map_dir:=/home/krowd/krowd_maps/20261022_161206 \
  map_filename:=global_map_ready.mm
```

원래 map_ready.simplemap/keyframes_ready.tum/trajectory_reference.csv는 역사적 진단 자료로 유지했다. 현재 native MM은 raw bag와 trajectory_body_reference.csv에서 생성한 결과다.
상세 검증: `/home/krowd/map_yaw_repair_20261026/REPORT.md`

---

# 예선·본선 맵 보정 및 오프라인 검증 — 2026-10-26

## 판정

- **예선: 보정 후보 생성 완료, 원본 백의 별도 스캔을 사용한 비교에서 개선 확인.** 전체 ROS localization 재생·실차 정지/코너 검증은 미실시. 즉시 대회 최종 합격으로 간주하지 않는다.
- **본선: yaw 보정 후보 생성 및 파일/표면 검증 완료.** 원본 `0913_1` 백과 해당 시점 ESKF body yaw가 없어 독립적인 정합 성능 개선은 미확인. 기존 본선 맵을 대체하지 않았다.
- 원본 맵, 운용 중 센서·제어 설정, latest 링크를 변경하지 않았다. 모니터링 대상 원본 파일 SHA256 재확인 결과 전부 동일하다.

## 대상과 결과물

| 구분 | 확인한 기존 수동 운용 맵 | 새 후보 폴더 | 후보의 선택 파일 |
|---|---|---|---|
| 예선 | `~/krowd_maps/20261022_161206/global_map_ready.mm` | `~/krowd_maps/20261022_161206_yawfixed_20261026` | `global_map_ready.mm` |
| 본선 | `~/krowd_maps/20260913_103732_repaired/global_map_repaired.mm` | `~/krowd_maps/20260913_103732_yawfixed_20261026` | `global_map_ready.mm` |

후보는 `MAP_CANDIDATE` 상태다. `READY_FOR_LOCALIZATION` 인증 표시는 새로 만들지 않았다.

## 발견한 원인과 보정 기준

1. `trajectory_reference.csv`의 방향은 차량 body yaw와 항상 같지 않다. 전방 약 2.5038m의 GNSS 안테나 course는 회전 시 body yaw보다 앞설 수 있다. 예선 원본 백에서는 이 차이가 최대 약 9.43°였다. 따라서 course에 모든 LiDAR yaw를 강제로 맞추는 보정은 부적절하다.
2. 예선 시작 keyframe에는 실제 큰 방향 오류가 있다. 예를 들어 KF5는 RTK로 검증된 ESKF body yaw 약 +1.1°에 대해 LiDAR pose 약 -111°였다. 초반 다른 프레임에도 약 34~35° 오류가 있다. 시작 구간 진단 시 RTK Fixed/heading verified가 true였으므로, 이 오류를 단순 RTK heading 미확정으로 설명할 수 없다.
3. 기존 SimpleMap의 pose만 수정한 후보와 시작부만 수정한 후보도 만들었지만, 별도 스캔 정합에서 XY 오차가 악화돼 선택하지 않았다. 원인은 pose 외에 저장된 deskewed observation의 왜곡, 기준 궤적의 오차 등이 남을 가능성이다. 이번 비교만으로 단일 원인을 확정하지 않는다.
4. 예선 선택 후보는 원본 백의 raw Ouster 스캔을 새로 변환해 만들었다. body yaw는 기록된 ESKF 전역 자세를 사용하고, 위치는 기존 map 기준 reference trajectory를 유지했다. 10ms 구간 deskew, stride4, 0.2m voxel을 적용했다. native MM의 localmap은 `mola::IncrementalPointCloud`이며 공분산 정합을 지원한다.
5. 본선은 원본 raw bag이 현재 접근 가능한 저장소에 없다. 안테나 궤적과 전방 레버암, 비홀로노믹 전진 모델로 body yaw를 추정해 원래 keyframe의 gravity/기울기를 유지하면서 회전시켰다. 이 모델을 예선의 실제 ESKF와 비교하면 차이 중앙값 0.29°, p95 1.30°, 최대 4.28°였다. 이는 본선의 실제 정확도를 보장하지 않는다.

## 예선 검증

원본 백: `/home/krowd/rosbags/mapping_1017_090530`.
전체 raw scan 848개 중 847개를 받아 433,892점의 선택 후보를 생성했다. 시작부·회전·후반에서 13개 non-keyframe raw scan을 선정했다. 비교용 맵에서는 이 13개를 **제외**하고 431,302점을 생성했다. 기존 맵에도 이들은 선택 keyframe observation이 아니다.

양쪽 맵에 같은 검증 스캔, 같은 초기 오차(+0.5m X, -0.4m Y, +4° yaw), 같은 알고리즘을 적용했다. 기준은 기록된 ESKF body yaw 및 기존 map reference position이다. 별도의 측량 ground truth가 아니며, 궤적을 지도 생성에도 이용했으므로 지리적 절대 정확도 검증과 구분한다.

### 공분산 기반 정합: Matcher_Cov2Cov / Solver_GaussNewton

| 지표 | 기존 맵 | 선택 후보의 hold-out 맵 |
|---|---:|---:|
| XY 오차 중앙값 | 0.464m | **0.212m** |
| XY 오차 p95 | 0.697m | **0.450m** |
| XY 오차 최대 | 0.715m | **0.492m** |
| yaw 절대 오차 중앙값 | 1.55° | **0.61°** |
| yaw 절대 오차 p95 | 3.76° | **1.90°** |
| yaw 절대 오차 최대 | 3.79° | **2.44°** |

13개 스냅샷 정합 비교다. MOLA 전체 런타임의 추적 주기·지연·heading_verified 게이트까지 검증한 것은 아니다. 모든 위치에서 2° 이내 목표를 충족했다고 주장하지 않는다.

별도 point-to-point Horn 정합에서도 XY 중앙값 0.360→0.238m, yaw p95 3.72→1.79°로 개선됐다. 일부는 MaxIterations로 종료됐으므로 해당 시험만으로 수렴을 보장하지 않는다.

`validate_ready_map.validate()` 구조·궤적·표면·경로 검사 결과 예선 실패 항목은 없다. native 파일 재로딩 및 Cov2Cov 실제 pairing도 성공했다. 표면 두께는 중앙값 0.0953m, p95 0.1684m이며, 기존 0.0887m/0.1771m와 비교하면 중앙값은 소폭 증가, p95는 감소했다.

## 본선 검증과 한계

- 538개 keyframe, native 점 수 1,337,180.
- 원래 위치·경로·지리 기준을 유지했다. position reference 비교 최대 0.456m는 기존과 같다.
- yaw 수정량 중앙값 1.38°, p95 7.33°, 최대 12.42°. 이는 추정 body heading과 기존 맵 차이이며 독립 측정된 yaw 오류가 아니다.
- 표면 두께 중앙값 0.08532→0.08363m, p95 0.18073→0.17854m로 소폭 개선됐다. 표면 선명도가 yaw 정확도를 증명하지 않는다.
- SimpleMap을 다시 읽어 확인한 자세와 저장 TUM 자세 차이는 0.0001° 미만이다.
- initializer PLY는 후보 native map export와 동일하게 구성했다. 독립 GNSS cloud로 재생성한 것이 아니며 `INITIALIZER_PROVENANCE.md`에 명시했다.
- 원본 `0913_1` 백을 확보하거나 현재 장소에서 후보와 기존 맵을 비교하기 전에는 본선 최종 운용 파일로 승격하지 않는다.

## 경로·프레임 보존

- 예선 waypoint 411개 / 약 585.25m, 본선 647개 / 약 771.04m.
- `waypoints.csv`, `corridor.csv` 바이트와 SHA256는 원본과 동일하다. yaw 보정에 맞춰 mission index를 바꾸지 않았다.
- native map georeference sidecar, 지리 origin, map frame은 원본 기준을 유지했다. 센서 TF나 ESKF 설정을 고치지 않았다.
- **별도 통합 `course_final.yaml` / `course_preliminary.yaml`은 더 오래된 맵을 가리킨다.** 이번 수동 운용 맵/후보와 경로점 개수도 다르다. 이 프로파일은 수정하지 않았으므로 통합 런치가 후보를 자동 선택한다고 생각하면 안 된다. mission/차선/구간 index 확인 후 별도로 전환해야 한다.

## 수동 비교 실행 명령어

다음은 후보를 명시적으로 선택하는 명령이다. 실행하지 않았으며 현재 노드를 바꾸지 않았다. 두 맵을 동시에 같은 ROS domain에 띄우지 않는다.

예선 후보:

```bash
ros2 launch mola_slam_mapping localization.launch.py \
  map_dir:=/home/krowd/krowd_maps/20261022_161206_yawfixed_20261026 \
  map_filename:=global_map_ready.mm
```

본선 후보 — 정합 개선 미확인, 비교용:

```bash
ros2 launch mola_slam_mapping localization.launch.py \
  map_dir:=/home/krowd/krowd_maps/20260913_103732_yawfixed_20261026 \
  map_filename:=global_map_ready.mm
```

최종 확인은 각 장소의 시작점·직선·코너 후·경로 끝에서 동일 timestamp의 map 기준 ESKF yaw와 LiDAR localization yaw/position을 비교한다. 메시지 수신 시각 차이를 yaw 오차로 오인하지 않는다. 비교 시 차량 제어 게이트/안전 기준을 완화하지 않는다.

## 재현 자료

작업 폴더 `/home/krowd/map_yaw_repair_20261026`:

- `prepare_candidates.py`: gravity를 보존하는 pose yaw 후보 생성.
- `build_body_reference_cloud.py`: raw scan 재구성과 hold-out scan 제외 옵션.
- `converter/`: native IncrementalPointCloud MM 생성 소스 및 빌드.
- `validation_scans/scans.json`: 13개 검증 스캔의 시각·기준 자세·초기 오차.
- `run_scan_comparison.py`, `offline_icp.yaml`, `covariance_smoke.yaml`: 두 정합 방식 비교.
- `preliminary_raw_scan_summary_body_cov.json`: 최종 선택 후보 hold-out 공분산 정합 요약.
- `preliminary_raw_scan_comparison_body_cov.json`: 스캔별 결과. 각 `icp_body_cov_*.log`는 상세 pairing/수렴 출력.
- `preliminary_raw_scan_summary_body.json`: 별도 Horn 비교.
- `preliminary_raw_scan_summary.json`, `..._startup.json`: 선택하지 않은 pose-only 후보의 악화 결과도 보존.
- `final_validation.json`: 원본 hash, 경로 hash, surface/구조 검사, serialized pose 비교.
- `preliminary_selected_mm_info.txt`, `final_mm_info.txt`: native layer 및 georeference 로딩 결과.
- 후보 폴더의 `offline_validation.json`, `yaw_repair_report.json`도 함께 참조.

`map_ready.simplemap`은 예선에서 pose-only **진단 후보**다. 최종 예선 native MM은 raw bag에서 직접 생성했으므로 이 SimpleMap을 다시 sm2mm로 변환하면 선택한 예선 맵이 재현되지 않는다. 최종 raw-rebuild provenance를 `georeference.yaml`에도 명시했다.
