# 트랙 A — 실스캔 기반 2.5D 결함 합성

Zivid organized XYZ/RGB 스캔의 표면에 찍힘·긁힘을 주입한다.
CPU 코드이며 CUDA, 모델 가중치, GPU가 필요하지 않다.
NPZ 캐시로 실행하면 Zivid SDK도 필요하지 않다. 원본 ZDF를 새로 변환할 때만 SDK가 필요하다.

## 로컬 검증 결과 (2026-09-08)

- Windows Python 3.11에서 회귀 테스트 16개 통과(메시 위상·결측·PLY/OBJ 되읽기 포함).
- 범퍼 1개 샘플(결함 4개), 브래킷 1개 샘플(결함 2개)을 실제 스캔으로 생성.
- 원본 유효점 보존, 변형 밖 XYZ 보존, 부품 밖 변형 0, 제외 영역 변형 0, 인스턴스 면적 일치 확인.
- PCD 저장 후 좌표·색상·점 순서 되읽기 및 픽셀 인덱스와 클래스 라벨 정합 확인.
- 출력 PCD 점 수: 범퍼 3,593,855점, 브래킷 2,924,678점.
- Linux/vast.ai 실행은 아직 미검증. 아래 순서로 서버에서 재현한다.
- 메시 추가 검증: 범퍼 2,065,313정점·4,105,131삼각형, 브래킷 1,749,509정점·3,448,722삼각형.
  두 부품 모두 PLY/OBJ 되읽기 및 메시 정점 라벨 정합 통과.

## vast.ai에서 실행

이 폴더의 코드와 `data/2-2_raw.npz`, `data/0023_raw.npz`만 업로드하면 된다.
캐시는 로컬 `C:/Users/biop9/zdf-synth/data`에 있다. 기존 out 폴더나 모델 파일은 필요하지 않다.
설정의 `data/2-2.zdf` 경로는 같은 폴더의 `2-2_raw.npz`를 우선 읽는다.
서버 대여와 데이터 업로드는 사용자가 진행한다. 이 코드에서 유료 서버를 생성하지 않는다.

Python 3.11 환경에서, 업로드한 프로젝트 폴더로 이동한 뒤 실행:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s . -p test_pipeline.py -v
python validate.py --data-dir data --out out/check_01 --pcd --compare
```

한글 확대 그림용 폰트가 없는 Ubuntu 환경에서는 `fonts-noto-cjk`를 설치하면 된다.
폰트 없이도 계산은 가능하고 비교 그림 제목은 영문이다.
설치된 의존성을 기록하려면 `python -m pip freeze > out/check_01/environment.txt`를 실행한다.
메모리 최대 사용량을 확인하려면 Linux에서 `/usr/bin/time -v`를 validate 명령 앞에 붙인다.
샘플은 순차 처리한다. 메모리 사용량을 측정하기 전에는 여러 프로세스로 동시에 생성하지 않는다.

확인할 파일:

- `out/check_01/validation.json`: 구조·라벨 검증 결과. 실패 시 프로세스가 비정상 종료한다.
- `out/check_01/compare/real_vs_synth.png`, `profile.png`, `stats.json`: 품질 검토.
- 각 부품의 `0000/ignore.png`, `rgb.png`, `overlay.png`: 기존 실물 결함 제외 범위와 새 결함 위치.
- `effective_config.yaml`: 실제 실행 경로와 파라미터.

출력 폴더는 기존 결과를 덮어쓰지 않는다. 재실행할 때 `check_02` 등 새 이름을 사용한다.

검증 후 작은 배치:

```bash
python inject.py --scan bumper_cover --samples 5 --seed 42 --mask-only --pcd --out out/batch_01
python inject.py --scan weld_bracket --samples 5 --seed 42 --mask-only --pcd --out out/batch_01
```

결함 전체가 들어갈 공간이 없으면 오류로 중단한다. 실패 후보를 강제로 배치하지 않는다.
크기/개수를 줄이거나 배치 위치를 검토하고 새 출력 폴더에서 재실행한다.
완료된 배치를 판별할 때 `run.json` 존재 여부를 확인한다. 중간 실패 폴더를 학습에 넣지 않는다.

## 폴리곤 오브젝트 출력

점군 외에 삼각형 면이 있는 메시를 저장할 수 있다. `surface.ply`는 정점 RGB를 포함하고,
`surface.obj`는 표준 OBJ 형상을 저장한다(재질/텍스처 없음).
인접한 원본 격자 점을 연결하므로 원본 좌표와 합성 변형을 유지한다. 평활화/다운샘플링하지 않는다.
결측과 부품 밖 영역을 연결하지 않고, 기본적으로 점 간격 중앙값의 3배보다 긴 모서리도 제거한다.
단일 시점에서 관측된 열린 표면이며, 뒷면·두께·결측 부위의 형상을 생성하지 않는다.
좌표 단위는 mm다. 가져오는 프로그램이 m 단위를 쓰면 0.001 배율을 적용한다.

원본 부품의 오브젝트만 생성(원래 실물 결함도 그대로 포함):

```bash
python mesh_export.py --scan bumper_cover --format both --out out/bumper_object_01
```

원본 실물 결함 제외 영역을 메시에서도 비우려면 `--exclude-known-defects`를 추가한다.
합성 샘플 메시에는 학습 제외 영역이 항상 빠진다.

결함 합성과 함께 점군·PLY·OBJ 생성:

```bash
python inject.py --scan bumper_cover --samples 1 --seed 42 --mask-only --pcd --mesh --mesh-format both --out out/mesh_batch_01
```

출력은 `out/mesh_batch_01/bumper_cover/0000/mesh/` 안의 `surface.ply`, `surface.obj`다.
`mesh.json`에는 정점·삼각형 수와 되읽기 검증 여부가 기록된다.
`vertex_pixel_indices.npy`는 원본 격자 픽셀과 메시 정점을 연결하고, 합성 메시의
`vertex_labels.npy`는 해당 정점의 결함 클래스 라벨이다. OBJ 로더는 정점을 재정렬할 수 있으므로
사이드카 라벨은 파일에 기록된 정점 순서(또는 검증된 PLY 순서)에 대응한다.
기존 점군 PLY와 달리 이 PLY에는 실제 `face` 요소가 들어 있다.
전체 해상도 메시이므로 수백만 개의 면이 생길 수 있다. 처음에는 샘플 1개로 확인한다.

서버에서 두 부품의 메시 생성·되읽기·라벨 검증:

```bash
python validate.py --data-dir data --out out/mesh_check_01 --mesh --mesh-format both
```

## 라벨과 제외 정책

- `seg.png`: scratch=0, dent=1, 배경=255. `inst.png`: 배경=0, 양수=인스턴스 ID.
- 마스크는 절대 변위의 최대값 대비 10% 초과 영역이다. 림의 바깥쪽 변위도 포함한다.
- `ignore.png`: 255인 곳은 학습 제외. `valid.png`: 유효 XYZ이면서 제외 영역 밖인 픽셀.
- 범퍼 실물 찍힘 중심 `(559,1114)` 주변 반경 220px는 잠정 제외한다. 학습 전에 범위 확인이 필요하다.
- 제외 영역은 출력 RGB를 검게, depth/normal을 0으로 만든다. 원본 XYZ는 수정하지 않고 PCD에서는 제외한다.
- 학습 로더는 ignore/valid 마스크를 반영해야 한다. 검은 영역을 정상 부품으로 학습하지 않는다.
- PCD의 `point_labels.npy`와 `point_pixel_indices.npy`는 점 순서와 일치한다.
  픽셀 인덱스는 원본 H×W 격자를 행 우선으로 펼친 인덱스다.
- `--mask-only`는 정확한 픽셀·점 라벨을 저장하며 YOLO 텍스트는 생성하지 않는다.
  YOLO를 요청했을 때 분리 영역/구멍/퇴화 폴리곤이 있으면 오류로 중단한다.
  가장 큰 외곽선만 남기거나 구멍을 채워 성공으로 보고하지 않는다.
- `overlay.png`는 축소 시각화다. 정답 마스크로 사용하지 않는다.
- `depth_mm.png`는 0.1mm 단위 uint16이며 정확한 XYZ는 PCD를 사용한다.

## 실물 비교의 현재 한계

`--radius`는 타원의 긴 축 방향 0 교차 반경, `--rim`은 솟은 림의 최대 높이(mm)다.
기존 계측의 11.27mm는 절반 깊이 반경이므로 같은 의미가 아니다.
기본 비교 반경 22.5mm·림 0.25mm는 검토용 후보이며 피팅 완료값이 아니다.
정반사 함수는 구현했지만 기본 가중치 0이다. 다음처럼 옵션을 주어 비교할 수 있다:

```bash
python compare.py --real 559 1114 --at 760 1560 --specular 0.5 --out out/compare_specular
```

곡면 적합 고리는 림 전체 바깥에 있어야 한다. 기본값은 32~42mm다.
로컬 비교 결과: 실물 최대 깊이 4.225mm, 합성 3.063mm, 단면 상관 0.972, RMSE 0.631mm.
기존 고리 18~30mm의 실물 깊이 3.059mm와 차이가 크므로 배경 곡면 적합에 민감하다.
높은 상관만으로 유사성 통과를 선언하지 않는다. 현재 실물과 합성의 RGB 대비 차이도 남아 있다.
다음 품질 작업은 결함 없는 주변 표면의 적합 영역 선정·깊이/반경 재계측·정반사 보정이다.
법선 각도 잔차 통계에는 실제 표면 구조도 포함되므로 순수 센서 노이즈로 해석하지 않는다.
기존 결측은 보존하지만 새 결함에 따른 센서 그림자/결측은 모델링하지 않는다.
