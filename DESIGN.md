# venhance — ローカル動画エンハンサー CLI 設計書

ローカル（macOS / Apple Silicon）で完結する動画エンハンスCLI。
2つの機能を提供する:

- **フレーム補間 (interpolate)**: 30fps → 60fps など、AIでフレームを生成してfpsを上げる
- **超解像 (upscale)**: 720p → 1440p/4K など、AIで解像度を上げる

コマンド名: `venhance`（リポジトリ名は `video-enhancer` などに後でリネーム可）

---

## 1. 基本方針

- **モデルは学習済みのものを使う**。自前学習はしない。
  - フレーム補間: **RIFE v4.x**（実用速度と品質のバランスで事実上の定番）
  - 超解像: **Real-ESRGAN**（実写向け）/ **Real-ESRGAN anime**（アニメ向け）
- **動画のデコード/エンコード/音声はすべてffmpegに任せる**。
  自分で書くのは「フレームを取り出す → モデルに通す → 書き戻す」の配管だけ。
- **中間ファイルを作らないストリーミング処理**を基本とする
  （rawvideoをパイプで流す。10分の1080p動画をPNG展開すると数十GBになるため）。

### 難易度について

超解像・フレーム補間とも、コアは「学習済みモデルにフレームを入れて出す」だけ。
本当に手間がかかるのは共通のパイプライン部分:

| 部分 | 難易度 | 備考 |
|---|---|---|
| ffmpeg入出力パイプ | 中 | 一度書けば両機能で共用 |
| 超解像（Real-ESRGAN適用） | 低 | 1フレーム独立。ループを回すだけ |
| フレーム補間（RIFE適用） | 中 | 前後フレームのペア管理、シーンチェンジ対応 |
| 音声・タイムスタンプ維持 | 中 | ffmpegの知識でカバー |

---

## 2. 技術スタック

| 項目 | 選定 | 理由 |
|---|---|---|
| 言語 | Python 3.12+ | モデルエコシステムがPython前提 |
| パッケージ管理 | uv | 高速、`uv tool install` でCLI配布も楽 |
| CLIフレームワーク | Typer | サブコマンド + 型ヒントベース |
| 推論 | PyTorch (MPSバックエンド) | Apple M5のGPUを使う。`device="mps"` |
| 動画I/O | ffmpeg (subprocess + パイプ) | Homebrewで導入。`imageio-ffmpeg`同梱も検討 |
| 進捗表示 | rich | プログレスバー、ETA表示 |

**代替案（Plan B）**: `rife-ncnn-vulkan` / `realesrgan-ncnn-vulkan` のビルド済みバイナリを
オーケストレーションする方式。Python側の実装が激減する反面、画像ファイル経由になり
ディスクI/Oが重い。PyTorch/MPSでの実装が詰まったときの逃げ道として温存する。
このためにも推論部分は Backend インターフェースで抽象化しておく（§5）。

---

## 3. CLI設計

```
venhance interp  INPUT [-o OUTPUT] --fps 60          # フレーム補間のみ
venhance interp  INPUT --factor 2                    # 倍率指定 (30→60)
venhance upscale INPUT [-o OUTPUT] --scale 2         # 超解像のみ (2x/4x)
venhance run     INPUT --fps 60 --scale 2            # 両方（補間→超解像の順）
venhance probe   INPUT                               # 入力情報の表示 (fps/解像度/コーデック)
venhance models  [download|list]                     # モデル重みの管理
```

共通オプション:

```
-o, --output PATH        省略時: input_60fps.mp4 / input_2x.mp4 のように自動命名
--model NAME             rife-v4.6 / realesrgan-x4 / realesrgan-anime など
--codec {h264,hevc,prores}   出力コーデック（デフォルト: hevc + videotoolbox）
--crf / --bitrate        品質指定
--chunk-secs N           チャンク処理の単位（デフォルト30秒）
--resume                 中断したジョブの再開
--dry-run                実行計画とffmpegコマンドを表示するだけ
```

### 処理順序（`run` で両方指定時）

**補間 → 超解像** の順で行う。低解像度のうちに補間する方が補間コストが安く、
超解像は生成後の全フレームに1枚ずつかければよい。1パスのストリームとして
`decode → RIFE → Real-ESRGAN → encode` を直結し、中間動画は作らない。

---

## 4. パイプラインアーキテクチャ

```
                    ┌─────────────────────────────────────────┐
 input.mp4 ──────►  │ ffmpeg (decode)                          │
                    │  -i in.mp4 -vf fps=CFR化 -pix_fmt rgb24  │
                    │  -f rawvideo pipe:1                      │
                    └───────────────┬─────────────────────────┘
                                    │ raw RGB フレーム (stdout)
                                    ▼
                    ┌─────────────────────────────────────────┐
                    │ venhance コア (Python)                    │
                    │  FrameSource → [Interpolator] →          │
                    │  [Upscaler] → FrameSink                  │
                    │  (各ステージはジェネレータで連結)          │
                    └───────────────┬─────────────────────────┘
                                    │ raw RGB フレーム (stdin)
                                    ▼
                    ┌─────────────────────────────────────────┐
                    │ ffmpeg (encode)                          │
                    │  -f rawvideo pipe:0 ... -c:v hevc_       │
                    │  videotoolbox  + 元動画から音声をコピー    │
                    └───────────────┬─────────────────────────┘
                                    ▼
                                output.mp4
```

- 各ステージは `Iterator[Frame]` を受けて `Iterator[Frame]` を返すジェネレータ。
  補間・超解像の有効/無効はステージの差し込みで表現する。
- **Interpolator**: 直前フレームを1枚保持し、ペア (t, t+1) をRIFEに渡して中間フレームを生成。
  シーンチェンジ検出（フレーム差分 or RIFE内蔵のフラグ）を行い、カット境界では
  補間せず前フレームを複製する（カットまたぎのモーフィング破綻を防ぐ）。
- **Upscaler**: 1フレームずつReal-ESRGANに通す。VRAM節約のため必要ならタイル分割
  （512pxタイル + オーバーラップ）で処理する。

### 音声・メタデータ

- 音声は再エンコードせず `-map 1:a -c:a copy` で元ファイルからコピー。
- 入力がVFR（可変フレームレート、スマホ撮影に多い）の場合はデコード時に
  `fps` フィルタでCFR化してから補間する。`probe` でVFRを検出したら警告を出す。
- 色空間 (BT.709など) とHDRメタデータは v1 ではSDR/BT.709のみサポートと割り切る。
  HDR入力は検出して警告。

### チャンク処理とレジューム

長尺動画対策として、`--chunk-secs`（デフォルト30秒）単位で分割処理し、
各チャンク完了ごとに一時ディレクトリ（`~/.cache/venhance/jobs/<hash>/`）へ
セグメントファイルを書く。全チャンク完了後にffmpegのconcat demuxerで結合。
`--resume` は完了済みセグメントをスキップする。ジョブの同一性は
「入力ファイルのハッシュ + オプション」で判定する。

---

## 5. モジュール構成

```
video-enhancer/
├── pyproject.toml
├── DESIGN.md
├── src/venhance/
│   ├── cli.py               # Typer エントリポイント
│   ├── probe.py             # ffprobe ラッパー（fps/解像度/VFR/HDR検出）
│   ├── pipeline.py          # ステージ連結・チャンク・レジューム制御
│   ├── io/
│   │   ├── decoder.py       # ffmpeg decode → フレームイテレータ
│   │   └── encoder.py       # フレームイテレータ → ffmpeg encode
│   ├── stages/
│   │   ├── base.py          # Stage プロトコル定義
│   │   ├── interpolate.py   # RIFEステージ（ペア管理・シーン検出込み）
│   │   └── upscale.py       # Real-ESRGANステージ（タイル分割込み）
│   ├── backends/
│   │   ├── base.py          # InferenceBackend プロトコル
│   │   ├── torch_mps.py     # PyTorch + MPS 実装（デフォルト）
│   │   └── ncnn.py          # ncnnバイナリ実装（Plan B、後回し可）
│   └── models/
│       └── registry.py      # モデル重みのDL・キャッシュ・検証（sha256）
└── tests/
    ├── fixtures/            # 数秒のテスト用クリップ（生成スクリプト付き）
    └── ...
```

モデル重みは初回実行時に GitHub Releases 等から `~/.cache/venhance/models/` へ
ダウンロードし、sha256で検証する（`venhance models download` で事前取得も可）。

---

## 6. パフォーマンス目安（Apple M5想定）

| 処理 | 想定速度 | 1080p/30fps 5分動画の所要 |
|---|---|---|
| RIFE補間 (1080p, 30→60) | 実測 約7fps処理 (fp32/MPS, M5) | 約25分（fp16化で短縮余地あり） |
| Real-ESRGAN 2x (1080p→4K) | 2〜6 fps処理 | 25〜75分 |
| 両方 | 超解像が支配的 | 〜1.5時間 |

超解像は圧倒的に重いので、`--scale` 指定時は開始前に推定所要時間を表示して
確認を促す。将来の最適化候補: Core ML変換（ANE活用）、fp16推論、バッチ推論。

### 最適化の実測メモ（M5, 720p入力, MPS）

| 施策 | 結果 | 判断 |
|---|---|---|
| fp16推論（RIFE） | 1.36倍高速、fp32比 PSNR 56dB | ✅ 採用（MPS/CUDAデフォルト、`--fp32` で無効化） |
| fp16推論（Real-ESRGAN） | 採用済み（当初から） | ✅ |
| バッチ推論（SR, batch=2/4） | 効果なし（むしろ約5%劣化。720pで既にGPU飽和） | ❌ 不採用 |
| channels_last | 効果なし | ❌ 不採用 |
| ダブルバッファ（転送と推論のオーバーラップ） | 効果なし（MPSの非同期キューで既に隠蔽済み） | ❌ 不採用 |
| デコード先読みスレッド | 不要（デコードは0.7ms/frameで律速でない） | ❌ 不採用 |

残る大きな伸びしろは Core ML / ANE 変換のみ。

---

## 7. マイルストーン

1. **M1: 骨格** — `probe` + decode→（無変換）→encode のパススルーが動く。
   音声コピー・チャンク結合含む。ここでパイプラインの正しさを固める。 ✅ 実装済み（チャンク分割は未実装、ストリーミング一発処理）
2. **M2: interp** — RIFE (torch/MPS) を組み込み `interp` 完成。シーン検出込み。 ✅ 実装済み（rife-v4.7 / v4.26、任意fps比対応）
3. **M3: upscale** — Real-ESRGAN + タイル分割で `upscale` 完成。 ✅ 実装済み（realesr-general / realesr-anime / realesrgan-x4plus、fp16推論、タイル分割対応。実測: 720p入力 2x で general 約3.6fps, anime 約6.6fps, x4plus 約0.12fps @ MPS fp16）
4. **M4: run/resume** — 複合パイプライン、レジューム、進捗ETA、モデル管理。 🔶 `run`（interp→upscale 1パス複合）と進捗ETA・モデル管理は実装済み。チャンク分割・レジュームは未実装。
5. **M5以降（任意）** — ncnnバックエンド、Core ML最適化、アニメ用モデル、
   バッチ処理（ディレクトリ一括）、HDR対応。

## 8. テスト方針

- `ffmpeg -f lavfi -i testsrc` で数秒のフィクスチャ動画を生成（リポジトリに置かない）。
- 検証項目: 出力のフレーム数（補間で2倍±1）、duration一致、解像度、音声ストリーム保持。
- 画質はSSIM/PSNRの回帰チェック程度に留める（絶対品質のテストはしない）。
- パイプ切断・中断（Ctrl-C）時に一時ファイルが残らない/レジュームできること。
