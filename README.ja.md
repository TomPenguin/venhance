# venhance

ローカル（macOS / Apple Silicon）で動作するAI動画エンハンサーCLI。
フレーム補間（30fps → 60fps など）と超解像（720p → 1440p など）をサポート。

[English README is here](README.md)

## なぜ venhance か

無料のAI動画エンハンスツールは存在しますが、macOS ではどれも一長一短です。
[Video2X](https://github.com/k4yt3x/video2x) は Vulkan 依存で macOS ネイティブビルドがなく、
`*-ncnn-vulkan` 系 CLI は連番画像ベースで動画・音声の配管を自前で組む必要があり、
その他は Windows 中心の GUI が主流です。venhance はその隙間を埋めます:

- **Apple Silicon ネイティブ推論** — PyTorch MPS + fp16。MoltenVK/Vulkan 変換層なし
- **動画 in → 動画 out** — コマンド1発。PNG 連番への展開は不要
- **中間ファイルなしのストリーミング処理** — rawvideo を ffmpeg とパイプで直結
- **VideoToolbox ハードウェアエンコード** — HEVC / H.264 / ProRes 出力
- **補間 + 超解像を1パスで** — 低解像度のうちに補間 → 全フレームを超解像
- **音声・色メタデータを維持** — 音声はそのままコピー、色空間タグはソースから引き継ぎ

## 必要なもの

- macOS (Apple Silicon)
- ffmpeg: `brew install ffmpeg`
- [uv](https://docs.astral.sh/uv/): `brew install uv`

## インストール

グローバルコマンドとしてインストール（推奨）:

```sh
uv tool install .        # リポジトリのルートで実行
uv tool update-shell     # ~/.local/bin がPATHに無い場合のみ（シェル再起動が必要）
```

以降はどこからでも `venhance ...` で実行できます。コード更新後は
`uv tool install . --force` で再インストールしてください。
アンインストールは `uv tool uninstall venhance`。

開発用にリポジトリ内で直接動かす場合:

```sh
uv sync
uv run venhance ...
```

## 使い方

```sh
# 入力動画の情報を表示
venhance probe input.mp4

# 30fps -> 60fps（初回はモデル重みを自動ダウンロード）
venhance interp input.mp4 --fps 60

# 倍率指定・出力先指定
venhance interp input.mp4 --factor 2 -o out.mp4

# 超解像 720p -> 1440p（初回はモデル重みを自動ダウンロード）
venhance upscale input.mp4 --scale 2

# 補間 + 超解像を1パスで（30fps 720p -> 60fps 1440p）
venhance run input.mp4 --fps 60 --scale 2

# モデル管理
venhance models list
venhance models download rife-v4.26
```

主なオプション（`interp`）:

| オプション | 説明 |
|---|---|
| `--fps` / `--factor` | 目標fps（`60`, `59.94`, `60000/1001`）か倍率のどちらか一方 |
| `--model` | `rife-v4.7`（デフォルト・高速）/ `rife-v4.26`（高品質寄り） |
| `--codec` | `hevc`（デフォルト）/ `h264` / `prores`（VideoToolboxハードウェアエンコード） |
| `--quality` | 1-100（デフォルト65） |
| `--scene-threshold` | シーンカット検出しきい値。カット境界は補間せず複製（0で無効） |
| `--device` | `auto` / `mps` / `cpu` |

主なオプション（`upscale`）:

| オプション | 説明 |
|---|---|
| `--scale` | 出力倍率。1より大きく4以下、小数可（デフォルト2） |
| `--model` | `realesr-general`（デフォルト・実写向け）/ `realesr-anime`（アニメ向け・最速）/ `realesrgan-x4plus`（最高品質・低速） |
| `--tile` | タイルサイズpx。省略時はモデルに応じて自動（compact系: 分割なし、x4plus: 512） |
| `--fp32` | fp16推論を無効化（デフォルトはMPS/CUDAでfp16） |
| `--codec` / `--quality` / `--device` | `interp` と共通 |

`run` は `interp` と `upscale` を1パスのストリームで両方行います
（低解像度のうちに補間 → 全フレームを超解像）。オプションは両者の合成で、
モデルは `--interp-model` / `--sr-model` で個別に指定します。

推論はデフォルトで fp16（MPS/CUDA時）。fp32との差は PSNR 47dB以上で視覚的に
同一のまま、フレーム補間が約1.4倍高速になります。`--fp32` で無効化できます。

音声は元動画からそのままコピーされ、色空間メタデータはソースから引き継がれます。

## 性能

Apple M5・720p入力・MPS + fp16 での実測値:

| 処理 | モデル | 速度 |
|---|---|---|
| フレーム補間 | rife-v4.7 | 約51 ms/フレーム |
| 超解像 2x | realesr-anime | 約152 ms/フレーム |
| 超解像 2x | realesr-general | 約276 ms/フレーム |
| 超解像 4x | realesrgan-x4plus | 約8.5 s/フレーム |

## 設計

[DESIGN.md](DESIGN.md) を参照。

## ライセンス

MIT — [LICENSE](LICENSE) を参照。

`src/venhance/vendor/rife_arch.py` は
[Fannovel16/ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation)（MIT）
由来で、[Practical-RIFE](https://github.com/hzwer/Practical-RIFE)（MIT）がベースです。
モデル重みも同リポジトリのGitHub Releasesから取得します。

`src/venhance/vendor/realesrgan_arch.py` は
[BasicSR](https://github.com/XPixelGroup/BasicSR)（Apache-2.0）由来で、
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)（BSD-3-Clause）で使われている
アーキテクチャです。モデル重みは Real-ESRGAN の GitHub Releases から取得します。
