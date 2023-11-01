# はじめに

今回はYOLO v7のHPEを用いて人体姿勢推定を実行してみたいと思います。
詳しい記事はQiitaの方をご覧ください：
https://qiita.com/yorul/items/323f930cb23b39baff1b

# 前準備
今回はGoogle Colabを使うので環境構築は特にいりません。

# YOLOv7で姿勢推定
## 00. セットアップ
最初はランタイムのタイプがちゃんとGPUになっているかを確認してみましょう。

```shell
!nvidia-smi
```
無料版のGPU性能だと、動画の長さによってメモリ不足で落ちる可能性があるので要注意です。

次はGoogleドライブをマウントします。
```python
from google.colab import drive
drive.mount('/content/drive')
%cd ./drive/MyDrive
```
公式のYOLOv7をcloneします。
```shell
!git clone https://github.com/WongKinYiu/yolov7
%cd yolov7
!pip install -r requirements.txt
```
最後に学習モデルをダウンロードします。
```shell
!wget -nc https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt --quiet
POSE_MODEL_WEIGHTS_PATH = f"/content/drive/MyDrive/yolov7/yolov7-w6-pose.pt"
```

CUDAが利用可能な場合はGPUを使用し、そうでない場合はCPUを使用します。
処理速度を上げるためFP16を使います。
気になる人はこちらの記事を参照してください：


https://qiita.com/arutema47/items/a507a3a8ee10654d5d1f

```python
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights = torch.load(POSE_MODEL_WEIGHTS_PATH, map_location=device)
pose_model = weights["model"]
_ = pose_model.float().eval()

if torch.cuda.is_available():
    pose_model.half().to(device)
```


## 01. 動画を用意する

ドライブのyolov7フォルダにアップロードします。
サンプル動画が必要な場合はこちらのコードを実行しましょう。
```shell
!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1C2eVNmLN64nvpd3VQcFYuKR0jbyvIvdI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1C2eVNmLN64nvpd3VQcFYuKR0jbyvIvdI" -O sample.mp4 && rm -rf /tmp/cookies.txt
```
自分で動画を用意する場合はパスを変更してください。
```python
SOURCE_VIDEO_A_PATH = f"./sample.mp4"
```

## 02. 関数を定義
動画からフレームを抽出する関数を定義します
引数には動画のパスを受け取り、`Yield式`で`NumPy`の`ndarray`（多次元配列）を一つずつ返します。

```python
from typing import Generator, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import cv2

%matplotlib inline
def generate_frames(video_file: str) -> Generator[np.ndarray, None, None]:
    video = cv2.VideoCapture(video_file)

    while video.isOpened():
        success, frame = video.read()

        if not success:
            break

        yield frame

    video.release()
```
ポーズ推定の前処理を行います（フレーム画像のリサイズ、ピクセル値の正規化など）

```python
from utils.datasets import letterbox
from torchvision import transforms

def pose_pre_process_frame(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    image = letterbox(frame, 960, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    if torch.cuda.is_available():
        image = image.half().to(device)

    return image
```

ポーズ推定モデルの出力を後処理するための関数を定義します。
`post_process_pose`関数は前処理でフレーム画像のサイズをスケーリングした結果に基づいて、ポーズ推定の結果にもスケーリングを行います。
`pose_post_process_output`関数は**非最大抑制**を行います。

https://meideru.com/archives/3538

```python
from typing import Tuple
from utils.general import non_max_suppression_kpt, non_max_suppression
from utils.plots import output_to_keypoint

def post_process_pose(pose: np.ndarray, image_size: Tuple, scaled_image_size: Tuple) -> np.ndarray:
    height, width = image_size
    scaled_height, scaled_width = scaled_image_size
    vertical_factor = height / scaled_height
    horizontal_factor = width / scaled_width
    result = pose.copy()
    for i in range(17):
        result[i * 3] = horizontal_factor * result[i * 3]
        result[i * 3 + 1] = vertical_factor * result[i * 3 + 1]
    return result

def pose_post_process_output(
    output: torch.tensor,
    confidence_trashold: float,
    iou_trashold: float,
    image_size: Tuple[int, int],
    scaled_image_size: Tuple[int, int]
) -> np.ndarray:
    output = non_max_suppression_kpt(
        prediction=output,
        conf_thres=confidence_trashold,
        iou_thres=iou_trashold,
        nc=pose_model.yaml['nc'],
        nkpt=pose_model.yaml['nkpt'],
        kpt_label=True)

    with torch.no_grad():
        output = output_to_keypoint(output)

        for idx in range(output.shape[0]):
            output[idx, 7:] = post_process_pose(
                output[idx, 7:],
                image_size=image_size,
                scaled_image_size=scaled_image_size
            )

    return output
```
## 03. 動画の姿勢推定の処理を行う

ここのステップは入力動画にスケルトンを描画したい場合だけ必要です。
これで推定結果を直観的に確認することができます。
```python
from utils.plots import plot_skeleton_kpts

def pose_annotate(image: np.ndarray, detections: np.ndarray) -> np.ndarray:
    annotated_frame = image.copy()
    for idx in range(detections.shape[0]):
        pose = detections[idx, 7:].T
        plot_skeleton_kpts(annotated_frame, pose, 3)
    return annotated_frame
```


```python
from dataclasses import dataclass

# 出力動画に関する情報を格納
@dataclass(frozen=True)
class VideoConfig:
    fps: float
    width: int
    height: int

# 出力動画を保存するためのcv2.VideoWriterオブジェクトを作成
def get_video_writer(target_video_path: str, video_config: VideoConfig) -> cv2.VideoWriter:
    video_target_dir = os.path.dirname(os.path.abspath(target_video_path))
    os.makedirs(video_target_dir, exist_ok=True)
    return cv2.VideoWriter(
        target_video_path,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=video_config.fps,
        frameSize=(video_config.width, video_config.height),
        isColor=True
    )
# 入力動画のフレーム数を取得
def get_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # フレーム数を返す
```
姿勢推定の処理を実行する関数を定義します。
```python
from tqdm.notebook import tqdm
import os

def process_and_annotate(source_video_path: str, target_video_path: str) -> None:

  #初期化
  video_config = VideoConfig(
      fps=25,  # フレームレート
      width=1920,  # 幅
      height=1080)  # 高さ
  video_writer = get_video_writer(
      target_video_path=target_video_path,
      video_config=video_config)
  
  frame_iterator = iter(generate_frames(video_file=source_video_path))

  # フレームの合計数を取得
  total = get_frame_count(source_video_path)

  # 各フレームを処理
  for frame in tqdm(frame_iterator, total=total):
      # フレームのコピーを作成
      annotated_frame = frame.copy()

      with torch.no_grad():
          # フレームのサイズを取得
          image_size = frame.shape[:2]

          # ポーズ推定の前処理
          pose_pre_processed_frame = pose_pre_process_frame(
              frame=frame,
              device=device)
          # スケールされたフレーム画像サイズを取得
          pose_scaled_image_size = tuple(pose_pre_processed_frame.size())[2:]

          # ポーズ推定モデルからの出力を取得
          pose_output = pose_model(pose_pre_processed_frame)[0].detach().cpu()
          # ポーズ推定の出力を後処理
          pose_output = pose_post_process_output(
              output=pose_output,
              confidence_trashold=0.25,
              iou_trashold=0.65,
              image_size=image_size,
              scaled_image_size=pose_scaled_image_size
          )
          # フレームにポーズ推定の結果を注釈として追加
          annotated_frame = pose_annotate(
              image=annotated_frame, detections=pose_output)

          # 保存
          video_writer.write(annotated_frame)

  video_writer.release()
```
最後は実行しましょう。
```python
process_and_annotate(SOURCE_VIDEO_PATH, f"./output/sample-out.mp4")
```
![シーケンス 01.gif](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3501427/1516b553-3f6c-dbd9-4e6b-8651242c37bf.gif)

動きがあまり激しくなかったら、そこそこの精度が出せると思います。


## 終わりに
留学生なので不自然な日本語があれば指摘してくれると嬉しいです。


## 参考にした記事
・https://tt-tsukumochi.com/archives/3549
・https://learnopencv.com/yolov7-pose-vs-mediapipe-in-human-pose-estimation/
・https://github.com/SkalskiP/sport/tree/master
