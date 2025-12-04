# RT-DETR 学習済みモデルの使用方法

学習済みモデル `rtdetrv2_pytorch\output\rtdetrv2_r18vd_120e_coco\last.pth` を使用する方法を説明します。

## 1. 推論(Inference)を実行する

画像に対して物体検出を実行する場合は、`rtdetrv2_torch.py` を使用します:

```bash
python rtdetrv2_pytorch/references/deploy/rtdetrv2_torch.py -c rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r rtdetrv2_pytorch/output/rtdetrv2_r18vd_120e_coco/last.pth -f path/to/your/image.jpg -d cuda:0
```

### パラメータ説明:
- `-c`: 設定ファイルのパス
- `-r`: 学習済みモデル(checkpoint)のパス
- `-f`: 推論対象の画像ファイルのパス
- `-d`: 使用するデバイス(`cuda:0` または `cpu`)

### 実行結果:
検出結果が `results_0.jpg` として保存されます。

## 2. 評価(Evaluation)を実行する

モデルの性能を評価する場合:

```bash
cd rtdetrv2_pytorch
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r output/rtdetrv2_r18vd_120e_coco/last.pth --test-only
```

### パラメータ説明:
- `--nproc_per_node`: 使用するGPU数
- `--test-only`: 評価モードで実行

## 3. 追加学習(Fine-tuning)を実行する

学習を続ける場合:

```bash
cd rtdetrv2_pytorch
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -t output/rtdetrv2_r18vd_120e_coco/last.pth --use-amp --seed=0
```

### パラメータ説明:
- `-t`: 追加学習の開始点となるcheckpointのパス
- `--use-amp`: 混合精度学習を使用
- `--seed`: 再現性のための乱数シード

## 4. ONNXにエクスポートする

推論を高速化するためにONNX形式にエクスポート:

```bash
cd rtdetrv2_pytorch
python tools/export_onnx.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r output/rtdetrv2_r18vd_120e_coco/last.pth --check
```

### パラメータ説明:
- `--check`: エクスポート後に検証を実行

エクスポートされたONNXモデルは `output/rtdetrv2_r18vd_120e_coco/` ディレクトリに保存されます。

## 5. Pythonコードから直接使用する

スクリプト内で使用する場合:

```python
import torch
import torchvision.transforms as T
from PIL import Image
from src.core import YAMLConfig

# 設定とモデルの読み込み
cfg = YAMLConfig('rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml')
checkpoint = torch.load('rtdetrv2_pytorch/output/rtdetrv2_r18vd_120e_coco/last.pth', map_location='cpu')

# EMAモデルまたは通常モデルのstate_dictを取得
if 'ema' in checkpoint:
    state = checkpoint['ema']['module']
else:
    state = checkpoint['model']

# モデルに重みをロード
cfg.model.load_state_dict(state)
model = cfg.model.deploy()
model.eval()

# 画像の準備
im_pil = Image.open('path/to/image.jpg').convert('RGB')
w, h = im_pil.size
orig_size = torch.tensor([w, h])[None]

transforms = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])
im_data = transforms(im_pil)[None]

# 推論の実行
with torch.no_grad():
    outputs = model(im_data)
    outputs = cfg.postprocessor.deploy()(outputs, orig_size)
    labels, boxes, scores = outputs

# 結果の取得
threshold = 0.6
for i in range(len(labels)):
    scr = scores[i]
    lab = labels[i][scr > threshold]
    box = boxes[i][scr > threshold]
    scrs = scores[i][scr > threshold]
    
    print(f"検出された物体: {len(lab)}個")
    for j in range(len(lab)):
        print(f"ラベル: {lab[j].item()}, スコア: {scrs[j].item():.2f}, バウンディングボックス: {box[j].tolist()}")
```

## 注意事項

1. **設定ファイル**: 学習時に使用した設定ファイルと同じものを使用してください。
2. **デバイス**: GPU使用時は `cuda:0`、CPU使用時は `cpu` を指定してください。
3. **画像サイズ**: モデルは入力画像を640x640にリサイズして処理します。
4. **閾値**: デフォルトの検出閾値は0.6です。必要に応じて調整してください。

## チェックポイントの内容

`last.pth` ファイルには以下の情報が含まれています:
- `model`: モデルの重み
- `ema`: Exponential Moving Average モデルの重み(使用している場合)
- `optimizer`: オプティマイザの状態
- `epoch`: エポック数
- その他の学習状態情報
