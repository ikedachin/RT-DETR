# RT-DETR CPU推論の改良

## 概要
`rtdetrv2_pytorch\references\deploy\rtdetrv2_torch.py`をCPU推論用に改良し、以下の機能を追加しました:
- クラス名表示機能(日本語対応)
- 出力ディレクトリ指定
- 検出領域の切り抜き機能
- テキストの視認性向上(中抜き文字)
- 推論の最適化
- エラーハンドリングの改善

---

## 変更点の詳細

### 1. コマンドライン引数の追加

**変更内容:**
```python
parser.add_argument('-c', '--config', type=str, help='path to model config file')
parser.add_argument('-r', '--resume', type=str, help='path to model checkpoint')
parser.add_argument('-f', '--im-file', type=str, help='path to input image file')
parser.add_argument('-d', '--device', type=str, default='cpu')
parser.add_argument('-o', '--output', type=str, default='output')
parser.add_argument('-t', '--training_config', type=str, default=None, 
                    help='path to training config file, optional')
```

**解説:**
- `-c, --config`: モデル設定ファイル(YAML)のパスを指定
- `-r, --resume`: 学習済みモデルの重みファイル(.pth)のパスを指定
- `-f, --im-file`: 推論対象の画像ファイルのパスを指定
- `-d, --device`: 使用デバイスを指定(デフォルト: `cpu`)
- `-o, --output`: 推論結果画像の出力先ディレクトリを指定(デフォルト: `output`)
- `-t, --training_config`: COCO形式のアノテーションファイルパスを指定し、クラス名を取得可能に(オプション)

---

### 2. モジュールインポートエラーの回避

**変更内容:**
```python
# Add the parent directory to the Python path
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
```

**解説:**
スクリプトの実行位置に関わらず、相対パスでのモジュールインポートを可能にします。`../..`を追加することで、`src`モジュールへのアクセスを保証します。

---

### 3. 必要なモジュールのインポート

**変更内容:**
```python
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn 
import torchvision.transforms as T
import numpy as np 
from PIL import Image, ImageDraw
from src.core import YAMLConfig
```

**解説:**
- `json`モジュールを追加してCOCO形式のアノテーションファイルを読み込み可能に
- `sys.path.insert(0, ...)`でスクリプトの実行位置に関わらず、相対パスでのモジュールインポートを可能に

---

### 4. クラス名取得関数の追加

**変更内容:**
```python
def load_class_names(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    categories = data['categories']
    class_names = {cat['id']: cat['name'] for cat in categories}
    return class_names
```

**解説:**
COCO形式のアノテーションファイルから、カテゴリIDとクラス名のマッピング辞書を生成します。`encoding='utf-8'`を指定することで日本語のクラス名にも対応しています。

---

### 5. 推論の最適化

**変更内容:**
```python
model = Model().to(args.device)
model.eval()  # Set model to evaluation mode

# ... (中略) ...

with torch.no_grad():  # Disable gradient computation for inference
    output = model(im_data, orig_size)
```

**解説:**
- `model.eval()`: BatchNormalizationやDropoutを評価モードに設定
- `torch.no_grad()`: 勾配計算を無効化し、メモリ使用量を削減して推論速度を向上

---

### 6. draw関数の改良

**変更内容:**
```python
def draw(images, labels, boxes, scores, output_dir, base_name, label_names=None, thrh=0.6):
    """
    Draw bounding boxes and labels on images and save them to the output directory.
    
    Args:
        images: List of PIL Image objects to draw on
        labels: Tensor of predicted class labels for each detection
        boxes: Tensor of bounding box coordinates [x1, y1, x2, y2] for each detection
        scores: Tensor of confidence scores for each detection
        output_dir (str): Directory path where the output images will be saved
        base_name (str): Base filename for the output image
        label_names (dict, optional): Dictionary mapping label IDs to class names
        thrh (float, optional): Confidence threshold for filtering detections
    """
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        # テンソルをCPUに移動
        scr = scores[i].detach().cpu()
        lab = labels[i][scr > thrh].detach().cpu()
        box = boxes[i][scr > thrh].detach().cpu()
        scrs = scores[i][scr > thrh].detach().cpu()

        # ラベル名または数値IDを表示
        for j, b in enumerate(box):
            # Draw bounding box
            draw.rectangle(list(b.numpy()), outline='red', width=2)
            
            # Prepare label text
            if label_names is not None:
                label_text = label_names.get(lab[j].item(), str(lab[j].item()))
            else:
                label_text = str(lab[j].item())
            text = f"{label_text} {round(scrs[j].item(), 2)}"
            
            # Calculate text position (above the box)
            text_x = b[0].item()
            text_y = b[1].item() - 15  # 15 pixels above the box
            
            # Draw text with outline (stroke) for hollow effect
            outline_color = 'black'
            text_color = 'white'
            
            # Draw outline by drawing text slightly offset in all directions
            for offset_x in [-1, 0, 1]:
                for offset_y in [-1, 0, 1]:
                    if offset_x != 0 or offset_y != 0:
                        draw.text((text_x + offset_x, text_y + offset_y), 
                                text=text, fill=outline_color)
            
            # Draw main text
            draw.text((text_x, text_y), text=text, fill=text_color)
        
        # 出力ディレクトリを自動作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        im.save(os.path.join(output_dir, f'{base_name}'))
```

**解説:**
- **テンソル処理**: `.detach().cpu()`で明示的にCPUに移動し、メモリ管理を改善
- **クラス名表示**: `label_names`が指定されていれば実際のクラス名を、なければ数値IDを表示
- **出力管理**: 
  - 指定されたディレクトリが存在しない場合は自動作成
  - 元のファイル名を保持(`base_name`パラメータを追加)
- **視認性向上**: 
  - バウンディングボックスの線幅を2ピクセルに設定
  - テキストをボックスの上部(15ピクセル上)に配置
  - 中抜き文字効果を実装(黒い縁取り + 白い文字)して背景色に関わらず視認性を向上

---

### 7. 検出領域の切り抜き機能追加

**変更内容:**
```python
def get_crop_area(image, labels, boxes, scores, base_name, label_names=None, thrh=0.6) -> list:
    crop_areas = []
    crop_area_datas = []
    scr = scores.detach().cpu()
    lab = labels[scr > thrh].detach().cpu()
    box = boxes[scr > thrh].detach().cpu()
    scrs = scores[scr > thrh].detach().cpu()

    for j, b in enumerate(box):
        # Ensure coordinates are integers and within image bounds
        x1, y1, x2, y2 = int(b[0].item()), int(b[1].item()), int(b[2].item()), int(b[3].item())
        
        # Crop the original image (not the drawn one)
        crop_area = image.crop((x1, y1, x2, y2))
        crop_areas.append(crop_area)
        crop_area_datas.append({
            'label': label_names.get(lab[j].item(), str(lab[j].item())) if label_names is not None else str(lab[j].item()),
            'score': round(scrs[j].item(), 2),
            'box': [x1, y1, x2, y2]
        })
    
    return crop_areas, crop_area_datas
```

**解説:**
- 検出された各バウンディングボックス領域を元画像から切り抜き
- 切り抜いた画像とメタデータ(ラベル、スコア、座標)を返す
- 後続処理(OCR、画像解説など)に渡すことが可能

---

### 8. main関数での統合

**変更内容:**
```python
def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)

    # クラス名の読み込み
    if args.training_config is not None:
        label_names = load_class_names(args.training_config)
    else:
        label_names = None

    # ... (モデル読み込み・推論) ...

    labels, boxes, scores = output
    print(f'labels: {labels.shape}\n boxes: {boxes.shape}\n scores: {scores.shape}')
    
    filename_with_ext = os.path.basename(args.im_file)
    b_name = os.path.basename(filename_with_ext)
    
    # First crop the original image (before drawing)
    crop_areas, crop_area_datas = get_crop_area(im_pil, labels[0], boxes[0], scores[0], b_name, label_names)
    
    # Then draw on the image
    draw([im_pil], labels, boxes, scores, args.output, b_name, label_names)
    
    # Save cropped areas
    for idx, crop_area in enumerate(crop_areas):
        crop_area.save(os.path.join(args.output, f'crop_{idx}_{b_name}'))
```

**解説:**
- コマンドライン引数で指定された学習設定ファイルからクラス名を読み込み
- 推論結果のテンソル形状を表示してデバッグを容易に
- 元のファイル名を取得して出力ファイル名に使用
- **重要**: 描画前に元画像から領域を切り抜き(描画後では線が入ってしまう)
- 切り抜いた領域を個別のファイルとして保存(`crop_0_xxx.jpg`, `crop_1_xxx.jpg`, ...)

---

## 使用例

```bash
python rtdetrv2_pytorch/references/deploy/rtdetrv2_torch.py \
    -c rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
    -r rtdetrv2_pytorch/output/rtdetrv2_r18vd_120e_coco/last.pth \
    -d cpu \
    -f ./coco_dataset/images/val/sample.png \
    -o output \
    -t coco_dataset/annotations/instances_train.json
```

**引数の説明:**
- `-c`: モデル設定ファイル(YAML)
- `-r`: 学習済みモデルの重みファイル(.pth)
- `-d`: 使用デバイス(`cpu`または`cuda`)
- `-f`: 推論対象の画像ファイル
- `-o`: 結果画像の出力先ディレクトリ
- `-t`: クラス名取得用のアノテーションファイル(オプション)

---

## 出力結果

推論完了後、指定した出力ディレクトリに以下のファイルが生成されます:

### 1. アノテーション済み画像
- `元のファイル名.jpg`: バウンディングボックスとクラス名(またはID)、信頼度スコアが描画された画像
  - バウンディングボックス: 赤色、線幅2ピクセル
  - テキスト: 白色の中抜き文字(黒い縁取り)でボックスの上部に表示

### 2. 切り抜き画像
- `crop_0_元のファイル名.jpg`: 1番目の検出領域
- `crop_1_元のファイル名.jpg`: 2番目の検出領域
- ...

各切り抜き画像は、検出されたバウンディングボックスの領域のみを含みます。OCRや画像解説などの後続処理に利用可能です。

---

## 今後の拡張案

コード内のコメントにあるように、以下の機能追加が想定されています:

1. **読み順推論**: 検出されたボックスとラベルの位置関係から、文書の読み順を自動推論
2. **OCR処理**: 切り抜いた画像に対してOCRを実行し、テキストを抽出
3. **画像解説**: 各切り抜き領域の内容を解説
4. **結果の統合**: 読み順に従ってテキストや解説を保存