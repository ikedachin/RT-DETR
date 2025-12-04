# RT-DETR ファインチューニング用データセット構造

RT-DETRをファインチューニングするには、**COCO形式**のデータセットを準備する必要があります。

## 📁 推奨ディレクトリ構造

```
dataset/
├── annotations/
│   ├── instances_train.json    # 訓練用アノテーション
│   └── instances_val.json      # 検証用アノテーション
├── train/                      # 訓練画像
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── val/                        # 検証画像
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

## 📝 COCOアノテーション形式 (JSON)

```json
{
    "images": [
        {
            "id": 1,
            "file_name": "image001.jpg",
            "width": 1920,
            "height": 1080
        },
        {
            "id": 2,
            "file_name": "image002.jpg",
            "width": 1920,
            "height": 1080
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 200, 50, 80],
            "area": 4000,
            "iscrowd": 0
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 2,
            "bbox": [300, 400, 60, 90],
            "area": 5400,
            "iscrowd": 0
        },
        {
            "id": 3,
            "image_id": 2,
            "category_id": 1,
            "bbox": [150, 250, 70, 50],
            "area": 3500,
            "iscrowd": 0
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "class_a"
        },
        {
            "id": 2,
            "name": "class_b"
        }
    ]
}
```

**注意**: 
- `bbox`は `[左上x座標, 左上y座標, 幅, 高さ]` の形式です。※ピクセル値
- `area`はバウンディングボックスの面積（`width × height`）です。※ピクセル²
- 1つの画像に複数のオブジェクトがある場合、同じ`image_id`で異なる`id`（アノテーションID）を持つエントリを複数作成します
- `images.id`は画像ごとにユニーク、`annotations.id`はデータセット全体でユニークである必要があります

### COCO形式の `bbox` 座標系

```
(0,0)────────────────────────────────► X軸
  │
  │        (x, y)
  │          ┌─────────────────┐
  │          │                 │
  │          │                 │ height
  │          │                 │
  │          └─────────────────┘
  │                width
  │
  ▼
 Y軸
bbox = [x, y, width, height]

例: bbox = [100, 200, 50, 80]
    → 左上座標 (100, 200) から 幅50px × 高さ80px のボックス
```

## ⚙️ 設定ファイルの修正

PyTorch版を使用する場合、`rtdetrv2_pytorch/configs/dataset` 内の設定ファイルを編集します：

```yaml
train_dataloader:
  dataset:
    img_folder: /path/to/dataset/train
    ann_file: /path/to/dataset/annotations/instances_train.json

val_dataloader:
  dataset:
    img_folder: /path/to/dataset/val
    ann_file: /path/to/dataset/annotations/instances_val.json
```

## 🔢 クラス数の変更

カスタムクラス数に合わせて、モデル設定も変更が必要です：

```yaml
RTDETRTransformerv2:
  num_classes: 80  # ← 自分のクラス数に変更
```

## 🚀 学習の開始

```bash
cd rtdetrv2_pytorch
$env:PYTHONUTF8 = "1" # Windows環境でUTF-8問題を避ける場合
python tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd.yml
python .\tools\train.py -c .\configs\rtdetrv2\rtdetrv2_r50vd_dsp_1x_coco.yml
```



詳細な設定は `rtdetrv2_pytorch` または `rtdetr_pytorch` フォルダ内のREADMEを参照してください。



# 参考
## 📖 `iscrowd` フィールドについて

`iscrowd`はオブジェクトが群衆（複数の重なり合ったオブジェクト）かどうかを示すフラグです。

| 値 | 意味 |
|----|------|
| `0` | 単一のオブジェクト（通常のバウンディングボックス） |
| `1` | 群衆領域（複数のオブジェクトが密集・重複している領域） |

### `iscrowd: 0` を使うケース（通常）


個別のオブジェクトがはっきり識別できる場合：

| シーン | 説明 |
|--------|------|
| 🚗 駐車場の車 | 各車が分離して見える |
| 🍎 テーブル上の果物 | 個々の果物が識別可能 |
| 👤 数人の人物 | 各人物を個別にアノテーション可能 |
| 📦 棚の商品 | 各商品が区別できる |

### `iscrowd: 1` を使うケース（群衆）

個別にアノテーションするのが困難/非現実的な場合：

| シーン | 説明 |
|--------|------|
| 👥 満員電車の乗客 | 人が重なり合って個別識別が困難 |
| 🐦 空を飛ぶ鳥の群れ | 大量の鳥が密集 |
| 🍇 ブドウの房 | 個々の粒を分離するのが非現実的 |
| 🎭 コンサート会場の観客 | 数百人が密集している |

### 視覚的な判断基準

```
iscrowd: 0                    iscrowd: 1
┌─────────────────┐          ┌─────────────────┐
│  ┌──┐    ┌──┐  │          │ ┌──┬──┬──┬──┐  │
│  │人│    │人│  │          │ │人│人│人│人│  │
│  └──┘    └──┘  │          │ ├──┼──┼──┼──┤  │
│       ┌──┐     │          │ │人│人│人│人│  │
│       │人│     │          │ └──┴──┴──┴──┘  │
│       └──┘     │          │   (密集群衆)    │
└─────────────────┘          └─────────────────┘
  各人を個別にBBox             1つの大きなBBoxで
  でアノテーション             群衆全体を囲む
```

### 実践的なアドバイス

**カスタムデータセットでは、ほとんどの場合 `iscrowd: 0` で問題ありません。**

`iscrowd: 1` は主にCOCOデータセットのような大規模データセットで、評価の公平性を保つために使われます。
