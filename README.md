# BRISC2025 Multi-Task Brain Tumor Model

A U-Net-style CNN encoder/decoder with a Transformer bottleneck, jointly
trained on T1-weighted brain MRI to perform:

- **Classification / detection** — no_tumor / glioma / meningioma / pituitary
- **Segmentation** — pixel-wise tumor mask
- **Direction-aware** — conditioned on scan plane (axial / coronal / sagittal / unknown)

Built on the [BRISC2025 dataset](https://www.kaggle.com/datasets/briscdataset/brisc2025) (Fateh et al., 2025).

## Architecture

```
Input (B, 1, H, W) T1-weighted MRI
        |
   CNN Encoder (4 stages, stride-2 downsampling each)
   s1 (H)  ->  s2 (H/2)  ->  s3 (H/4)  ->  s4 (H/8)     <- skip connections saved
        |                                    |
        v                                    |
   bottleneck_conv (H/16)                    |
        |                                    |
   Transformer Bottleneck                    |
     - flatten to tokens + positional embed  |
     - + direction embedding (global cond.)  |
     - N x self-attention + MLP blocks       |
     - reshape back to spatial map           |
        |                                    |
        +------------------+                 |
        |                  |                 |
   GAP -> cls_head    CNN Decoder (4 stages, upsample + concat skips)
        |             s4 -> s3 -> s2 -> s1  <-+
        v                  |
  cls_logits (B, 4)   seg_head -> seg_logits (B, 1, H, W)
```

- **Why the transformer sits only at the bottleneck**: self-attention is
  O(n^2) in token count, so it's only cheap at the lowest-resolution feature
  map. That's also exactly where *global* context (tumor location relative
  to the whole brain) matters more than local texture.
- **Why skip connections come from the CNN, not the transformer**: they
  preserve the fine boundary detail a pure ViT bottleneck would lose.
- **Why direction is injected at the bottleneck, not concatenated as an
  input channel**: it's a global, non-spatial property of the whole image,
  so it's added once to every token (same mechanism as the positional
  embedding) rather than duplicated across every pixel through 4 CNN stages.

## Dataset

[BRISC2025](https://www.kaggle.com/datasets/briscdataset/brisc2025): 6,000
contrast-enhanced T1-weighted MRI scans (5,000 train / 1,000 test), balanced
across 4 classes, annotated by certified radiologists. 4,793 images
(all tumor-bearing ones) include pixel-level segmentation masks; `no_tumor`
images have none. Images span axial, sagittal, and coronal planes.

```bibtex
@article{fateh2025brisc,
  title={Brisc: Annotated dataset for brain tumor segmentation and classification with swin-hafnet},
  author={Fateh, Amirreza and Rezvani, Yasin and Moayedi, Sara and Rezvani, Sadjad and Fateh, Fatemeh and Fateh, Mansoor and Abolghasemi, Vahid},
  journal={arXiv preprint arXiv:2506.14318},
  year={2025}
}
```

## Repo structure

```
configs/          hyperparameters, paths (config.yaml)
data/index/        small, committed CSV index (image paths, not images)
scripts/           things you run
  data_index.py       build the train/test CSV index
  train.py              training entry point (CLI)
src/brisc/         importable library code
  dataset.py           Brisc2025Dataset
  losses.py             dice_loss, masked_bce_loss, MultiTaskLoss
  models/               blocks.py, transformer.py, multitask_model.py
```


## Setup

```bash
git clone <your-repo-url>
cd brisc2025-multitask
pip install -r requirements.txt
```

Kaggle credentials for `scripts/download_data.py`: place `kaggle.json` at
`~/.kaggle/kaggle.json`, or export `KAGGLE_USERNAME` / `KAGGLE_KEY` as
environment variables. Never commit `kaggle.json` (already in `.gitignore`).

## Usage

```bash
# 1) download the raw dataset from Kaggle
python scripts/download_data.py --out_dir data/raw

# 2) build the train/test CSV index -- probe first to confirm the actual
#    downloaded folder structure matches what build_index.py expects
python scripts/build_index.py --raw_root data/raw --probe
python scripts/build_index.py --raw_root data/raw --out_csv data/index/brisc2025_index.csv

# 3) sanity-check model shapes and parameter count
python scripts/model_summary.py

# 4) train
python scripts/train.py \
  --csv_path data/index/brisc2025_index.csv \
  --epochs 50 \
  --batch_size 16 \
  --image_size 512
```

Checkpoints (`best_model.pt`, tracked by validation Dice; `last_model.pt`)
are written to `--out_dir` (default `checkpoints/`).

### Key training args

| Flag | Default | Notes |
|---|---|---|
| `--image_size` | 512 | must be divisible by 16 (4 downsampling stages) |
| `--base_ch` | 32 | encoder channel width; doubles each stage (32→64→128→256) |
| `--embed_dim` | 256 | transformer token dimension |
| `--block_num` | 4 | number of transformer blocks in the bottleneck |
| `--num_heads` | 8 | attention heads |

## Results

_TODO: fill in once you have a trained checkpoint —
val accuracy, per-class F1, and mean Dice._

| Model | Val Accuracy | Val Dice |
|---|---|---|
| _pending_ | | |

## Notes / known gotchas

- `scripts/build_index.py` assumes the folder layout documented by the
  BRISC2025 Kaggle release (`classification_task/<split>/<class>/`,
  `segmentation_task/<split>/{images,masks}/`) — verify with `--probe`
  before relying on it, since Kaggle releases occasionally rename
  top-level folders.
- `image_size` must be a multiple of 16 — the encoder downsamples 4 times
  (2^4=16), and the decoder needs each upsampled stage to land exactly on
  its matching skip connection's resolution.
- `no_tumor` images have no segmentation mask; `losses.py` masks the
  segmentation loss out for those samples (`has_mask=False`) so the decoder
  isn't trained toward blank masks on healthy scans.

## License

