# Multimodal Compatibility Modeling via Exploring the Consistent and Complementary Correlations [ACM MM 2021]


## Authors

**Weili Guan**<sup>1</sup>, **Haokun Wen**<sup>2</sup>, **Xuemeng Song**<sup>2</sup>, **Chung-Hsing Yeh**<sup>1</sup>, **Xiaojun Chang**<sup>3</sup>, **Liqiang Nie**<sup>2</sup>

<sup>1</sup> Monash University, Melbourne, VIC, Australia  
<sup>2</sup> Shandong University, Jinan, Shandong, China  
<sup>3</sup> RMIT University, Melbourne, VIC, Australia

## Links

- **Paper**: [ACM DL](https://dl.acm.org/doi/10.1145/3474085.3475392)

---

## Repository Structure

```text
mm_ocm/
├── compatibility/       # Code for outfit compatibility estimation
└── fill_in_the_blank/   # Code for fill-in-the-blank
```

---

## Dataset Preparation

This project uses the **Polyvore Outfits** dataset (nondisjoint split) and **Polyvore Outfits-D** (disjoint split).

Make sure the dataset is placed in the current working directory before running any training or evaluation scripts. Create a `results/` folder in the corresponding task directory to save model outputs:

```bash
mkdir results
```

---

## Usage

### Fashion Compatibility Estimation

Source code is located in the `./compatibility` folder.

#### Polyvore Outfits (nondisjoint)

```bash
python train.py --polyvore_split nondisjoint --batch_size 16 --epoch_num 30 \
  --lr 1e-4 --model_dir ./results
```

#### Polyvore Outfits-D (disjoint)

```bash
python train.py --polyvore_split disjoint --batch_size 16 --epoch_num 30 \
  --lr 1e-4 --model_dir ./results
```

---

### Fill-in-the-Blank

Source code is located in the `./fill_in_the_blank` folder.

#### Polyvore Outfits (nondisjoint)

> **Note**: We observed serious overfitting when training directly on `polyvore_outfits/nondisjoint/fill_in_the_blank_train.json`, which we attribute to characteristics of the dataset itself. Therefore, for the nondisjoint split, we directly use the saved model pre-trained on outfit compatibility estimation to obtain the reported results.

Ensure that `img_model.pt` and `text_model.pt` (pre-trained on compatibility estimation) are placed in `./fitb_on_polyvore_outfits/`, then run:

```bash
python ./fitb_on_polyvore_outfits/test.py
```

#### Polyvore Outfits-D (disjoint)

```bash
python train.py --polyvore_split disjoint --batch_size 16 --epoch_num 30 \
  --lr 1e-4 --model_dir ./results
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{guan2021multimodal,
  title     = {Multimodal Compatibility Modeling via Exploring the Consistent and Complementary Correlations},
  author    = {Guan, Weili and Wen, Haokun and Song, Xuemeng and Yeh, Chung-Hsing and Chang, Xiaojun and Nie, Liqiang},
  booktitle = {Proceedings of the ACM International Conference on Multimedia},
  pages     = {2299--2307},
  year      = {2021}
}
```
