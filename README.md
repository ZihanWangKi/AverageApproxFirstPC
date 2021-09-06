# Avg ~ PC1
<h5 align="center">"Average" Approximates "First Principal Component"? An Empirical Analysis on Representations from Neural Language Models</h5>

Our paper is available [here](https://arxiv.org/abs/2104.08673). Accepted as a short paper in EMNLP'21.
## Motivation

In progress.

## Scripts
#### Environment
Packages can be installed via `pip install -r requirements.txt`.

#### Reproduce
1. First run `embed_corpus.py` to obtain the embeddings for a certain corpus with a certain language model.
2. Then, run `calculate_properties.py` to get the absolute cosine similarities between first PC and the average of the embeddings.
3. Calculations for other properties in the paper are in progress.

## Citation
Please cite the following paper if you found our dataset or framework useful. Thanks!

>Zihan Wang, Chengyu Dong, and Jingbo Shang. ""Average" Approximates "First Principal Component"? An Empirical Analysis on Representations from Neural Language Models" arXiv preprint arXiv:2104.08673 (2021).

```
@misc{wang2020xclass,
      title={"Average" Approximates "First Principal Component"? An Empirical Analysis on Representations from Neural Language Models}, 
      author={Zihan Wang and Chengyu Dong and Jingbo Shang},
      year={2021},
      eprint={2104.08673},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
