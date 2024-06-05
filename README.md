# Wise-layer Knowledge Editor (WilKE)

This is the relevant code for the paper [WilKE: Wise-Layer Knowledge Editor for Lifelong Knowledge Editing](https://arxiv.org/abs/2402.10987).



# Requirements

```shell
conda create -n wilke python=3.10
```

```shell
pip install -r requirements.txt
```



To achieve the baseline (KE & MEND), download the necessary open source model (Please refer to [ROME](https://github.com/kmeng01/rome/tree/main)).

- Download related KE (for GPT2-XL)：

```shell
wget https://rome.baulab.info/data/weights/efk-1tok-gpt2-xl.pt -P baselines/efk/weights
```

- Download related MEND (for GPT2-XL & GPT-J)：

```shell
wget https://rome.baulab.info/data/weights/mend-10tok-gpt2-xl.pt -P baselines/mend/weights
```

```shell
wget https://rome.baulab.info/data/weights/mend-10tok-gpt-j-6b.pt -P baselines/mend/weights
```



To achieve the baseline (ROME&MEMIT), download the necessary open source data (Please refer to [ROME](https://github.com/kmeng01/rome/tree/main)).

```shell
wget -r -np -nH --cut-dirs=2 https://rome.baulab.info/data/stats -P data
```

```shell
wget -r -np -nH --cut-dirs=2 -A "*" https://rome.baulab.info/data/dsets/ -P data
```



# Filtered data

Based on the CounterFact dataset, we filter the data that would cause toxicity flash when editing GPT2-XL and GPT-J, respectively. For details, see `data/`.



# Quick Start

You can use WilKE to perform 1024 steps of continuous knowledge editing on GPT2-XL with the following command, and you can modify `--alg_name` and `--model_name` to use other knowledge editing methods and language models.

```shell
python3 -m experiments.evaluate --alg_name=WilKE --model_name=gpt2-xl --hparams_fname=gpt2-xl.json --ds_name=counterfact --dataset_size_limit=2048 --edit_times=1024 --skip_generation_tests
```

You can then use the `experiments/summarize*` jupyter notebook files to summarize the results of your runs.



# Acknowledgements

Our code is built based on [ROME](https://github.com/kmeng01/rome/tree/main) and [MEMIT](https://github.com/kmeng01/memit/tree/main), so we would like to thank the original authors.



# How to Cite

```latex
@article{hu2024wilke,
  title={WilKE: Wise-Layer Knowledge Editor for Lifelong Knowledge Editing},
  author={Hu, Chenhui and Cao, Pengfei and Chen, Yubo and Liu, Kang and Zhao, Jun},
  journal={arXiv preprint arXiv:2402.10987},
  year={2024}
}
```

