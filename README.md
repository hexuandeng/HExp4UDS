# Holistic Exploration on UDS Parsing

## Download Corresponding Datasets

```bash
mkdir datasets
cd datasets
wget 'https://data.statmt.org/news-crawl/en/news.2021.en.shuffled.deduped.gz'
wget 'https://nlp.stanford.edu/data/glove.840B.300d.zip'
unzip glove.840B.300d.zip
gzip -d news.2021.en.shuffled.deduped.gz
```

## Environment 

First install [PredPatt](https://github.com/hltcoe/PredPatt) and [decomp](https://github.com/decompositional-semantics-initiative/decomp) following the instructions in the link, then run:
```bash
pip install -r requirements.txt
```

## Experiment

For naive model training, run:

```bash
python heuds/main.py train --task UDSTask --arch Bert_UDS --save-dir 'Bert_naive' --encoder-output-dim 1024 --layer-in-use 0,0,1,1,1,1,1
```

For model training with additional syntactic information, run:

```bash
python heuds/main.py train --task UDSTask --arch Bert_UDS --save-dir 'Bert_incorpsyn' --encoder-output-dim 1024 --contact-ud --syntax-edge-gcn
```

For our best model training with additional syntactic information and data augmentation method, run:

```bash
python heuds/main.py train --task UDSTask --arch Bert_Syntactic --save-dir 'Bert_syntactic' --encoder-output-dim 1024
python heuds/main.py generate --task ConlluTask --arch Bert_Syntactic --save-dir 'Bert_syntactic' --encoder-output-dim 1024 --mono-file datasets/news.2021.en.shuffled.deduped --conllu-file datasets/news.conllu
python heuds/main.py train --task PredPattTask --arch Bert_UDS --save-dir 'Bert_best_pretrained' --max-epoch 30 --encoder-output-dim 1024 --layer-in-use 1,1,1,1,1,0,0 --conllu datasets/news.conllu --name news --validate-interval -1 --contact-ud --syntax-edge-gcn
python heuds/main.py train --task UDSTask --arch Bert_UDS --save-dir 'Bert_best' --pretrained-model-dir 'Bert_best_pretrained' --encoder-output-dim 1024 --lr 2e-5 --pretrained-lr 1e-6 --contact-ud --syntax-edge-gcn
```

Replace "train" to "test" for model evaluation.


## Citation

If you find this work helpful, please consider citing as follows:

```bibtex
@inproceedings{deng-etal-2024-holistic,
    title = "Holistic Exploration on Universal Decompositional Semantic Parsing: Architecture, Data Augmentation, and {LLM} Paradigm",
    author = "Deng, Hexuan  and
      Zhang, Xin  and
      Zhang, Meishan  and
      Liu, Xuebo  and
      Zhang, Min",
    editor = "Wong, Kam-Fai  and
      Zhang, Min  and
      Xu, Ruifeng  and
      Li, Jing  and
      Wei, Zhongyu  and
      Gui, Lin  and
      Liang, Bin  and
      Zhao, Runcong",
    booktitle = "Proceedings of the 10th SIGHAN Workshop on Chinese Language Processing (SIGHAN-10)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.sighan-1.6",
    pages = "45--57"
}
```
