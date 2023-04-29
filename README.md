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

[PredPatt](https://github.com/hltcoe/PredPatt)
[decomp](https://github.com/decompositional-semantics-initiative/decomp)
pip install -r requirements.txt
