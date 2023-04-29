import stanza
stanza.download('en')       # download English model
nlp = stanza.Pipeline('en') # initialize English neural pipeline
with open("datasets/news.2021.en.shuffled.uds.clean", 'r', encoding='utf-8') as f,\
    open("datasets/uds_stanza.conllu", 'w', encoding='utf-8') as w:
    for line in f:
        doc = nlp(line.strip())
        w.write(doc._sentences[0].comments[0] + '\n')
        doc = doc.to_dict()[0]
        for item in doc:
            if 'feats' not in item:
                item['feats'] = '_'
            if 'lemma' not in item:
                item['lemma'] = item['text']
            w.write(str(item['id']) + '\t' + item['text'] + '\t' + item['lemma'] + '\t' + item['upos'] + '\t' + item['xpos'] + \
                    '\t' + item['feats'] + '\t' + str(item['head']) + '\t' + item['deprel'] + '\t' + '_' + '\t' + '_' + '\n')
        w.write('\n')
