from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from transformers import BertTokenizer, RobertaTokenizer, BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large", do_lower_case=True) #  do_lower_case is true by default

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True) #  do_lower_case is true by default

# tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)

nltk_tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
url_commom_words = ['http', 'com', 'cn', 'org', 'https', 'www', 'html', 'htm', 'asp', 'js']

def filter_url(url):
    string_tokens = nltk_tokenizer.tokenize(url)
    return [tok for tok in string_tokens if tok not in url_commom_words]

with open("/home/tangyubao/wikipassage/document_passages.json") as fin, \
    open("/home/tangyubao/wikipassage/document_passages_bart_tokenized.txt", 'a+') as fout:
    for idx, line in enumerate(tqdm(fin, total=8841823, desc="Complete collection preprocessing")):
        data = line.rstrip().split('\t')
        if len(data) < 2:
            continue
        docid = data[0]
        url = data[1]
        url_toks = filter_url(url)
        # print(url, url_toks)
        doc = " ".join(url_toks+data[2:]).lower()
        tokens = tokenizer.tokenize(" ".join(doc.split()[:512]))[:512]
        if idx < 4:
            print(doc)
            print(tokens)
        fout.write("{}\t{}\t{}\t{}\t{}\n".format('-', docid, '-', ' '.join(tokens), '-'))

