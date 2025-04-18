import bz2
import json
from glob import glob
from pathlib import Path
from tqdm import tqdm
import bm25s

DATASET_DIR = Path(__file__).parent.parent / "dataset"


def generate_indexing_queries_from_bz2(bz2file):
    with bz2.open(bz2file, 'rt') as f:
        body = [json.loads(line) for line in f]
        subset_body = [
            {
                "id": doc["id"],
                "url": doc["url"],
                "title": doc["title"],
                "text": doc["text"]
            } for doc in body
        ]

    return subset_body


def main():

    filelist = glob(str(DATASET_DIR / "HotpotQA" / "enwiki-20171001-pages-meta-current-withlinks-abstracts/*/wiki_*.bz2"))

    print('Making indexing queries...')
    corpus_json = []
    for file in tqdm(filelist):
        documents_from_file = generate_indexing_queries_from_bz2(file)
        corpus_json.extend(documents_from_file)
    corpus_text = [f"{doc["title"]}\n\n{''.join(doc["text"])}" for doc in corpus_json]
    corpus_tokens = bm25s.tokenize(corpus_text, stopwords="en")
    retriever = bm25s.BM25(corpus=corpus_json)
    retriever.index(corpus_tokens)
    retriever.save(DATASET_DIR / "HotpotQA" / "wikipedia_index_bm25")


def test():
    retriever = bm25s.BM25.load(DATASET_DIR / "HotpotQA" / "wikipedia_index_bm25", load_corpus=True, mmap=True)
    tokenized = bm25s.tokenize("Michael Jackson", stopwords="en")
    out = retriever.retrieve(tokenized, k=5)
    print(out.documents[0])

if __name__ == '__main__':
    main()