import datasets
import numpy as np
import random
import pickle
import torch
from tqdm import tqdm


def embed_document_with_transformers(doc, model, tokenizer, has_sos):
    tokens = tokenizer.encode(doc, add_special_tokens=True)
    input_ids = torch.tensor([tokens], device=model.device)
    with torch.no_grad():
        output = model(input_ids, output_hidden_states=True)
    layers = output[2]
    layer_embeddings = [layer.squeeze(0).detach().cpu().numpy()[1: -1] if has_sos else
                        layer.squeeze(0).detach().cpu().numpy() for layer in layers]
    layer_embeddings = np.array(layer_embeddings)
    return tokens[1: -1] if has_sos else tokens[:], layer_embeddings


def get_corpus(corpus_name, document_size):
    corpus = datasets.load_dataset(corpus_name)
    if corpus_name == "ag_news":
        corpus = corpus["train"]["text"]
    elif corpus_name == "dbpedia_14":
        corpus = corpus["train"]["content"]
    else:
        # + custom datasets
        assert False
    n = len(corpus)
    selected_indicies = list(range(n))
    random.seed(42)
    random.shuffle(selected_indicies)
    selected_indicies = selected_indicies[: document_size]
    return [corpus[idx] for idx in selected_indicies]


# corpus:
# ag_news, dbpedia_14
def embed_corpus_with_transformers(corpus_name="ag_news",
                                   document_size=4000,
                                   language_model="bert-base-cased",
                                   random_init=False):
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(language_model)
    if random_init:
        config = transformers.AutoConfig.from_pretrained(language_model)
        model = transformers.AutoModel.from_config(config)
    else:
        model = transformers.AutoModel.from_pretrained(language_model)
    model.eval()
    model = model.to(torch.device("cuda"))
    tokens = []
    embeddings = []
    corpus = get_corpus(corpus_name, document_size)
    for doc in tqdm(corpus):
        # bert, roberta, xlnet have start of sentence (and end of sentence) tokens, while gpt does not.
        token, embedding = embed_document_with_transformers(doc, model, tokenizer, language_model != "gpt2")
        tokens.append(token)
        embeddings.append(embedding)
    with open(f"{corpus_name}.{language_model}.pk", "wb") as f:
        pickle.dump({
            "tokens": tokens,
            "embeddings": embeddings
        }, f, protocol=4)


def embed_corpus_with_static(corpus_name="ag_news",
                             document_size=4000,
                             language_model="skip_gram"):
    # You can download 6 & 8 from http://vectors.nlpl.eu/repository/
    path_to_word_vectors = {
        "skip_gram": f"static_embeddings/skip_gram/model.txt",
        "glove": f"static_embeddings/glove/model.txt",
    }
    word2index = {}
    embedding_matrix = None
    with open(path_to_word_vectors[language_model], "r") as f:
        for i, line in tqdm(enumerate(f.readlines())):
            if i == 0:
                print(line)
                vocab_size, emb_dim = list(map(int, line.split()))
                embedding_matrix = np.zeros((vocab_size, emb_dim))
                continue
            else:
                word = line.split()[:-emb_dim]
                if len(word) == 1:
                    word = word[0]
                else:
                    word = " ".join(word)
                embs = list(map(float, line.split()[-emb_dim:]))

                embedding_matrix[i - 1] = embs
                word2index[word] = len(word2index)

    corpus = get_corpus(corpus_name, document_size)
    unk_cnt = 0
    total_cnt = 0
    embeddings = []
    tokens = []
    for doc in tqdm(corpus):
        embedding = []
        token = []
        for word in doc.split():
            total_cnt += 1
            if word not in word2index:
                unk_cnt += 1
                continue
            word_embedding = embedding_matrix[word2index[word]]
            embedding.append(word_embedding)
            token.append(word)
        # assert len(token) > 0, corpus[idx]
        if len(token) == 0:
            embeddings.append(np.zeros((1, emb_dim)))
        else:
            embeddings.append(np.array(embedding))
        tokens.append(token)
    print(unk_cnt, total_cnt)
    with open(f"{corpus_name}.{language_model}.pk", "wb") as f:
        pickle.dump({
            "tokens": tokens,
            "embeddings": embeddings
        }, f, protocol=4)


def embed_corpus_with_elmo(corpus_name="ag_news",
                      document_size=4000,
                      language_model="elmo"):
    from allennlp.modules.elmo import Elmo, batch_to_ids
    # code from https://github.com/allenai/allennlp/issues/2245
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    model = Elmo(options_file, weight_file, 1, dropout=0)
    model.eval()
    model = model.to(torch.device("cuda"))
    tokens = []
    embeddings = []
    corpus = get_corpus(corpus_name, document_size)
    for doc in tqdm(corpus):
        token, ids = doc.split(), batch_to_ids([doc.split()])
        ids = ids.cuda(torch.device('cuda'))
        with torch.no_grad():
            hidden_states = model(ids)
        embedding = hidden_states["elmo_representations"][0][0]
        embedding = embedding.detach().cpu().numpy()
        tokens.append(token)
        embeddings.append(embedding)
    with open(f"{corpus_name}.{language_model}.pk", "wb") as f:
        pickle.dump({
            "tokens": tokens,
            "embeddings": embeddings
        }, f, protocol=4)

if __name__ == '__main__':
    # examples
    embed_corpus_with_transformers(corpus_name="ag_news", language_model="bert-base-cased")
    embed_corpus_with_static(corpus_name="ag_news", language_model="skip_gram")
    embed_corpus_with_elmo(corpus_name="dbpedia_14", language_model="elmo")
