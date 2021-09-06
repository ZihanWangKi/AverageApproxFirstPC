import numpy as np
import pickle
from sklearn.decomposition import PCA
from tqdm import tqdm

def cosine_similarity_embedding(emb_a, emb_b):
    assert emb_a.shape == emb_b.shape
    return np.dot(emb_a, emb_b) / np.linalg.norm(emb_a) / np.linalg.norm(emb_b)

# embddings: L * D,
def calc_average_vs_first_pc(embeddings):
    pca = PCA(n_components=1, random_state=42).fit(embeddings.transpose((1, 0)))
    first_component = pca.transform(embeddings.transpose((1, 0))).squeeze()
    average = np.average(embeddings, axis=0)
    assert first_component.shape == average.shape, f"({first_component.shape})==({average.shape})"
    return cosine_similarity_embedding(first_component, average)

def property_eval(embeddings):
    cosine_sim = []
    for doc_embedding in tqdm(embeddings):
        cosine_sim.append(calc_average_vs_first_pc(doc_embedding))
    cosine_sim = abs(np.array(cosine_sim))
    return cosine_sim

def test_property(corpus_name, language_model, multi_layers=True):
    repr_path = f"{corpus_name}.{language_model}.pk"
    with open(repr_path, "rb") as f:
        data = pickle.load(f)
        embeddings = data["embeddings"]
    if multi_layers:
        embeddings = [sent_embedding[-1] for sent_embedding in embeddings]
    print("doc length", np.average([len(x) for x in embeddings]))
    cosine_sims = property_eval(embeddings)
    print(f"max {np.max(cosine_sims)} "
          f"min {np.min(cosine_sims)} "
          f"mean {np.average(cosine_sims)}")

if __name__ == '__main__':
    test_property(corpus_name="ag_news", language_model="bert-base-cased")
    test_property(corpus_name="ag_news", language_model="skip_gram", multi_layers=False)
    test_property(corpus_name="dbpedia_14", language_model="elmo", multi_layers=False)