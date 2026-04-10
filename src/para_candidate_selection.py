from sklearn.neighbors import NearestNeighbors
import numpy as np


# STEP 3: Matched Paragraph Candidate Retrieval
# -------------------------------------
def build_paragraph_index(paragraphs, model):
    """

    :param paragraphs:
    :param model:
    :return:
    """
    texts_en = [
        f"passage: {p['para_en']}"
        for p in paragraphs
        ]

    texts_fr = [
        f"passage: {p['para']}"
        for p in paragraphs
        ]

    # Encode both languages
    embeddings_en = model.encode(
        texts_en,
        normalize_embeddings=True,
        show_progress_bar=True
        )

    embeddings_fr = model.encode(
        texts_fr,
        normalize_embeddings=True,
        show_progress_bar=True
        )

    # Average embeddings
    embeddings = (embeddings_en + embeddings_fr) / 2

    # re-normalize after averaging
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Build index
    index = NearestNeighbors(
        n_neighbors=min(20, len(paragraphs)),
        metric="cosine"
        )
    index.fit(embeddings)

    return index


def retrieve_paragraph_candidates(
    paragraphs,
    index,
    i,
    model,
    min_sim=0.55,
    max_k=20,
    window_size=2,
    final_cap=6
):
    """

    :param paragraphs:
    :param index:
    :param i:
    :param model:
    :param min_sim:
    :param max_k:
    :param window_size:
    :param final_cap:
    :return:
    """

    n = len(paragraphs)
    target_para = paragraphs[i]

    # 1. SEMANTIC RETRIEVAL
    emb_en = model.encode(
        [f"query: {target_para['para_en']}"],
        normalize_embeddings=True
        )
    emb_fr = model.encode(
        [f"query: {target_para.get('para', target_para['para'])}"],
        normalize_embeddings=True
        )

    # average + normalize
    q_emb = (emb_en + emb_fr) / 2
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

    k = min(max_k, n)

    # index contains all the paragraph embeddings
    distances, indices = index.kneighbors(q_emb, n_neighbors=k)

    semantic = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx <= i:
            continue  # enforce forward-only

        score = 1 - dist  # cosine similarity

        if score >= min_sim:
            semantic.append((idx, score))

    # 2. PROXIMITY WINDOW (next forward paragraphs may likely be related)
    window = []
    for j in range(i + 1, min(n, i + 1 + window_size)):
        window.append((j, 1.0))

    # 3. MERGE 1 and 2
    merged = {}

    for idx, score in semantic:
        merged[idx] = max(merged.get(idx, 0), score)

    for idx, score in window:
        merged[idx] = max(merged.get(idx, 0), score)

    # 4. SORT + SELECT
    sorted_items = sorted(merged.items(), key=lambda x: -x[1])

    final = [idx + 1 for idx, _ in sorted_items[:final_cap]]

    return final


def generate_para_candidates(doc, model):
    """

    :param doc:
    :param model:
    :return:
    """
    paragraphs = doc["body"]["paragraphs"]

    index = build_paragraph_index(paragraphs, model)

    results = {}

    for i in range(len(paragraphs)):
        candidates = retrieve_paragraph_candidates(
            paragraphs,
            index,
            i,
            model
        )

        results[i + 1] = candidates

    return results