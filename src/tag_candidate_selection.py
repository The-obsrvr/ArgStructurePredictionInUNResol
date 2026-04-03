from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

# -------------------------------------
# Step 2: Tag Candidate Retrieval
# -------------------------------------

# 2. BUILD INDEX
def build_tag_index(tags, model):
    """

    :param tags:
    :param model:
    :return:
    """
    texts = [t["text"] for t in tags]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    index = NearestNeighbors(n_neighbors=min(50, len(tags)), metric='cosine').fit(embeddings)

    id2tag = {i: tags[i] for i in range(len(tags))}

    return index, id2tag, embeddings


# 3. RETRIEVAL (THRESHOLD)
def retrieve_tag_candidates(
    paragraph,
    model,
    tag_index,
    id2tag,
    tag_embeddings,
    min_sim=0.55,
    max_k=10,
    final_cap=7
):
    # bilingual query
    emb_en = model.encode([f"query: {paragraph['para_en']}"], normalize_embeddings=True)
    emb_fr = model.encode([f"query: {paragraph['para']}"], normalize_embeddings=True)

    q_emb = (emb_en + emb_fr) / 2

    k = min(max_k, len(id2tag))

    distances, indices = tag_index.kneighbors(q_emb, n_neighbors=k)

    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        score = 1 - dist  # cosine similarity
        if score >= min_sim:
            tag = id2tag[idx]
            candidates.append((tag, score, idx))

    # fallback to top k tags if no candidate passes threshold
    if len(candidates) == 0:
        candidates = [(id2tag[idx], 1 - dist, idx)
                      for dist, idx in zip(distances[0][:3], indices[0][:3])]

    # 4. check for duplicates or too similar tags.
    selected = []
    selected_embs = []

    DUPLICATE_THRESHOLD = 0.97

    for tag, score, idx in candidates:
        emb = tag_embeddings[idx]

        is_duplicate = False
        for e in selected_embs:
            if np.dot(e, emb) > DUPLICATE_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            selected.append(tag)
            selected_embs.append(emb)

        # keep maximum selected tags to final cap
        if len(selected) >= final_cap:
            break

    return selected


def generate_tag_candidates_for_paragraph(doc, model, tag_index, id2tag, tag_embeddings):
    """

    :param doc:
    :param model:
    :param tag_index:
    :param id2tag:
    :param tag_embeddings:
    :return:
    """
    results: dict[Any, Any] = {}

    for p in doc["body"]["paragraphs"]:
        candidates = retrieve_tag_candidates(
            p,
            model,
            tag_index,
            id2tag,
            tag_embeddings
        )

        results[p["para_number"]] = candidates

    return results
