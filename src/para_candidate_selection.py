from sklearn.neighbors import NearestNeighbors


# -------------------------------------
# STEP 3: Matched Paragraph Candidate Retrieval
# -------------------------------------

def build_paragraph_index(paragraphs, model):
    texts = [
        f"passage: {p['para_en']}"
        for p in paragraphs
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    index = NearestNeighbors(
        n_neighbors=min(20, len(paragraphs)),
        metric="cosine"
        )
    index.fit(embeddings)

    return index, embeddings


def retrieve_paragraph_candidates(
    paragraphs,
    index,
    embeddings,
    i,
    model,
    min_sim=0.45,
    max_k=10,
    window_size=2,
    final_cap=6
):
    """
    i = target index (0-based)
    return: list of source paragraph indices (1-based)
    """

    n = len(paragraphs)
    target_para = paragraphs[i]

    # 1. SEMANTIC RETRIEVAL
    query = f"query: {target_para['para_en']}"
    q_emb = model.encode([query], normalize_embeddings=True)

    k = min(max_k, len(paragraphs))

    distances, indices = index.kneighbors(q_emb, n_neighbors=k)

    semantic = []
    for dist, idx in zip(distances[0], indices[0]):
        score = 1 - dist  # cosine similarity

        if score >= min_sim:
            semantic.append((idx, score))

    # 2. PROXIMITY WINDOW (next immediate paragraphs may likely be related)
    window = []
    for j in range(i + 1, min(n, i + 1 + window_size)):
        window.append((j, 1.0))  # strong prior

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
    paragraphs = doc["body"]["paragraphs"]

    index, embeddings = build_paragraph_index(paragraphs, model)

    results = {}

    for i in range(len(paragraphs)):
        candidates = retrieve_paragraph_candidates(
            paragraphs,
            index,
            embeddings,
            i,
            model
        )

        results[i + 1] = candidates

    return results