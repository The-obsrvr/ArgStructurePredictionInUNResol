import copy
import re
from collections import Counter
import json
from typing import Any

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors


def load_tags(csv_path):
    """

    :param csv_path:
    :return:
    """
    df = pd.read_csv(csv_path, sep=";")

    tags = []
    for _, row in df.iterrows():
        dimension = str(row["Dimensions"])
        category = str(row["Categories"])
        code = str(row["CODE"])

        # hierarchical text representation
        text = f"passage: {dimension} | {category} | {code}"

        tags.append({
            "code": code,
            "dimension": dimension,
            "category": category,
            "text": text
        })

    return tags


# -------------------------------------
# STEP 1 Document Level LLM reasoning
# -------------------------------------

def build_structure_prompt(doc):
    paragraphs = doc["body"]["paragraphs"]

    para_text = []
    for p in paragraphs:
        para_text.append(
            f"[{p['para_number']}]\nFR: {p['para']}\nEN: {p['para_en']}\n"
        )

    para_block = "\n".join(para_text)

    prompt = f"""
You are an expert in UN resolution analysis.

TASK:
Classify each paragraph as:
- preambular (context/justification)
- operative (actions/recommendations)

Use:
- French for structural cues
- English for meaning

Rules:
- Preambular usually appear first
- Operative contain actions or recommendations
- Every paragraph must be classified exactly once

Return STRICT JSON:

{{
  "preambular_para": [],
  "operative_para": [],
  "think": ""
}}

INPUT Paragraphs:
{para_block}

- Output ONLY valid JSON
- The final answer must start with {{ and end with }}
"""
    return prompt.strip()


def run_document_level_llm(model, tokenizer, prompt, temperature=0.1):
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4096,
        do_sample=True,
        temperature=temperature,
        top_p=0.9
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # extract thinking
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

    return thinking, content





def validate_structure(output, doc):
    n = len(doc["body"]["paragraphs"])

    pre = set(output["preambular_para"])
    op = set(output["operative_para"])

    assert pre.isdisjoint(op), "Overlap between pre and op"

    all_ids = set(range(1, n + 1))
    assigned = pre | op

    assert assigned == all_ids, f"Incomplete assignment: {assigned}"

    for idx in assigned:
        assert 1 <= idx <= n, f"Invalid index: {idx}"

    return True


def merge_structures(out1, out2, think1, think2, n):
    final_pre = set()
    final_op = set()

    for i in range(1, n + 1):
        in_pre1 = i in out1["preambular_para"]
        in_pre2 = i in out2["preambular_para"]

        # majority vote (2 runs → agreement or fallback)
        if in_pre1 and in_pre2:
            final_pre.add(i)
        elif (not in_pre1) and (not in_pre2):
            final_op.add(i)
        else:
            # disagreement → fallback heuristic
            # prefer later paragraphs as operative
            if i > n // 2:
                final_op.add(i)
            else:
                final_pre.add(i)

    # reasoning selection based on agreement
    score1 = len(set(out1["preambular_para"]) & final_pre)
    score2 = len(set(out2["preambular_para"]) & final_pre)

    final_think = think1 if score1 >= score2 else think2

    return {
        "preambular_para": sorted(list(final_pre)),
        "operative_para": sorted(list(final_op)),
        "think": final_think
    }


def parse_output(content):
    match = re.search(r"\{.*}", content, re.DOTALL)

    if match is None:
        raise ValueError(f"No JSON found in model output:\n{content[:500]}")

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON:\n{match.group()[:500]}")


def run_structure_self_consistency(model, tokenizer, doc):
    # doc = load_document(doc_path)
    prompt = build_structure_prompt(doc)

    # run twice (self consistency)
    think1, content1 = run_document_level_llm(model, tokenizer, prompt, temperature=0.1)
    think2, content2 = run_document_level_llm(model, tokenizer, prompt, temperature=0.2)

    out1 = parse_output(content1)
    out2 = parse_output(content2)

    n = len(doc["body"]["paragraphs"])
    # merge the outputs of the self-consistency runs
    merged = merge_structures(out1, out2, think1, think2, n)

    validate_structure(merged, doc)

    return merged


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
    min_sim=0.45,
    max_k=10,
    final_cap=15
):
    # bilingual query
    query_text = f"query: {paragraph['para']} {paragraph['para_en']}"

    q_emb = model.encode([query_text], normalize_embeddings=True)

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
    max_k=12,
    window_size=3,
    final_cap=10
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

    # 2. PROXIMITY WINDOW
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

# -------------------------------------
# Step 4: Paragraph Level LLM Reasoning
# -------------------------------------

def build_paragraph_prompt(para, candidate_tags, relation_candidates, all_paragraphs):

    tag_block = "\n".join([
        f"{t['code']}: {t['dimension']} | {t['category']}"
        for t in candidate_tags
        ]
        )

    candidate_para_block = "\n\n".join([
        f"[{p['para_number']}]\nEN: {p['para_en']}\nFR: {p['para']}"
        for p in all_paragraphs
        if p["para_number"] in relation_candidates
        ]
        )


    return f"""
You are an expert in UN resolution analysis.

TASK:
For each paragraph, you must:

1. Assign relevant tags from the provided candidate_tags
2. Identify which OTHER paragraphs are related to this paragraph
3. Assign one or more relation types for each matched paragraph

-------------------------------------

INPUT (for current paragraph):

- para_number: {para["para_number"]}
- text_fr: {para["para"]}
- text_en: {para["para_en"]}

- candidate_tags: {tag_block}
- relation_candidates: {candidate_para_block}
- allowed_paragraph_ids: {relation_candidates}

-------------------------------------

DEFINITIONS

TAGS:
- Select ONLY from candidate_tags
- Output ONLY tag codes (e.g., "A1", "B2")
- A paragraph may have multiple tags
- Only include tags clearly supported by the paragraph


-------------------------------------

RELATIONS

- para_number is the TARGET paragraph (current paragraph)
- matched_paras must contain OTHER paragraphs that relate to this paragraph

FORMAT:
matched_paras = {
  "X": (relation_type_1, relation_type_2, ...),
  ...
}

Where:
- X is the source paragraph_number shown in brackets [X]
- X must be chosen from allowed_paragraph_ids ONLY

-------------------------------------
RELATION TYPES

supporting:
- the OTHER paragraph supports or agrees with this paragraph

complemental:
- the OTHER paragraph adds related information to this paragraph

contradictive:
- the OTHER paragraph contradicts or challenges this paragraph

modifying:
- the OTHER paragraph refines or adjusts this paragraph

-------------------------------------

IMPORTANT RULES

- Only select paragraph indices from allowed_paragraph_ids
- Do NOT include the current paragraph itself
- Relations can exist between ANY paragraphs in relation_candidates (no ordering restriction)

- A pair of paragraphs may have MULTIPLE relation types
- Include ALL applicable relation types

- Only create a relation if there is a clear semantic connection

-------------------------------------

REASONING REQUIREMENTS

- Think carefully before answering
- When identifying relations:
  • determine which parts of the current paragraph relate to which parts of the other paragraph
  • use this to decide the correct relation type(s)

-------------------------------------

OUTPUT (STRICT JSON FORMAT)

{
  "para_number": {para["para_number"]},
  "tags": [],
  "matched_paras": {
    "X": ("relation_type", ...),
    ...
  },
  "think": ""
}

-------------------------------------

THINK FIELD

- Explain ONLY why the selected tags apply
- Keep it concise (1–2 sentences)

-------------------------------------

CONSTRAINTS

- Output JSON only
- Do NOT invent tags
- Do NOT include paragraphs outside relation_candidates
- Do NOT include empty relation entries
"""


def run_paragraph_level_llm(model, tokenizer, prompt, temperature=0.1):
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=temperature,
        top_p=0.9
    )

    output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

    return thinking, content


def merge_tags(tags1, tags2):
    counts = Counter(tags1 + tags2)
    return [t for t, c in counts.items() if c >= 1]  # union


def merge_relations_strict(r1, r2):
  """
  use intersection in the two outputs to ensure higher precision and agreement
  """
  merged = {}
  keys = set(r1.keys()) | set(r2.keys())
  for k in keys:
      set1 = set(r1.get(k, []))
      set2 = set(r2.get(k, []))
      common = set1 & set2
      if common:
          merged[k] = list(common)
  return merged


def reasoning_score(pred_tags, final_tags):
    pred = set(pred_tags)
    final = set(final_tags)
    return len(pred & final) / max(len(pred), 1)


def merge_outputs(o1, o2, t1, t2):
    final_tags = merge_tags(o1["tags"], o2["tags"])
    final_relations = merge_relations_strict(o1["matched_paras"], o2["matched_paras"])

    score1 = reasoning_score(o1["tags"], final_tags)
    score2 = reasoning_score(o2["tags"], final_tags)

    final_think = t1 if score1 >= score2 else t2

    return {
        "para_number": o1["para_number"],
        "tags": final_tags,
        "matched_pars": final_relations,
        "think": final_think
    }


def validate_paragraph_output(output, para_number, relation_candidates):
    assert output["para_number"] == para_number

    # relations valid
    for target in output["matched_pars"].keys():
        t = int(target)
        # assert t < para_number, f"Invalid relation {para_number}->{t}"
        assert t in relation_candidates, f"Target {t} not in candidates"

    # tags non-empty list
    assert isinstance(output["tags"], list)

    # think exists
    assert isinstance(output["think"], str) and len(output["think"]) > 0


def process_paragraph(
    model,
    tokenizer,
    para,
    candidate_tags,
    relation_candidates,
    all_paragraphs
):
    prompt = build_paragraph_prompt(
        para,
        candidate_tags,
        relation_candidates,
        all_paragraphs
    )

    # run twice
    t1, c1 = run_paragraph_level_llm(model, tokenizer, prompt, temperature=0.1)
    t2, c2 = run_paragraph_level_llm(model, tokenizer, prompt, temperature=0.2)

    o1 = parse_output(c1)
    o2 = parse_output(c2)

    merged = merge_outputs(o1, o2, t1, t2)

    validate_paragraph_output(
        merged,
        para["para_number"],
        relation_candidates
    )

    return merged


def run_para_level_reasoning(model, tokenizer, doc, tag_candidates, relation_candidates):
    """

    :param model:
    :param tokenizer:
    :param doc:
    :param tag_candidates:
    :param relation_candidates:
    :return:
    """
    outputs = {}

    paragraphs = doc["body"]["paragraphs"]

    for para in paragraphs:
        i = para["para_number"]

        result = process_paragraph(
            model,
            tokenizer,
            para,
            tag_candidates[i],
            relation_candidates[i],
            paragraphs
        )

        outputs[i] = result

    return outputs


# -------------------------------------
# Step 5: Update doc
# -------------------------------------

def update_document(doc, step1_out, step4_out):

    doc = copy.deepcopy(doc)

    preambular = step1_out["preambular_para"]
    operative = step1_out["operative_para"]
    think = step1_out["think"]

    # META
    doc["preambular_para"] = preambular
    doc["operative_para"] = operative
    doc["think"] = think

    # BODY
    for para in doc["body"]["paragraphs"]:
        pid = para["para_number"]

        para["type"] = "preambular" if pid in preambular else ("operative" if pid in operative else None)

        if pid in step4_out:
            para["tags"] = step4_out[pid]["tags"]
            para["matched_pars"] = step4_out[pid]["matched_pars"]
            para["think"] = step4_out[pid]["think"]

    return doc