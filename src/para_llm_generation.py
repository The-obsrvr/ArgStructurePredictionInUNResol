import json
import re
from collections import Counter
from doc_llm_generation import extract_json_block, run_qwen_generation


# Step 4: Paragraph Level LLM Reasoning
def build_paragraph_prompt(para, candidate_tags, relation_candidates, all_paragraphs):
    """

    :param para:
    :param candidate_tags:
    :param relation_candidates:
    :param all_paragraphs:
    :return:
    """

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
For the given TARGET paragraph, you must:

1. Assign relevant tags from the provided candidate_tags
2. Identify which OTHER paragraphs are meaningfully related
3. Assign one or more relation types for each related paragraph

-------------------------------------

INPUT (TARGET PARAGRAPH)

- para_number: {para["para_number"]}
- text_fr: {para["para"]}
- text_en: {para["para_en"]}

- candidate_tags:
{tag_block}

- relation_candidates:
{candidate_para_block}

- allowed_paragraph_ids:
{relation_candidates}

-------------------------------------

MULTIPLE TAG CLASSIFICATION

- Select ONLY from candidate_tags
- Output ONLY tag codes (e.g., "A1", "B2")
- A paragraph may have multiple tags
- Include ONLY tags clearly supported by the paragraph content
- EXPLAIN clearly why the tags have been selected

-------------------------------------

RELATIONS (MULTI-LABEL)

- A pair of paragraphs may have MULTIPLE relation types
- Assign ALL relation types that are clearly supported

-------------------------------------

RELATION DECISION PROCESS (VERY IMPORTANT)

For EACH candidate paragraph:

STEP 1 — Check if a meaningful relation exists:
- If there is NO clear semantic connection → DO NOT include this paragraph

STEP 2 — If a relation exists, evaluate EACH relation type independently:

- supporting:
  The other paragraph reinforces, agrees with, or provides justification

- contradictive:
  The other paragraph opposes, restricts, or challenges

- modifying:
  The other paragraph refines, specifies, or adds conditions

- complemental:
  The other paragraph adds related but independent information

STEP 3 — Assign a confidence score and include ALL relation types that are above threshold of 0.5:
- 0.9–1.0 → very strong, explicit relation
- 0.7–0.9 → clear relation with strong evidence
- 0.5–0.7 → moderate relation
- < 0.5 → weak → DO NOT include 
- Only include relations with confidence ≥ 0.5
- Confidence must reflect how clearly the relation is supported by the text

-------------------------------------

REASONING GUIDELINES

- Think carefully before selecting:
- For each relation:
  • identify the specific idea in the target paragraph  
  • identify the corresponding idea in the candidate paragraph  
  • base the relation ONLY on these aligned parts  

-------------------------------------

OUTPUT (STRICT JSON FORMAT)

{{
  "para_number": {para["para_number"]},
  "tags": ["tag_code_1", "tag_code_2", ...],
  "matched_paras": {{
    "X": [
      {{"type": "relation_type", "confidence": 0.0}}
    ]
  }},
  "think": "Briefly explain why the selected tags apply"
}}

-------------------------------------

OUTPUT RULES

- Output ONLY valid JSON
- Do NOT include <think> tags
- The final answer MUST start with {{ and end with }}
- Do NOT return empty JSON {{}}

-------------------------------------

CONSTRAINTS

- Only use tag codes from candidate_tags
- Only use paragraph IDs from allowed_paragraph_ids
- Each relation MUST include a confidence score
- Do NOT include relations with confidence < 0.6
- Do NOT include the target paragraph itself
- Do NOT include empty relation entries
"""


# MERGE Self-consistency outputs
def merge_tags(tags1, tags2):
    """

    :param tags1:
    :param tags2:
    :return:
    """
    counts = Counter(tags1 + tags2)
    # take union of tags rather than intersection to optimize on recall
    return [t for t, c in counts.items() if c >= 1]


def merge_relations_strict(r1, r2):
  """

  :param r1:
  :param r2:
  :return:
  """
  merged = {}
  keys = set(r1.keys()) | set(r2.keys())
  for k in keys:
      set1 = set(r1.get(k, []))
      set2 = set(r2.get(k, []))
      # take intersection of relations rather than union to optimize on precision
      common = set1 & set2
      if common:
          merged[k] = list(common)
  return merged


def reasoning_score(pred_tags, final_tags):
    """

    :param pred_tags:
    :param final_tags:
    :return:
    """
    pred = set(pred_tags)
    final = set(final_tags)
    return len(pred & final) / max(len(pred), 1)


def merge_outputs(o1, o2, t1, t2):
    """

    :param o1: output 1 from prompt 1
    :param o2: output 2 from prompt 2
    :param t1: output thinking 1 from prompt 1
    :param t2: output thinking 2 from prompt 2
    :return: structured merged output
    """
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


# VALIDATE Paragraph Outputs
def validate_paragraph_output(output, para_number, relation_candidates):
    """

    :param output:
    :param para_number:
    :param relation_candidates:
    :return:
    """
    assert output["para_number"] == para_number

    # relations valid
    for target in output["matched_pars"].keys():
        t = int(target)
        # assert t < para_number, f"Invalid relation {para_number}->{t}"
        assert t in relation_candidates, f"Target {t} not in candidates"

    # tags non-empty list
    assert isinstance(output["tags"], list)

    # thinking exists
    assert isinstance(output["think"], str) and len(output["think"]) > 0



def fallback_paragraph_output(para_number):
    """
    No Fallback logic. Return empty JSON and report accordingly in final prediction.
    :param para_number:
    :return:
    """
    return {
        "para_number": para_number,
        "tags": [],
        "matched_pars": {},
        "think": "fallback"
    }


def parse_output_safe(text):
    """

    :param text:
    :return:
    """
    json_str = extract_json_block(text)

    if json_str is None:
        raise ValueError(f"No JSON found:\n{text[:300]}")

    json_str = json_str.strip()

    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    while json_str.count("{") > json_str.count("}"):
        json_str += "}"

    while json_str.count("[") > json_str.count("]"):
        json_str += "]"

    data = json.loads(json_str)

    return data


def process_paragraph(
    model,
    model_name,
    tokenizer,
    para,
    candidate_tags,
    relation_candidates,
    all_paragraphs,
    self_consistency=False
):
    """

    :param model:
    :param tokenizer:
    :param para:
    :param candidate_tags:
    :param relation_candidates:
    :param all_paragraphs:
    :param self_consistency:
    :return:
    """
    prompt = build_paragraph_prompt(
        para,
        candidate_tags,
        relation_candidates,
        all_paragraphs
    )

    if self_consistency:
        # run twice (same command as doc-level run LLM)
        t1, c1 = run_qwen_generation(model, model_name, tokenizer, prompt, temperature=0.1)
        t2, c2 = run_qwen_generation(model, model_name, tokenizer, prompt, temperature=0.2)

        o1 = parse_output_safe(c1)
        o2 = parse_output_safe(c2)

        output = merge_outputs(o1, o2, t1, t2)

        validate_paragraph_output(
            output,
            para["para_number"],
            relation_candidates
            )
        return output

    else:
        # do generation with retries (without self-consistency)
        for i in range(3):
            print(f"Attempt {i} for para_number {para['para_number']}")
            thinking, content = run_qwen_generation(
                model, model_name, tokenizer, prompt, temperature=0.1
                )

            try:
                parsed = parse_output_safe(content)

                # normalize relations (handle confidence)
                raw_rel = parsed.get("matched_paras") or parsed.get("matched_pars") or {}

                clean_rel = {}

                for k, rel_list in raw_rel.items():
                    try:
                        k_int = int(k)
                    except:
                        print("paragraph number is not a readable integer value.")
                        continue

                    if k_int not in relation_candidates:
                        print("related para number not in relation candidates")
                        continue

                    valid_types = []

                    for r in rel_list:
                        # case 1: with confidence score
                        if isinstance(r, dict):
                            r_type = r.get("type")
                            conf = r.get("confidence", 0)
                            # hard filter to ensure only above threshold relations are kept, even though LLM is asked to preserve only those.
                            if r_type and conf >= 0.5:
                                valid_types.append(r_type)

                        # case 2: with confidence
                        elif isinstance(r, str):
                            valid_types.append(r)

                    # add all valid types identified into the final relation list
                    if valid_types:
                        # remove duplicates
                        clean_rel[k_int] = list(set(valid_types))

                # validate tags
                tags = parsed.get("tags", [])
                if not isinstance(tags, list):
                    tags = []

                output = {
                    "para_number": parsed.get("para_number", para["para_number"]),
                    "tags": tags,
                    "matched_pars": clean_rel,
                    "think": parsed.get("think", thinking)
                    }

                validate_paragraph_output(
                    output,
                    para["para_number"],
                    relation_candidates
                    )

                return output

            except Exception as e:
                print(f"[ERROR] Attempt failed: {str(e)}")
                if i > 0:
                    prompt += "\n\nIMPORTANT: Do NOT miss any paragraph. Ensure full coverage. Your previous answer did NOT include complete JSON. You MUST output JSON."

                continue

        return fallback_paragraph_output(para['para_number'])


def run_para_level_reasoning(model, model_name, tokenizer, doc, tag_candidates, relation_candidates, self_consistency=False):
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
            model_name,
            tokenizer,
            para,
            tag_candidates[i],
            relation_candidates[i],
            paragraphs,
            self_consistency
        )

        outputs[i] = result

    return outputs
