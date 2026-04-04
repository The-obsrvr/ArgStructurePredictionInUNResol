import re
import json

# -------------------------------------
# STEP 1 Document Level LLM reasoning
# -------------------------------------

def build_structure_prompt(doc):
    paragraphs = doc["body"]["paragraphs"]

    para_text = []

    def compress_para(text, max_len=200):
        text = text.strip().replace("\n", " ")
        return text[:max_len]
    for p in paragraphs:
        para_text.append(
            f"{p['para_number']}. {compress_para(p['para'])}"
            )


    para_block = "\n".join(para_text)

    # prompt = f"""
    # You are an expert in UN resolution analysis.
    #
    # TASK:
    # Classify each paragraph as:
    # - preambular (context/justification)
    # - operative (actions/recommendations)
    #
    # Rules (VERY IMPORTANT):
    #
    # Preambular paragraphs (French cues):
    # - Start with: "Rappelant", "Reconnaissant", "Considérant", "Notant", "Soulignant"
    # - Often end with commas (,)
    # - Provide context, background, justification
    #
    # Operative paragraphs (French cues):
    # - Start with: "Décide", "Demande", "Encourage", "Prie", "Exhorte"
    # - Often numbered and action-oriented
    # - Contain clear actions, instructions, or recommendations
    #
    # STRICT REASONING:
    # - If a paragraph contains a clear operative verb or action → classify as operative
    # - If it only provides context → preambular
    #
    # CONFIDENCE SCORING (VERY IMPORTANT):
    # For EACH paragraph, assign a confidence score between 0 and 1:
    # - 0.9–1.0 → explicit cue word or very clear meaning
    # - 0.7–0.89 → strong but indirect signal
    # - 0.5–0.69 → somewhat unclear
    # - <0.5 → uncertain or ambiguous
    #
    # Return STRICT JSON:
    #
    # {{
    #   "preambular_para": [list of paragraph numbers],
    #   "operative_para": [list of paragraph numbers],
    #   "confidence": {{
    #       "1": 0.95,
    #       "2": 0.80,
    #       "3": 0.60
    #   }},
    #   "think": "Explain classification using French cues and meaning"
    # }}
    #
    # INPUT Paragraphs:
    # {para_block}
    #
    # RULES:
    # - Every paragraph MUST appear in exactly one class
    # - Every paragraph MUST have a confidence score
    # - Confidence keys MUST be strings of paragraph numbers
    # - Output ONLY valid JSON
    # - Do NOT return empty JSON {{}}
    # """
    # return prompt.strip()

    prompt = f"""
    You are an expert in UN resolution analysis written in French.
    
    TASK:
    1. Identify how preambular and operative paragraphs are distinguished in THIS document.
    2. Use discourse markers and linguistic cues (NOT ordering).
    3. Apply ONE consistent rule to classify ALL paragraphs.
    
    DEFINITIONS:
    
    Preambular paragraphs:
    - Provide context, justification, background
    - Often begin with: "Considérant", "Rappelant", "Reconnaissant", "Notant", "Soulignant"
    - Often end with commas
    - Do NOT contain actions
    
    Operative paragraphs:
    - Contain actions, recommendations, or directives
    - May include verbs like: "Décide", "Demande", "Encourage"
    - May be structured or directive
    
    IMPORTANT:
    - The document may NOT be ordered
    - You MUST rely on linguistic/discourse cues
    - You MUST produce ONE GLOBAL reasoning
    
    RETURN STRICT JSON:
    
    {{
      "preambular_para": [2, 3, 6, ...],
      "operative_para": [1, 4, 5, ...],
      "reasoning": "One unified explanation of how you distinguished preambular vs operative"
    }}
    
    RULES:
    - Do NOT omit any paragraph. Every paragraph must appear exactly once.
    - MUST be either "preambular" or "operative"
    - Reasoning must explain the rule used
    - Output ONLY valid JSON
    
    INPUT:
    {para_block}
    """
    return prompt.strip()


def run_qwen_generation(model, tokenizer, prompt, temperature=0.1, max_tokens=2048):
    """

    """
    messages = [
        {"role": "system", "content": "You are a strict JSON generator."},
        {"role": "user", "content": prompt}
        ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    if not text.strip():
        raise ValueError("Empty chat template")

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9
    )

    output_ids = outputs[0][len(inputs.input_ids[0]):]
    full_output = tokenizer.decode(output_ids, skip_special_tokens=True)

    if "</think>" in full_output:
        thinking, content = full_output.split("</think>", 1)
    else:
        thinking, content = "", full_output

    return thinking.strip(), content.strip()


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


def extract_json_block(text):
    if "</think>" in text:
        text = text.split("</think>")[-1]

    start = text.find("{")
    if start == -1:
        return None

    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1

        if brace_count == 0:
            return text[start:i+1]

    return text[start:]


def parse_output_safe(text):
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

    return json.loads(json_str)


def validate_structure(output, doc):
    """

    """
    n = len(doc["body"]["paragraphs"])

    pre = set(output["preambular_para"])
    op = set(output["operative_para"])

    assert pre.isdisjoint(op)
    assert pre | op == set(range(1, n+1))


def run_structure_self_consistency(model, tokenizer, doc, self_consistency=True, max_retries=2):

    prompt = build_structure_prompt(doc)
    n = len(doc["body"]["paragraphs"])

    def process_output(output):
        pre = output.get("preambular_para", [])
        op = output.get("operative_para", [])
        reasoning = output.get("reasoning", "")

        if not isinstance(pre, list) or not isinstance(op, list):
            raise ValueError("Invalid structure lists")

        pre_set = set(pre)
        op_set = set(op)

        if pre_set & op_set:
            raise ValueError("Overlap between preambular and operative")

        if pre_set | op_set != set(range(1, n + 1)):
            raise ValueError("Missing or extra paragraph assignments")

        return {
            "preambular_para": sorted(pre),
            "operative_para": sorted(op),
            "think": reasoning
            }

    # -------------------------------------
    # SELF-CONSISTENCY MODE
    # -------------------------------------
    if self_consistency:

        for attempt in range(max_retries):
            try:
                think1, content1 = run_qwen_generation(
                    model, tokenizer, prompt, temperature=0.1
                )
                think2, content2 = run_qwen_generation(
                    model, tokenizer, prompt, temperature=0.2
                )

                out1 = parse_output_safe(content1)
                out2 = parse_output_safe(content2)

                res1 = process_output(out1)
                res2 = process_output(out2)

                # Agreement → return
                if (
                    res1["preambular_para"] == res2["preambular_para"]
                    and res1["operative_para"] == res2["operative_para"]
                ):
                    return res1

                # Disagreement → pick richer reasoning
                def reasoning_score(text):
                    return len(text.split())

                chosen = res1 if reasoning_score(res1["think"]) >= reasoning_score(res2["think"]) else res2

                chosen["think"] += "\n\n[Self-consistency: disagreement resolved by selecting richer reasoning]"
                return chosen

            except Exception:
                continue

        # ✅ SAFE FALLBACK
        return {
            "preambular_para": [],
            "operative_para": [],
            "think": "LLM failed to classify structure after multiple attempts (self-consistency mode). Returned empty lists."
        }

    # -------------------------------------
    # Without self-consistency
    # -------------------------------------
    else:

        for attempt in range(max_retries):
            try:
                thinking, content = run_qwen_generation(
                    model, tokenizer, prompt, temperature=0.1
                )

                output = parse_output_safe(content)
                result = process_output(output)

                return result

            except Exception:
                continue

        # ✅ SAFE FALLBACK
        return {
            "preambular_para": [],
            "operative_para": [],
            "think": "LLM failed to classify structure after multiple attempts (single-run mode). Returned empty lists."
        }
