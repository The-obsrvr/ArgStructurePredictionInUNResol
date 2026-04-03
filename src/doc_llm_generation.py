import re
import json

# -------------------------------------
# STEP 1 Document Level LLM reasoning
# -------------------------------------

def build_structure_prompt(doc):
    paragraphs = doc["body"]["paragraphs"]

    para_text = []
    for p in paragraphs:
        para_text.append(
            f"[Paragraph number: {p['para_number']}]\nFR: {p['para']}\n"
        )

    para_block = "\n".join(para_text)

    prompt = f"""
    You are an expert in UN resolution analysis.

    TASK:
    Classify each paragraph as:
    - preambular (context/justification)
    - operative (actions/recommendations)

    Rules (VERY IMPORTANT):

    Preambular paragraphs (French cues):
    - Start with: "Rappelant", "Reconnaissant", "Considérant", "Notant", "Soulignant"
    - Often end with commas (,)
    - Provide context, background, justification

    Operative paragraphs (French cues):
    - Start with: "Décide", "Demande", "Encourage", "Prie", "Exhorte"
    - Often numbered and action-oriented
    - Contain clear actions, instructions, or recommendations

    STRICT REASONING:
    - If a paragraph contains a clear operative verb or action → classify as operative
    - If it only provides context → preambular

    CONFIDENCE SCORING (VERY IMPORTANT):
    For EACH paragraph, assign a confidence score between 0 and 1:
    - 0.9–1.0 → explicit cue word or very clear meaning
    - 0.7–0.89 → strong but indirect signal
    - 0.5–0.69 → somewhat unclear
    - <0.5 → uncertain or ambiguous

    Return STRICT JSON:

    {{
      "preambular_para": [list of paragraph numbers],
      "operative_para": [list of paragraph numbers],
      "confidence": {{
          "1": 0.95,
          "2": 0.80,
          "3": 0.60
      }},
      "think": "Explain classification using French cues and meaning"
    }}

    INPUT Paragraphs:
    {para_block}

    RULES:
    - Every paragraph MUST appear in exactly one class
    - Every paragraph MUST have a confidence score
    - Confidence keys MUST be strings of paragraph numbers
    - Output ONLY valid JSON
    - Do NOT return empty JSON {{}}
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


def run_structure_self_consistency(model, tokenizer, doc, self_consistency=True):

    prompt = build_structure_prompt(doc)

    def get_label(output, i):
        if i in output.get("preambular_para", []):
            return "pre"
        elif i in output.get("operative_para", []):
            return "op"
        return None

    def get_conf(output, i):
        conf_dict = output.get("confidence", {})
        return float(conf_dict.get(str(i), 0.0))

    if self_consistency:
        # run twice
        think1, content1 = run_qwen_generation(model, tokenizer, prompt, temperature=0.1)
        think2, content2 = run_qwen_generation(model, tokenizer, prompt, temperature=0.2)

        out1 = parse_output_safe(content1)
        out2 = parse_output_safe(content2)

        n = len(doc["body"]["paragraphs"])

        final_pre = set()
        final_op = set()

        think_trace = []  # keep reasoning trace

        for i in range(1, n + 1):
            l1 = get_label(out1, i)
            l2 = get_label(out2, i)

            c1 = get_conf(out1, i)
            c2 = get_conf(out2, i)

            # CASE 1: agreement
            if l1 == l2 and l1 is not None:
                if l1 == "pre":
                    final_pre.add(i)
                else:
                    final_op.add(i)

                think_trace.append(f"[{i}] agree → {l1} (c1={c1:.2f}, c2={c2:.2f})")

            # CASE 2: disagreement → use confidence
            else:
                if c1 > c2:
                    chosen = l1
                    chosen_conf = c1
                    source = "run1"
                else:
                    chosen = l2
                    chosen_conf = c2
                    source = "run2"

                if chosen == "pre":
                    final_pre.add(i)
                else:
                    final_op.add(i)

                think_trace.append(
                    f"[{i}] conflict → {chosen} from {source} (c1={c1:.2f}, c2={c2:.2f})"
                )

            # safety fallback (very rare)
            if l1 is None and l2 is None:
                if i > n // 2:
                    final_op.add(i)
                    think_trace.append(f"[{i}] fallback → op (no labels)")
                else:
                    final_pre.add(i)
                    think_trace.append(f"[{i}] fallback → pre (no labels)")

        # choose best reasoning block (same as your logic)
        score1 = len(set(out1.get("preambular_para", [])) & final_pre)
        score2 = len(set(out2.get("preambular_para", [])) & final_pre)

        base_think = think1 if score1 >= score2 else think2

        merged = {
            "preambular_para": sorted(list(final_pre)),
            "operative_para": sorted(list(final_op)),
            "think": base_think + "\n\nMERGE TRACE:\n" + "\n".join(think_trace)
        }

        validate_structure(merged, doc)
        return merged

    else:
        LOW_CONF_THRESHOLD = 0.6
        MAX_LOW_CONF_RATIO = 0.4  # tolerate up to 40% uncertain paragraphs

        for attempt in range(3):
            thinking, content = run_qwen_generation(model, tokenizer, prompt)

            try:
                output = parse_output_safe(content)

                if "preambular_para" not in output or "operative_para" not in output:
                    continue

                n = len(doc["body"]["paragraphs"])

                conf_dict = output.get("confidence", {})
                low_conf_count = 0

                for i in range(1, n + 1):
                    conf = float(conf_dict.get(str(i), 0.0))
                    if conf < LOW_CONF_THRESHOLD:
                        low_conf_count += 1

                low_conf_ratio = low_conf_count / n

                # Accept only if confidence is good enough
                if low_conf_ratio <= MAX_LOW_CONF_RATIO:
                    validate_structure(output, doc)

                    output["think"] = (
                            thinking
                            + f"\n\nCONFIDENCE CHECK: low={low_conf_count}/{n} "
                              f"(ratio={low_conf_ratio:.2f}) → accepted"
                    )

                    return output

                else:
                    # retry with trace
                    continue

            except Exception:
                continue

        # fallback (only after confidence failures)
        n = len(doc["body"]["paragraphs"])
        split = n // 2

        return {
            "preambular_para": list(range(1, split + 1)),
            "operative_para": list(range(split + 1, n + 1)),
            "think": (
                "fallback rule-based logic applied\n"
                f"Reason: confidence too low across attempts (threshold={LOW_CONF_THRESHOLD})"
            )
            }
