import re
import json
from collections import Counter


# STEP 1 Document Level LLM reasoning
def build_structure_prompt(doc):
    """

    :param doc:
    :return:
    """
    paragraphs = doc["body"]["paragraphs"]

    para_text = []

    def compress_para(text, max_len=120):
        text = text.strip().replace("\n", " ")
        return text[:max_len]
    for p in paragraphs:
        para_text.append(
            f"{p['para_number']}. FR: {compress_para(p['para'])}"
            )

    n = len(paragraphs)

    para_block = "\n".join(para_text)

    prompt = f"""
    You are an expert in UN resolution analysis written in French.
    
    TASK:
    1. Identify how preambular and operative paragraphs are distinguished in THIS document.
    2. Use discourse markers and linguistic cues.
    3. Apply ONE consistent rule to classify ALL paragraphs.
    RETURN STRICT JSON
    
    DEFINITIONS:
    
    Preambular paragraphs (in FRENCH):
    - Provide context, justification, background
    - May begin with: "Considérant", "Rappelant", "Reconnaissant", "Notant", "Soulignant"
    - Often end with commas
    - Do NOT contain actions
    
    Operative paragraphs (in FRENCH):
    - Contain actions, recommendations, or directives
    - May include verbs like: "Décide", "Demande", "Encourage"
    - May be structured, numbered and action-oriented
    
    IMPORTANT:
    Think briefly inside <think></think>, then answer.    
    
    - Numbered paragraphs (1., I., II.) → operative
    - Usually after first operative → all following are operative
    
    OUTPUT (STRICT JSON FORMAT)
    
    {{
      "preambular_para": [list of paragraph numbers],
      "operative_para": [list of paragraph numbers],
      "think": "One unified explanation of how you distinguished preambular vs operative"
    }}
    
    RULES:
    - Do NOT omit any paragraph. Every paragraph must appear exactly ONCE.
    - MUST be either "preambular" or "operative"
    - Think must explain the rule used
    - Output ONLY valid JSON
    
    INPUT:
    {para_block}
    """
    return prompt.strip()


def run_qwen_generation(model, model_name: str, tokenizer, prompt, temperature=0.1, max_tokens=3072, top_p=1.0):
    """

    :param model_name:
    :param model:
    :param tokenizer:
    :param prompt:
    :param temperature:
    :param max_tokens:
    :param top_p:
    :return:
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict reasoning model.\n"
                "You MUST think step-by-step inside <think></think>.\n"
                "Then output valid JSON."
            )
            },
        {"role": "user", "content": prompt}
        ]

    if model_name == "qwen":
        enable_thinking = True
    else:
        enable_thinking = False

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )

    if not text.strip():
        raise ValueError("Empty chat template")

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p
    )

    output_ids = outputs[0][len(inputs.input_ids[0]):]
    full_output = tokenizer.decode(output_ids, skip_special_tokens=True)

    if "</think>" in full_output:
        thinking, content = full_output.split("</think>", 1)
    else:
        thinking, content = "", full_output

    return thinking.strip(), content.strip()


def extract_json_block(text):
    """

    :param text:
    :return:
    """
    for start in [m.start() for m in re.finditer(r"\{", text)]:
        brace_count = 0

        # parse through the text string and mark the labels
        for i in range(start, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1

            if brace_count == 0:
                candidate = text[start:i+1]

                # try parsing immediately
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    break  # not valid JSON → try next '{'

    return None


def parse_output_safe(text):
    """
    Basic output cleaning and processing to retrieve the JSON structure in readable format
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

    if "preambular_para" not in data or "operative_para" not in data:
        raise ValueError("Missing keys")

    data["preambular_para"] = [int(x) for x in data["preambular_para"]]
    data["operative_para"] = [int(x) for x in data["operative_para"]]
    return data


def validate_output(output, doc):
    """

    :param output:
    :param doc:
    :return:
    """
    n = len(doc["body"]["paragraphs"])

    pre = set(output["preambular_para"])
    op = set(output["operative_para"])

    if pre & op:
        raise ValueError("Overlap detected")

    if pre | op != set(range(1, n + 1)):
        raise ValueError("Missing or extra paragraphs")

    return {
        "preambular_para": sorted(pre),
        "operative_para": sorted(op),
        "think": output.get("think", "")
    }


def fallback(doc):
    """

    :param doc:
    :return:
    """
    pre, op = [], []
    transition = False

    for p in doc["body"]["paragraphs"]:
        text = p["para"].strip().lower()
        pid = p["para_number"]

        # bullet / numbering detection → operative
        if re.match(r"^(\d+[\.\)]{1,2}\s|[ivxlcdm]+[\.\)]{1,2}\s)", text, re.IGNORECASE):
            transition = True
            op.append(pid)
            continue

        # transition verbs
        if any(w in text for w in ["soumet", "recommande", "décide", "demande"]):
            transition = True

        if transition:
            op.append(pid)
        elif text.startswith(("considérant", "rappelant", "notant", "reconnaissant", "soulignant")):
            pre.append(pid)
        else:
            pre.append(pid)

    return {
        "preambular_para": pre,
        "operative_para": op,
        "think": "Fallback heuristic applied (bullet + transition + cues)"
    }


def merge_outputs(out1, out2, doc):
    """

    :param out1:
    :param out2:
    :param doc:
    :return:
    """
    n = len(doc["body"]["paragraphs"])

    votes = {i: [] for i in range(1, n + 1)}

    for i in out1["preambular_para"]:
        votes[i].append("pre")
    for i in out1["operative_para"]:
        votes[i].append("op")

    for i in out2["preambular_para"]:
        votes[i].append("pre")
    for i in out2["operative_para"]:
        votes[i].append("op")

    final_pre, final_op = [], []
    transition_seen = False

    # do majority voting with default preambular
    for i in range(1, n + 1):
        label = Counter(votes[i]).most_common(1)[0][0] if votes[i] else "pre"

        # adopting convention of UN resolution if operative is labeled, then all subsequent paras are also likely operative
        if label == "op":
            transition_seen = True

        if transition_seen:
            final_op.append(i)
        else:
            final_pre.append(i)

    # choose better reasoning
    def reasoning_score(text):
        """

        :param text:
        :return:
        """
        if not text:
            return 0

        score = 0

        # longer = more detailed
        score += len(text.split())

        # bonus for key signals
        keywords = ["considérant", "transition", "soumet", "recommande", "numérot", "structure"]
        score += sum(5 for k in keywords if k in text.lower())

        return score

    best_think = out1["think"] if reasoning_score(out1["think"]) >= reasoning_score(out2["think"]) else out2["think"]

    return {
        "preambular_para": final_pre,
        "operative_para": final_op,
        "think": best_think
    }


def run_structure_self_consistency(
    model,
    tokenizer,
    doc,
    model_name: str = "qwen",
    self_consistency=True,
    max_retries=1
):
    """

    :param model:
    :param tokenizer:
    :param doc:
    :param self_consistency:
    :param max_retries:
    :return:
    """
    prompt = build_structure_prompt(doc)

    # self-consistency mode
    if self_consistency:
        for attempt in range(max_retries):
            try:
                think1, content1 = run_qwen_generation(
                    model, model_name, tokenizer, prompt,
                    temperature=0.1, top_p=0.9
                    )
                think2, content2 = run_qwen_generation(
                    model, model_name, tokenizer, prompt,
                    temperature=0.15, top_p=0.9
                    )

                out1 = parse_output_safe(content1)
                out1["think"] = think1

                out2 = parse_output_safe(content2)
                out2["think"] = think2

                merged = merge_outputs(out1, out2, doc)

                return validate_output(merged, doc)

            except Exception as e:

                print(f"[ERROR] Attempt failed: {str(e)}")

                if attempt > 0:
                    prompt += "\n\nIMPORTANT: Do NOT miss any paragraph. Ensure full coverage. Your previous answer did NOT include complete JSON. You MUST output JSON."

                continue

    # No self-consistency mode
    else:
        for attempt in range(max_retries):
            try:
                think, content = run_qwen_generation(
                    model, model_name, tokenizer, prompt,
                    temperature=0.0
                    )

                out = parse_output_safe(content)
                out["think"] = think

                return validate_output(out, doc)

            except Exception as e:

                print(f"[ERROR] Attempt failed: {str(e)}")

                if attempt > 0:
                    prompt += "\n\nIMPORTANT: Do NOT miss any paragraph. Ensure full coverage. Your previous answer did NOT include complete JSON. You MUST output JSON."

                continue

    # in case of failed generation, fallback logic is used.
    return fallback(doc)
