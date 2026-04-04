import json
import os
import copy

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from sentence_transformers import SentenceTransformer
import pandas as pd

from doc_llm_generation import run_structure_self_consistency
# from para_candidate_selection import generate_para_candidates
# from tag_candidate_selection import generate_tag_candidates_for_paragraph, build_tag_index
# from para_llm_generation import run_para_level_reasoning


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
# Step 5: Update doc
# -------------------------------------
def update_document(doc, step1_out, step4_out):

    doc = copy.deepcopy(doc)

    preambular = step1_out["preambular_para"]
    operative = step1_out["operative_para"]
    think = step1_out["think"]

    # META
    doc["METADATA"]["structure"]["preambular_para"] = preambular
    doc["METADATA"]["structure"]["operative_para"] = operative
    doc["METADATA"]["structure"]["think"] = think

    # BODY
    for para in doc["body"]["paragraphs"]:
        pid = para["para_number"]

        para["type"] = "preambular" if pid in preambular else ("operative" if pid in operative else None)

        # if pid in step4_out:
        #     para["tags"] = step4_out[pid]["tags"]
        #     para["matched_pars"] = step4_out[pid]["matched_pars"]
        #     para["think"] = step4_out[pid]["think"]

    return doc


def main():
    """
    Process UN resolution documents to identify argumentative paragraphs, classify them and relate them to each other.

    :return:
    """
    """
    outputs2: gpu 4
    outputs: gpu7
    outputs3: 5
    output4: 6
    """
    input_folder = "outputs"
    output_folder = "outputs11"
    # create output folder
    os.makedirs(output_folder, exist_ok=True)

    # define embedding model
    # embed_model = SentenceTransformer("intfloat/multilingual-e5-large")

    # define LLM model
    llm_model_name = "Qwen/Qwen3-8B"
    HF_token = "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"

    is_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if is_bf16 else torch.float16

    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        device_map="auto",
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=HF_token,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        llm_model_name,
        token=HF_token
        )

    # load tags
    # tags_path = "Data/education_dimensions_updated.csv"
    # tags = load_tags(tags_path)
    # tag_index, id2tag, tag_embeddings = build_tag_index(tags, embed_model)

    # load test folder
    for file in os.listdir(input_folder):

        if not file.endswith(".json"):
            continue
        print(f"Processing {file}")

        with open(os.path.join(input_folder, file), "r", encoding="utf-8") as f:

            doc = json.load(f)

        para_level_output = None

        # Step 1: Document Level LLM reasoning
        doc_level_output = run_structure_self_consistency(llm_model, tokenizer, doc, self_consistency=True)
        # this returns preambular list, operative list and thinking for this step
        print("Step 1 complete")

        # # Step 2: Tag Candidate Retrieval
        # tag_candidates = generate_tag_candidates_for_paragraph(doc, embed_model, tag_index, id2tag, tag_embeddings)
        # # this returns the top tag candidates for each paragraph in the doc
        # print("Step 2 complete")
        #
        # # Step 3: Matched Paragraph Retrieval
        # paragraph_candidates = generate_para_candidates(doc, embed_model)
        # # this returns the top matched paragraph candidates for each paragraph in the doc
        # print("Step 3 complete")
        # # Step 4: Paragraph level LLM reasoning
        # para_level_output = run_para_level_reasoning(llm_model, tokenizer, doc, tag_candidates, paragraph_candidates, self_consistency=False)
        # # this returns the predicted tags and matched_para for each paragraph in the doc
        # print("Step 4 complete")
        # Step 5: Merge the outputs to the required schema and save

        updated_doc = update_document(doc, doc_level_output, para_level_output)
        print("Step 5 complete")
        with open(os.path.join(output_folder, file), "w", encoding="utf-8") as f:
            json.dump(updated_doc, f, indent=2, ensure_ascii=False)
        print("Saved {}".format(file))
    print("Processing complete")

if __name__ == "__main__":
    main()
