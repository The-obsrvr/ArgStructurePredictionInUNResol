import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from inference import (build_tag_index, run_structure_self_consistency,
                       generate_tag_candidates_for_paragraph, generate_para_candidates, run_para_level_reasoning,
                       update_document, load_tags
                       )


def main():
    """
    Process UN resolution documents to identify argumentative paragraphs, classify them and relate them to each other.

    :return:
    """

    input_folder = "Data/test-data"
    output_folder = "outputs"
    # create output folder
    os.makedirs(output_folder, exist_ok=True)

    # define embedding model
    embed_model = SentenceTransformer("intfloat/multilingual-e5-large")

    # define LLM model
    llm_model_name = "Qwen/Qwen3-8B-Instruct"
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
    tags_path = ""
    tags = load_tags(tags_path)
    tag_index, id2tag, tag_embeddings = build_tag_index(tags, embed_model)

    # load test folder
    for file in os.listdir(input_folder):

        if not file.endswith(".json"):
            continue
        print(f"Processing {file}")

        with open(os.path.join(input_folder, file), "r", encoding="utf-8") as f:

            doc = json.load(f)

        # Step 1: Document Level LLM reasoning
        doc_level_output = run_structure_self_consistency(llm_model, tokenizer, doc)
        # this returns preambular list, operative list and thinking for this step

        # Step 2: Tag Candidate Retrieval
        tag_candidates = generate_tag_candidates_for_paragraph(doc, embed_model, tag_index, id2tag, tag_embeddings)
        # this returns the top tag candidates for each paragraph in the doc

        # Step 3: Matched Paragraph Retrieval
        paragraph_candidates = generate_para_candidates(doc, embed_model)
        # this returns the top matched paragraph candidates for each paragraph in the doc

        # Step 4: Paragraph level LLM reasoning
        para_level_output = run_para_level_reasoning(llm_model, tokenizer, doc, tag_candidates, paragraph_candidates)
        # this returns the predicted tags and matched_para for each paragraph in the doc

        # Step 5: Merge the outputs to the required schema and save
        updated_doc = update_document(doc, doc_level_output, para_level_output)

        with open(os.path.join(output_folder, file), "w", encoding="utf-8") as f:
            json.dump(updated_doc, f, indent=2, ensure_ascii=False)

    print("Processing complete")


if __name__ == "__main__":
    main()
