# Modeling Argument Structure in Bilingual UN Resolutions via Retrieval-Augmented and Self-Consistent Reasoning LLMs

Team: HybridArguer at UZH Shared Task @ ArgMining Workshop 2026 : [Reconstructing the Reasoning in United Nations Resolutions](https://shared-task-argmining.linguistik.uzh.ch/)



The repository contains the scripts to extract argument structures in form of pre-defined JSON scripts from bilingual (French and English) UN resolutions documents. 

---

## Data

The dataset used in this work is derived from the Shared Task on \textit{Reconstructing the Reasoning in United Nations Resolutions}, organized by the University of Zurich (UZH) as part of the 13th Workshop on Argument Mining (2026). The training set comprises 2,695 bilingual (French–English) documents, while the held-out test set includes approximately 45 resolutions distributed across 90 JSON files with a fixed schema.

---

## Methodology

Our approach comprises four stages: 

(1) a reasoning LLM classifies paragraphs collectively as \textit{preambular} or \textit{operative}; 

(2) embedding-based similarity retrieves tag candidates for each paragraph;  

(3), similar to 2., candidate source paragraphs are selected under a chronological constraint; 

(4) the LLM processes each target paragraph individually to assign tags from its candidate pool and predict one or more predefined relation types with its candidate source paragraphs. 

Outputs from the first and final stages are consolidated into the final structured representation.  

---

## Execution 

---

## Acknowledgements

This research work has received funding from the European Union's Horizon Europe research and innovation programme under the Marie Skłodowska-Curie Grant Agreement No. 101073351. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Research Executive Agency (REA). Neither the European Union nor the granting authority can be held responsible for them.

---

## Citation

