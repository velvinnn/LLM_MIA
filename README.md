# Membership Inference Attacks (MIA) in language models.
The code for case study in the manuscript, 'Does Data Contamination Detection Work (Well) for LLMs? A Survey and Evaluation on Detection Assumptions'

## Data download
1. Pre-training datasets 
    * The [Pile dataset](https://huggingface.co/datasets/ArmelR/the-Pile-splitted), used by the [Pythia model families](https://huggingface.co/EleutherAI/pythia-70m). We randomly sampled 11 domains for experiments.
    * The [Semantic Scholar](https://huggingface.co/datasets/allenai/peS2o) and [Algebraic Stack](https://huggingface.co/datasets/EleutherAI/proof-pile-2) datasets, used by the [OLMo-7B model](https://huggingface.co/allenai/OLMo-7B). We randomly sampled 5 domains from the Algebraic Stack dataset for experiments.

2. Post-training datasets
    * **Supervised fine-tuning (sft)**: The [UltraChat dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), used by [Zephyr-7B-β model](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta).
    *  **reinforcement learning with verifiable rewards (RLVR)** [RLVR-GSM dataset](https://huggingface.co/datasets/allenai/RLVR-GSM), used by the [OLMo-2-Instruct models families (7B and 13B)](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct).

## Data pre-processing
1. We consider each domain of a big data as an individual dataset.
2. Given that some datasets can contain up to 200GB data, we always download the the top 5 data subfiles from the huggingface website. We combine those 5 subfiles into a single file, representing a dataset. 
3. For each dataset, we randomly sample 1,000 instances respectively from the train (seen) and test (unseen) splits. If there is no test splits, we sample from the validation splits. If there are fewer than 1,000 instances in an unseen split, we use all instances from that set. Domains with fewer than 100 unseen instances are excluded.
4. Save each split from every domain into a .jsonl file, where each line contains an json object, storing the text instance in {'text':'...'}. 
5. Specify your corresponding file directories in the python file, file_paths.py


## Validating assumptions
see the jupyter notebook data_contamination.ipynb
