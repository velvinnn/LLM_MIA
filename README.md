# Data-Contamination
The papers and code for the case study in the manuscript, ['Does Data Contamination Detection Work (Well) for LLMs? A Survey and Evaluation on Detection Assumptions'](https://arxiv.org/abs/2410.18966)

# Prior Papers Reviewed in Our Survey
## Instance Similarity
1. Jesse Dodge, Maarten Sap, Ana Marasovic, William Agnew, Gabriel Ilharco, Dirk Groeneveld, Margaret Mitchell, and Matt Gardner. 2021. [Documenting large webtext corpora: A case study on the colossal clean crawled corpus.](https://aclanthology.org/2021.emnlp-main.98.pdf) In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 1286–1305.
2. Aparna Elangovan, Jiayuan He, and Karin Verspoor. 2021. [Memorization vs. generalization: Quantifying data leakage in nlp performance evaluation.](https://arxiv.org/pdf/2102.01818) In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 1325–1335.
3. Yucheng Li, Yunhao Guo, Frank Guerin, and Chenghua Lin. 2024. [An open-source data contamination report for large language models](https://aclanthology.org/2024.findings-emnlp.30.pdf). In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 528–541.
4. Martin Riddell, Ansong Ni, and Arman Cohan. 2024. [Quantifying contamination in evaluating code generation capabilities of language models](https://arxiv.org/pdf/2403.04811). arXiv preprint arXiv:2403.04811.
5. Chunyuan Deng, Yilun Zhao, Xiangru Tang, Mark Gerstein, and Arman Cohan. 2024b. [Investigating data contamination in modern benchmarks for large language models.](https://arxiv.org/pdf/2311.09783) In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8698–8711.
6. Shuo Yang, Wei-Lin Chiang, Lianmin Zheng, Joseph E. Gonzalez, and Ion Stoica. 2023. [Rethinking benchmark and contamination for language models with rephrased samples.](https://arxiv.org/pdf/2311.04850) Preprint, arXiv:2311.04850.
7. Aleksandra Piktus, Christopher Akiki, Paulo Villegas, Hugo Laurençon, Gérard Dupont, Sasha Luccioni, Yacine Jernite, and Anna Rogers. 2023. [The roots search tool: Data transparency for llms.](https://aclanthology.org/2023.acl-demo.29/) In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pages 304–314.
8. Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. 2022. [Deduplicating training data makes language models better.](https://arxiv.org/pdf/2107.06499) In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8424–8445.
9. Marc Marone and Benjamin Van Durme. 2023. [Data portraits: Recording foundation model training data.](https://proceedings.neurips.cc/paper_files/paper/2023/file/3112ee706d21d734c15532c1239773e1-Paper-Datasets_and_Benchmarks.pdf) In Advances in Neural Information Processing Systems, volume 36, pages 15121–15135.
   
## Probability Analysis
### Absolute Probability
1. Congzheng Song and Vitaly Shmatikov. 2019. [Auditing data provenance in text-generation models](https://dl.acm.org/doi/pdf/10.1145/3292500.3330885). In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’19, page 196–206, New York, NY, USA. Association for Computing Machinery.
2. Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettlemoyer. 2023. [Detecting pretraining data from large language models.](https://arxiv.org/pdf/2310.16789) In NeurIPS 2023 Workshop on Regulatable ML.
3. Matthieu Meeus, Igor Shilov, Manuel Faysse, and YvesAlexandre de Montjoye. 2024b. [Copyright traps for large language models](https://arxiv.org/pdf/2402.09363). In Forty-first International Conference on Machine Learning.
4. Pratyush Maini, Hengrui Jia, Nicolas Papernot, and Adam Dziedzic. 2024b. [Llm dataset inference: Did you train on my dataset?](https://arxiv.org/pdf/2406.06443) The 1st Workshop on Data Contamination (CONDA).
5. Johnny Wei, Ryan Wang, and Robin Jia. 2024. [Proving membership in LLM pretraining data via data watermarks.](https://arxiv.org/pdf/2402.10892) In Findings of the Association for Computational Linguistics ACL 2024, pages 13306–13320, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics.
6. Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. 2023. [Beyond the imitation game: Quantifying and extrapolating the capabilities of language models.](https://arxiv.org/pdf/2206.04615) Transactions on Machine Learning Research.
7. Yucheng Li. 2023. [Estimating contamination via perplexity: Quantifying memorisation in language model evaluation.](https://arxiv.org/pdf/2309.10677) arXiv preprint arXiv:2309.10677.

**Critiques on Those Approaches**
1. Jasper Dekoninck, Mark Niklas Müller, Maximilian Baader, Marc Fischer, and Martin Vechev. 2024. [Evading data contamination detection for language models is (too) easy.](https://arxiv.org/pdf/2402.02823)
2. Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, and Hannaneh Hajishirzi. 2024. [Do membership inference attacks work on large language models?](https://arxiv.org/pdf/2402.07841) In Conference on Language Modeling (COLM).
3.  Pratyush Maini, Hengrui Jia, Nicolas Papernot, and Adam Dziedzic. 2024b. [Llm dataset inference: Did you train on my dataset?](https://arxiv.org/pdf/2406.06443) The 1st Workshop on Data Contamination (CONDA).
4.  Jialun Cao, Wuqi Zhang, and Shing-Chi Cheung. 2024. [Concerned with data contamination? assessing countermeasures in code language model.](https://arxiv.org/pdf/2403.16898) arXiv preprint arXiv:2403.16898.
5.  Matthieu Meeus, Igor Shilov, Shubham Jain, Manuel Faysse, Marek Rei, and Yves-Alexandre de Montjoye. 2024c. [Sok: Membership inference attacks on llms are rushing nowhere (and how to fix it).](https://arxiv.org/abs/2406.17975) arXiv preprint arXiv:2406.17975.

### Reference Probability by An Instance
1. Justus Mattern, Fatemehsadat Mireshghallah, Zhijing Jin, Bernhard Schoelkopf, Mrinmaya Sachan, and Taylor Berg-Kirkpatrick. 2023. [Membership inference attacks against language models via neighborhood comparison.](https://arxiv.org/pdf/2305.18462) In Findings of the Association for Computational Linguistics: ACL 2023, pages 11330– 11343.
2. Pratyush Maini, Hengrui Jia, Nicolas Papernot, and Adam Dziedzic. 2024b. [Llm dataset inference: Did you train on my dataset?](https://arxiv.org/pdf/2406.06443) The 1st Workshop on Data Contamination (CONDA).
3. Yonatan Oren, Nicole Meister, Niladri S. Chatterji, Faisal Ladhak, and Tatsunori Hashimoto. 2024. [Proving test set contamination in black-box language models.](https://arxiv.org/pdf/2310.17623) In The Twelfth International Conference on Learning Representations.

**Critiques on Those Approaches**
1. Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, and Hannaneh Hajishirzi. 2024. [Do membership inference attacks work on large language models?](https://arxiv.org/pdf/2402.07841) In Conference on Language Modeling (COLM).
2. Pratyush Maini, Hengrui Jia, Nicolas Papernot, and Adam Dziedzic. 2024b. [Llm dataset inference: Did you train on my dataset?](https://arxiv.org/pdf/2406.06443) The 1st Workshop on Data Contamination (CONDA).
3. Matthieu Meeus, Igor Shilov, Shubham Jain, Manuel Faysse, Marek Rei, and Yves-Alexandre de Montjoye. 2024c. [Sok: Membership inference attacks on llms are rushing nowhere (and how to fix it).](https://arxiv.org/abs/2406.17975) arXiv preprint arXiv:2406.17975.

### Reference Probability by Another LM
1. Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. 2021. [Extracting training data from large language models.](https://arxiv.org/pdf/2012.07805) In 30th USENIX Security Symposium (USENIX Security 21), pages 2633–2650.
2. Pratyush Maini, Hengrui Jia, Nicolas Papernot, and Adam Dziedzic. 2024b. [Llm dataset inference: Did you train on my dataset?](https://arxiv.org/pdf/2406.06443) The 1st Workshop on Data Contamination (CONDA).
3. Fatemehsadat Mireshghallah, Kartik Goyal, Archit Uniyal, Taylor Berg-Kirkpatrick, and Reza Shokri. 2022. [Quantifying privacy risks of masked language models using membership inference attacks.](https://aclanthology.org/2022.emnlp-main.570.pdf) In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 8332– 8347, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.
4. Santiago Zanella-Béguelin, Lukas Wutschitz, Shruti Tople, Victor R"uhle, Andrew Paverd, Olga Ohri- ¨ menko, Boris K"opf, and Marc Brockschmidt. 2020. ¨ [Analyzing information leakage of updates to natural language models.](https://dl.acm.org/doi/pdf/10.1145/3372297.3417880) In Proceedings of the 2020 ACM SIGSAC conference on computer and communications security, pages 363–375.
5. Matthieu Meeus, Shubham Jain, Marek Rei, and YvesAlexandre de Montjoye. 2024a. [Did the neurons read your book? document-level membership inference for large language models.](https://www.usenix.org/system/files/usenixsecurity24-meeus.pdf) In 33rd USENIX Security Symposium (USENIX Security 24), pages 2369–2385.

**Critiques on Those Approaches**
1. Jasper Dekoninck, Mark Niklas Müller, Maximilian Baader, Marc Fischer, and Martin Vechev. 2024. [Evading data contamination detection for language models is (too) easy.](https://arxiv.org/pdf/2402.02823)
2. Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, and Hannaneh Hajishirzi. 2024. [Do membership inference attacks work on large language models?](https://arxiv.org/pdf/2402.07841) In Conference on Language Modeling (COLM).
3.  Pratyush Maini, Hengrui Jia, Nicolas Papernot, and Adam Dziedzic. 2024b. [Llm dataset inference: Did you train on my dataset?](https://arxiv.org/pdf/2406.06443) The 1st Workshop on Data Contamination (CONDA).
4.  Jialun Cao, Wuqi Zhang, and Shing-Chi Cheung. 2024. [Concerned with data contamination? assessing countermeasures in code language model.](https://arxiv.org/pdf/2403.16898) arXiv preprint arXiv:2403.16898.
5.  Matthieu Meeus, Igor Shilov, Shubham Jain, Manuel Faysse, Marek Rei, and Yves-Alexandre de Montjoye. 2024c. [Sok: Membership inference attacks on llms are rushing nowhere (and how to fix it).](https://arxiv.org/abs/2406.17975) arXiv preprint arXiv:2406.17975.

### Others
1. bhyuday Jagannatha, Bhanu Pratap Singh Rawat, and Hong Yu. 2021. [Membership inference attack susceptibility of clinical language models.](https://arxiv.org/pdf/2104.08305) arXiv preprint arXiv:2104.08305.
2. Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramèr, and Nicholas Carlini. 2023. [Counterfactual memorization in neural language models.](https://proceedings.neurips.cc/paper_files/paper/2023/file/7bc4f74e35bcfe8cfe43b0a860786d6a-Paper-Conference.pdf) Advances in Neural Information Processing Systems, 36:39321–39362.

## Instance Generation and Instance Selection
### Verbatim Memorization
1. Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, and Chiyuan Zhang. 2022. [Quantifying memorization across neural language models.](https://arxiv.org/pdf/2202.07646#page=2.88) In The Eleventh International Conference on Learning Representations.
2. Nikhil Kandpal, Eric Wallace, and Colin Raffel. 2022. [Deduplicating training data mitigates privacy risks in language models.](https://proceedings.mlr.press/v162/kandpal22a/kandpal22a.pdf) In International Conference on Machine Learning, pages 10697–10707. PMLR.
3. Inbal Magar and Roy Schwartz. 2022. [Data contamination: From memorization to exploitation.](https://aclanthology.org/2022.acl-short.18.pdf) In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 157–165.
4. André Vicente Duarte, Xuandong Zhao, Arlindo L. Oliveira, and Lei Li. 2024. [DE-COP: Detecting copyrighted content in language models training data.](https://arxiv.org/pdf/2402.09910) In Forty-first International Conference on Machine Learning.
5. Kushal Tirumala, Aram Markosyan, Luke Zettlemoyer, and Armen Aghajanyan. 2022. [Memorization without overfitting: Analyzing the training dynamics of large language models.](https://proceedings.neurips.cc/paper_files/paper/2022/file/fa0509f4dab6807e2cb465715bf2d249-Paper-Conference.pdf) Advances in Neural Information Processing Systems, 35:38274–38290.
6. Avi Schwarzschild, Zhili Feng, Pratyush Maini, Zachary C Lipton, and J Zico Kolter. 2024. [Rethinking llm memorization through the lens of adversarial compression.](https://arxiv.org/pdf/2404.15146) The 1st Workshop on Data Contamination (CONDA).
7. Shahriar Golchin and Mihai Surdeanu. 2023a. [Data contamination quiz: A tool to detect and estimate contamination in large language models.](https://arxiv.org/pdf/2311.06233) CoRR, abs/2311.06233. 

### Key Information Generation
1. Chunyuan Deng, Yilun Zhao, Xiangru Tang, Mark Gerstein, and Arman Cohan. 2024b. [Investigating data contamination in modern benchmarks for large language models.](https://arxiv.org/pdf/2311.09783) In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8698–8711.
2. Federico Ranaldi, Elena Sofia Ruzzetti, Dario Onorati, Leonardo Ranaldi, Cristina Giannone, Andrea Favalli, Raniero Romagnoli, and Fabio Massimo Zanzotto. 2024. [Investigating the impact of data contamination of large language models in text-to-sql translation.](https://arxiv.org/pdf/2402.08100) arXiv preprint arXiv:2402.08100.
3. Kent Chang, Mackenzie Cramer, Sandeep Soni, and David Bamman. 2023. [Speak, memory: An archaeology of books known to chatgpt/gpt-4.](https://arxiv.org/pdf/2305.00118) In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7312–7327.
4. Xudong Pan, Mi Zhang, Shouling Ji, and Min Yang. 2020. [Privacy risks of general-purpose language models.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9152761) In 2020 IEEE Symposium on Security and Privacy (SP), pages 1314–1331.
5. Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. 2021. [Extracting training data from large language models.](https://arxiv.org/pdf/2012.07805) In 30th USENIX Security Symposium (USENIX Security 21), pages 2633–2650.
6. Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos, and Dawn Song. 2019. [The secret sharer: Evaluating and testing unintended memorization in neural networks.](https://www.usenix.org/system/files/sec19-carlini.pdf) In 28th USENIX Security Symposium (USENIX Security 19), pages 267–284, Santa Clara, CA. USENIX Association.
7. Chuang Liu, Renren Jin, Mark Steedman, and Deyi Xiong. 2024. [Evaluating Chinese large language models on discipline knowledge acquisition via memorization and robustness assessment.](https://aclanthology.org/2024.conda-1.1/) In Proceedings of the 1st Workshop on Data Contamination (CONDA), pages 1–12, Bangkok, Thailand. Association for Computational Linguistics.
8. Shahriar Golchin and Mihai Surdeanu. 2023a. [Data contamination quiz: A tool to detect and estimate contamination in large language models.](https://arxiv.org/pdf/2311.06233) CoRR, abs/2311.06233. 
### Generation Variation
1. Yihong Dong, Xue Jiang, Huanyu Liu, Zhi Jin, and Ge Li. 2024. [Generalization or memorization: Data contamination and trustworthy evaluation for large language models.](https://arxiv.org/pdf/2402.15938) arXiv preprint arXiv:2402.15938.
### Metadata-based Memorization
1. Oscar Sainz, Jon Ander Campos, Iker García-Ferrero, Julen Etxaniz, and Eneko Agirre. 2023b. [Did chatgpt cheat on your test?](https://hitz-zentroa.github.io/lm-contamination/blog/) Accessed: 2024-09-09.
2. Antonia Karamolegkou, Jiaang Li, Li Zhou, and Anders Søgaard. 2023. [Copyright violations and large language models.](https://arxiv.org/pdf/2310.13771) In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7403–7412, Singapore. Association for Computational Linguistics.

**Critiques on Those Approaches**
1. Jasper Dekoninck, Mark Niklas Müller, Maximilian Baader, Marc Fischer, and Martin Vechev. 2024. [Evading data contamination detection for language models is (too) easy.](https://arxiv.org/pdf/2402.02823)

## Answer Memorization
1. Chuang Liu, Renren Jin, Mark Steedman, and Deyi Xiong. 2024. [Evaluating Chinese large language models on discipline knowledge acquisition via memorization and robustness assessment.](https://aclanthology.org/2024.conda-1.1/) In Proceedings of the 1st Workshop on Data Contamination (CONDA), pages 1–12, Bangkok, Thailand. Association for Computational Linguistics.
2. Behzad Mehrbakhsh, Dario Garigliotti, Fernando Martínez-Plumed, and Jose Hernandez-Orallo. 2024. [Confounders in instance variation for the analysis of data contamination.](https://aclanthology.org/2024.conda-1.2.pdf) In Proceedings of the 1st Workshop on Data Contamination (CONDA), pages 13–21, Bangkok, Thailand. Association for Computational Linguistics.
3. Wen-wai Yim, Yujuan Fu, Asma Ben Abacha, and Meliha Yetisgen. 2024. [To err is human, how about medical large language models? comparing pretrained language models for medical assessment errors and reliability.](https://aclanthology.org/2024.lrec-main.1409/) In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LRECCOLING 2024), pages 16211–16223, Torino, Italia. ELRA and ICCL.
4. Yongshuo Zong, Tingyang Yu, Bingchen Zhao, Ruchika Chavhan, and Timothy Hospedales. 2023. [Fool your (vision and) language model with embarrassingly simple permutations.](https://arxiv.org/abs/2310.01651) arXiv preprint arXiv:2310.01651.
5. Yasaman Razeghi, Robert L Logan IV, Matt Gardner, and Sameer Singh. 2022. [Impact of pretraining term frequencies on few-shot numerical reasoning.](https://arxiv.org/abs/2202.07206) In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 840–854.

# Case Study
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

# Citation
```console
@inproceedings{fu2025data_contamination,
  author    = {Yujuan Velvin Fu and {\"O}zlem Uzuner and Meliha Yetişgen and Fei Xia},
  title     = {Does Data Contamination Detection Work (Well) for LLMs? A Survey and Evaluation on Detection Assumptions},
  booktitle = {Findings of the North American Chapter of the Association for Computational Linguistics (NAACL 2025)},
  year      = {2025},
   url={https://arxiv.org/abs/2410.18966}, 
}
```


