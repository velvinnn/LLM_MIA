# model_path
BioMistral=''


# the base directories for all files
base_path=''
base_path2=''
output_path='data'

# domain = [seen_file,unseen_file]
AS_tex=[f'{base_path}/algebraic-stack/tex_train.jsonl',f'{base_path}/algebraic-stack/tex_test.jsonl']
AS_python=[f'{base_path}/algebraic-stack/python_train.jsonl',f'{base_path}/algebraic-stack/python_test.jsonl']
AS_julia=[f'{base_path}/algebraic-stack/julia_train.jsonl',f'{base_path}/algebraic-stack/julia_test.jsonl']
AS_fortran=[f'{base_path}/algebraic-stack/fortran_train.jsonl',f'{base_path}/algebraic-stack/fortran_test.jsonl']
AS_cpp=[f'{base_path}/algebraic-stack/cpp_train.jsonl',f'{base_path}/algebraic-stack/cpp_test.jsonl']
AS_agda=[f'{base_path}/algebraic-stack/agda_train.jsonl',f'{base_path}/algebraic-stack/agda_test.jsonl']
AS_c=[f'{base_path}/algebraic-stack/c_train.jsonl',f'{base_path}/algebraic-stack/c_test.jsonl']
AS_gap=[f'{base_path}/algebraic-stack/gap_train.jsonl',f'{base_path}/algebraic-stack/gap_test.jsonl']
AS_haskell=[f'{base_path}/algebraic-stack/haskell_train.jsonl',f'{base_path}/algebraic-stack/haskell_test.jsonl']
AS_idris=[f'{base_path}/algebraic-stack/idris_train.jsonl',f'{base_path}/algebraic-stack/idris_test.jsonl']
AS_isa=[f'{base_path}/algebraic-stack/isa_proofsteps_train.jsonl',f'{base_path}/algebraic-stack/isa_proofsteps_test.jsonl']
AS_lean=[f'{base_path}/algebraic-stack/lean_proofsteps_train.jsonl',f'{base_path}/algebraic-stack/lean_proofsteps_test.jsonl']
AS_maple=[f'{base_path}/algebraic-stack/maple_train.jsonl',f'{base_path}/algebraic-stack/maple_test.jsonl']
AS_r=[f'{base_path}/algebraic-stack/r_train.jsonl',f'{base_path}/algebraic-stack/r_test.jsonl']
AS_github_coq=[f'{base_path}/algebraic-stack/Github-Coq_train.jsonl',f'{base_path}/algebraic-stack/Github-Coq_test.jsonl']
AS_github_isabelle=[f'{base_path}/algebraic-stack/Github-Isabelle_train.jsonl',f'{base_path}/algebraic-stack/Github-Isabelle_test.jsonl']
AS_github_lean=[f'{base_path}/algebraic-stack/Github-Lean_train.jsonl',f'{base_path}/algebraic-stack/Github-Lean_test.jsonl']
AS_github_MATLAB=[f'{base_path}/algebraic-stack/Github-MATLAB_train.jsonl',f'{base_path}/algebraic-stack/github-MATLAB_test.jsonl']
semantic_scholar=[f'{base_path}/peS2o/train.jsonl',f'{base_path}/peS2o/validation.jsonl']
UltraChat=[f'{base_path}/UltraChat/train.jsonl',f'{base_path}/UltraChat/test.jsonl']
RLVR_GSM=[f'{base_path}/UltraChat/train.jsonl',f'{base_path}/UltraChat/test.jsonl']

DC_2006smoker=[f'{base_path2}/DC/2006smoker/train.json',f'{base_path2}/DC/2006smoker/evaluation.json']
DC_2008obesity=[f'{base_path2}/DC/2008obesity/train.json',f'{base_path2}/DC/2008obesity/evaluation.json']
DC_2018cohort=[f'{base_path2}/DC/2018cohort/train.json',f'{base_path2}/DC/2018cohort/evaluation.json']
DC_2024SemEval=[f'{base_path2}/DC/2024SemEval/train.json',f'{base_path2}/DC/2024SemEval/evaluation.json']
DC_Reason2Stop=[f'{base_path2}/DC/Reason2Stop/train.json',f'{base_path2}/DC/Reason2Stop/evaluation.json']
DC_Reason2StopQA=[f'{base_path2}/DC/Reason2StopQA/train.json',f'{base_path2}/DC/Reason2StopQA/evaluation.json']
DC_MTSample=[f'{base_path2}/DC/MTSample/train.json',f'{base_path2}/DC/MTSample/evaluation.json']
DC_CZI_DRSM=[f'{base_path2}/DC/CZI_DRSM/train.json',f'{base_path2}/DC/CZI_DRSM/evaluation.json']
NLI_snli=[f'{base_path2}/NLI/snli/train.json',f'{base_path2}/NLI/snli/evaluation.json']
NLI_multinli=[f'{base_path2}/NLI/multinli/train.json',f'{base_path2}/NLI/multinli/evaluation.json']
NLI_BioNLI=[f'{base_path2}/NLI/BioNLI/train.json',f'{base_path2}/NLI/BioNLI/evaluation.json']
STS_STS_B=[f'{base_path2}/STS/STS-B/train.json',f'{base_path2}/STS/STS-B/evaluation.json']
summarization_CDSR=[f'{base_path2}/summarization/CDSR/train.json',f'{base_path2}/summarization/CDSR/evaluation.json']
summarization_PubMedSum=[f'{base_path2}/summarization/PubMedSum/train.json',f'{base_path2}/summarization/PubMedSum/evaluation.json']
summarization_AciDemo=[f'{base_path2}/summarization/AciDemo/train.json',f'{base_path2}/summarization/AciDemo/evaluation.json']
events_DrugProt=[f'{base_path2}/events/DrugProt/train.json',f'{base_path2}/events/DrugProt/evaluation.json']
events_BioRed=[f'{base_path2}/events/BioRed/train.json',f'{base_path2}/events/BioRed/evaluation.json']
events_tmVar=[f'{base_path2}/events/tmVar/train.json',f'{base_path2}/events/tmVar/evaluation.json']
events_NLM_Gene=[f'{base_path2}/events/NLM-Gene/train.json',f'{base_path2}/events/NLM-Gene/evaluation.json']
events_GNormPlus=[f'{base_path2}/events/GNormPlus/train.json',f'{base_path2}/events/GNormPlus/evaluation.json']
events_2006deid=[f'{base_path2}/events/2006deid/train.json',f'{base_path2}/events/2006deid/evaluation.json']
events_2009medication=[f'{base_path2}/events/2009medication/train.json',f'{base_path2}/events/2009medication/evaluation.json']
events_2011coreference=[f'{base_path2}/events/2011coreference/train.json',f'{base_path2}/events/2011coreference/evaluation.json']
events_2012temporal=[f'{base_path2}/events/2012temporal/train.json',f'{base_path2}/events/2012temporal/evaluation.json']
events_2014PHI=[f'{base_path2}/events/2014PHI/train.json',f'{base_path2}/events/2014PHI/evaluation.json']
events_2018ade=[f'{base_path2}/events/2018ade/train.json',f'{base_path2}/events/2018ade/evaluation.json']
events_2022sdoh=[f'{base_path2}/events/2022sdoh/train.json',f'{base_path2}/events/2022sdoh/evaluation.json']
events_GENIA=[f'{base_path2}/events/GENIA/train.json',f'{base_path2}/events/GENIA/evaluation.json']
events_linnaeus=[f'{base_path2}/events/linnaeus/train.json',f'{base_path2}/events/linnaeus/evaluation.json']
events_BC4CHEMD=[f'{base_path2}/events/BC4CHEMD/train.json',f'{base_path2}/events/BC4CHEMD/evaluation.json']
events_PICO_data=[f'{base_path2}/events/PICO-data/train.json',f'{base_path2}/events/PICO-data/evaluation.json']
events_PubMedPICO=[f'{base_path2}/events/PubMedPICO/train.json',f'{base_path2}/events/PubMedPICO/evaluation.json']
events_ClinicalIE_medication_attr=[f'{base_path2}/events/ClinicalIE_medication_attr/train.json',f'{base_path2}/events/ClinicalIE_medication_attr/evaluation.json']
events_ClinicalIE_medication_status=[f'{base_path2}/events/ClinicalIE_medication_status/train.json',f'{base_path2}/events/ClinicalIE_medication_status/evaluation.json']
events_BioASQ=[f'{base_path2}/events/BioASQ/train.json',f'{base_path2}/events/BioASQ/evaluation.json']
RE_DrugProt=[f'{base_path2}/RE/DrugProt/train.json',f'{base_path2}/RE/DrugProt/evaluation.json']
RE_BioRed=[f'{base_path2}/RE/BioRed/train.json',f'{base_path2}/RE/BioRed/evaluation.json']
RE_2011coreference=[f'{base_path2}/RE/2011coreference/train.json',f'{base_path2}/RE/2011coreference/evaluation.json']
RE_2012temporal=[f'{base_path2}/RE/2012temporal/train.json',f'{base_path2}/RE/2012temporal/evaluation.json']
RE_euadr=[f'{base_path2}/RE/euadr/train.json',f'{base_path2}/RE/euadr/evaluation.json']
chat_wiki_medical_terms=[f'{base_path2}/chat/wiki_medical_terms/train.json',f'{base_path2}/chat/wiki_medical_terms/evaluation.json']
chat_medical_chat=[f'{base_path2}/chat/medical_chat/train.json',f'{base_path2}/chat/medical_chat/evaluation.json']
chat_MedDialog=[f'{base_path2}/chat/MedDialog/train.json',f'{base_path2}/chat/MedDialog/evaluation.json']

Github=[f"{base_path}/pile/Github_train.jsonl",f"{base_path}/pile/Github_test.jsonl"]
FreeLaw=[f"{base_path}/pile/FreeLaw_train.jsonl",f"{base_path}/pile/FreeLaw_test.jsonl"]
Enron_Emails=[f"{base_path}/pile/Enron Emails_train.jsonl",f"{base_path}/pile/Enron Emails_test**.jsonl"]
ArXiv=[f"{base_path}/pile/ArXiv_train.jsonl",f"{base_path}/pile/ArXiv_test**.jsonl"]
OpenWeb_Text2=[f"{base_path}/pile/OpenWebText2_train.jsonl",f"{base_path}/pile/OpenWebText2_test.jsonl"]
Open_Subtitles=[f"{base_path}/pile/OpenSubtitles_train.jsonl",f"{base_path}/pile/OpenSubtitles_test.jsonl"]
Hacker_News=[f"{base_path}/pile/HackerNews_train.jsonl",f"{base_path}/pile/HackerNews_test**.jsonl"]
YoutubeSubtitles=[f"{base_path}/pile/YoutubeSubtitles_train.jsonl",f"{base_path}/pile/YoutubeSubtitles_test**.jsonl"]
Pile_CC=[f"{base_path}/pile/Pile-CC_train.jsonl",f"{base_path}/pile/Pile-CC_test.jsonl"]

