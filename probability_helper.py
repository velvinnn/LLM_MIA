import gc
import re
import json
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,T5ForConditionalGeneration
import json
from tqdm import tqdm
import numpy as np 
import pandas as pd
from glob import glob
import seaborn as sns
import random
from scipy.stats import pearsonr
import argparse
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM, utils,GPTNeoXForCausalLM,GPTNeoXTokenizerFast
#from bertviz import model_view,head_view
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from numpy.linalg import norm
from hf_olmo import OLMoForCausalLM  # pip install ai2-olmo
import numpy as np
import pandas as pd
import os
import json
from glob import glob
from scipy.stats import ks_2samp
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import entropy
from sklearn import metrics
# pltting pairs  
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import seaborn as sns

plot_figure=False
ACCURACY='accuracy'
AUC='AUC'
PAIRS='pairs'
token_overhead=0.000001
ppl_keys=[f'ppl_{i}' for i in [50,100,200]]+['perplexity'] # 
entropy_ps=[5, 10, 15, 20, 25]
rank_thresholds=[1,3,5,10,15,20,25] 
token_thresholds=[5,15,25]
rows=ppl_keys  +  [f'Min {k}% token' for k in token_thresholds] + [f'Mem {k}' for k in rank_thresholds] + [f'Entropy {k}' for k in entropy_ps] + ['PPL_seen','PPL_unseen']

def load_model(base_model_id,device):
    if 't5' in base_model_id:
        model=T5ForConditionalGeneration.from_pretrained(base_model_id, output_attentions=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    elif 'pythia' in base_model_id:
        model=GPTNeoXForCausalLM.from_pretrained(base_model_id, output_attentions=True,attn_implementation="eager").to(device)
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(base_model_id)
    elif base_model_id=='allenai/OLMo-7B':
        model=OLMoForCausalLM.from_pretrained(base_model_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    else:
        model =AutoModelForCausalLM.from_pretrained(base_model_id, output_attentions=True,attn_implementation="eager").to(device)  # Configure model to return attention values
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    return model,tokenizer

def find_ranks(sentence,base_model_id,model,tokenizer,device,start_token=1):
    tokens = tokenizer(sentence, return_tensors='pt', return_offsets_mapping=True,truncation=True,max_length=2048).to(device)
    offset_mapping = tokens.pop('offset_mapping').squeeze()
    offset_mapping = [offset_mapping[i,1].item() for i in range(offset_mapping.shape[0])]

    all_tokens=[]
    for token_id in tokens['input_ids'].squeeze().tolist():
        token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        all_tokens.append(token)


    with torch.no_grad():
        outputs = model(tokens['input_ids'], labels=tokens['input_ids'])
    loss, logits = outputs[:2]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens['input_ids'][:, 1:].contiguous()

    # same as shi et al from min tok prob,https://github.com/swj0419/detect-pretrain-code/issues/5
    log_probs = F.log_softmax(shift_logits, dim=-1) # (batch_size, num_tokens,vocabulary_size)

    # Gather the log-probs corresponding to the correct tokens
    log_probs_for_correct_tokens = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    # Calculate per-token probabilities
    token_probs = torch.exp(log_probs_for_correct_tokens).tolist()[0]

    # Calculate sentence perplexity
    sentence_log_prob = log_probs_for_correct_tokens.mean()
    perplexity={'perplexity':torch.exp(-sentence_log_prob).item()}
    for length in [5,10,25,50,100,200,500,1000]:
        if log_probs_for_correct_tokens.shape[1]>=length:
            sentence_log_prob = log_probs_for_correct_tokens[0,:length].mean()
            perplexity[f'ppl_{length}']=torch.exp(-sentence_log_prob).item()
        else:
            break


    ranks=[]
    # Predict each token considering the previous ones
    for i in range(log_probs.shape[1]):
        rank= torch.sum(log_probs[0,i,:]>log_probs[0,i,tokens['input_ids'][0,i]]).item()
        ranks.append(rank)
          # Get the probability of the actual token

    for k in entropy_ps:
        perplexity[f'Entropy {k}']=[]
    for i in range(log_probs.shape[1]):
        topk=torch.topk(log_probs[0,i,:], entropy_ps[-1]).values#.tensor()
        for k in entropy_ps[::-1]:
            topk= torch.topk(topk, k).values
            perplexity[f'Entropy {k}'].append(-torch.sum(topk * torch.exp(topk)).item())
    for k in entropy_ps:
        perplexity[f'Entropy {k}']=np.average(perplexity[f'Entropy {k}'])     
    return offset_mapping,all_tokens,ranks,token_probs,perplexity

def calc_entropy(probs,p,normalized=True):
    # entropy across all text
    # pk = np.array(ls)  # fair coin
    # H = entropy(pk)
    
    # always normalize the probs
    probs = probs/ np.sum(probs)
    # Sort the probabilities in descending order
    sorted_probs = np.sort(probs)[::-1]
    
    if p<1: # top p sampling
        # Select the top probabilities whose sum reaches the desired p
        cumulative_probs = np.cumsum(sorted_probs)
        #print(cumulative_probs)
        top_p_idx = np.where(cumulative_probs >= p)[0][0]
        top_probs = sorted_probs[:top_p_idx + 1]
    else: # top k sampling
        top_probs = sorted_probs[:int(p)]

    if normalized:
        # Normalize the top_probs so they sum to p
        normalized_probs = top_probs / np.sum(top_probs)
        
        # Calculate the entropy
        entropy = -np.sum(normalized_probs * np.log(normalized_probs))
        
        return entropy
    else:
        return -np.sum(top_probs * np.log(top_probs))

def plot_density(data,xlabel,outname,lower_bound=None,upper_bound=None,description=''):
    f = plt.figure(figsize=(4, 2), dpi=200)
    temp=[]
    for key, ls in data.items():
        temp+=ls
    lower_bound=np.percentile(temp,10)
    # if 'Entropy' in outname:
    #     lower_bound=np.percentile(temp,40)
    upper_bound=np.percentile(temp,90)
    for key, ls in data.items():
        key='seen' if 'train' in key.lower() else 'unseen'
        if lower_bound is not None and upper_bound is not None:
            plt.hist(ls, bins=100, weights=np.ones(len(ls)) / len(ls), alpha=0.5, 
                    #label=f'{key}, mean {round(np.mean(ls),1)}, var {round(np.std(ls),1)}, support {len(ls)}',
                    label=f'{key}',
                    range=[lower_bound,upper_bound]
                    )
        else:
            plt.hist(ls, bins=100, weights=np.ones(len(ls)) / len(ls), alpha=0.5, 
                    label=f'{key}',
                    )         
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    #plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.legend(loc='upper left')
    plt.ylabel('percentage')
    plt.xlabel(xlabel.replace('ppl','PPL').replace('Mem 2','Mem 1'), labelpad=20)
    #plt.xticks(rotation=45)
    # if 'Entropy' in outname and '0.' not in outname:
    #     plt.xscale('log')
    #plt.title(f'{domain}, {model_name},\n{description}')
    outdir='/'.join(outname.split('/')[:-1])
    if '.png' not in outname:
        outname+='.png'
    os.makedirs(outdir,exist_ok=True)
    plt.tight_layout()
    plt.savefig(outname)
    f.clear()
    plt.close(f)
    return

def compare_distribution(list1,list2,side='less'):
    statistic, p_value = ks_2samp(list1,list2, alternative=side) #{‘two-sided’, ‘less’, ‘greater’}, optional
    if p_value < 0.05:
        return round(p_value,3)
    else:
        return round(p_value,3)

# def calculate_accuracy(l1,l2,direction='smaller'):
#     l1=[v for v in l1 if not np.isnan(v) and v!=float("inf") and v!=float("-inf")]
#     l2=[v for v in l2 if not np.isnan(v) and v!=float("inf") and v!=float("-inf")]
#     if len(l1)==0 or len(l2)==0:
#         return -1,'NA'
#     if direction=='smaller':
#         y=np.array([1]*len(l1)+[2]*len(l2))
#     else:
#         y=np.array([2]*len(l1)+[1]*len(l2))
#     pred = np.array(l1+l2)
#     fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    
#     return round(metrics.auc(fpr, tpr)*100,1),'NA'
def calculate_accuracy(l1,l2,direction='smaller'):
    l1=[v for v in l1 if not np.isnan(v) and v!=float("inf") and v!=float("-inf")]
    l2=[v for v in l2 if not np.isnan(v) and v!=float("inf") and v!=float("-inf")]
    if len(l1)==0 or len(l2)==0:
        return -1,'NA'
    y=np.array([1]*len(l1)+[2]*len(l2))
    if direction!='smaller':
        l1=[-v for v in l1]
        l2=[-v for v in l2]
    pred = np.array(l1+l2)
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    
    return round(metrics.auc(fpr, tpr)*100,1),'NA'


def plot_and_calculate(train,test,train_domain,test_domain,model_tag,plot_figure=True): #'perplexity'
    if os.path.isfile(train) and os.path.isfile(test):
        scores={}
        train=json.loads(open(train).read())
        test=json.loads(open(test).read())
        row_idx=0
        # first calculate the perplexity
        for ppl_key in ppl_keys:
            p_train=[dic['perplexity'][ppl_key] for dic in train if ppl_key in dic['perplexity']]
            p_test=[dic['perplexity'][ppl_key] for dic in test if ppl_key in dic['perplexity']]
            if len(p_train)>10 and len(p_test)>10:
                accuracy, p_value=calculate_accuracy(p_train, p_test,direction='smaller')
                scores[ppl_key]=accuracy
                if plot_figure:
                    plot_density({'train':p_train,'test':p_test},ppl_key,
                                 f'data/analysis/plots/density/{train_domain}_{test_domain}_{ppl_key.lower()}_{model_tag}')
                    #print(f'data/analysis/plots/density/{train_domain}_{test_domain}_{ppl_key.lower()}_{model_tag}')
                if ppl_key=='ppl_200':
                    scores['PPL_seen']=[np.mean(p_train),np.std(p_train)]
                    scores['PPL_unseen']=[np.mean(p_test),np.std(p_test)]
        # token  probabilities
        for row_type,direction in [('token_probs','larger')]:#,('ranks','smaller')
            for k in token_thresholds:
                ls=[[],[]]
                for idx,source in enumerate([train,test]):
                    for dic in source:
                        percentile=np.percentile(dic[row_type], k) 
                        if row_type=='ranks':
                            ls[idx].append(np.average([v for v in dic[row_type] if v<=percentile and str(v) != 'nan'])) 
                        else:
                            ls[idx].append(np.average([v for v in dic[row_type] if v<=percentile and str(v) != 'nan']))    
                accuracy, p_value=calculate_accuracy(ls[0], ls[1], direction=direction)
                scores[f'Min {k}% token']=accuracy
                if plot_figure:
                    plot_density({'train':ls[0],'test':ls[1]},row_type,
                                 f'data/analysis/plots/density/{train_domain}_{test_domain}_Min_{k}_token_{model_tag}')
        
        # percentage of memorization
        for rank_threshold in rank_thresholds:
            ls=[[],[]]
            for idx,source in enumerate([train,test]):
                for dic in source:
                    ls[idx].append(len([v for v in dic['ranks'] if v<=rank_threshold])/len(dic['ranks'])) 
            accuracy, p_value=calculate_accuracy(ls[0], ls[1],  direction='larger')
            scores[f'Mem {rank_threshold}']=accuracy
            if plot_figure:
                plot_density({'train':ls[0],'test':ls[1]},f'Mem {rank_threshold}',
                                 f'data/analysis/plots/density/{train_domain}_{test_domain}_Mem_{k}_{model_tag}')

        
        direction='smaller'
        for p in entropy_ps:
            e_train=[dic['perplexity'][f'Entropy {p}'] for dic in train if ppl_key in dic['perplexity']]
            e_test=[dic['perplexity'][f'Entropy {p}'] for dic in test if ppl_key in dic['perplexity']]
            accuracy, p_value=calculate_accuracy(e_train, e_test, direction=direction)
            if plot_figure:
                plot_density({'train':e_train,'test':e_test},f'Entropy {p}',
                                 f'data/analysis/plots/density/{train_domain}_{test_domain}_Entropy_{p}_{model_tag}')
            scores[f'Entropy {p}']=accuracy        
    return scores

text_before="""
\\begin{table*}[h]
\\centering
\\resizebox{0.98\\textwidth}{!}{
\\begin{tabular}{llccccccccccc}
\\hline
\\multicolumn{2}{l}{\\textbf{Assumptions \\& Metric}} 
"""
#& \\textbf{Github} & \\textbf{\\begin{tabular}[c]{@{}c@{}}Free-\\\\ Law\\end{tabular}} & \\textbf{ArXiv} & \\textbf{\\begin{tabular}[c]{@{}c@{}}PubMed\\\\ Central\\end{tabular}} & \\textbf{\\begin{tabular}[c]{@{}c@{}}Enron\\\\ Emails\end{tabular}} & \\textbf{\\begin{tabular}[c]{@{}c@{}}PubMed\\\\ Abstracts\end{tabular}} & \\textbf{\\begin{tabular}[c]{@{}c@{}}Open-\\\\ Subtitles\end{tabular}} & \\textbf{\\begin{tabular}[c]{@{}c@{}}OpenWeb-\\\\ Text2\end{tabular}} & \\textbf{\\begin{tabular}[c]{@{}c@{}}Youtube-\\\\ Subtitles\end{tabular}} & \\textbf{\\begin{tabular}[c]{@{}c@{}}Hacker-\\\\ News\end{tabular}} & \\textbf{Pile-CC} \\\\\hline

def get_text_after(model_tag,dataset):
    text_after="""
\hline
\end{tabular}
}
\caption{Average contamination detection AUC for the \\textbf{{model_tag}} model,  under different domains within
the {dataset} dataset. `PPL\_200' represents the average perplexity $\pm$ STD, from the first 200 tokens within every instance.  The color {\color[HTML]{38761D} green} represents AUCs higher than 60.
}
\label{tab:results_{model_tag}}
\end{table*}
"""
    return text_after.replace("{model_tag}",f"{model_tag}").replace("{dataset}",f"{dataset}")


latex_lines={
    '\multirow{6}{*}{A\\ref{as:prob_absolute}}  & PPL\_50':'ppl_50',
    ' & PPL\_100':'ppl_100',
    ' & PPL\_200':'ppl_200',
    ' & Min 5\% token':'Min 5% token',
    ' & Min 15\% token':'Min 15% token',
    ' & Min 25\% token':'Min 25% token',
    ' \cline{1-2}\multirow{3}{*}{A\\ref{as:exact_memorization}} & Mem 5':'Mem 5',
    ' & Mem 15':'Mem 15',
    ' & Mem 25':'Mem 25',
    '\cline{1-2} \multirow{3}{*}{A\\ref{as:Generation_Variation}} & Entropy 5':'Entropy 5',
    ' & Entropy 15':'Entropy 15',
    ' & Entropy 25':'Entropy 25',
    '\hline\multirow{2}{*}{PPL_200} & Seen':'PPL_seen',
    ' & Unseen':'PPL_unseen'
}

def plot_heat_map(data,xs,ys,outname):
    fig, ax = plt.subplots(figsize=(6,5), dpi=900)
    sns.heatmap(data, annot=data, fmt="", cmap='RdYlGn', ax=ax, cbar=False,annot_kws={'color':'black'})
    heatmap = plt.pcolor(data, cmap='RdYlGn')
    plt.colorbar(heatmap, cmap='RdYlGn')
    # Show all ticks and label them with the respective list entries
    ax.set_xticks([v+0.5 for v in np.arange(len(xs))], labels=xs)
    ax.set_yticks([v+0.5 for v in np.arange(len(ys))], labels=ys)
    
    plt.xlabel('Domain of the unseen instances')
    plt.ylabel('Domain of the seen instances')
    #ax.set_title(title)
    fig.tight_layout()
    plt.savefig(outname)
    return

#