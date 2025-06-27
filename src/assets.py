from torch import nn
import torch
from typing import Dict
from cl_dataset import GaussianDistribution
import copy

task_config = {
    "task1590_diplomacy_text_generation": "configs/SuperNI/task1590_diplomacy_text_generation",
    "task181_outcome_extraction": "configs/SuperNI/task181_outcome_extraction",
    "task591_sciq_answer_generation": "configs/SuperNI/task591_sciq_answer_generation",
    "task1729_personachat_generate_next": "configs/SuperNI/task1729_personachat_generate_next",
    "task1572_samsum_summary": "configs/SuperNI/task1572_samsum_summary",
    "task1510_evalution_relation_extraction": "configs/SuperNI/task1510_evalution_relation_extraction",
    "task748_glucose_reverse_cause_event_detection": "configs/SuperNI/task748_glucose_reverse_cause_event_detection",
    "task002_quoref_answer_generation": "configs/SuperNI/task002_quoref_answer_generation",
    "task1687_sentiment140_classification": "configs/SuperNI/task1687_sentiment140_classification",
    "task511_reddit_tifu_long_text_summarization": "configs/SuperNI/task511_reddit_tifu_long_text_summarization",
    "task875_emotion_classification": "configs/SuperNI/task511_reddit_tifu_long_text_summarization",
    "task639_multi_woz_user_utterance_generation": "configs/SuperNI/task639_multi_woz_user_utterance_generation",
    "task1290_xsum_summarization": "configs/SuperNI/task1290_xsum_summarization",
    "task073_commonsenseqa_answer_generation": "configs/SuperNI/task073_commonsenseqa_answer_generation",
    "task363_sst2_polarity_classification": "configs/SuperNI/task363_sst2_polarity_classification",
    "dbpedia": "configs/Long_Sequence/dbpedia",
    "amazon": "configs/Long_Sequence/amazon",
    "agnews": "configs/Long_Sequence/agnews",
    "yahoo": "configs/Long_Sequence/yahoo",
    "yelp": "configs/Long_Sequence/yelp",
    "copa": "configs/Long_Sequence/copa",
    "mnli": "configs/Long_Sequence/mnli",
    "cb": "configs/Long_Sequence/cb",
    "imdb": "configs/Long_Sequence/imdb",
    "multirc": "configs/Long_Sequence/multirc",
    "sst2": "configs/Long_Sequence/sst2",
    "boolq": "configs/Long_Sequence/boolq",
    "rte": "configs/Long_Sequence/rte",
    "wic": "configs/Long_Sequence/wic",
    "qqp": "configs/Long_Sequence/qqp",
}

def lora_state_dict_A(model: nn.Module, bias: str = 'none', task_name=None) -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_A' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_A' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_A')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError

def lora_state_dict_B(model: nn.Module, bias: str = 'none', task_name=None) -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_B' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_B' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_B')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError



def lora_state_dict_distribution(model: nn.Module,task_id=None) -> Dict[str, GaussianDistribution
]:
    k={}
    for layer_ids,layer in enumerate(model.model.layers):
        q_lora_distrebution=copy.deepcopy(layer.self_attn.distribution_q)
        q_lora_distrebution.mean=q_lora_distrebution.mean.to('cpu')
        q_lora_distrebution.var=q_lora_distrebution.var.to('cpu')
        v_lora_distrebution=copy.deepcopy(layer.self_attn.distribution_v)
        v_lora_distrebution.mean=v_lora_distrebution.mean.to('cpu')
        v_lora_distrebution.var=v_lora_distrebution.var.to('cpu')
        k['layers.'+str(layer_ids)+'.task.'+str(task_id)+'.q']=q_lora_distrebution
        k['layers.'+str(layer_ids)+'.task.'+str(task_id)+'.v']=v_lora_distrebution
    return k


def lora_state_dict_distribution_T5(model: nn.Module,task_id=None) -> Dict[str, GaussianDistribution
]:
    k={}
    for layer_ids,layer in enumerate(model.encoder.block):
        q_lora_distrebution=copy.deepcopy(layer.layer[0].SelfAttention.distribution_q)
        q_lora_distrebution.mean=q_lora_distrebution.mean.to('cpu')
        q_lora_distrebution.var=q_lora_distrebution.var.to('cpu')
        v_lora_distrebution=copy.deepcopy(layer.layer[0].SelfAttention.distribution_v)
        v_lora_distrebution.mean=v_lora_distrebution.mean.to('cpu')
        v_lora_distrebution.var=v_lora_distrebution.var.to('cpu')
        
        k[f'encoder.block.{layer_ids}.layer.0.task.{task_id}.q']=q_lora_distrebution
        k[f'encoder.block.{layer_ids}.layer.0.task.{task_id}.v']=v_lora_distrebution
    return k



def lora_state_dict_distribution_T5_all(model: nn.Module,task_id=None) -> Dict[str, GaussianDistribution
]:
    k={}
    for layer_ids,layer in enumerate(model.encoder.block):
        q_lora_distrebution=copy.deepcopy(layer.layer[0].SelfAttention.distribution_q)
        q_lora_distrebution.mean=q_lora_distrebution.mean.to('cpu')
        q_lora_distrebution.var=q_lora_distrebution.var.to('cpu')
        v_lora_distrebution=copy.deepcopy(layer.layer[0].SelfAttention.distribution_v)
        v_lora_distrebution.mean=v_lora_distrebution.mean.to('cpu')
        v_lora_distrebution.var=v_lora_distrebution.var.to('cpu')
        
        k[f'encoder.block.{layer_ids}.layer.0.task.{task_id}.q']=q_lora_distrebution
        k[f'encoder.block.{layer_ids}.layer.0.task.{task_id}.v']=v_lora_distrebution
    
    for layer_ids,layer in enumerate(model.decoder.block):
        q_lora_distrebution_0=copy.deepcopy(layer.layer[0].SelfAttention.distribution_q)
        q_lora_distrebution_0.mean=q_lora_distrebution_0.mean.to('cpu')
        q_lora_distrebution_0.var=q_lora_distrebution_0.var.to('cpu')
        q_lora_distrebution_1=copy.deepcopy(layer.layer[1].EncDecAttention.distribution_q)
        q_lora_distrebution_1.mean=q_lora_distrebution_1.mean.to('cpu')
        q_lora_distrebution_1.var=q_lora_distrebution_1.var.to('cpu')


        v_lora_distrebution_0=copy.deepcopy(layer.layer[0].SelfAttention.distribution_v)
        v_lora_distrebution_0.mean=v_lora_distrebution_0.mean.to('cpu')
        v_lora_distrebution_0.var=v_lora_distrebution_0.var.to('cpu')
        v_lora_distrebution_1=copy.deepcopy(layer.layer[1].EncDecAttention.distribution_v)
        v_lora_distrebution_1.mean=v_lora_distrebution_1.mean.to('cpu')
        v_lora_distrebution_1.var=v_lora_distrebution_1.var.to('cpu')
        
        k[f'decoder.block.{layer_ids}.layer.0.task.{task_id}.q']=q_lora_distrebution_0
        k[f'decoder.block.{layer_ids}.layer.0.task.{task_id}.v']=v_lora_distrebution_0
        k[f'decoder.block.{layer_ids}.layer.1.task.{task_id}.q']=q_lora_distrebution_1
        k[f'decoder.block.{layer_ids}.layer.1.task.{task_id}.v']=v_lora_distrebution_1
    return k



def merge_distributions(dist1, dist2):
    if dist1.mean.size(0) != dist2.mean.size(0):
        raise ValueError("Distributions must have the same dimension to be merged.")

    total_n = dist1.n + dist2.n
    
    if total_n == 0:
        new_dist = GaussianDistribution(dim=dist1.mean.size(0))
    else:
        new_mean = (dist1.n * dist1.mean + dist2.n * dist2.mean) / total_n
        
        mean_diff1 = dist1.mean - new_mean
        mean_diff2 = dist2.mean - new_mean
        
        new_var = (
            (dist1.n * (dist1.var + mean_diff1**2) +
             dist2.n * (dist2.var + mean_diff2**2)) / total_n
        )
        
        new_dist = GaussianDistribution(dim=dist1.mean.size(0))
        new_dist.mean = new_mean
        new_dist.var = new_var
        new_dist.n = total_n

    return new_dist



def prompt_state_dict(model: nn.Module, bias: str = 'none', task_name=None) -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    return {k: my_state_dict[k].to('cpu') for k in my_state_dict if 'current_prompt' in k}