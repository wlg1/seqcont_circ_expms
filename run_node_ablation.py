
from dataset import Dataset
# from generate_data import *
from metrics import *
from head_ablation_fns import *
from mlp_ablation_fns import *
from node_ablation_fns import *
from loop_node_ablation_fns import *

import pickle
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gp")

    args = parser.parse_args()
    model_name = args.model

    ### Load Model ###
    model = HookedTransformer.from_pretrained(
        model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )

    ### Load Datasets ###
    task = "numerals"  # choose: numerals, numwords, months
    prompt_types = ['done', 'lost', 'names']
    num_samps_per_ptype = 512 #768 512

    save_files = True
    run_on_other_tasks = True
    prompts_list = []

    for i in prompt_types:
        file_name = f'/content/seqcont_circ_expms/data/{task}/{task}_prompts_{i}.pkl'
        with open(file_name, 'rb') as file:
            filelist = pickle.load(file)

        print(filelist[0]['text'])
        prompts_list += filelist [:num_samps_per_ptype]

    pos_dict = {}
    for i in range(len(model.tokenizer.tokenize(prompts_list[0]['text']))):
        pos_dict['S'+str(i)] = i

    dataset = Dataset(prompts_list, pos_dict, model.tokenizer)

    file_name = f'/content/seqcont_circ_expms/data/{task}/randDS_{task}.pkl'
    with open(file_name, 'rb') as file:
        prompts_list_2 = pickle.load(file)

    dataset_2 = Dataset(prompts_list_2, pos_dict, model.tokenizer)

    #### Get orig score ####
    model.reset_hooks(including_permanent=True)
    ioi_logits_original = model(dataset.toks)
    orig_score = get_logit_diff(ioi_logits_original, dataset)
    print(orig_score)

    ##############
    ### Node Ablation Iteration ###

    # threshold = 20
    # curr_circ_heads = []
    # curr_circ_mlps = []
    # prev_score = 100
    # new_score = 0
    # iter = 1
    # all_comp_scores = []
    # while prev_score != new_score:
    #     print('\nbackw prune, iter ', str(iter))
    #     old_circ_heads = curr_circ_heads.copy() # save old before finding new one
    #     old_circ_mlps = curr_circ_mlps.copy()
    #     curr_circ_heads, curr_circ_mlps, new_score, comp_scores = find_circuit_backw(model, dataset, dataset_2, curr_circ_heads, curr_circ_mlps, orig_score, threshold)
    #     if old_circ_heads == curr_circ_heads and old_circ_mlps == curr_circ_mlps:
    #         break
    #     all_comp_scores.append(comp_scores)
    #     print('\nfwd prune, iter ', str(iter))
    #     # track changes in circuit as for some reason it doesn't work with scores
    #     old_circ_heads = curr_circ_heads.copy()
    #     old_circ_mlps = curr_circ_mlps.copy()
    #     curr_circ_heads, curr_circ_mlps, new_score, comp_scores = find_circuit_forw(model, dataset, dataset_2, curr_circ_heads, curr_circ_mlps, orig_score, threshold)
    #     if old_circ_heads == curr_circ_heads and old_circ_mlps == curr_circ_mlps:
    #         break
    #     all_comp_scores.append(comp_scores)
    #     iter += 1