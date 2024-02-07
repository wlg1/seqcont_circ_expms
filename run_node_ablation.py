
from dataset import Dataset
from generate_data import *
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