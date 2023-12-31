#https://github.com/callummcdougall/ARENA_2.0/tree/main/chapter1_transformers/exercises/part3_indirect_object_identification

from dataset import Dataset
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
import einops
from functools import partial
import torch as t
from torch import Tensor
from typing import Dict, Tuple, List
from jaxtyping import Float, Bool

lst = [(layer, head) for layer in range(12) for head in range(12)]
CIRCUIT = {
    "number mover": lst,
    # "number mover 4": lst,
    "number mover 3": lst,
    "number mover 2": lst,
    "number mover 1": lst,
}

SEQ_POS_TO_KEEP = {
    "number mover": "end",
    # "number mover 4": "S4",
    "number mover 3": "S3",
    "number mover 2": "S2",
    "number mover 1": "S1",
}

# # Simple test: do the indices for head 9.9 (which is a name mover head) match the positions of the "end" tokens?

# heads_circuit = get_heads_circuit(ioi_dataset)
# assert (heads_circuit[(9, 9)] == ioi_dataset.word_idx["end"]).all()


def get_heads_and_posns_to_keep(
    means_dataset: Dataset,
    model: HookedTransformer,
    circuit: Dict[str, List[Tuple[int, int]]],
    seq_pos_to_keep: Dict[str, str],
) -> Dict[int, Bool[Tensor, "batch seq head"]]:
    '''
    Returns a dictionary mapping layers to a boolean mask giving the indices of the 
    z output which *shouldn't* be mean-ablated.

    The output of this function will be used for the hook function that does ablation.
    '''
    heads_and_posns_to_keep = {}
    batch, seq, n_heads = len(means_dataset), means_dataset.max_len, model.cfg.n_heads

    for layer in range(model.cfg.n_layers):

        mask = t.zeros(size=(batch, seq, n_heads))

        for (head_type, head_list) in circuit.items():
            seq_pos = seq_pos_to_keep[head_type]
            indices = means_dataset.word_idx[seq_pos] # modify this for key vs query pos. curr, this is query
            for (layer_idx, head_idx) in head_list:
                if layer_idx == layer:
                    mask[:, indices, head_idx] = 1

        heads_and_posns_to_keep[layer] = mask.bool()

    return heads_and_posns_to_keep


# # Simple test: check the correct mask is produced when the heads circuit is "just keep layer 0, head 0, sequence position 0"

# heads_circuit_test = {(0, 0): t.full(size=(len(ioi_dataset),), fill_value=0)}
# heads_and_posns_to_keep = get_heads_and_posns_to_keep(heads_circuit_test, ioi_dataset, model)

# # Check all masks for layers after the first one are zero
# for layer in range(1, model.cfg.n_layers):
#     assert (heads_and_posns_to_keep[layer] == False).all()

# # Check mask for first layer is one at sequence position 0 for head 0, and zero everywhere else
# layer0_mask = heads_and_posns_to_keep[0]
# assert layer0_mask.shape == (len(ioi_dataset), ioi_dataset.max_len, model.cfg.n_heads)
# assert (layer0_mask[:, 1:, :] == False).all()
# assert (layer0_mask[:, 0, 1:] == False).all()
# assert (layer0_mask[:, 0, 0] == True).all()



def hook_fn_mask_z(
    z: Float[Tensor, "batch seq head d_head"],
    hook: HookPoint,
    heads_and_posns_to_keep: Dict[int, Bool[Tensor, "batch seq head"]],
    means: Float[Tensor, "layer batch seq head d_head"],
) -> Float[Tensor, "batch seq head d_head"]:
    '''
    Hook function which masks the z output of a transformer head.

    heads_and_posns_to_keep
        Dict created with the get_heads_and_posns_to_keep function. This tells
        us where to mask.

    means
        Tensor of mean z values of the means_dataset over each group of prompts
        with the same template. This tells us what values to mask with.
    '''
    # Get the mask for this layer, and add d_head=1 dimension so it broadcasts correctly
    mask_for_this_layer = heads_and_posns_to_keep[hook.layer()].unsqueeze(-1).to(z.device)

    # Set z values to the mean 
    z = t.where(mask_for_this_layer, z, means[hook.layer()])

    return z



def compute_means_by_template(
    means_dataset: Dataset, 
    model: HookedTransformer
) -> Float[Tensor, "layer batch seq head_idx d_head"]:
    '''
    Returns the mean of each head's output over the means dataset. This mean is
    computed separately for each group of prompts with the same template (these
    are given by means_dataset.groups).
    '''
    # Cache the outputs of every head
    _, means_cache = model.run_with_cache(
        means_dataset.toks.long(),
        return_type=None,
        names_filter=lambda name: name.endswith("z"),
    )
    # Create tensor to store means
    n_layers, n_heads, d_head = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head
    batch, seq_len = len(means_dataset), means_dataset.max_len
    means = t.zeros(size=(n_layers, batch, seq_len, n_heads, d_head), device=model.cfg.device)

    # Get set of different templates for this data
    for layer in range(model.cfg.n_layers):
        z_for_this_layer: Float[Tensor, "batch seq head d_head"] = means_cache[utils.get_act_name("z", layer)]
        for template_group in means_dataset.groups:
            z_for_this_template = z_for_this_layer[template_group]
            z_means_for_this_template = einops.reduce(z_for_this_template, "batch seq head d_head -> seq head d_head", "mean")
            means[layer, template_group] = z_means_for_this_template

    return means



def add_mean_ablation_hook(
    model: HookedTransformer, 
    means_dataset: Dataset, 
    circuit: Dict[str, List[Tuple[int, int]]] = CIRCUIT,
    seq_pos_to_keep: Dict[str, str] = SEQ_POS_TO_KEEP,
    is_permanent: bool = True,
) -> HookedTransformer:
    '''
    Adds a permanent hook to the model, which ablates according to the circuit and 
    seq_pos_to_keep dictionaries.

    In other words, when the model is run on ioi_dataset, every head's output will 
    be replaced with the mean over means_dataset for sequences with the same template,
    except for a subset of heads and sequence positions as specified by the circuit
    and seq_pos_to_keep dicts.
    '''
    
    model.reset_hooks(including_permanent=True)

    # Compute the mean of each head's output on the ABC dataset, grouped by template
    means = compute_means_by_template(means_dataset, model)
    
    # Convert this into a boolean map
    heads_and_posns_to_keep = get_heads_and_posns_to_keep(means_dataset, model, circuit, seq_pos_to_keep)

    # Get a hook function which will patch in the mean z values for each head, at 
    # all positions which aren't important for the circuit
    hook_fn = partial(
        hook_fn_mask_z, 
        heads_and_posns_to_keep=heads_and_posns_to_keep, 
        means=means
    )

    # Apply hook
    model.add_hook(lambda name: name.endswith("z"), hook_fn, is_permanent=is_permanent)

    return model

