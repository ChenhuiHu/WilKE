import json
import copy
import typing
import shutil
import random
from itertools import islice
from time import time
from pathlib import Path
from typing import Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import (
    AttributeSnippets,
    get_tfidf_vectorizer,
    CounterFactKnownDataset
)

from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact

from baselines.efk import EFKHyperParams, apply_ke_to_model
from baselines.kn import KNHyperParams, apply_kn_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from rome import ROMEHyperParams, apply_rome_to_model
from memit import MEMITHyperParams, apply_memit_to_model
from wilke import WilKEHyperParams, apply_wilke_to_model

from util import nethook
from util.globals import *
from util.hparams import HyperParams

# lawke/lawke_hparams.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def swap_targets(requested_rewrite):        
    requested_rewrite_copy = copy.deepcopy(requested_rewrite)
    requested_rewrite_copy["target_new"], requested_rewrite_copy["target_true"] = requested_rewrite_copy["target_true"], requested_rewrite_copy["target_new"]
    return requested_rewrite_copy


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


def eval_retention_counterfact(
    model,
    tok,
    prefixes,
    suffixes
):
    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix, suffix in zip(prefixes, suffixes)
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    
    a_tok = [tok(f" {n}")["input_ids"] for n in suffixes]
    a_len = [len(n) for n in a_tok]
    
    targets_correct = []
    
    # Batched inference
    batch_size = 32
    
    with torch.no_grad():
        # Iterate through all batches
        for batch_idx, i in enumerate(range(0, len(prompt_tok["input_ids"]), batch_size)):
            start_idx = i
            end_idx = min(i + batch_size, len(prompt_tok["input_ids"]))
            logits = model(input_ids=prompt_tok["input_ids"][start_idx:end_idx], attention_mask=prompt_tok["attention_mask"][start_idx:end_idx]).logits
            
            # Iterate over the current batch
            for j in range(logits.size(0)):
                cur_len = a_len[batch_idx*batch_size+j]
                correct = True
                
                # Iterate through each token of the correct answer
                for k in range(cur_len):
                    cur_tok = a_tok[batch_idx*batch_size+j][k]
                    if logits[j, prefix_lens[batch_idx*batch_size+j] + k - 1, :].argmax().item() != cur_tok:
                        correct = False
                        break
                targets_correct.append(correct)
            torch.cuda.empty_cache()
                
    return targets_correct.count(True)


ALG_DICT = {
    "KE": (EFKHyperParams, apply_ke_to_model),
    "KN": (KNHyperParams, apply_kn_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "WilKE": (WilKEHyperParams, apply_wilke_to_model),
}

DS_DICT = {
    "counterfact": (CounterFactKnownDataset, compute_rewrite_quality_counterfact),
}
    
def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    edit_times: int,
    skip_generation_tests: bool,
    conserve_memory: bool,
    dir_name: str,
):
    
    retention_ckpt = sorted([2**i for i in range(int(edit_times.bit_length()))] + list(range(50, 1001, 50)))
    
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory    
    alg_dir = RESULTS_DIR / dir_name
    if alg_dir.exists():
        id_list = [
            int(str(x).split("_")[-1])
            for x in alg_dir.iterdir()
            if str(x).split("_")[-1].isnumeric()
        ]
        run_id = 0 if not id_list else max(id_list) + 1
    else:
        run_id = 0
    run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = HPARAMS_DIR / alg_name / hparams_fname
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")
    
    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path
        
    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None
    
    ds_class, ds_eval_method = DS_DICT[ds_name]
    if model_name == "EleutherAI/gpt-j-6B":
        print(f"Loading data from: data/{ds_name}_EleutherAI_gpt-j-6B_known.json")
        ds = CounterFactKnownDataset(f"data/{ds_name}_EleutherAI_gpt-j-6B_known.json", size=2048)
    else:
        print(f"Loading data from: data/{ds_name}_{model_name}_known.json")
        ds = CounterFactKnownDataset(f"data/{ds_name}_{model_name}_known.json", size=2048)
        
    seed_value = 9 
    random.seed(seed_value)
    random.shuffle(ds.data)
        
    retention_prompts = [record["requested_rewrite"]["prompt"].format(record["requested_rewrite"]["subject"]) for record in ds]
    retention_answer = [record["requested_rewrite"]["target_true"]["str"] for record in ds]
    
    edit_retention_prompts = retention_prompts[:1024]
    edit_retention_answer = retention_answer[:1024]
    orig_retention_prompts = retention_prompts[1024:]
    orig_retention_answer = retention_answer[1024:]
    
    print(f"EDIT FIRST: {edit_retention_prompts[0], edit_retention_answer[0]}")
    print(f"ORIG FIRST: {orig_retention_prompts[0], orig_retention_answer[0]}")
    
    
    cur_edit = 0
    # Iterate through dataset
    for record_chunks in chunks(ds, 1):
        
        if cur_edit > edit_times:
            break
        cur_edit += 1
        
        for record in record_chunks:
            print(f"Current process edit {cur_edit} on case " + str(record["case_id"]))
        
        case_result_template = str(run_dir / "edit_{}_on_case_{}.json")
        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=None) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()
        start = time()
        edited_model, weights_copy, *remaining = apply_algo(
            model,
            tok,
            [
                {"case_id": record["case_id"], **record["requested_rewrite"]}
                for record in record_chunks
            ],
            hparams,
            copy=False,
            return_orig_weights=True,
            **args_conserve_memory,
            **etc_args,
        )
        exec_time = time() - start
        print("Execution took", exec_time)
        # Evaluate new model
        start = time()
        gen_test_vars = [snips, vec]
        for record in record_chunks:
            out_file = Path(case_result_template.format(cur_edit, record["case_id"]))
            if out_file.exists():
                print(f"Skipping {out_file}; already exists")
                continue
            metrics = {
                "case_id": record["case_id"],
                #"grouped_case_ids": case_ids,
                #"num_edits": num_edits,
                "edit_times": cur_edit,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": ds_eval_method(
                    edited_model,
                    tok,
                    record,
                    *(gen_test_vars),
                ),
            }
            # Dump metrics in .json
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)
        
        print("Evaluation took", time() - start)
        
        # Roll back edits
        kwargs = {"edited_layer": remaining[0]} if alg_name == "WilKE" else {}
        model, weights_copy = apply_algo(
            edited_model,
            tok,
            [
                {"case_id": record["case_id"], **swap_targets(record["requested_rewrite"])}
                for record in record_chunks
            ],
            hparams,
            copy=False,
            return_orig_weights=True,
            **args_conserve_memory,
            **etc_args,
            **kwargs
        )
        
        ## Restore original weights
        #with torch.no_grad():
        #    for k, v in weights_copy.items():
        #        nethook.get_parameter(model, k)[...] = v.to("cuda")

        if cur_edit in retention_ckpt:
            edit_retention = eval_retention_counterfact(model, tok, edit_retention_prompts[:cur_edit], edit_retention_answer[:cur_edit])
            orig_retention = eval_retention_counterfact(model, tok, orig_retention_prompts, orig_retention_answer)
            with open(run_dir / f"retention_of_edit_{cur_edit}.json", "w") as f:
                json.dump({"edit_retention": edit_retention, "edit_length": len(edit_retention_prompts[:cur_edit]),
                           "orig_retention": orig_retention, "orig_length": len(orig_retention_prompts)}, f, indent=1)
            print(f"On edit {cur_edit} retention rate: {edit_retention}/{len(edit_retention_prompts[:cur_edit])}")
            print(f"On edit {cur_edit} retention rate: {orig_retention}/{len(orig_retention_prompts)}")
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["KE","KN", "MEND", "ROME", "MEMIT", "WilKE"],
        default="WilKE",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["counterfact"],
        default="counterfact",
        help="Dataset to perform evaluations on.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=10000,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--edit_times",
        type=int,
        default=1024,
        help="Number of edit to perform.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.set_defaults(skip_generation_tests=True, conserve_memory=False)
    args = parser.parse_args()
    
    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.edit_times,
        args.skip_generation_tests,
        args.conserve_memory,
        dir_name=args.alg_name,
    )