import os
import sys
from copy import deepcopy
from typing import Dict, List

import hydra
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.globals import *

from .algs.efk import EFK
from .efk_hparams import EFKHyperParams

def apply_ke_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: EFKHyperParams,
    copy=False,
    return_orig_weights=False,
    return_orig_weights_device="cuda",
):
    
    if copy:
        model = deepcopy(model)
        
    train_ds = (
        "counterfact-" if hparams.counterfact else ("zsre-" if hparams.zsre else "")
    )
    modelcode = "gpt2xl" if hparams.model_name == "gpt2-xl" else "gpt-j-6b"
    model_filename = f"efk-{hparams.n_toks}tok-{train_ds}gpt2-xl.pt"
    model_dir = "baselines/efk/weights"
    
    with hydra.initialize(config_path="config", job_name="run"):
        config = hydra.compose(
            config_name="config",
            overrides=[
                "+alg=efk",
                "+experiment=gen",
                f"+model={modelcode}",
                f"data.path=data/{hparams.n_toks}token/data/self_sample/",
            ],
        )
        
    alg = EFK(model, config)
    d = torch.load(f"{model_dir}/{model_filename}")
    alg.load_state_dict(deepcopy(d["model"]))
    
    del d
    torch.cuda.empty_cache()
    
    alg.editor.cuda()

    for request in requests:
        request_rewrite = deepcopy(request)
        target = " " + request_rewrite["target_new"]["str"]
        sentence = (
            request_rewrite["prompt"].format(request_rewrite["subject"]) + target
        )
        target_tokens = tok(target)["input_ids"]
        tokens = torch.tensor(tok(sentence)["input_ids"])[None]
        label_tokens = tokens.clone()
        label_tokens[0][: -len(target_tokens)] = -100
        edit_inner = dict(
            input_ids=tokens.clone().cuda(),
            attention_mask=torch.ones_like(tokens).cuda(),
            labels=label_tokens.clone().cuda(),
        )
        cond = dict(
            input_ids=tokens.clone().cuda(),
            attention_mask=torch.ones_like(tokens).cuda(),
        )
        
        edited_model = alg.edit(edit_inner, cond)
        
    del alg
    torch.cuda.empty_cache()
        
    return edited_model, {}

#class EfkRewriteExecutor:
#    method_name = "KE"
#
#    def __init__(self) -> None:
#        self.is_init = False
#
#    def init_model(self, model, tok, params, is_first=True):        
#        train_ds = (
#            "counterfact-" if params.counterfact else ("zsre-" if params.zsre else "")
#        )
#
#        modelcode = "gpt2xl" if params.model_name == "gpt2-xl" else "gpt-j-6b"
#        model_filename = f"efk-{params.n_toks}tok-{train_ds}gpt2-xl.pt"
#        model_dir = "baselines/efk/weights"
#        
#        with hydra.initialize(config_path="config", job_name="run"):
#            config = hydra.compose(
#                config_name="config",
#                overrides=[
#                    "+alg=efk",
#                    "+experiment=gen",
#                    f"+model={modelcode}",
#                    f"data.path=data/{params.n_toks}token/data/self_sample/",
#                ],
#            )
#
#        #def add_padding(tokenizer, model):
#        #    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
#        #    model.resize_token_embeddings(len(tokenizer))
#        #    model.transformer.wte.weight.data[
#        #        -1
#        #    ] = model.transformer.wte.weight.data.mean(0)
#
#        # Load the gpt2xl and tokenizer
#        self.model = model
#        self.tokenizer = tok
#        #if is_first:
#        #    add_padding(self.tokenizer, self.model)
#
#        if not is_first:
#            del self.alg.editor
#        
#        self.alg = EFK(self.model, config)
#        d = torch.load(f"{model_dir}/{model_filename}")
#        self.alg.load_state_dict(deepcopy(d["model"]))
#        
#        del d
#        
#        self.alg.editor.cuda()
#        self.is_init = True
#
#    def reset_model(self):
#        self.is_init = False
#        del self.model, self.tokenizer, self.alg
#
#    def apply_to_model(
#        self,
#        model: AutoModelForCausalLM,
#        tok: AutoTokenizer,
#        requests: List[Dict],
#        hparams: EFKHyperParams,
#        copy=False,
#        return_orig_weights=False,
#        return_orig_weights_device="cuda",
#    ):
#        """
#        Processes a request, for example
#        {'prompt': '{} has the position of',
#         'subject': 'Charles Herman Helmsing',
#         'relation_id': 'P39',
#         'target_new': {'str': 'President', 'id': 'Q11696'},
#         'target_true': {'str': 'bishop', 'id': 'Q29182'}}
#        Returns an edited GPT model.
#        """
#
#        if copy:
#            print("Copy model!!!")
#            model = deepcopy(model)
#
#        if not self.is_init:
#            print("Init model!!!")
#            self.init_model(model, tok, hparams, is_first=True)
#
#        for request in requests:
#            #self.init_model(model, tok, hparams, is_first=False)
#            request_rewrite = deepcopy(request)
#
#            target = " " + request_rewrite["target_new"]["str"]
#            sentence = (
#                request_rewrite["prompt"].format(request_rewrite["subject"]) + target
#            )
#            target_tokens = self.tokenizer(target)["input_ids"]
#            tokens = torch.tensor(self.tokenizer(sentence)["input_ids"])[None]
#            label_tokens = tokens.clone()
#            label_tokens[0][: -len(target_tokens)] = -100
#            edit_inner = dict(
#                input_ids=tokens.clone().cuda(),
#                attention_mask=torch.ones_like(tokens).cuda(),
#                labels=label_tokens.clone().cuda(),
#            )
#            cond = dict(
#                input_ids=tokens.clone().cuda(),
#                attention_mask=torch.ones_like(tokens).cuda(),
#            )
#
#            weights_copy = {}
#            if return_orig_weights:
#                for k, v in model.named_parameters():
#                    if k not in weights_copy:
#                        weights_copy[k] = (
#                            v.detach().to(return_orig_weights_device).clone()
#                        )
#
#            
#            edited_model = self.alg.edit(edit_inner, cond)
#            #self.alg.model = edited_model
#            #torch.cuda.empty_cache()
#            
#            del edit_inner
#            del cond
#            #torch.cuda.empty_cache()
#            
#        return edited_model, weights_copy
#    
#