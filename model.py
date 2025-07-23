import torch
import bitsandbytes as bnb
from transformers import LlamaForCausalLM, LlamaTokenizerFast

class CADPolicy:
    def __init__(self, device='cuda'):
        self.device = device
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            'meta-llama/Llama-2-7b', use_fast=True
        )
        self.model = LlamaForCausalLM.from_pretrained(
            'meta-llama/Llama-2-7b',
            load_in_8bit=True,
            device_map='auto'
        )

    def generate(self, prompt, max_length=128):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        out = self.model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def get_log_probs(self, prompt, code_str):
        combined = prompt + code_str
        tokens = self.tokenizer(combined, return_tensors='pt').to(self.device)
        labels = tokens.input_ids.clone()
        outputs = self.model(**tokens, labels=labels)
        # sum log‑prob ≈ −loss * N_tokens
        return - outputs.loss * labels.size(1)

    def parameters(self):
        return self.model.parameters()
