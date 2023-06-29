import os

import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402
from transformers import AutoModel

BASE_MODEL = 'decapoda-research/llama-7b-hf'
# BASE_MODEL = os.environ.get("BASE_MODEL", None)
# assert (
#     BASE_MODEL
# ), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=huggyllama/llama-7b`"  # noqa: E501
### temporary test ###
base_model = AutoModel.from_pretrained('decapoda-research/llama-7b-hf')

# Print the state_dict
print("Base Model State Dict:")
for param_tensor in base_model.state_dict():
    print(param_tensor, "\t", base_model.state_dict()[param_tensor].size())

# Load the finetuned model
lora_model = PeftModel.from_pretrained(
    base_model='decapoda-research/llama-7b-hf',
    model_path='/content/gdrive/MyDrive/alpaca-lora/lora-alpaca',
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

# Print the state_dict
print("Finetuned Model State Dict:")
for param_tensor in lora_model.state_dict():
    print(param_tensor, "\t", lora_model.state_dict()[param_tensor].size())

# Load the finetuned weights directly
weights = torch.load('/content/gdrive/MyDrive/alpaca-lora/lora-alpaca')

# Print the weights
print("Directly Loaded Weights:")
for param_tensor in weights:
    print(param_tensor, "\t", weights[param_tensor].size())
### end test
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    "/content/gdrive/MyDrive/alpaca-lora/lora-alpaca",
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

lora_weight = lora_model.base_model.model.model.layers[
    0
].self_attn.q_proj.weight
print(first_weight_old)
print(first_weight)
assert torch.allclose(first_weight_old, first_weight)


# merge weights - new merging method from peft
lora_model = lora_model.merge_and_unload()

lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

LlamaForCausalLM.save_pretrained(
    base_model, "./hf_ckpt", state_dict=deloreanized_sd, max_shard_size="400MB"
)
