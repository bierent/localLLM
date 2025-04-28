from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

print(f"CUDA beschikbaar: {torch.cuda.is_available()}")
print(f"GPU naam: {torch.cuda.get_device_name(0)}")

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"  # Dit zorgt voor automatische device toewijzing
)



text = 'Are you a nerd?'
inputs = tokenizer(text, return_tensors='pt')
inputs = {k: v.to(model.device) for k, v in inputs.items()}
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))