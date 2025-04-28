from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import time
import gradio as gr

##Quantitizen dus opdelen in discrete stapjes beetje generaliseren
quantization_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_use_double_quant_nested=True
)

#model moet eerste keer iets van 20gb downloaden erna gesaved
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=quantization_config, device_map="auto", torch_dtype=torch.float16)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
model.config.use_cache=True

#def warm_up_gpu():
#    #haha mooi
#    dummy_text = "This is a warm-up prompt."
#    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True)
#    inputs = {k: v.to(model.device) for k, v in inputs.items()}
#    with torch.no_grad():
#        for _ in range(3):
#            _ = model.generate(**inputs, max_length=20)

def get_gpu_usage():
    return torch.cuda.memory_allocated() / 1024**2


#kan sneller gezet worden maar kwali verminderd dan
def generate_response(prompt, max_tokens=256, temperature=0.6):
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    inputs=tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs=model.generate(**inputs, max_new_tokens=int(max_tokens), 
                           temperature=float(temperature), 
                           top_p=0.9, top_k=50, 
                           um_beams=1, do_sample=True, repetition_penalty=1.2, no_repeat_ngram_size=2, 
                           early_stopping=True, use_cache=True, pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

#if __name__ == "__main__":
 #   print(f"CUDA beschikbaar: {torch.cuda.is_available()}")
  #  print(f"GPU naam: {torch.cuda.get_device_name(0)}")
   # print(f"GPU geheugen: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    #print("\nWarming up GPU....")
    #warm_up_gpu()
    #test_prompts = [
     #   "Wat zijn de drie belangrijkste planeten in ons zonnestelsel?",
      #  "Schrijf een korte samenvatting over machine learning.",
    #    "Leg het verschil uit tussen Python en Java."
    #]

    #for prompt in test_prompts:
     #   print("\nPrompt:", prompt)
      #  print("\nAntwoord:", generate_response(prompt))
       # print("-" * 50)

#GUI in browser
gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=4, label="Jouw vraag"),
    outputs=gr.Textbox(lines=8, label="Antwoord"),
    title="Mistral-7B Chatbot",
    description="Stel een vraag aan het lokale Mistral-7B model."
).launch()