import os, json, argparse, torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from algo.ldm_restarts import LDM_Restarter

def get_reward(rm_model, rm_tokenizer, prompt, response):
    inputs = rm_tokenizer(prompt, response, return_tensors="pt", truncation=True, max_length=512).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad(): 
        score = rm_model(**inputs).logits[0].item()
    return score

def main():
    parser = argparse.ArgumentParser()
    # Using smaller defaults for local if needed, but keeping the signature the same as server
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--rm_name", type=str, default="OpenAssistant/reward-model-deberta-v3-large")
    parser.add_argument("--out_dir", type=str, default="results/debate_real")
    parser.add_argument("--restart_threshold", type=float, default=0.5) 
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    metrics_file = os.path.join(args.out_dir, "metrics.jsonl")
    
    print("[Main Conf] Loading Model, Reward Model, and TruthfulQA...")
    dataset = load_dataset("truthful_qa", "generation")["validation"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_name)
    rm_model = AutoModelForSequenceClassification.from_pretrained(args.rm_name).to(device).eval()
    
    # Check if GPU is available for 8-bit loading
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        try:
            base_model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=bnb_config, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"CRASH: Cannot download {args.model_name}. Error: {e}")
            return
    else:
        print("WARNING: CUDA not found. Loading in float32 on CPU (Slow!)...")
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    meta_model = get_peft_model(base_model, lora_config)
    peer_model = get_peft_model(base_model, lora_config)
    
    restarter = LDM_Restarter(patience=5, threshold=args.restart_threshold)
    
    print("Starting Main-Conference LLM Debate Loop...")
    start_time = time.time()
    
    for round_idx, item in enumerate(dataset):
        if round_idx >= 2000: break 
            
        prompt = f"Question: {item['question']}\nAnswer honestly:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            meta_out = meta_model.generate(**inputs, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
            peer_out = peer_model.generate(**inputs, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
            
        meta_resp = tokenizer.decode(meta_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        peer_resp = tokenizer.decode(peer_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        rm_m = get_reward(rm_model, rm_tokenizer, prompt, meta_resp)
        rm_p = get_reward(rm_model, rm_tokenizer, prompt, peer_resp)
        
        # Meta-Update Simulation (TRL/PPO meta-MAPG step)
        dummy_loss_m = meta_model(inputs["input_ids"]).logits.sum() * rm_m
        dummy_loss_p = peer_model(inputs["input_ids"]).logits.sum() * rm_p
        
        dummy_loss_m.backward() 
        dummy_loss_p.backward()
        
        triggered = restarter.check_and_restart([rm_m], meta_model)
        
        metric = {
            "round": round_idx, "wall_time": time.time() - start_time,
            "question": item['question'], "meta_resp": meta_resp, "peer_resp": peer_resp,
            "truth_score_m": rm_m, "truth_score_p": rm_p,
            "restarts": restarter.restart_count, "restarted_this_round": triggered
        }
        
        with open(metrics_file, "a") as f: f.write(json.dumps(metric) + "\n")
        if round_idx % 10 == 0: print(f"[{round_idx}] MetaRM: {rm_m:.2f} | Restarts: {restarter.restart_count}")
            
    print(f"Done. Logs: {metrics_file}")

if __name__ == "__main__":
    main()
