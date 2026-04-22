import os, json, argparse, torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from algo.ldm_restarts import LDM_Restarter

def get_reward(rm_model, rm_tokenizer, prompt, response):
    inputs = rm_tokenizer(prompt, response, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad(): 
        score = rm_model(**inputs).logits[0].item()
    return score

def main():
    parser = argparse.ArgumentParser()
    # Llama-3.3-70B: PROVEN to work on 4xV100 in 4-bit NF4 (ran 45 rounds before torch import bug)
    # Llama-4-Scout: DOES NOT FIT on V100s (MoE 16-expert architecture too large)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--rm_name", type=str, default="OpenAssistant/reward-model-deberta-v3-large")
    parser.add_argument("--out_dir", type=str, default="results/debate_real")
    parser.add_argument("--restart_threshold", type=float, default=0.5) 
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    metrics_file = os.path.join(args.out_dir, "metrics.jsonl")
    
    # RESUME LOGIC: skip already-completed rounds
    start_round = 0
    prior_restarts = 0
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                if lines:
                    last_entry = json.loads(lines[-1])
                    start_round = last_entry["round"] + 1
                    prior_restarts = last_entry.get("restarts", 0)
            print(f"[Resume] Picking up from Round {start_round} (prior restarts: {prior_restarts})")
        except Exception as e:
            print(f"[Resume] Could not parse metrics file ({e}), starting fresh.")

    print(f"[Config] Model: {args.model_name}")
    print(f"[Config] Reward Model: {args.rm_name}")
    print(f"[Config] Restart Threshold: {args.restart_threshold}")
    
    dataset = load_dataset("truthful_qa", "generation")["validation"]
    
    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_name)
    rm_model = AutoModelForSequenceClassification.from_pretrained(args.rm_name).to("cuda").eval()
    
    # 4-bit NF4 double quantization — proven stable on 4xV100 for 70B
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16 
    )

    print("[Loading] Base model with NF4 quantization...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        quantization_config=bnb_config, 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    meta_model = get_peft_model(base_model, lora_config)
    peer_model = get_peft_model(base_model, lora_config)
    
    # Optimizers — MANDATORY to call zero_grad() each round to prevent OOM
    opt_m = torch.optim.AdamW(meta_model.parameters(), lr=1e-5)
    opt_p = torch.optim.AdamW(peer_model.parameters(), lr=1e-5)

    restarter = LDM_Restarter(patience=5, threshold=args.restart_threshold)
    restarter.restart_count = prior_restarts  # restore counter on resume
    
    print(f"[Start] Debate loop targeting 1000 rounds...")
    start_time = time.time()
    
    for round_idx, item in enumerate(dataset):
        if round_idx < start_round: 
            continue  # skip completed rounds
        if round_idx >= 1000: 
            break
            
        prompt = f"Question: {item['question']}\nAnswer honestly:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            meta_out = meta_model.generate(**inputs, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
            peer_out = peer_model.generate(**inputs, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
            
        meta_resp = tokenizer.decode(meta_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        peer_resp = tokenizer.decode(peer_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        rm_m = get_reward(rm_model, rm_tokenizer, prompt, meta_resp)
        rm_p = get_reward(rm_model, rm_tokenizer, prompt, peer_resp)
        
        # Meta-MAPG gradient step — zero_grad FIRST to prevent memory leak
        opt_m.zero_grad()
        opt_p.zero_grad()
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
        
        with open(metrics_file, "a") as f: 
            f.write(json.dumps(metric) + "\n")
            f.flush()
        if round_idx % 5 == 0: 
            print(f"[{round_idx}] MetaRM: {rm_m:.2f} | PeerRM: {rm_p:.2f} | Restarts: {restarter.restart_count} | Time: {time.time()-start_time:.0f}s")
            
    print(f"Done. Total time: {time.time()-start_time:.0f}s | Logs: {metrics_file}")

if __name__ == "__main__":
    main()
