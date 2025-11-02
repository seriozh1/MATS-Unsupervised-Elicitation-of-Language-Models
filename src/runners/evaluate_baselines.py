import os
import sys
import csv
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.model_querying.prompt_creation import create_truthfulqa_prompt
from src.model_querying.solution_extraction import get_yes_no
from src.tools.dataloaders import load_truthfulqa_test

load_dotenv()

api_key = os.getenv('HYPERBOLIC_API_KEY')
if not api_key:
    raise ValueError("HYPERBOLIC_API_KEY not found in .env file")

client = OpenAI(
    api_key=api_key, 
    base_url="https://api.hyperbolic.xyz/v1",
    timeout=30.0
)

base_model = "meta-llama/Meta-Llama-3.1-405B"
instruct_model = "meta-llama/Meta-Llama-3.1-405B-Instruct"

test_data = load_truthfulqa_test()
MAX_SAMPLES = None
test_data = test_data[:MAX_SAMPLES] if MAX_SAMPLES is not None else test_data

def evaluate_zero_shot(model_name, model_label, use_completions=True, use_hhh=False):
    correct = 0
    total = 0
    results = []
    
    for item in tqdm(test_data, desc=f"Zero-shot ({model_label})"):
        prompt = create_truthfulqa_prompt(item['question'], item['choice'], use_hhh=use_hhh)
        
        prediction = None
        text = ""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                if use_completions:
                    response = client.completions.create(
                        model=model_name,
                        prompt=prompt,
                        max_tokens=5,
                        temperature=0.0
                    )
                    text = response.choices[0].text.strip().lower()
                    prediction = get_yes_no(text)
                else:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=5,
                        temperature=0.0
                    )
                    text = (response.choices[0].message.content or "").strip().lower()
                    prediction = get_yes_no(text)
                
                if prediction is not None:
                    break
                elif attempt < max_retries - 1:
                    continue
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                else:
                    print(f"\nError on sample {item.get('question', 'unknown')[:50]} after {max_retries} attempts: {e}")
                    import traceback
                    traceback.print_exc()
                    break
        
        if prediction is not None:
            is_correct = (1 if prediction else 0) == item['label']
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                'model': model_name,
                'model_label': model_label,
                'question': item['question'],
                'choice': item['choice'],
                'true_label': item['label'],
                'model_response': text,
                'prediction': prediction,
                'is_correct': is_correct
            })
        else:
            print(f"Warning: Could not parse prediction after {max_retries} attempts. Question: '{item['question'][:50]}...' | Last model output: '{text[:200]}'")
    
    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, total, results

results_dir = Path(__file__).parent.parent.parent / 'results'
results_dir.mkdir(exist_ok=True)
plots_dir = results_dir / 'plots'
plots_dir.mkdir(exist_ok=True)

print("Evaluating zero-shot baselines on TruthfulQA...\n")

print("1. Zero-shot (Base model)...")
accuracy_base, total_base, results_base = evaluate_zero_shot(base_model, "Base", use_completions=True, use_hhh=False)

csv_path_base = results_dir / 'baseline_results_base.csv'
with open(csv_path_base, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['model', 'model_label', 'question', 'choice', 'true_label', 'model_response', 'prediction', 'is_correct']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results_base)
print(f"Saved base model results to {csv_path_base}")

print("\n2. Zero-shot (Chat/Instruct model)...")
accuracy_chat, total_chat, results_chat = evaluate_zero_shot(instruct_model, "Chat", use_completions=False, use_hhh=False)

csv_path_chat = results_dir / 'baseline_results_chat.csv'
with open(csv_path_chat, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['model', 'model_label', 'question', 'choice', 'true_label', 'model_response', 'prediction', 'is_correct']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results_chat)
print(f"Saved chat model results to {csv_path_chat}")

print(f"\nResults:")
print(f"Zero-shot (Base): {accuracy_base:.2f}% ({total_base}/{len(test_data)})")
print(f"Zero-shot (Chat): {accuracy_chat:.2f}% ({total_chat}/{len(test_data)})")

