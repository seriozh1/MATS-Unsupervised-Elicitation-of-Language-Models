import os
import sys
import json
import math
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

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
MAX_SAMPLES = 10
test_data = test_data[:MAX_SAMPLES] if MAX_SAMPLES is not None else test_data

train_data_path = Path(__file__).parent.parent.parent / "data" / "truthfulqa_train.json"
with open(train_data_path, 'r') as f:
    train_data = json.load(f)

def evaluate_zero_shot(model_name, model_label, use_completions=True, use_hhh=False):
    correct = 0
    total = 0
    
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
            if (1 if prediction else 0) == item['label']:
                correct += 1
            total += 1
        else:
            print(f"Warning: Could not parse prediction after {max_retries} attempts. Question: '{item['question'][:50]}...' | Last model output: '{text[:200]}'")
    
    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, total

def create_icm_prompt(question, choice, demonstrations):
    demo_text = ""
    for demo in demonstrations:
        label_text = "True" if demo['label'] == 1 else "False"
        demo_text += f"Question: {demo['question']}\nClaim: {demo['choice']}\nI think this claim is {label_text}\n\n"
    
    prompt = f"{demo_text}Question: {question}\nClaim: {choice}\nI think this claim is"
    return prompt

def get_logprobs_for_tokens(model_name, prompt):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                top_logprobs=20
            )
            
            if not response.choices[0].logprobs or not response.choices[0].logprobs.content:
                if attempt < max_retries - 1:
                    continue
                return {}
            
            top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            logprobs_dict = {}
            for logprob_item in top_logprobs:
                token = logprob_item.token.strip().lower()
                logprobs_dict[token] = logprob_item.logprob
            
            return logprobs_dict
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            else:
                print(f"Error getting logprobs after {max_retries} attempts: {e}")
                return {}
    return {}

def calculate_mutual_predictability_score(item, demonstrations):
    if len(demonstrations) == 0:
        return 0.0
    
    prompt = create_icm_prompt(item['question'], item['choice'], demonstrations)
    logprobs_dict = get_logprobs_for_tokens(base_model, prompt)
    
    eps = 1e-5
    prob_true = eps
    prob_false = eps
    
    for token, logprob in logprobs_dict.items():
        if 'true' in token:
            prob_true += math.exp(logprob)
        elif 'false' in token:
            prob_false += math.exp(logprob)
    
    score = math.log(prob_true) - math.log(prob_false)
    return score

def evaluate_with_demonstrations(demonstrations, label_type):
    correct = 0
    total = 0
    
    for item in tqdm(test_data, desc=f"Evaluating {label_type}"):
        score = calculate_mutual_predictability_score(item, demonstrations)
        prediction = 1 if score > 0 else 0
        if prediction == item['label']:
            correct += 1
        total += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy

print("Evaluating all methods on TruthfulQA...\n")

print("1. Zero-shot (Base model)...")
accuracy_base, total_base = evaluate_zero_shot(base_model, "Base", use_completions=False, use_hhh=False)

print("\n2. Zero-shot (Chat/Instruct model)...")
accuracy_chat, total_chat = evaluate_zero_shot(instruct_model, "Chat", use_completions=False, use_hhh=False)

print("\n3. ICM (using learned labels)...")
icm_results_path = Path(__file__).parent.parent.parent / 'results' / 'icm_demonstrations.json'
if icm_results_path.exists():
    with open(icm_results_path, 'r') as f:
        icm_demonstrations = json.load(f)
    accuracy_icm = evaluate_with_demonstrations(icm_demonstrations, "ICM")
else:
    print("ICM demonstrations not found. Run icm_algorithm.py first.")
    accuracy_icm = 0

print("\n4. Golden Labels (using true labels)...")
golden_demonstrations = [deepcopy(item) for item in train_data[:50]]
accuracy_golden = evaluate_with_demonstrations(golden_demonstrations, "Golden Labels")

print(f"\nResults:")
print(f"Zero-shot (Base): {accuracy_base:.2f}%")
print(f"Zero-shot (Chat): {accuracy_chat:.2f}%")
print(f"ICM: {accuracy_icm:.2f}%")
print(f"Golden Labels: {accuracy_golden:.2f}%")

plt.figure(figsize=(10, 6))
methods = ['Zero-shot\n(Base)', 'Zero-shot\n(Chat)', 'ICM', 'Golden\nLabels']
accuracies = [accuracy_base, accuracy_chat, accuracy_icm, accuracy_golden]
colors = ['#C8A8E9', '#A18CD1', '#8B7AB8', '#6B5B95']

bars = plt.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
bars[1].set_hatch('oo')
bars[1].set_edgecolor('black')

plt.ylabel('Accuracy (%)', fontsize=12)
plt.title(f'TruthfulQA - All Methods\n({MAX_SAMPLES} test samples, 50 train samples)', fontsize=14, fontweight='bold')
plt.ylim([0, 100])
plt.grid(axis='y', alpha=0.3, linestyle='--')

for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plot_dir = Path(__file__).parent.parent.parent / 'plots'
plot_dir.mkdir(exist_ok=True)
plot_path = plot_dir / 'all_methods_results.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nSaved plot to {plot_path}")

