import os
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import matplotlib.pyplot as plt

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
test_data = test_data[:MAX_SAMPLES]

def evaluate_zero_shot(model_name, model_label, use_completions=True, use_hhh=False):
    correct = 0
    total = 0
    
    for item in tqdm(test_data, desc=f"Zero-shot ({model_label})"):
        prompt = create_truthfulqa_prompt(item['question'], item['choice'], use_hhh=use_hhh)
        
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
                if (1 if prediction else 0) == item['label']:
                    correct += 1
                total += 1
            else:
                print(f"Warning: Could not parse prediction. Question: '{item['question'][:50]}...' | Model output: '{text[:200]}'")
        except Exception as e:
            print(f"\nError on sample {item.get('question', 'unknown')[:50]}: {e}")
            import traceback
            traceback.print_exc()
    
    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, total

print("Evaluating zero-shot baselines on TruthfulQA...\n")

# print("1. Zero-shot (Base model)...")
accuracy_base = 0
# accuracy_base, total_base = evaluate_zero_shot(base_model, "Base", use_completions=True, use_hhh=False)

print("\n2. Zero-shot (Chat/Instruct model)...")
accuracy_chat, total_chat = evaluate_zero_shot(instruct_model, "Chat", use_completions=False, use_hhh=False)

print(f"\nResults:")
# print(f"Zero-shot (Base): {accuracy_base:.2f}% ({total_base}/{MAX_SAMPLES})")
print(f"Zero-shot (Chat): {accuracy_chat:.2f}% ({total_chat}/{MAX_SAMPLES})")

plt.figure(figsize=(8, 6))
methods = ['Zero-shot', 'Zero-shot\n(Chat)']
accuracies = [accuracy_base, accuracy_chat]
colors = ['#C8A8E9', '#A18CD1']

bars = plt.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title(f'TruthfulQA - Zero-shot Baselines\n({MAX_SAMPLES} samples)', fontsize=14, fontweight='bold')
plt.ylim([0, 100])
plt.grid(axis='y', alpha=0.3, linestyle='--')

for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plot_dir = Path(__file__).parent.parent.parent / 'plots'
plot_dir.mkdir(exist_ok=True)
plot_path = plot_dir / 'baseline_results.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nSaved plot to {plot_path}")

