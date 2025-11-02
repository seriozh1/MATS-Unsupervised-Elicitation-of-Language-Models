import os
import sys
import json
import csv
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from copy import deepcopy

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.tools.dataloaders import load_truthfulqa_test
from src.model_querying.solution_extraction import get_yes_no

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

test_data = load_truthfulqa_test()
MAX_SAMPLES = None
test_data = test_data[:MAX_SAMPLES] if MAX_SAMPLES is not None else test_data

train_data_path = Path(__file__).parent.parent.parent / "data" / "truthfulqa_train.json"
with open(train_data_path, 'r') as f:
    train_data = json.load(f)

def create_fewshot_prompt(question, choice, demonstrations):
    demo_text = ""
    for demo in demonstrations:
        label_text = "True" if demo['label'] == 1 else "False"
        demo_text += f"Question: {demo['question']}\nClaim: {demo['choice']}\n\nIs this claim true or false? Respond with only the word \"True\" or \"False\".\n\nAnswer: {label_text}\n\n"
    
    prompt = f"{demo_text}Question: {question}\nClaim: {choice}\n\nIs this claim true or false? Respond with only the word \"True\" or \"False\".\n\nAnswer:"
    return prompt

def evaluate_golden(demonstrations):
    correct = 0
    total = 0
    results = []
    
    for item in tqdm(test_data, desc="Evaluating Golden Labels"):
        prompt = create_fewshot_prompt(item['question'], item['choice'], demonstrations)
        
        prediction = None
        text = ""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = client.completions.create(
                    model=base_model,
                    prompt=prompt,
                    max_tokens=5,
                    temperature=0.0
                )
                text = response.choices[0].text.strip().lower()
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
                    break
        
        if prediction is not None:
            prediction_int = 1 if prediction else 0
            is_correct = prediction_int == item['label']
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                'method': 'Golden Labels',
                'question': item['question'],
                'choice': item['choice'],
                'true_label': item['label'],
                'model_response': text,
                'prediction': prediction_int,
                'is_correct': is_correct
            })
        else:
            print(f"Warning: Could not parse prediction after {max_retries} attempts. Question: '{item['question'][:50]}...' | Last model output: '{text[:200]}'")
    
    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, results

results_dir = Path(__file__).parent.parent.parent / 'results'

print("Evaluating Golden Labels on TruthfulQA...\n")

golden_demonstrations = [deepcopy(item) for item in train_data]
accuracy_golden, results_golden = evaluate_golden(golden_demonstrations)

csv_path_golden = results_dir / 'golden_results.csv'
with open(csv_path_golden, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['method', 'question', 'choice', 'true_label', 'model_response', 'prediction', 'is_correct']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results_golden)

print(f"Golden Labels Accuracy: {accuracy_golden:.2f}%")
print(f"Saved Golden Labels results to {csv_path_golden}")
