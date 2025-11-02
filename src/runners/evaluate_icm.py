import os
import sys
import json
import csv
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

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

def create_fewshot_prompt(question, choice, demonstrations):
    demo_text = ""
    for demo in demonstrations:
        label_text = "True" if demo['label'] == 1 else "False"
        demo_text += f"Question: {demo['question']}\nClaim: {demo['choice']}\n\nIs this claim true or false? Respond with only the word \"True\" or \"False\".\n\nAnswer: {label_text}\n\n"
    
    prompt = f"{demo_text}Question: {question}\nClaim: {choice}\n\nIs this claim true or false? Respond with only the word \"True\" or \"False\".\n\nAnswer:"
    return prompt

def evaluate_icm(demonstrations):
    correct = 0
    total = 0
    results = []
    
    for item in tqdm(test_data, desc="Evaluating ICM"):
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
                'method': 'ICM',
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

print("Evaluating ICM on TruthfulQA...\n")

icm_results_path = results_dir / 'icm_demonstrations.json'
if icm_results_path.exists():
    with open(icm_results_path, 'r') as f:
        icm_demonstrations = json.load(f)
    accuracy_icm, results_icm = evaluate_icm(icm_demonstrations)
    
    csv_path_icm = results_dir / 'icm_results.csv'
    with open(csv_path_icm, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['method', 'question', 'choice', 'true_label', 'model_response', 'prediction', 'is_correct']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_icm)
    
    print(f"ICM Accuracy: {accuracy_icm:.2f}%")
    print(f"Saved ICM results to {csv_path_icm}")
else:
    print("ICM demonstrations not found. Run icm_algorithm.py first.")
