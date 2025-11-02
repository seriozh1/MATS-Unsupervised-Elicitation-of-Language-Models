import os
import sys
import json
import math
import random
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from copy import deepcopy

sys.path.append(str(Path(__file__).parent.parent.parent))
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

def create_icm_prompt(question, choice, demonstrations):
    demo_text = ""
    for demo in demonstrations:
        label_text = "True" if demo['label'] == 1 else "False"
        demo_text += f"Question: {demo['question']}\nClaim: {demo['choice']}\nI think this claim is {label_text}\n\n"
    
    prompt = f"{demo_text}Question: {question}\nClaim: {choice}\nI think this claim is"
    return prompt

def get_logprobs_for_tokens(model_name, prompt, tokens_to_check):
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
            return {}
        
        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        logprobs_dict = {}
        for logprob_item in top_logprobs:
            token = logprob_item.token.strip().lower()
            logprobs_dict[token] = logprob_item.logprob
        
        return logprobs_dict
    except Exception as e:
        print(f"Error getting logprobs: {e}")
        return {}

def calculate_mutual_predictability_score(item, demonstrations, alpha=50):
    if len(demonstrations) == 0:
        return 0.0
    
    prompt = create_icm_prompt(item['question'], item['choice'], demonstrations)
    logprobs_dict = get_logprobs_for_tokens(base_model, prompt, ['true', 'false'])
    
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

def get_temperature(iteration, initial_temp, final_temp, decay_rate):
    return max(final_temp, initial_temp / (1 + 2 * math.log(1 + iteration)))

def run_icm(train_data, num_seed=8, max_iterations=100, alpha=50, initial_T=10, final_T=0.01, decay=0.99):
    demonstrations = []
    
    random_init_labels = [1] * (num_seed // 2) + [0] * (num_seed // 2)
    random.shuffle(random_init_labels)
    
    for i in range(num_seed):
        item = deepcopy(train_data[i])
        item['label'] = random_init_labels[i]
        demonstrations.append(item)
    
    unlabeled = train_data[num_seed:]
    
    for item in unlabeled:
        score = calculate_mutual_predictability_score(item, demonstrations, alpha)
        item['score'] = score
        item['label'] = 1 if score > 0 else 0
        demonstrations.append(item)
    
    best_demonstrations = deepcopy(demonstrations)
    best_score = sum([d['score'] * (1 if d['label'] == 1 else -1) for d in demonstrations])
    
    flip_cnt = 0
    for iteration in tqdm(range(max_iterations), desc="ICM iterations"):
        example_id = random.randint(0, len(demonstrations) - 1)
        
        tmp_demonstrations = deepcopy(demonstrations)
        old_label = tmp_demonstrations[example_id]['label']
        new_label = 1 - old_label
        
        tmp_demonstrations[example_id]['label'] = new_label
        
        new_score = 0
        for i, demo in enumerate(tmp_demonstrations):
            other_demos = [d for j, d in enumerate(tmp_demonstrations) if j != i]
            score = calculate_mutual_predictability_score(demo, other_demos, alpha)
            demo['score'] = score
            new_score += score * (1 if demo['label'] == 1 else -1)
        
        current_score = sum([d['score'] * (1 if d['label'] == 1 else -1) for d in demonstrations])
        
        T = get_temperature(flip_cnt, initial_T, final_T, decay)
        accept_prob = math.exp((new_score - current_score) / T)
        
        if random.random() < accept_prob:
            demonstrations = tmp_demonstrations
            flip_cnt += 1
            if new_score > best_score:
                best_score = new_score
                best_demonstrations = deepcopy(demonstrations)
    
    return best_demonstrations

def evaluate_with_icm_labels(test_data, icm_demonstrations):
    correct = 0
    total = len(test_data)
    
    for item in tqdm(test_data, desc="Evaluating with ICM labels"):
        score = calculate_mutual_predictability_score(item, icm_demonstrations)
        prediction = 1 if score > 0 else 0
        if prediction == item['label']:
            correct += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy

if __name__ == "__main__":
    train_data_path = Path(__file__).parent.parent.parent / "data" / "truthfulqa_train.json"
    with open(train_data_path, 'r') as f:
        train_data = json.load(f)
    
    test_data = load_truthfulqa_test()
    MAX_SAMPLES = 10
    test_data = test_data[:MAX_SAMPLES]
    train_data = train_data[:50]
    
    print("Running ICM algorithm...")
    icm_demonstrations = run_icm(train_data, num_seed=8, max_iterations=50, alpha=50)
    
    print("\nEvaluating on test set...")
    icm_accuracy = evaluate_with_icm_labels(test_data, icm_demonstrations)
    
    print(f"\nICM Accuracy: {icm_accuracy:.2f}%")
    
    results_dir = Path(__file__).parent.parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'icm_demonstrations.json', 'w') as f:
        json.dump(icm_demonstrations, f, indent=2)
    
    print(f"Saved ICM demonstrations to {results_dir / 'icm_demonstrations.json'}")

