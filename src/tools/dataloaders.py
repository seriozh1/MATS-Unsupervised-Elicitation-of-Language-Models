import json

def load_truthfulqa_test(path="data/truthfulqa_test.json"):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

