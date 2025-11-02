import csv
from pathlib import Path
import matplotlib.pyplot as plt

def load_results_from_csv(csv_path):
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results

def calculate_accuracy_from_results(results):
    if len(results) == 0:
        return 0.0, 0
    correct = sum(1 for r in results if r['is_correct'] == 'True')
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, total

def plot_all_methods(results_dir, output_filename='all_methods_results.png'):
    results_dir = Path(results_dir)
    
    base_csv = results_dir / 'baseline_results_base.csv'
    chat_csv = results_dir / 'baseline_results_chat.csv'
    icm_csv = results_dir / 'icm_results.csv'
    golden_csv = results_dir / 'golden_results.csv'
    
    results_base = load_results_from_csv(base_csv) if base_csv.exists() else []
    results_chat = load_results_from_csv(chat_csv) if chat_csv.exists() else []
    results_icm = load_results_from_csv(icm_csv) if icm_csv.exists() else []
    results_golden = load_results_from_csv(golden_csv) if golden_csv.exists() else []
    
    accuracy_base, total_base = calculate_accuracy_from_results(results_base)
    accuracy_chat, total_chat = calculate_accuracy_from_results(results_chat)
    accuracy_icm, total_icm = calculate_accuracy_from_results(results_icm)
    accuracy_golden, total_golden = calculate_accuracy_from_results(results_golden)
    
    total_samples = max(total_base, total_chat, total_icm, total_golden)
    
    plt.figure(figsize=(10, 6))
    methods = ['Zero-shot\n(Base)', 'Zero-shot\n(Chat)', 'ICM', 'Golden\nLabels']
    accuracies = [accuracy_base, accuracy_chat, accuracy_icm, accuracy_golden]
    colors = ['#B565A7', '#B565A7', '#7FC7D9', '#E89B3C']
    
    bars = plt.bar(methods, accuracies, color=colors, alpha=0.9, edgecolor='black', linewidth=1.2)
    bars[1].set_hatch('oo')
    bars[1].set_edgecolor('black')
    
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'TruthfulQA - All Methods\n({total_samples} test samples)', fontsize=14, fontweight='bold')
    plt.ylim([0, 100])
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / output_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {plot_path}")
    return plot_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = Path(__file__).parent.parent.parent / 'results'
    
    plot_all_methods(results_dir)

