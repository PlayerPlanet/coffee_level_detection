"""
Test and visualization utilities for inference results.
"""
import os
import json
import matplotlib.pyplot as plt
import pytest

ANNOTATIONS_PATH = 'inference_coffee_level_annotations.json'

def load_annotations(json_path=ANNOTATIONS_PATH):
    """Load inference results from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def plot_level_distribution(data):
    """Plot coffee level distribution as a bar chart."""
    dist = data.get('level_distribution', {})
    levels = list(map(int, dist.keys()))
    counts = [dist[str(l)] for l in levels]
    plt.figure(figsize=(8, 4))
    plt.bar(levels, counts, color='saddlebrown')
    plt.xlabel('Coffee Level')
    plt.ylabel('Count')
    plt.title('Coffee Level Distribution (Inference)')
    plt.tight_layout()
    plt.show()

@pytest.mark.visual
def test_plot_level_distribution():
    """Test: visualize coffee level distribution from inference results."""
    data = load_annotations()
    plot_level_distribution(data)

# Optionally, add more tests for summary stats, etc.


if __name__=="__main__":
    test_plot_level_distribution()