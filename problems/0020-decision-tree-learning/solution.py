import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    h = 0
    prop = Counter(labels)
    n = len(labels)

    if not labels:
        return 0

    for k, v in prop.items():
        h+=(v/n)*math.log2(v/n)
    return -h

def calculate_information_gain(examples, attr, target_attr):
    total_entropy = calculate_entropy([example[target_attr] for example in examples])
    values = set(example[attr] for example in examples)
    attr_entropy = 0
    for value in values:
        value_subset = [example[target_attr] for example in examples if example[attr] == value]
        value_entropy = calculate_entropy(value_subset)
        attr_entropy += (len(value_subset) / len(examples)) * value_entropy
    return total_entropy - attr_entropy

def majority_class(examples, target_attr):
    counts = Counter([example[target_attr] for example in examples])
    max_count = max(counts.values())
    # Return alphabetically first among ties
    candidates = [label for label, count in counts.items() if count == max_count]
    return sorted(candidates)[0]
    
def learn_decision_tree(examples, attributes, target_attr):
    if not examples:
        return 'No examples'
    if all(example[target_attr] == examples[0][target_attr] for example in examples):
        return examples[0][target_attr]
    if not attributes:
        return majority_class(examples, target_attr)
    
    # Calculate gains, use attribute list order for tie-breaking
    gains = [(attr, calculate_information_gain(examples, attr, target_attr)) for attr in attributes]
    best_attr = max(gains, key=lambda x: x[1])[0]
    tree = {best_attr: {}}
    
    # Process values in sorted order for consistent output
    for value in sorted(set(example[best_attr] for example in examples)):
        subset = [example for example in examples if example[best_attr] == value]
        new_attributes = [a for a in attributes if a != best_attr]
        subtree = learn_decision_tree(subset, new_attributes, target_attr)
        tree[best_attr][value] = subtree
    
    return tree




