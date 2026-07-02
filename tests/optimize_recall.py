# tests/optimize_recall.py
import json
import os
import requests
from collections import defaultdict

def analyze_trace_performance():
    """Deep dive into trace performance to optimize recall"""
    
    traces_dir = 'data/traces'
    all_insights = defaultdict(list)
    
    for filename in sorted(os.listdir(traces_dir)):
        if not filename.endswith('.json'):
            continue
            
        with open(os.path.join(traces_dir, filename)) as f:
            trace = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"ANALYZING: {filename}")
        
        # Extract trace info
        persona = trace.get('persona', {})
        expected = trace.get('expected_shortlist', [])
        conversation = trace.get('conversation', [])
        
        print(f"Expected assessments: {expected}")
        print(f"Conversation turns: {len(conversation)}")
        
        # Simulate conversation turn by turn
        messages = []
        for i, turn in enumerate(conversation):
            messages.append(turn)
            
            r = requests.post('http://localhost:8000/chat', 
                            json={"messages": messages})
            
            if r.status_code != 200:
                print(f"  Turn {i}: API error {r.status_code}")
                continue
            
            data = r.json()
            reply = data.get('reply', '')
            recs = data.get('recommendations', [])
            
            print(f"  Turn {i}: {'RECOMMENDS' if recs else 'CLARIFIES'}")
            if recs:
                actual_names = [r['name'] for r in recs]
                matched = set(actual_names) & set(expected)
                recall = len(matched) / len(expected) if expected else 0
                print(f"    Recall: {recall:.2%}")
                print(f"    Matched: {matched}")
                print(f"    Missing: {set(expected) - set(actual_names)}")
                
                # Track insights
                if recall < 0.5:
                    all_insights['low_recall'].append({
                        'trace': filename,
                        'recall': recall,
                        'missing': list(set(expected) - set(actual_names))
                    })
            
            # Stop if conversation is complete
            if data.get('end_of_conversation'):
                break
    
    # Print optimization suggestions
    print("\n" + "="*60)
    print("OPTIMIZATION SUGGESTIONS")
    print("="*60)
    
    if all_insights['low_recall']:
        print("\nTraces with low recall:")
        for insight in all_insights['low_recall']:
            print(f"  {insight['trace']}: {insight['recall']:.1%}")
            print(f"    Missing assessments: {insight['missing']}")
        
        print("\nPossible fixes:")
        print("1. Check if missing assessments are in catalog")
        print("2. Improve query construction for these cases")
        print("3. Adjust similarity threshold in retrieval")
        print("4. Add synonym expansion for role/skill matching")

if __name__ == '__main__':
    analyze_trace_performance()