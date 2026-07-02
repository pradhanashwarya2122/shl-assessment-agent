# tests/analyze_traces.py
import json
import os
import requests

def analyze_trace(filename):
    with open(f'data/traces/{filename}') as f:
        trace = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Trace: {filename}")
    print(f"Persona: {json.dumps(trace.get('persona', {}), indent=2)}")
    print(f"Expected assessments: {trace.get('expected_shortlist', [])}")
    print(f"\nConversation flow:")
    
    for i, turn in enumerate(trace.get('conversation', [])):
        print(f"  Turn {i}: [{turn['role']}] {turn['content'][:100]}...")
    
    # Test against our API
    print(f"\nTesting against API...")
    messages = trace['conversation']
    response = requests.post('http://localhost:8000/chat', json={"messages": messages})
    
    if response.status_code == 200:
        data = response.json()
        print(f"  Agent reply: {data['reply'][:100]}...")
        print(f"  Recommendations: {len(data['recommendations'])}")
        
        # Calculate recall
        expected = set(trace.get('expected_shortlist', []))
        actual = set(r['name'] for r in data['recommendations'])
        recall = len(expected & actual) / len(expected) if expected else 0
        print(f"  Recall@10: {recall:.2%}")
    else:
        print(f"  Error: {response.status_code}")

# Analyze all traces
for filename in sorted(os.listdir('data/traces')):
    if filename.endswith('.json'):
        analyze_trace(filename)