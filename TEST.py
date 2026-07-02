# save as evaluate_recall.py
import json, requests, os, re

BASE_URL = "https://web-production-ada63.up.railway.app"  # Your deployed URL

# Load expected shortlists
with open('data/expected_shortlists.json', 'r') as f:
    expected = json.load(f)

traces_dir = 'data/traces'
all_results = {}
total_recall = 0
trace_count = 0

for fname in sorted(os.listdir(traces_dir)):
    if not fname.endswith('.md'):
        continue
    
    print(f"\n{'='*60}")
    print(f"EVALUATING: {fname}")
    
    with open(os.path.join(traces_dir, fname), 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract user messages in order
    user_messages = re.findall(
        r'\*\*User\*\*\s*\n?\s*>\s*(.+?)(?=\n\*\*Agent\*\*|\n###|\n_|\Z)', 
        content, 
        re.DOTALL
    )
    
    messages = []
    final_recs = []
    turns_taken = 0
    
    for i, user_msg in enumerate(user_messages):
        user_msg = user_msg.strip()
        # Clean special characters
        user_msg = user_msg.replace('\u2019', "'").replace('\u2018', "'")
        user_msg = user_msg.replace('\u201c', '"').replace('\u201d', '"')
        user_msg = user_msg.replace('\u2013', '-').replace('\u2014', '--')
        user_msg = user_msg.replace('\u2026', '...')
        user_msg = user_msg.replace('\n', ' ').strip()
        user_msg = ' '.join(user_msg.split())
        
        messages.append({"role": "user", "content": user_msg})
        turns_taken = i + 1
        
        print(f"  Turn {i+1}: {user_msg[:80]}...")
        
        try:
            r = requests.post(
                f"{BASE_URL}/chat",
                json=messages,
                timeout=120
            )
            
            if r.status_code == 422:
                print(f"    ❌ Validation error: {r.text[:150]}")
                continue
            
            if r.status_code != 200:
                print(f"    ❌ HTTP {r.status_code}")
                continue
            
            data = r.json()
            messages.append({"role": "assistant", "content": data['reply']})
            
            if data.get('recommendations'):
                final_recs = data['recommendations']
                names = [r['name'][:50] for r in final_recs[:5]]
                print(f"    📋 {len(final_recs)} recommendations: {names}")
            
            if data.get('end_of_conversation'):
                print(f"    🏁 Agent ended conversation")
                break
                
        except requests.Timeout:
            print(f"    ⏰ Timeout (cold start?)")
            break
        except Exception as e:
            print(f"    ❌ Error: {e}")
            break
    
    # Calculate recall
    expected_set = set(expected.get(fname, []))
    actual_set = set(r['name'] for r in final_recs) if final_recs else set()
    
    if expected_set:
        matched = expected_set & actual_set
        recall = len(matched) / len(expected_set)
        total_recall += recall
        trace_count += 1
        
        all_results[fname] = {
            'recall': round(recall, 3),
            'turns': turns_taken,
            'expected_count': len(expected_set),
            'matched_count': len(matched),
            'expected': list(expected_set),
            'got': list(actual_set),
            'matched': list(matched),
            'missing': list(expected_set - actual_set),
            'extra': list(actual_set - expected_set)[:5],
        }
        
        print(f"\n  📊 Recall: {recall:.1%} ({len(matched)}/{len(expected_set)})")
        if matched:
            print(f"  ✅ Matched: {[m[:60] for m in matched]}")
        if expected_set - actual_set:
            print(f"  ❌ Missing: {[m[:60] for m in list(expected_set - actual_set)]}")

# Summary
mean_recall = total_recall / trace_count if trace_count > 0 else 0
print(f"\n{'='*60}")
print(f"📊 FINAL RESULTS")
print(f"   Mean Recall@10: {mean_recall:.1%}")
print(f"   Traces evaluated: {trace_count}/10")

# Per-trace breakdown
print(f"\n📋 PER-TRACE BREAKDOWN:")
for fname, result in sorted(all_results.items()):
    bar = '█' * int(result['recall'] * 20) + '░' * (20 - int(result['recall'] * 20))
    print(f"   {fname}: {result['recall']:.0%} {bar} ({result['matched_count']}/{result['expected_count']})")

# Save detailed results
with open('data/recall_results.json', 'w') as f:
    json.dump({
        'mean_recall': round(mean_recall, 3),
        'traces_evaluated': trace_count,
        'details': all_results
    }, f, indent=2)

print(f"\n✅ Detailed results saved to data/recall_results.json")