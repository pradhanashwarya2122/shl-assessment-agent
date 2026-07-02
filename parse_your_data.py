# parse_your_data.py - RUN THIS IMMEDIATELY
import json
import os

# 1. Check what files you actually have
print("📁 Files in data directory:")
for root, dirs, files in os.walk('data'):
    for file in files:
        print(f"  {os.path.join(root, file)}")

# 2. Read the catalog HTML structure FIRST
with open('data/catalog_raw.html', 'r', encoding='utf-8') as f:
    html_content = f.read()[:5000]  # First 5000 chars
    print("\n📄 Catalog HTML preview:")
    print(html_content)
    print("\n... (truncated)")

# 3. Analyze trace structure
import glob
trace_files = glob.glob('data/traces/**/*.json', recursive=True)
if not trace_files:
    trace_files = glob.glob('data/traces/*.json')

print(f"\n📊 Found {len(trace_files)} trace files")

if trace_files:
    with open(trace_files[0]) as f:
        first_trace = json.load(f)
    print("\n🔍 First trace structure:")
    print(json.dumps(first_trace, indent=2)[:2000])
    
    # Extract critical info
    print(f"\n  Persona keys: {list(first_trace.get('persona', {}).keys())}")
    print(f"  Conversation turns: {len(first_trace.get('conversation', []))}")
    print(f"  Expected shortlist: {first_trace.get('expected_shortlist', [])}")