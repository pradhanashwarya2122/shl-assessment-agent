# save as check_traces.py
import os

traces_dir = 'data/traces'
for fname in sorted(os.listdir(traces_dir))[:2]:  # Just first 2
    fpath = os.path.join(traces_dir, fname)
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"\n{'='*50}")
    print(f"FILE: {fname} ({len(content)} chars)")
    print(f"FIRST 600 chars:\n{content[:600]}")
    print(f"\nLAST 400 chars:\n{content[-400:]}")