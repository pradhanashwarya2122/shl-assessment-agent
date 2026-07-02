import json

with open('data/product_catalog_fixed.json', 'r', encoding='utf-8') as f:
    catalog = json.load(f)

for item in catalog:
    if 'Verify Interactive Process Monitoring' in item.get('name', ''):
        print(json.dumps(item, indent=2))