# scraper/analyze_catalog.py
from bs4 import BeautifulSoup
import json

with open('data/catalog_raw.html', 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f.read(), 'html.parser')

# IMPORTANT: Inspect the HTML structure FIRST
# Look for individual test solution cards vs pre-packaged job solutions
# Common SHL structure: tables, divs with specific classes

# Print all unique class names to understand structure
classes = set()
for elem in soup.find_all(True):
    if elem.get('class'):
        classes.update(elem.get('class'))
print("Found classes:", sorted(classes)[:50])

# Find product containers - ADJUST THESE SELECTORS
products = soup.find_all('div', class_='product-item')  # Adjust!
print(f"\nFound {len(products)} potential products")

# Check first product structure
if products:
    print("\nFirst product HTML structure:")
    print(products[0].prettify()[:500])