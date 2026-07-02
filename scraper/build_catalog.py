# scraper/build_catalog.py
import json
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin

class SHLScraper:
    def __init__(self, html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            self.soup = BeautifulSoup(f.read(), 'lxml')
        self.base_url = "https://www.shl.com"
        self.assessments = []
        
    def is_individual_test(self, product):
        """Filter out pre-packaged job solutions"""
        # Check for keywords indicating individual test
        text = product.get_text().lower()
        exclude_keywords = ['job solution', 'pre-packaged', 'complete package', 'bundle']
        return not any(kw in text for kw in exclude_keywords)
    
    def extract_test_type(self, product):
        """Extract SHL test type code (K, P, B, S, A, etc.)"""
        text = product.get_text().upper()
        
        type_mapping = {
            'KNOWLEDGE': 'K',
            'PERSONALITY': 'P',
            'BEHAVIORAL': 'B',
            'SITUATIONAL': 'S',
            'ABILITY': 'A',
            'COGNITIVE': 'C',
            'APTITUDE': 'A',
            'DEVELOPMENT': 'D'
        }
        
        for key, code in type_mapping.items():
            if key in text:
                return code
        return 'K'  # default to Knowledge
    
    def scrape(self):
        """Main scraping logic - CUSTOMIZE BASED ON ACTUAL HTML"""
        # Try multiple selectors based on SHL's structure
        product_selectors = [
            'tr.catalog-row',
            'div.product-card',
            'li.assessment-item',
            'div[class*="product"]',
            'article.assessment'
        ]
        
        products = []
        for selector in product_selectors:
            products = self.soup.select(selector)
            if products:
                break
        
        if not products:
            # Fallback: try to find any table rows or list items
            products = self.soup.find_all(['tr', 'li', 'div'], 
                                         class_=re.compile(r'product|assessment|item'))
        
        for product in products:
            if not self.is_individual_test(product):
                continue
                
            # Extract name - try common patterns
            name = None
            for tag in ['h3', 'h4', 'a', 'td']:
                name_elem = product.find(tag, class_=re.compile(r'name|title|heading'))
                if name_elem:
                    name = name_elem.get_text(strip=True)
                    break
            
            if not name:
                continue
                
            # Extract URL
            url = None
            link = product.find('a', href=True)
            if link:
                url = urljoin(self.base_url, link['href'])
            
            if not url:
                continue
            
            # Extract description
            desc_elem = product.find(['p', 'div', 'span'], 
                                    class_=re.compile(r'desc|summary|text'))
            description = desc_elem.get_text(strip=True)[:500] if desc_elem else ""
            
            # Extract additional fields
            duration = self.extract_field(product, r'(\d+)\s*(?:min|minute)')
            languages = self.extract_field(product, r'available in\s+(.+)', multi=True)
            job_level = self.extract_field(product, r'(?:junior|mid|senior|entry|graduate)')
            
            assessment = {
                'name': name,
                'url': url,
                'test_type': self.extract_test_type(product),
                'description': description,
                'duration': duration,
                'languages': languages,
                'job_level': job_level,
                'remote_testing': 'remote' in description.lower()
            }
            
            self.assessments.append(assessment)
        
        return self.assessments
    
    def extract_field(self, product, pattern, multi=False):
        """Extract specific field using regex"""
        text = product.get_text()
        matches = re.findall(pattern, text, re.IGNORECASE)
        if multi:
            return matches
        return matches[0] if matches else None
    
    def save(self, path='data/assessments.json'):
        """Save parsed catalog"""
        with open(path, 'w') as f:
            json.dump(self.assessments, f, indent=2)
        
        # Validation
        names = [a['name'] for a in self.assessments]
        urls = [a['url'] for a in self.assessments]
        
        print(f"✅ Scraped {len(self.assessments)} individual test solutions")
        print(f"Sample names: {names[:5]}")
        print(f"Unique test types: {set(a['test_type'] for a in self.assessments)}")
        
        return len(self.assessments)

# Run it
scraper = SHLScraper('data/catalog_raw.html')
scraper.scrape()
scraper.save()