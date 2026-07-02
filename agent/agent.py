# agent/agent.py
import json
import re
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler

class SHLAgent:
    def __init__(self, catalog_path='data/product_catalog_fixed.json'):
        # Load catalog
        with open(catalog_path, 'r', encoding='utf-8') as f:
            raw_catalog = json.load(f)
        
        # Normalize catalog: ensure 'url' field exists (from 'link')
        self.catalog = []
        for item in raw_catalog:
            normalized = {
                'name': item.get('name', ''),
                'url': item.get('link', item.get('url', '')),
                'test_type': self._derive_test_type(item),
                'description': item.get('description', ''),
                'job_levels': item.get('job_levels', []),
                'languages': item.get('languages', []),
                'duration': item.get('duration', ''),
                'remote': item.get('remote', False),
                'keys': item.get('keys', []),
            }
            self.catalog.append(normalized)
        
        # Build indices
        self._build_indices()
        
        # Off-topic patterns
        self.off_topic = [
            (r'(?i)(ignore|forget|override).*(instruction|prompt)', 'injection'),
            (r'(?i)(legal|lawyer|sue|attorney)', 'legal'),
            (r'(?i)(salary|compensation|pay|bonus|benefits)', 'compensation'),
            (r'(?i)(interview.question|how.to.interview|recruiting.process)', 'hiring_advice'),
            (r'(?i)(other.company|other.vendor|not.SHL)', 'competitor'),
        ]
        
        print(f"✅ Agent ready: {len(self.catalog)} assessments loaded")
    
    def _derive_test_type(self, item):
        """Derive test type code from catalog data"""
        keys = item.get('keys', [])
        if isinstance(keys, str):
            keys = [keys]
        keys_str = ' '.join(keys).lower()
        desc = item.get('description', '').lower()
        name = item.get('name', '').lower()
        combined = f"{keys_str} {desc} {name}"
        
        if any(w in combined for w in ['personality', 'behavior', 'behaviour', 'opq']):
            return 'P'
        if any(w in combined for w in ['cognitive', 'ability', 'aptitude', 'verify', 'g+']):
            return 'A'
        if any(w in combined for w in ['situational', 'judgment', 'scenario', 'sjt']):
            return 'S'
        if any(w in combined for w in ['knowledge', 'skill', 'technical', 'coding', 'java', 'python', 'programming']):
            return 'K'
        if any(w in combined for w in ['biodata', 'biographical']):
            return 'B'
        if any(w in combined for w in ['development', '360', 'report']):
            return 'D'
        return 'K'  # default
    
    def _build_indices(self):
        """Build BM25 + embedding indices"""
        self.documents = []
        for item in self.catalog:
            doc = f"{item['name']} {item['description']} {' '.join(item.get('keys', []))} {' '.join(item.get('job_levels', []))}"
            self.documents.append(doc.lower())
        
        # BM25
        tokenized = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        
        # Embeddings (use smaller model for speed)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(self.documents, show_progress_bar=False)
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
    
    def process(self, messages):
        """Process conversation, return {reply, recommendations, end_of_conversation}"""
        # Get last user message
        last_user = ''
        for m in reversed(messages):
            if m.get('role') == 'user':
                last_user = m.get('content', '')
                break
        
        if not last_user:
            return self._respond("How can I help you find the right SHL assessment?", [], False)
        
        # 1. Check off-topic
        refusal = self._check_off_topic(last_user)
        if refusal:
            return refusal
        
        # 2. Check comparison request
        comp = self._check_comparison(last_user)
        if comp:
            return self._handle_comparison(comp)
        
        # 3. Extract facts from ALL user messages
        facts = self._extract_facts(messages)
        
        # 4. Decide: clarify or recommend
        if self._can_recommend(facts):
            recs = self._search(facts)
            reply = self._build_reply(facts, recs)
            return self._respond(reply, recs, True)
        else:
            question = self._clarify(facts)
            return self._respond(question, [], False)
    
    def _extract_facts(self, messages):
        """Extract facts from all user messages"""
        all_text = ' '.join([m.get('content', '') for m in messages if m.get('role') == 'user']).lower()
        
        facts = {'role': None, 'skills': [], 'test_types': [], 'seniority': None, 'remote': False}
        
        # Role
        role_patterns = [
            r'(?:hiring|need|seeking|for)\s+(?:a|an)?\s*([^.!?]{5,40}?(?:developer|engineer|manager|analyst|tester|designer|lead|architect))',
            r'(java|python|javascript|react|angular|node|\.net)\s*(?:developer|engineer)',
            r'(senior|junior|mid.level|graduate)\s+(\w+(?:\s+\w+){0,2})',
        ]
        for pat in role_patterns:
            m = re.search(pat, all_text)
            if m:
                facts['role'] = m.group(0).strip()
                break
        
        # Skills
        skill_words = ['java', 'python', 'javascript', 'typescript', 'react', 'angular', 'vue', 'node', 'sql', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'microservices', 'api', 'agile', 'scrum', 'stakeholder', 'communication', 'leadership', 'cloud', 'devops', 'machine learning', 'ai', 'data science']
        facts['skills'] = list(set(s for s in skill_words if s in all_text))
        
        # Test types
        if any(w in all_text for w in ['personality', 'behavioral', 'behaviour', 'soft skill', 'behavior']):
            facts['test_types'].append('P')
        if any(w in all_text for w in ['technical', 'coding', 'programming', 'knowledge', 'skill test', 'technical skill']):
            facts['test_types'].append('K')
        if any(w in all_text for w in ['cognitive', 'aptitude', 'reasoning', 'ability', 'intelligence', 'verify']):
            facts['test_types'].append('A')
        if any(w in all_text for w in ['situational', 'judgment', 'sjt', 'scenario']):
            facts['test_types'].append('S')
        
        # Seniority
        if any(w in all_text for w in ['senior', 'lead', 'principal', 'architect', 'head', 'cxo', 'director', '15', '10+']):
            facts['seniority'] = 'senior'
        elif any(w in all_text for w in ['mid', 'intermediate', '4', '5', '6', '7', '8', '9']):
            facts['seniority'] = 'mid'
        elif any(w in all_text for w in ['junior', 'entry', 'graduate', 'fresher', 'trainee', '0-3', '1', '2', '3']):
            facts['seniority'] = 'junior'
        
        # Remote
        facts['remote'] = 'remote' in all_text
        
        return facts
    
    def _can_recommend(self, facts):
        """Check if we have enough to recommend"""
        has_role_or_skill = bool(facts['role'] or facts['skills'])
        has_test = bool(facts['test_types'])
        return has_role_or_skill and has_test
    
    def _search(self, facts, k=10):
        """Search catalog"""
        query_parts = []
        if facts['role']:
            query_parts.append(facts['role'])
        if facts['skills']:
            query_parts.extend(facts['skills'][:5])
        if facts['seniority']:
            query_parts.append(facts['seniority'])
        
        type_map = {'P': 'personality behavioral', 'K': 'technical knowledge skills', 'A': 'cognitive ability aptitude', 'S': 'situational judgment scenario'}
        for t in facts['test_types']:
            if t in type_map:
                query_parts.append(type_map[t])
        
        query = ' '.join(query_parts).lower().strip()
        if not query:
            return []
        
        # BM25
        tok_query = query.split()
        bm25_scores = self.bm25.get_scores(tok_query)
        
        # Embedding
        q_emb = self.model.encode([query])
        q_emb = q_emb / np.linalg.norm(q_emb)
        emb_scores = np.dot(q_emb, self.embeddings.T)[0]
        
        # Normalize & combine
        scaler = MinMaxScaler()
        bm25_norm = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
        emb_norm = scaler.fit_transform(emb_scores.reshape(-1, 1)).flatten()
        
        combined = 0.5 * bm25_norm + 0.5 * emb_norm
        
        # Get top results, filter by test type
        top_idx = np.argsort(combined)[::-1]
        results = []
        for idx in top_idx:
            if combined[idx] < 0.05:
                continue
            item = self.catalog[idx]
            if facts['test_types'] and item['test_type'] not in facts['test_types']:
                # Still include if it's a close match
                if combined[idx] < 0.3:
                    continue
            results.append({'name': item['name'], 'url': item['url'], 'test_type': item['test_type']})
            if len(results) >= k:
                break
        
        return results
    
    def _clarify(self, facts):
        """Generate clarifying question"""
        if not facts['role'] and not facts['skills']:
            return "What role or skills are you looking to assess?"
        if not facts['test_types']:
            return "What type of assessment do you need? (e.g., technical skills, personality, cognitive ability)"
        return "Could you share more about the role, such as seniority level or specific skills required?"
    
    def _build_reply(self, facts, recs):
        """Build natural reply"""
        parts = []
        if facts['role']:
            parts.append(f"for {facts['role']}")
        if facts['seniority']:
            parts.append(f"at {facts['seniority']} level")
        ctx = ' '.join(parts) if parts else 'matching your criteria'
        return f"Here are {len(recs)} SHL assessments {ctx}:"
    
    def _check_off_topic(self, msg):
        """Check if message is off-topic"""
        for pattern, reason in self.off_topic:
            if re.search(pattern, msg):
                replies = {
                    'injection': "I can only help with SHL assessment recommendations.",
                    'legal': "I cannot provide legal advice. I help find SHL assessments.",
                    'compensation': "I focus on assessments, not compensation.",
                    'hiring_advice': "I help with assessment selection, not hiring processes.",
                    'competitor': "I only work with SHL's catalog.",
                }
                return self._respond(replies.get(reason, "I can only help with SHL assessments."), [], False)
        return None
    
    def _check_comparison(self, msg):
        """Check if comparison request"""
        patterns = [
            r'(?:compare|difference|diff)\s+(?:between\s+)?(.+?)\s+(?:and|vs\.?|versus)\s+(.+)',
            r'(.+?)\s+vs\.?\s+(.+)',
        ]
        for pat in patterns:
            m = re.search(pat, msg, re.IGNORECASE)
            if m:
                return (m.group(1).strip(), m.group(2).strip())
        return None
    
    def _handle_comparison(self, names):
        """Compare two assessments"""
        a1 = next((a for a in self.catalog if names[0].lower() in a['name'].lower()), None)
        a2 = next((a for a in self.catalog if names[1].lower() in a['name'].lower()), None)
        if not a1 or not a2:
            return self._respond("I can only compare assessments from the SHL catalog. Please verify the names.", [], False)
        reply = f"**{a1['name']}**: {a1.get('description','')[:200]}\n**{a2['name']}**: {a2.get('description','')[:200]}\nBoth are SHL Individual Test Solutions."
        return self._respond(reply, [], False)
    
    def _respond(self, reply, recommendations, end):
        """Standard response"""
        return {'reply': reply, 'recommendations': recommendations, 'end_of_conversation': end}