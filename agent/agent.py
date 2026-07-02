import json, re, os, numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler


_TITLE_SUFFIXES = (
    'developer|engineer|architect|analyst|manager|tester|designer|lead|programmer|'
    'agent|operator|assistant|staff|representative|nurse|technician|coordinator|'
    'specialist|consultant|associate|officer|clerk|advisor|supervisor|executive|'
    'scientist|researcher|administrator|director|recruiter|trainer|therapist|mechanic'
)


class SHLAgent:
    def __init__(self, catalog_path='data/product_catalog_fixed.json'):
        with open(catalog_path, 'r', encoding='utf-8') as f:
            raw_catalog = json.load(f)

        self.catalog = []
        for item in raw_catalog:
            self.catalog.append({
                'name': item.get('name', ''),
                'url': item.get('link', item.get('url', '')),
                'test_type': self._derive_test_type(item),
                'description': item.get('description', ''),
                'job_levels': item.get('job_levels', []) if isinstance(item.get('job_levels'), list) else [],
                'languages': item.get('languages', []) if isinstance(item.get('languages'), list) else [],
                'duration': str(item.get('duration', '')),
                'remote': item.get('remote', False),
                'keys': item.get('keys', []) if isinstance(item.get('keys'), list) else [],
            })

        self._build_indices()

        self.type_thresholds = {
            'K': {'floor': 0.12, 'ratio': 0.35},
            'P': {'floor': 0.06, 'ratio': 0.18},
            'A': {'floor': 0.07, 'ratio': 0.20},
            'S': {'floor': 0.07, 'ratio': 0.20},
            'B': {'floor': 0.08, 'ratio': 0.22},
            None: {'floor': 0.10, 'ratio': 0.30},
        }

        self.off_topic = [
            (r'(?i)(ignore|forget|override|disregard).*(instruction|prompt|rule|system)', 'injection'),
            (r'(?i)(legal|lawyer|sue|attorney|compliance|regulation)', 'legal'),
            (r'(?i)(salary|compensation|pay\s|bonus|benefits|package)', 'compensation'),
            (r'(?i)(interview.question|how.to.interview|recruiting.process|hire.*process)', 'hiring_advice'),
            (r'(?i)(other.company|other.vendor|not.SHL|different.provider)', 'competitor'),
            (r'(?i)(hack|exploit|cheat|bypass|manipulate.test)', 'security'),
        ]

        print(f"✅ Agent ready: {len(self.catalog)} assessments")

    def _derive_test_type(self, item):
        keys = item.get('keys', [])
        if isinstance(keys, str):
            keys = [keys]
        keys_str = ' '.join(keys).lower()
        desc = str(item.get('description', '')).lower()
        name = str(item.get('name', '')).lower()
        combined = f"{keys_str} {desc} {name}"

        if any(w in combined for w in ['personality', 'behavioral', 'behaviour', 'opq']):
            return 'P'
        if any(w in combined for w in ['cognitive', 'ability', 'aptitude', 'verify g+', 'verify interactive']):
            return 'A'
        if any(w in combined for w in ['situational', 'judgment', 'scenario', 'sjt']):
            return 'S'
        if any(w in combined for w in ['knowledge', 'skill', 'technical', 'coding', 'java', 'python',
                                        'programming', 'development', 'framework', 'design pattern',
                                        'web service', 'enterprise']):
            return 'K'
        if any(w in combined for w in ['biodata', 'biographical']):
            return 'B'
        return 'K'

    def _build_indices(self):
        self.documents = []
        for item in self.catalog:
            doc_parts = [item['name'], item['description'], ' '.join(item.get('keys', []))]
            if item.get('job_levels'):
                doc_parts.append(' '.join(item['job_levels']))
            self.documents.append(' '.join(doc_parts).lower())

        tokenized = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)

        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(self.documents, show_progress_bar=False)
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

    # ==================================================================
    # MAIN ENTRY POINT
    # ==================================================================
    def process(self, messages):
        user_turns = sum(1 for m in messages if m.get('role') == 'user')

        if user_turns > 8:
            facts = self._extract_facts(messages)
            recs = self._search(facts)
            return self._respond("Here are my final recommendations based on our discussion.", recs[:10], True)

        last_user = ''
        for m in reversed(messages):
            if m.get('role') == 'user':
                last_user = m.get('content', '')
                break

        if not last_user:
            return self._respond("How can I help you find the right SHL assessment?", [], False)

        refusal = self._check_off_topic(last_user)
        if refusal:
            return refusal

        comp = self._check_comparison(last_user)
        if comp:
            return self._handle_comparison(comp)

        facts = self._extract_facts(messages)

        last_lower = last_user.lower()
        is_refinement = any(w in last_lower for w in
                             ['add', 'also include', 'also add', 'instead', 'remove',
                              'actually', 'change', 'update', 'no,', 'wait'])

        no_pref = any(p in last_lower for p in [
            'no preference', "don't know", 'not sure', 'whatever', 'any',
            'no idea', 'i have no', 'no specific', 'not particular', 'up to you',
        ])

        has_role_or_skill = bool(facts['role'] or facts['skills'])
        has_test_type = bool(facts['test_types'])
        is_job_desc = len(last_user) > 200

        can_recommend = (
            (has_role_or_skill and has_test_type)
            or (is_job_desc and has_role_or_skill)
            or (user_turns >= 3 and has_role_or_skill)
            or (no_pref and has_role_or_skill)
        )

        if can_recommend:
            if no_pref and not has_test_type:
                facts['test_types'] = ['K']
            recs = self._search(facts)
            reply = self._build_reply(facts, recs, is_refinement)
            return self._respond(reply, recs, True)
        elif has_role_or_skill and not has_test_type:
            return self._respond(
                "What type of assessment are you looking for? For example, technical skills, "
                "personality, cognitive ability, or a combination?", [], False)
        else:
            question = self._clarify(facts)
            return self._respond(question, [], False)

    # ==================================================================
    # FACT EXTRACTION (EXPANDED)
    # ==================================================================
    def _extract_facts(self, messages):
        all_text = ' '.join(
            [m.get('content', '') for m in messages if m.get('role') == 'user']
        ).lower()

        facts = {
            'role': None, 'skills': [], 'test_types': [],
            'seniority': None, 'remote': False, 'years': None,
        }

        role_patterns = [
            rf'hiring\s+(?:a|an)?\s*([^.!?]{{3,60}}?(?:{_TITLE_SUFFIXES}))',
            rf'(?:need|seeking|looking for|for)\s+(?:a|an)?\s*([^.!?]{{3,60}}?(?:{_TITLE_SUFFIXES}))',
            rf'screen(?:ing)?\s+(?:\d+\s+)?(?:entry[- ]level\s+)?([^.!?]{{3,60}}?(?:{_TITLE_SUFFIXES}))',
            rf'(java|python|javascript|react|angular|node|\.net|ruby|golang|rust|php)\s*(?:developer|engineer|programmer)',
        ]
        for pat in role_patterns:
            m = re.search(pat, all_text)
            if m:
                role_text = m.group(0).strip()
                role_text = re.sub(
                    r'^(?:hiring|need|seeking|looking for|for|screen(?:ing)?)\s+(?:\d+\s+)?(?:a|an)?\s*(?:entry[- ]level\s+)?',
                    '', role_text,
                )
                facts['role'] = role_text.strip()
                break

        if not facts['role']:
            skill_roles = {
                'java': 'Java Developer', 'python': 'Python Developer',
                'javascript': 'JavaScript Developer', 'react': 'React Developer',
                'angular': 'Angular Developer', 'node': 'Node.js Developer',
            }
            for skill, role_name in skill_roles.items():
                if skill in all_text:
                    facts['role'] = role_name
                    break

        if not facts['role']:
            domain_role_map = [
                (r'contact cent(?:re|er)|call cent(?:re|er)|inbound call', 'contact centre agent'),
                (r'customer service', 'customer service agent'),
                (r'plant operator|chemical facility|process operator|manufacturing', 'plant operator'),
                (r'admin(?:istrative)?\s+assistant|clerical|office admin', 'admin assistant'),
                (r'health(?:care)?|medical|clinical|hospital|nurse|hipaa|patient', 'healthcare staff'),
                (r'sales\b|account\s+executive|business development', 'sales representative'),
                (r'warehouse|logistics|supply chain|forklift', 'warehouse operative'),
                (r'graduate|campus|fresher|trainee|entry.level|recent grad', 'graduate'),
                (r'financial|finance|accounting|banking|audit', 'financial analyst'),
                (r'rust\s+(?:developer|engineer|programmer)', 'Rust developer'),
                (r'full.stack|fullstack', 'full stack developer'),
                (r'frontend|front.end|ui\s+developer', 'frontend developer'),
                (r'backend|back.end|api\s+developer', 'backend developer'),
                (r'data\s+(?:scientist|analyst|engineer)', 'data professional'),
                (r'project\s+manager|programme\s+manager|scrum\s+master', 'project manager'),
                (r'security|cyber|infosec|penetration', 'security professional'),
                (r'operator|technician|mechanic|electrician', 'technical operator'),
            ]
            for pattern, role_name in domain_role_map:
                if re.search(pattern, all_text):
                    facts['role'] = role_name
                    break

        skill_words = [
            'java', 'python', 'javascript', 'typescript', 'react', 'angular', 'vue', 'node',
            'sql', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'microservices', 'api',
            'rest', 'graphql', 'agile', 'scrum', 'cloud', 'devops', 'machine learning',
            'ai', 'data', 'spring', 'hibernate', 'git', 'spring boot',
            'stakeholder', 'communication', 'leadership',
            'excel', 'word', 'powerpoint', 'outlook', 'ms office', 'microsoft office',
            'safety', 'dependability', 'customer service', 'call center', 'contact center',
            'finance', 'accounting', 'banking', 'numerical', 'statistics',
            'healthcare', 'medical', 'hipaa', 'clinical', 'patient',
            'sales', 'retail', 'selling', 'account management',
            'rust', 'c++', 'c#', 'golang', 'php', 'ruby', 'swift', 'kotlin',
            'networking', 'linux', 'unix', 'windows', 'infrastructure',
            'full stack', 'frontend', 'backend', 'mobile', 'web',
            'bilingual', 'spanish', 'french', 'german', 'language',
            'writing', 'reading', 'listening', 'speaking', 'comprehension',
            'problem solving', 'critical thinking', 'analytical',
            'project management', 'time management', 'organization',
        ]
        facts['skills'] = list(set(s for s in skill_words if s in all_text))

        if any(w in all_text for w in ['personality', 'behavioral', 'behaviour',
                                        'soft skill', 'behavior', 'behavioural']):
            facts['test_types'].append('P')
        if any(w in all_text for w in ['technical', 'coding', 'programming', 'knowledge',
                                        'skill test', 'technical skill', 'hard skill', 'simulation']):
            facts['test_types'].append('K')
        if any(w in all_text for w in ['cognitive', 'aptitude', 'reasoning', 'ability',
                                        'intelligence', 'verify', 'mental']):
            facts['test_types'].append('A')
        if any(w in all_text for w in ['situational', 'judgment', 'sjt', 'scenario']):
            facts['test_types'].append('S')

        if any(w in all_text for w in ['senior', 'lead', 'principal', 'architect', 'head', 'cxo',
                                        'director', 'vp', '15+', '10+', '15 years', '10 years',
                                        '8 years']):
            facts['seniority'] = 'senior'
        elif any(w in all_text for w in ['mid', 'intermediate', '4 year', '5 year', '6 year',
                                          '7 year', '8 year', '9 year']):
            facts['seniority'] = 'mid'
        elif any(w in all_text for w in ['junior', 'entry', 'graduate', 'fresher', 'trainee',
                                          '0-3', '1 year', '2 year', '3 year']):
            facts['seniority'] = 'junior'

        exp_match = re.search(r'(\d+)[\+]?\s*(?:years|yrs|year)', all_text)
        if exp_match:
            facts['years'] = int(exp_match.group(1))
        facts['remote'] = 'remote' in all_text

        assistant_text = ' '.join(
            [m.get('content', '') for m in messages if m.get('role') == 'assistant']
        ).lower()

        if not facts['test_types']:
            if any(w in assistant_text for w in ['personality', 'behavioral', 'opq']):
                facts['test_types'].append('P')
            if any(w in assistant_text for w in ['cognitive', 'ability', 'aptitude', 'verify']):
                facts['test_types'].append('A')
            if any(w in assistant_text for w in ['knowledge', 'skill', 'technical', 'simulation']):
                facts['test_types'].append('K')

        if not facts['role']:
            role_from_assistant = re.search(
                rf'(?:for|hiring|role|position|screening)\s+([^.!?]{{5,60}}?(?:{_TITLE_SUFFIXES}))',
                assistant_text,
            )
            if role_from_assistant:
                facts['role'] = role_from_assistant.group(1).strip()

        return facts

    # ==================================================================
    # QUERY & SEARCH (WITH NAME MATCHING BOOST)
    # ==================================================================
    def _build_query(self, facts, type_hint=None):
        parts = []
        if facts.get('role'):
            parts.append(facts['role'])
            parts.extend(facts['role'].lower().split())
        if type_hint == 'K' and facts.get('skills'):
            parts.extend(facts['skills'][:5])
        if facts.get('seniority'):
            parts.append(facts['seniority'])

        role_synonyms = {
            'contact centre agent':  'call center customer service phone inbound entry level',
            'contact centre':        'call center customer service phone inbound',
            'contact center':        'call center customer service phone inbound',
            'customer service agent': 'call center customer service phone entry level',
            'admin assistant':       'administrative clerical office ms excel word',
            'administrative assistant': 'administrative clerical office ms excel word',
            'plant operator':        'manufacturing industrial safety chemical dependability',
            'healthcare staff':      'medical hipaa patient clinical hospital',
            'healthcare':            'medical hipaa patient clinical hospital',
            'graduate':              'entry level campus fresh graduate trainee',
            'sales representative':  'selling account business development opq mq',
            'sales':                 'selling retail account business development',
            'financial analyst':     'finance accounting banking numerical statistics',
            'rust developer':        'rust systems programming linux infrastructure',
            'full stack developer':  'full stack web frontend backend javascript',
            'frontend developer':    'frontend ui react angular javascript web',
            'backend developer':     'backend api server database sql microservices',
            'data professional':     'data science analytics machine learning sql statistics',
            'project manager':       'project management agile scrum leadership stakeholder',
            'security professional': 'security cybersecurity infosec network protection',
            'technical operator':    'operator technician maintenance repair industrial',
        }
        if facts.get('role'):
            role_lower = facts['role'].lower()
            for key, synonyms in role_synonyms.items():
                if key in role_lower:
                    parts.append(synonyms)
                    break

        type_map = {
            'P': 'personality behavioral',
            'K': 'technical knowledge skills',
            'A': 'cognitive ability aptitude',
            'S': 'situational judgment scenario',
        }
        if type_hint:
            parts.append(type_map.get(type_hint, ''))
        else:
            for t in facts.get('test_types', []):
                parts.append(type_map.get(t, ''))

        query = ' '.join(p for p in parts if p).lower().strip()
        if not query:
            query = ' '.join(facts.get('skills', [])).lower().strip()
        return query

    def _score_all(self, query):
        if not query:
            return []
        tok_query = query.split()
        bm25_scores = self.bm25.get_scores(tok_query)
        q_emb = self.model.encode([query])
        q_emb = q_emb / np.linalg.norm(q_emb)
        emb_scores = np.dot(q_emb, self.embeddings.T)[0]

        scaler = MinMaxScaler()
        bm25_norm = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
        emb_norm = scaler.fit_transform(emb_scores.reshape(-1, 1)).flatten()
        combined = 0.5 * bm25_norm + 0.5 * emb_norm

        query_tokens = set(tok_query)
        for idx, item in enumerate(self.catalog):
            item_keys = ' '.join(item.get('keys', [])).lower()
            key_tokens = set(item_keys.split())
            exact_key_matches = query_tokens & key_tokens

            name_tokens = set(item['name'].lower().split())
            exact_name_matches = query_tokens & name_tokens

            if exact_key_matches:
                combined[idx] += 0.15 * len(exact_key_matches)
            if exact_name_matches:
                combined[idx] += 0.1 * len(exact_name_matches)

        scored = []
        for idx, item in enumerate(self.catalog):
            scored.append({
                'name': item['name'], 'url': item['url'],
                'test_type': item['test_type'], 'score': float(combined[idx]),
            })
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored

    def _get_threshold_config(self, test_type=None):
        return self.type_thresholds.get(test_type, self.type_thresholds[None])

    def _dynamic_threshold(self, scored_list, test_type=None):
        config = self._get_threshold_config(test_type)
        floor, ratio = config['floor'], config['ratio']
        if not scored_list:
            return floor
        return max(floor, scored_list[0]['score'] * ratio)

    def _filter_relevant(self, scored_list, test_type=None):
        threshold = self._dynamic_threshold(scored_list, test_type=test_type)
        filtered = [s for s in scored_list if s['score'] >= threshold]
        if not filtered and scored_list:
            filtered = scored_list[:1]
        return filtered

    def _search(self, facts, k=10):
        requested_types = list(facts.get('test_types') or [])
        if not requested_types and facts.get('skills'):
            requested_types = ['K']

        if len(requested_types) <= 1:
            type_hint = requested_types[0] if requested_types else None
            query = self._build_query(facts, type_hint=type_hint)
            scored = self._score_all(query)
            relevant = self._filter_relevant(scored, test_type=type_hint)
            return [{'name': r['name'], 'url': r['url'], 'test_type': r['test_type']}
                    for r in relevant[:k]]

        per_type_k = max(2, k // len(requested_types))
        results, seen_names = [], set()

        role_keywords = set()
        if facts.get('role'):
            role_keywords.update(facts['role'].lower().split())
        if facts.get('skills'):
            role_keywords.update(facts['skills'])
        generic_words = {'developer', 'engineer', 'a', 'an', 'the', 'for', 'with',
                         'and', 'or', 'in', 'of', 'to', 'is', 'on', 'at', 'by'}
        role_keywords -= generic_words

        for t in requested_types:
            query = self._build_query(facts, type_hint=t)
            scored = self._score_all(query)
            type_scored = [s for s in scored if s['test_type'] == t]
            relevant = self._filter_relevant(type_scored, test_type=t)

            if t != 'K' and role_keywords:
                def role_relevance(item):
                    cat_item = next((c for c in self.catalog if c['name'] == item['name']), None)
                    if not cat_item:
                        return 0
                    text = (cat_item['name'] + ' ' + cat_item['description']).lower()
                    matches = sum(1 for kw in role_keywords if kw in text)
                    universal = any(w in cat_item['name'].lower() for w in
                                   ['opq', 'mq', 'personality questionnaire', 'verify',
                                    'cognitive', 'leadership'])
                    return matches + (2 if universal else 0)
                relevant.sort(key=role_relevance, reverse=True)

            for item in relevant[:per_type_k]:
                if item['name'] not in seen_names:
                    results.append(item)
                    seen_names.add(item['name'])

        if len(results) < k:
            combined_query = self._build_query(facts, type_hint=None)
            scored = self._score_all(combined_query)
            relevant = self._filter_relevant(scored, test_type=None)
            for item in relevant:
                if len(results) >= k:
                    break
                if item['name'] not in seen_names:
                    results.append(item)
                    seen_names.add(item['name'])

        results.sort(key=lambda x: x['score'], reverse=True)
        return [{'name': r['name'], 'url': r['url'], 'test_type': r['test_type']}
                for r in results[:k]]

    # ==================================================================
    # REPLY HELPERS
    # ==================================================================
    def _clarify(self, facts):
        if not facts['role'] and not facts['skills']:
            return "What specific role or skills are you looking to assess?"
        if not facts['test_types']:
            return ("What type of assessment do you need? For example, technical skills, "
                    "personality, cognitive ability, or a combination?")
        return "Could you share more about the role, such as seniority level or specific requirements?"

    def _build_reply(self, facts, recs, is_refinement=False):
        if not recs:
            return "I couldn't find exact matches. Could you broaden your criteria?"
        role_text = facts.get('role', 'your role')
        level_text = f" at {facts['seniority']} level" if facts.get('seniority') else ""
        type_text = ""
        if facts.get('test_types'):
            type_names = {'P': 'personality', 'K': 'knowledge', 'A': 'ability', 'S': 'situational judgment'}
            type_text = " (" + ", ".join(type_names.get(t, t) for t in facts['test_types']) + ")"
        if is_refinement:
            return f"Updated recommendations for {role_text}{level_text}{type_text}:"
        return f"Here are {len(recs)} SHL assessments for {role_text}{level_text}{type_text}:"

    # ==================================================================
    # GUARDRAILS
    # ==================================================================
    def _check_off_topic(self, msg):
        for pattern, reason in self.off_topic:
            if re.search(pattern, msg):
                replies = {
                    'injection':     "I can only help with SHL assessment recommendations.",
                    'legal':         "I cannot provide legal advice. I help find SHL assessments.",
                    'compensation':  "I focus on assessment selection, not compensation.",
                    'hiring_advice': "I help with SHL assessment selection, not hiring processes.",
                    'competitor':    "I only work with SHL's Individual Test Solutions catalog.",
                    'security':      "I can only assist with legitimate assessment selection needs.",
                }
                return self._respond(replies.get(reason, "I can only help with SHL assessments."), [], False)
        return None

    def _check_comparison(self, msg):
        patterns = [
            r'(?:compare|difference|diff)\s+(?:between\s+)?(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)(?:$|\?|\.)',
            r'(.+?)\s+vs\.?\s+(.+?)(?:$|\?|\.)',
        ]
        for pat in patterns:
            m = re.search(pat, msg, re.IGNORECASE)
            if m:
                return (m.group(1).strip().rstrip('?.'), m.group(2).strip().rstrip('?.'))
        return None

    def _handle_comparison(self, names):
        a1 = next((a for a in self.catalog if names[0].lower() in a['name'].lower()), None)
        a2 = next((a for a in self.catalog if names[1].lower() in a['name'].lower()), None)
        if not a1 or not a2:
            return self._respond(
                "I can only compare assessments from the SHL catalog. Please verify the names.",
                [], False,
            )
        reply = (
            f"**{a1['name']}** vs **{a2['name']}**:\n\n"
            f"**{a1['name']}**: Type: {a1['test_type']} | Duration: {a1.get('duration','N/A')} | "
            f"Remote: {'Yes' if a1.get('remote') else 'No'}\n"
            f"**{a2['name']}**: Type: {a2['test_type']} | Duration: {a2.get('duration','N/A')} | "
            f"Remote: {'Yes' if a2.get('remote') else 'No'}\n\n"
            f"Both are Individual Test Solutions from SHL's catalog."
        )
        return self._respond(reply, [], False)

    def _respond(self, reply, recommendations, end):
        return {'reply': reply, 'recommendations': recommendations, 'end_of_conversation': end}