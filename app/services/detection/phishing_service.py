"""
KAVACH AI Pro - Advanced Phishing Detection Service
Multi-factor URL and content analysis with 25+ heuristic rules.
Designed to be augmented with XGBoost model when trained weights are available.
"""

import re
import math
import time
import logging
from typing import Dict, Optional, List
from urllib.parse import urlparse, parse_qs

from app.core.config import settings
from app.utils.security import FileValidator
import whois
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _extract_url_features(url: str) -> list:
    """Extract 20 structural features from a URL for XGBoost classification.
    This is a self-contained copy that mirrors training/train_phishing.py:extract_url_features().
    Kept inline to avoid runtime import errors when the training package is not on sys.path.
    """
    import re
    import math
    from urllib.parse import urlparse

    parsed = urlparse(url if '://' in url else f'http://{url}')
    domain = parsed.netloc.lower()
    path = parsed.path.lower()

    features = []
    # 1. URL length
    features.append(len(url))
    # 2. Domain length
    features.append(len(domain))
    # 3. Path length
    features.append(len(path))
    # 4. Number of dots
    features.append(url.count('.'))
    # 5. Number of hyphens
    features.append(url.count('-'))
    # 6. Number of underscores
    features.append(url.count('_'))
    # 7. Number of slashes
    features.append(url.count('/'))
    # 8. Number of query parameters
    features.append(url.count('?') + url.count('&'))
    # 9. Has @ symbol
    features.append(1 if '@' in url else 0)
    # 10. Has IP address
    features.append(1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain.split(':')[0]) else 0)
    # 11. Is HTTPS
    features.append(1 if parsed.scheme == 'https' else 0)
    # 12. Subdomain count
    features.append(max(0, len(domain.split('.')) - 2))
    # 13. Path depth
    features.append(len([p for p in path.split('/') if p]))
    # 14. Number of digits in URL
    features.append(sum(c.isdigit() for c in url))
    # 15. Number of special characters
    features.append(len(re.findall(r'[%~\|\\{}^`\[\]!#$]', url)))
    # 16. URL entropy
    if url:
        freq: dict = {}
        for c in url:
            freq[c] = freq.get(c, 0) + 1
        length = len(url)
        entropy = -sum((count / length) * math.log2(count / length) for count in freq.values())
        features.append(entropy)
    else:
        features.append(0)
    # 17. Has suspicious TLD
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.buzz', '.click', '.link']
    features.append(1 if any(domain.endswith(tld) for tld in suspicious_tlds) else 0)
    # 18. Has non-standard port
    features.append(1 if ':' in domain and not domain.endswith((':443', ':80')) else 0)
    # 19. Contains suspicious keywords
    suspicious_words = ['login', 'verify', 'secure', 'account', 'update', 'confirm', 'banking', 'password']
    features.append(sum(1 for w in suspicious_words if w in url.lower()))
    # 20. Domain has numbers
    domain_name = domain.split('.')[0] if domain else ''
    features.append(sum(c.isdigit() for c in domain_name))

    return features


class PhishingDetector:
    """
    Hybrid phishing detection:
    1. XGBoost ML model (if trained model available) for primary classification
    2. Heuristic rules for explainability and threat breakdown
    """

    def __init__(self):
        self.ml_model = None
        self.advanced_phishing_model = None # Stub for ViT + RoBERTa MultiModal model
        self._advanced_loaded = False
        self._load_model()
        # self._load_advanced_model() is now called lazily inside analyze()
        
    def _load_advanced_model(self):
        """
        Load the Advanced Phishing Model (HuggingFace URL Classifier).
        """
        try:
            from transformers import pipeline
            import torch
            
            model_name = "Eason918/malicious-url-detector-v2"
            logger.info(f"Loading SOTA Phishing URL model: {model_name}...")
            
            device = 0 if torch.cuda.is_available() else -1
            self.advanced_phishing_model = pipeline(
                "text-classification", 
                model=model_name, 
                device=device
            )
            
            logger.info(f"Successfully loaded Advanced Phishing (URL) model from HuggingFace.")
        except Exception as e:
            logger.error(f"Failed to load Advanced Phishing model: {e}")
            self.advanced_phishing_model = None

    def _load_model(self):
        """Load trained XGBoost model if available and cryptographically valid"""
        import os
        model_name = getattr(settings, 'PHISHING_MODEL', 'phishing_xgb.pkl')
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'models', 'pretrained', model_name)
        paths = [
            model_path,
            f'models/pretrained/{model_name}',
            f'../models/pretrained/{model_name}',
            f'../../models/pretrained/{model_name}'
        ]
        for path in paths:
            try:
                if not os.path.exists(path):
                    continue
                
                # Cryptographic check
                expected_hash = getattr(settings, 'PHISHING_MODEL_SHA256', None)
                if not FileValidator.verify_model_hash(path, expected_hash):
                    logger.warning(f"Phishing model {path} failed cryptographic validation. Skipping this path.")
                    continue
                
                import sys
                sys.modules['__main__'].extract_url_features = lambda x: []
                import joblib
                data = joblib.load(path)
                self.ml_model = data.get('model') if isinstance(data, dict) else data
                logger.info(f"Loaded phishing XGBoost model from {path}")
                return
            except Exception as e:
                logger.warning(f"Could not load phishing model from {path}: {e}")
                continue
        logger.warning("No valid trained phishing model found. Activating Degraded Heuristic-Only Mode.")

    # Known suspicious TLDs frequently used in phishing
    SUSPICIOUS_TLDS = {
        '.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.buzz',
        '.club', '.work', '.click', '.link', '.info', '.site', '.online',
        '.icu', '.live', '.rest', '.fit', '.cam'
    }

    # Legitimate brands commonly impersonated
    IMPERSONATED_BRANDS = [
        'paypal', 'apple', 'microsoft', 'google', 'amazon', 'netflix',
        'facebook', 'instagram', 'whatsapp', 'linkedin', 'twitter',
        'chase', 'wellsfargo', 'bankofamerica', 'citibank', 'hsbc',
        'dropbox', 'adobe', 'spotify', 'zoom', 'slack',
        'fedex', 'ups', 'usps', 'dhl', 'irs', 'gov'
    ]

    # Homoglyph mapping (characters that look like Latin letters)
    HOMOGLYPHS = {
        'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y',
        'х': 'x', 'ѕ': 's', 'і': 'i', 'ј': 'j', 'ԁ': 'd', 'ɡ': 'g',
        'ɩ': 'l', 'ν': 'v', 'ω': 'w', '0': 'o', '1': 'l', '!': 'i'
    }

    # Urgency / social engineering keywords
    URGENCY_KEYWORDS = [
        'urgent', 'immediate', 'alert', 'verify now', 'account suspended',
        'action required', 'limited time', 'expires today', 'act now',
        'confirm your identity', 'unauthorized access', 'security alert',
        'unusual activity', 'click here immediately', 'final warning',
        'your account will be', 'verify your account', 'update payment',
        'won a prize', 'congratulations', 'selected winner'
    ]

    AUTHORITY_KEYWORDS = [
        'official notice', 'legal action', 'law enforcement', 'court order',
        'irs', 'tax return', 'government', 'federal', 'compliance',
        'regulatory', 'mandatory', 'investigation'
    ]

    def analyze(self, url: str, content: Optional[str] = None) -> Dict:
        if not self._advanced_loaded:
            self._load_advanced_model()
            self._advanced_loaded = True
            
        start_time = time.time()
        threats: List[str] = []
        scores = {}

        parsed = urlparse(url if '://' in url else f'http://{url}')
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        query = parsed.query

        # === URL STRUCTURAL ANALYSIS ===

        # 1. URL Length (0-8 pts)
        url_len = len(url)
        if url_len > 150:
            scores['url_length'] = 8
            threats.append(f"Extremely long URL ({url_len} chars)")
        elif url_len > 100:
            scores['url_length'] = 5
            threats.append(f"Suspiciously long URL ({url_len} chars)")
        elif url_len > 75:
            scores['url_length'] = 3
        else:
            scores['url_length'] = 0

        # 2. URL Entropy (0-8 pts) - high entropy = random/encoded
        entropy = self._calculate_entropy(url)
        if entropy > 4.5:
            scores['url_entropy'] = 8
            threats.append("High URL entropy (randomized/encoded characters)")
        elif entropy > 3.8:
            scores['url_entropy'] = 4
        else:
            scores['url_entropy'] = 0

        # 3. @ symbol in URL (0-10 pts)
        if '@' in url:
            scores['at_symbol'] = 10
            threats.append("URL contains @ symbol (credential harvesting indicator)")

        # 4. IP address as domain (0-10 pts)
        if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain.split(':')[0]):
            scores['ip_domain'] = 10
            threats.append("IP address used instead of domain name")

        # 5. No HTTPS (0-6 pts)
        if parsed.scheme != 'https':
            scores['no_ssl'] = 6
            threats.append("No SSL/TLS encryption (HTTP only)")
        else:
            scores['no_ssl'] = 0

        # 6. Suspicious TLD (0-6 pts)
        domain_parts = domain.split('.')
        tld = '.' + domain_parts[-1] if domain_parts else ''
        if tld in self.SUSPICIOUS_TLDS:
            scores['suspicious_tld'] = 6
            threats.append(f"Suspicious top-level domain ({tld})")
        else:
            scores['suspicious_tld'] = 0

        # 7. Excessive subdomains (0-6 pts)
        subdomain_count = len(domain.split('.')) - 2
        if subdomain_count > 3:
            scores['subdomains'] = 6
            threats.append(f"Excessive subdomains ({subdomain_count} levels)")
        elif subdomain_count > 2:
            scores['subdomains'] = 3
        else:
            scores['subdomains'] = 0

        # 8. Port number in URL (0-5 pts)
        if ':' in domain and not domain.endswith(':443') and not domain.endswith(':80'):
            scores['unusual_port'] = 5
            threats.append("Non-standard port number in URL")
        else:
            scores['unusual_port'] = 0

        # 9. URL path depth (0-5 pts)
        path_depth = len([p for p in path.split('/') if p])
        if path_depth > 5:
            scores['path_depth'] = 5
            threats.append(f"Deep URL path ({path_depth} levels)")
        else:
            scores['path_depth'] = 0

        # 10. Special characters in URL (0-5 pts)
        special_count = len(re.findall(r'[%~\|\\{}^`\[\]]', url))
        if special_count > 3:
            scores['special_chars'] = 5
            threats.append("Unusual special characters in URL")
        else:
            scores['special_chars'] = 0

        # === DOMAIN REPUTATION ANALYSIS ===

        # 11. Suspicious keywords in domain (0-8 pts)
        suspicious_domain_words = ['secure', 'account', 'verify', 'login', 'signin',
                                    'update', 'confirm', 'banking', 'password', 'auth']
        domain_word_hits = sum(1 for w in suspicious_domain_words if w in domain)
        if domain_word_hits >= 2:
            scores['domain_keywords'] = 8
            threats.append("Multiple suspicious keywords in domain name")
        elif domain_word_hits == 1:
            scores['domain_keywords'] = 4
        else:
            scores['domain_keywords'] = 0

        # 12. Brand impersonation (0-10 pts)
        for brand in self.IMPERSONATED_BRANDS:
            if brand in domain and brand not in domain.split('.')[-2]:
                scores['brand_impersonation'] = 10
                threats.append(f"Possible brand impersonation: '{brand}' in subdomain")
                break
        else:
            scores['brand_impersonation'] = 0

        # 13. Homoglyph detection (0-10 pts)
        homoglyph_found = False
        for char in domain:
            if char in self.HOMOGLYPHS:
                homoglyph_found = True
                break
        if homoglyph_found:
            scores['homoglyph'] = 10
            threats.append("Homoglyph characters detected (visual spoofing)")
        else:
            scores['homoglyph'] = 0

        # 14. Double extension in path (0-5 pts)
        if re.search(r'\.[a-z]{2,4}\.[a-z]{2,4}$', path):
            scores['double_extension'] = 5
            threats.append("Double file extension in URL path")
        else:
            scores['double_extension'] = 0

        # 15. URL shortener (0-3 pts)
        shorteners = ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly', 'is.gd',
                       'buff.ly', 'rebrand.ly', 'cutt.ly']
        if any(s in domain for s in shorteners):
            scores['url_shortener'] = 3
            threats.append("URL shortener detected (hides real destination)")
        else:
            scores['url_shortener'] = 0

        # === CONTENT ANALYSIS ===
        if content:
            content_lower = content.lower()

            # 16. Urgency tactics (0-10 pts)
            urgency_hits = sum(1 for kw in self.URGENCY_KEYWORDS if kw in content_lower)
            if urgency_hits >= 3:
                scores['urgency'] = 10
                threats.append(f"Multiple urgency tactics detected ({urgency_hits} indicators)")
            elif urgency_hits >= 1:
                scores['urgency'] = 5
                threats.append("Urgency tactics detected")
            else:
                scores['urgency'] = 0

            # 17. Authority impersonation (0-8 pts)
            authority_hits = sum(1 for kw in self.AUTHORITY_KEYWORDS if kw in content_lower)
            if authority_hits >= 2:
                scores['authority'] = 8
                threats.append("Authority impersonation language detected")
            elif authority_hits >= 1:
                scores['authority'] = 4
            else:
                scores['authority'] = 0

            # 18. Suspicious links in content (0-5 pts)
            link_count = len(re.findall(r'https?://[^\s]+', content))
            if link_count > 3:
                scores['content_links'] = 5
                threats.append(f"Multiple links in content ({link_count})")
            else:
                scores['content_links'] = 0

            # 19. Personal info request (0-8 pts)
            pii_keywords = ['ssn', 'social security', 'credit card', 'bank account',
                           'routing number', 'pin number', 'mother maiden', 'date of birth']
            pii_hits = sum(1 for kw in pii_keywords if kw in content_lower)
            if pii_hits > 0:
                scores['pii_request'] = 8
                threats.append("Request for personal/financial information")
            else:
                scores['pii_request'] = 0

            # 20. Grammar/spelling indicators (0-3 pts)
            poor_grammar = ['kindly', 'dear customer', 'dear user', 'dear sir/madam',
                           'do the needful', 'revert back']
            grammar_hits = sum(1 for g in poor_grammar if g in content_lower)
            if grammar_hits > 0:
                scores['grammar'] = 3
            else:
                scores['grammar'] = 0
        else:
            scores['urgency'] = 0
            scores['authority'] = 0
            scores['content_links'] = 0
            scores['pii_request'] = 0
            scores['grammar'] = 0

        # === FINAL SCORING ===
        total_score = sum(scores.values())

        # Determine if severe anomalies are present
        has_severe_anomaly = False
        severe_threats = []

        # 1. High-risk TLDs (.tk, .ml, .cf, .gq)
        high_risk_tlds = {'.tk', '.ml', '.cf', '.gq'}
        if tld in high_risk_tlds:
            has_severe_anomaly = True
            severe_threats.append(f"High-risk TLD detected: {tld}")

        # 2. Brand impersonation keywords combined with unofficial domains
        impersonation_keywords = ['paypal', 'amazon', 'netflix', 'banking', 'verification']
        official_domains = {
            'paypal': ['paypal.com', 'paypal.co.uk', 'paypal.net'],
            'amazon': ['amazon.com', 'amazon.co.uk', 'amazon.in', 'amazon.de', 'amazon.fr', 'amazon.it', 'amazon.es', 'amazon.ca', 'amazon.co.jp', 'amazon.com.mx', 'amazon.com.br', 'amazon.com.au'],
            'netflix': ['netflix.com', 'netflix.net'],
        }

        domain_lower = domain.lower()
        for kw in impersonation_keywords:
            if kw in domain_lower:
                is_official = False
                if kw in official_domains:
                    for off_dom in official_domains[kw]:
                        if domain_lower == off_dom or domain_lower.endswith('.' + off_dom):
                            is_official = True
                            break
                else:
                    # 'banking' and 'verification' have no official domains, so any presence in a domain is unofficial
                    is_official = False
                
                if not is_official:
                    has_severe_anomaly = True
                    severe_threats.append(f"Brand/service impersonation detected: '{kw}' in unofficial domain '{domain}'")
                    break

        # 3. Presence of '@' symbols or structural IP addresses instead of traditional domain names
        if '@' in url:
            has_severe_anomaly = True
            severe_threats.append("Presence of '@' symbol in URL")

        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        domain_without_port = domain.split(':')[0]
        if re.match(ip_pattern, domain_without_port):
            has_severe_anomaly = True
            severe_threats.append("Structural IP address used instead of domain name")

        # 4. WHOIS Domain Age Check (Newly registered domains are extremely risky)
        if not re.match(ip_pattern, domain_without_port) and domain_without_port not in ('localhost', '127.0.0.1'):
            try:
                domain_info = whois.whois(domain_without_port)
                creation_date = domain_info.creation_date
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]
                
                if creation_date:
                    # Handle both naive and aware datetimes
                    creation_date = creation_date.replace(tzinfo=None)
                    age_days = (datetime.now() - creation_date).days
                    if age_days < 30:
                        has_severe_anomaly = True
                        severe_threats.append(f"Domain is newly registered (only {age_days} days old)")
                        scores['new_domain'] = 10
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"WHOIS lookup failed for {domain_without_port}: {error_msg}")
                # In Windows/Firewall environments, port 43 might be blocked (getaddrinfo failed)
                # We should NOT flag legitimate domains as severe threats just because the local network failed the lookup.
                if 'getaddrinfo failed' not in error_msg and 'No match for' in error_msg:
                    has_severe_anomaly = True
                    severe_threats.append("Domain does not exist or WHOIS record is completely missing")

        # Use ML model if available for primary prediction
        ml_prediction = None
        ml_confidence = None
        ml_active = False

        if self.advanced_phishing_model is not None:
            try:
                # Text classification pipeline expects just the text string
                text_input = url
                results = self.advanced_phishing_model(text_input)
                # results format: [{'label': 'phishing', 'score': 0.99}]
                
                if isinstance(results, list) and len(results) > 0:
                    best_match = results[0]
                    if isinstance(best_match, list):
                        best_match = max(best_match, key=lambda x: x['score'])
                        
                    label = best_match['label'].lower()
                    
                    # Typical labels for URL classification: phishing, malicious, bad, label_1
                    ml_prediction = 'phishing' in label or 'malicious' in label or 'bad' in label or label == 'label_1'
                    ml_confidence = float(best_match['score'] * 100)
                    
                    ml_active = True
                    logger.info(f"Advanced ML prediction: is_phishing={ml_prediction}, confidence={ml_confidence:.1f}% (Label: {label})")
            except Exception as e:
                logger.warning(f"Advanced ML prediction failed, falling back to XGBoost: {e}")

        if not ml_active and self.ml_model is not None:
            try:
                import numpy as np
                features = _extract_url_features(url)
                features_array = np.array([features])
                ml_pred_val = int(self.ml_model.predict(features_array)[0])
                ml_proba = self.ml_model.predict_proba(features_array)[0]
                
                ml_prediction = bool(ml_pred_val)
                # Confidence = probability of being phishing (class 1)
                # Safe URLs get low confidence; phishing URLs get high confidence
                if len(ml_proba) > 1:
                    ml_confidence = float(ml_proba[1] * 100)
                else:
                    ml_confidence = float(max(ml_proba) * 100)
                ml_active = True
                logger.info(f"XGBoost ML prediction: is_phishing={ml_prediction}, confidence={ml_confidence:.1f}%")
            except Exception as e:
                logger.warning(f"ML prediction failed, using heuristics: {e}")
                ml_active = False

        # Determine initial verdict
        is_phishing = False
        confidence = 0.0
        
        if ml_active and ml_prediction is not None:
            is_phishing = ml_prediction
            confidence = ml_confidence
        else:
            is_phishing = total_score > 20
            confidence = min(99.0, max(1.0, float(total_score)))

        # Fallback/Override Check
        # Override if ML model is missing OR if ML is active but gives low confidence in its prediction
        # (e.g. phishing probability < 80% or ML predicts safe (is_phishing=False) but severe anomalies are present)
        is_low_confidence = (ml_confidence is None) or (is_phishing and ml_confidence < 80.0) or (not is_phishing)
        
        override_triggered = False
        if has_severe_anomaly:
            if not ml_active or not is_phishing or is_low_confidence:
                is_phishing = True
                confidence = max(confidence, 95.0)  # Elevate confidence to high threat
                override_triggered = True
                for st in severe_threats:
                    if st not in threats:
                        threats.append(st)
                logger.info(f"Fallback/Override triggered due to severe anomalies. Set is_phishing=True (confidence={confidence}%)")

        processing_time = round(time.time() - start_time, 4)

        # Risk level
        if confidence > 80 and is_phishing:
            risk_level = "Critical"
            action = "DO NOT VISIT — High-confidence phishing detection"
        elif confidence > 60 and is_phishing:
            risk_level = "High"
            action = "AVOID — Multiple phishing indicators detected"
        elif is_phishing:
            risk_level = "Medium"
            action = "CAUTION — Some suspicious characteristics found"
        elif total_score > 15:
            risk_level = "Low"
            action = "Proceed with caution — minor indicators present"
        else:
            risk_level = "Safe"
            action = "No significant phishing indicators detected"

        if ml_active and ml_prediction is not None:
            if self.advanced_phishing_model is not None:
                method = "HuggingFace Phishing URL ML (Override Active)" if override_triggered else "HuggingFace Phishing URL ML"
            else:
                method = "XGBoost ML + Heuristic (Override Active)" if override_triggered else "XGBoost ML + Heuristic"
        else:
            method = "Heuristic Only (Strict Override Active)" if override_triggered else "Heuristic Only"

        return {
            "is_phishing": is_phishing,
            "confidence": float(round(confidence, 1)),
            "url": url,
            "threats": threats,
            "risk_level": risk_level,
            "url_entropy": round(entropy, 2),
            "detection_method": method,
            "domain_analysis": {
                "domain": domain,
                "tld": tld,
                "subdomain_count": subdomain_count,
                "has_ssl": parsed.scheme == 'https',
                "path_depth": path_depth
            },
            "explanation": f"[{method}] Risk level: {risk_level}. "
                          f"{'Multiple threat indicators found' if is_phishing else 'Low risk profile'}. "
                          f"Analyzed {len([s for s in scores.values() if s > 0])}/{len(scores)} risk factors.",
            "recommended_action": action,
            "processing_time": processing_time,
            "analysis_details": scores
        }

    @staticmethod
    def _calculate_entropy(text: str) -> float:
        """Calculate Shannon entropy of a string"""
        if not text:
            return 0.0
        freq = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
        length = len(text)
        entropy = -sum((count / length) * math.log2(count / length)
                       for count in freq.values())
        return entropy


phishing_detector = PhishingDetector()
