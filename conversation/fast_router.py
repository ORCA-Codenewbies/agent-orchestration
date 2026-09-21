# conversation/fast_router.py
import re
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class FastRouteResult:
    matched: bool
    intent: Optional[str]
    confidence: float
    endpoints: Optional[Tuple[str, str]] = None

SIGNAL_WORDS = {
    "hazard_alert": {
        "en": ["cyclone", "lightning", "alert", "warning", "storm", "hurricane", "hazard"],
        "bn-Latn": ["baj", "cyclone", "alert", "warning", "jhor", "toofan", "bipod"],
        "bn": ["ঘূর্ণিঝড়", "বজ্রপাত", "সতর্কতা", "ঝড়", "বিপদ"],
        "hi-Latn": ["aandhi", "tufaan", "bijli", "alert", "warning", "khatra"],
        "hi": ["आंधी", "तूफान", "बिजली", "अलर्ट", "चेतावनी", "खतरा"]
    },
    "nearest_pfz": {
        "en": ["pfz", "potential fishing zone", "fish", "nearest", "where", "spot", "catch", "best zone", "fishing zone"],
        "bn-Latn": ["pfz", "mach", "fishing", "kothay", "nearest", "spot", "machh dhorar", "bhalo zone", "shobcheye bhalo", "valo zone", "bhalo jayga", "mach dhorar"],
        "bn": ["মাছ", "কোথায়", "পিএফজেড", "মাছ ধরার", "ভালো জোন"],
        "hi-Latn": ["pfz", "machli", "kahan", "paas", "spot", "fishing", "machhli pakadne", "achha zone", "best zone"],
        "hi": ["मछली", "कहाँ", "पास", "जगह", "मछली पकड़ने", "अच्छा ज़ोन"]
    },
    "marine_safety_forecast": {
        "en": ["safe", "venture", "tomorrow", "sea", "go", "trip", "forecast"],
        "bn-Latn": ["safe", "jawa", "samudra", "sea", "niraapood", "kal", "trip"],
        "bn": ["নিরাপদ", "যাওয়া", "সমুদ্র", "কাল", "ভ্রমণ"],
        "hi-Latn": ["safe", "jaana", "samundar", "kal", "trip", "surakshit"],
        "hi": ["सुरक्षित", "जाना", "समुद्र", "कल", "यात्रा"]
    },
    "marine_conditions": {
        "en": ["conditions", "weather", "wave", "wind", "current", "how is"],
        "bn-Latn": ["obostha", "weather", "wave", "wind", "dheu", "batas", "kemon"],
        "bn": ["অবস্থা", "আবহাওয়া", "ঢেউ", "বাতাস", "কেমন"],
        "hi-Latn": ["mausam", "wave", "wind", "hawa", "kaisa", "conditions"],
        "hi": ["मौसम", "लहर", "हवा", "कैसा", "हालत"]
    },
    "productivity_analysis": {
        "en": ["chlorophyll", "sst", "sea surface temperature", "productive region", "fish productivity", "productive fishing area", "favourable", "temperature"],
        "bn-Latn": ["chlorophyll beshi", "pani-r temperature", "sea temperature", "productive area", "mach beshi", "fishing er jonno bhalo area", "chlorophyll", "tapmatra"],
        "bn": ["ক্লোরোফিল", "সমুদ্রের তাপমাত্রা", "উৎপাদনশীল", "বেশি মাছ", "তাপমাত্রা"],
        "hi-Latn": ["chlorophyll zyada", "samundar ka temperature", "sea surface temperature", "productive area", "machhli ke liye achha area", "fishing ke liye best region", "chlorophyll", "sst", "temperature", "tapman"],
        "hi": ["क्लोरोफिल", "समुद्र का तापमान", "उत्पादक क्षेत्र", "मछली के लिए अच्छा", "तापमान"]
    }
}

def extract_route_endpoints(query_lower: str) -> Tuple[Optional[str], Optional[str]]:
    # En: from X to Y
    m = re.search(r'\bfrom\s+(.+?)\s+to\s+(.+?)(?:\s+plan|\s+route|\s+porjonto|\s+tak|\s+jabo|\s+karo|$)', query_lower)
    if m: return m.group(1).strip(), m.group(2).strip()
    
    # Benglish: X theke Y
    m = re.search(r'\b(.+?)\s+theke\s+(.+?)(?:\s+porjonto|\s+safe|\s+route|\s+plan|\s+koro|$)', query_lower)
    if m: return m.group(1).strip(), m.group(2).strip()
    
    # Hinglish: X se Y
    m = re.search(r'\b(.+?)\s+se\s+(.+?)(?:\s+tak|\s+safe|\s+route|\s+plan|\s+karo|$)', query_lower)
    if m: return m.group(1).strip(), m.group(2).strip()
    
    return None, None

def detect_specialized_capability(query_text: str, language: str) -> FastRouteResult:
    """
    Detects specialized benchmark intents with high confidence using multilingual heuristics.
    """
    query_lower = query_text.lower()
    
    # Explicit route queries should not be fast-routed as they require A* execution
    route_kws = [
        "safest route", "safe route", "route for a vessel", "navigate", "safe navigation", "route",
        "raasta", "marg", "रास्ता", "मार्ग", "रूट", "सुरक्षित रास्ता", # Hindi/Hinglish
        "rasta", "poth", "রাস্তা", "পথ", "নিরাপদ রুট" # Bengali/Benglish
    ]
    if any(kw in query_lower for kw in route_kws):
        origin, dest = extract_route_endpoints(query_lower)
        if origin and dest:
            return FastRouteResult(matched=True, intent="safe_route", confidence=0.95, endpoints=(origin, dest))
        return FastRouteResult(matched=False, intent=None, confidence=0.0)
    
    # Normalize language mapping slightly
    lang = language.lower()
    if lang in ["bn_en", "bn-latn"]:
        lang = "bn-Latn"
    elif lang in ["hi_en", "hi-latn"]:
        lang = "hi-Latn"
    elif lang not in ["en", "bn", "hi"]:
        lang = "en"
        
    best_intent = None
    best_score = 0.0
    
    for intent, lang_dict in SIGNAL_WORDS.items():
        signals = lang_dict.get(lang, lang_dict.get("en", []))
        
        # Cross-pollinate English signals to romanized languages since code-switching is common
        if lang in ["bn-Latn", "hi-Latn"]:
            signals = signals + lang_dict.get("en", [])
            
        matched_count = sum(1 for word in set(signals) if re.search(r'\b' + re.escape(word) + r'\b', query_lower))
        
        # Calculate a simple confidence score
        if intent == "productivity_analysis":
            # Require multiple distinct signals for productivity to avoid false positives with generic 'fish' queries
            chlo_signals = ["chlorophyll", "ক্লোরোফিল"]
            sst_signals = ["sst", "sea surface temperature", "sea temperature", "samundar ka temperature", "pani-r temperature", "সমুদ্রের তাপমাত্রা", "समुद्र का तापमान", "temperature", "tapmatra", "tapman", "তাপমাত্রা", "तापमान"]
            
            has_chlo = any(re.search(r'\b' + re.escape(word) + r'\b', query_lower) for word in chlo_signals)
            has_sst = any(re.search(r'\b' + re.escape(word) + r'\b', query_lower) for word in sst_signals)
            
            if has_chlo and has_sst:
                score = 0.98
            elif matched_count >= 2:
                score = 0.9
            else:
                score = 0.5 # Below threshold
        elif intent == "nearest_pfz":
            # Require multiple distinct signals for fishing zones
            fish_signals = ["machh", "mach", "machli", "मछली", "মাছ", "fish"]
            zone_signals = ["zone", "spot", "jayga", "jagah", "जगह", "bhalo", "achha"]
            
            has_fish = any(word in query_lower for word in fish_signals)
            has_zone = any(word in query_lower for word in zone_signals)
            
            if has_fish and has_zone:
                score = 0.95
            elif "pfz" in query_lower:
                score = 0.99
            elif matched_count > 0:
                if matched_count == 1:
                    score = 0.6
                elif matched_count == 2:
                    score = 0.9
                else:
                    score = 0.99
            else:
                score = 0.0
        else:
            if matched_count > 0:
                if matched_count == 1:
                    score = 0.6
                elif matched_count == 2:
                    score = 0.9
                else:
                    score = 0.99
            else:
                score = 0.0
                
        if score > best_score:
            best_score = score
            best_intent = intent

    if best_score >= 0.85:
        return FastRouteResult(matched=True, intent=best_intent, confidence=best_score)
        
    return FastRouteResult(matched=False, intent=None, confidence=best_score)
