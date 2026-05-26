
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import re
import base64
import json
from datetime import datetime

app = FastAPI(
    title="LLM Prompt Injection Detector API",
    description="Detects and classifies prompt injection attacks in LLM applications",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────

class DetectRequest(BaseModel):
    prompt: str
    context: Optional[str] = "general"
    language: Optional[str] = "en"

class DetectResponse(BaseModel):
    is_safe: bool
    risk_score: float
    attack_type: str
    confidence: float
    explanation: str
    recommendation: str
    detected_patterns: list
    timestamp: str


# ─────────────────────────────────────────
# ATTACK PATTERN RULES (Rule-Based Layer)
# ─────────────────────────────────────────

INJECTION_PATTERNS = {
    "direct_injection": [
        r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",
        r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"forget\s+(all\s+)?(previous|prior|your)\s+instructions?",
        r"override\s+(your\s+)?(instructions?|rules?|guidelines?)",
        r"do not\s+follow\s+(your\s+)?(instructions?|rules?)",
        r"new\s+instructions?\s*:",
        r"updated?\s+instructions?\s*:",
    ],
    "jailbreak": [
        r"\bDAN\b",
        r"do\s+anything\s+now",
        r"jailbreak",
        r"you\s+are\s+now\s+(a\s+)?(evil|bad|unrestricted|free|unfiltered)",
        r"act\s+as\s+(a\s+)?(evil|bad|unrestricted|hacker|malicious)",
        r"pretend\s+(you\s+are|to\s+be)\s+(a\s+)?(evil|hacker|unrestricted)",
        r"(you\s+have\s+)?(no\s+)(rules?|restrictions?|limits?|guidelines?|ethics?)",
        r"without\s+(any\s+)?(restrictions?|limits?|guidelines?|ethics?)",
        r"developer\s+mode",
        r"god\s+mode",
    ],
    "data_exfiltration": [
        r"(print|show|reveal|display|output|tell\s+me)\s+(your\s+)?(system\s+prompt|instructions?|prompt|api\s+key|password|secret)",
        r"what\s+(is|are)\s+your\s+(system\s+prompt|instructions?|rules?|guidelines?)",
        r"repeat\s+(everything|all)\s+(above|before|prior)",
        r"output\s+everything\s+above",
        r"(show|tell)\s+me\s+your\s+(context|memory|training)",
    ],
    "role_hijack": [
        r"you\s+are\s+now\s+(a\s+)?(different|another|new)\s+(ai|assistant|bot|model)",
        r"switch\s+(to\s+)?(a\s+)?(different|another)\s+(mode|personality|role)",
        r"from\s+now\s+on\s+(you\s+are|act\s+as|pretend\s+to\s+be)",
        r"your\s+(new\s+)?(name|identity|role|persona)\s+is",
        r"(be|become|act\s+as)\s+(evil|malicious|unethical|harmful)",
    ],
    "indirect_injection": [
        r"\[system\]",
        r"\[assistant\]",
        r"\[human\]",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"###\s*instruction",
        r"###\s*system",
        r"\{\{.*\}\}",
    ]
}

# Tamil/Hindi common injection patterns
MULTILINGUAL_PATTERNS = {
    "tamil_injection": [
        r"முந்தைய\s+அறிவுறுத்தல்களை\s+புறக்கணி",   # Ignore previous instructions
        r"நீ\s+இப்போது",                              # You are now
    ],
    "hindi_injection": [
        r"पिछले\s+निर्देशों\s+को\s+अनदेखा\s+करें",  # Ignore previous instructions
        r"अब\s+तुम",                                   # You are now
    ]
}

ENCODED_PATTERNS = [
    # Base64 encoded "ignore instructions"
    r"aWdub3JlIGluc3RydWN0aW9ucw==",
    r"aWdub3JlIGFsbCBwcmV2aW91cyBpbnN0cnVjdGlvbnM=",
]


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def decode_base64_in_text(text: str) -> str:
    """Detect and decode base64 encoded segments in text."""
    b64_pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
    matches = b64_pattern.findall(text)
    decoded_parts = []
    for match in matches:
        try:
            decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
            if decoded.isprintable():
                decoded_parts.append(decoded)
        except Exception:
            pass
    return text + " " + " ".join(decoded_parts)


def normalize_text(text: str) -> str:
    """Normalize unicode, remove zero-width chars, lowercase."""
    import unicodedata
    text = unicodedata.normalize('NFKD', text)
    # Remove zero-width characters
    zero_width = ['\u200b', '\u200c', '\u200d', '\ufeff', '\u00ad']
    for char in zero_width:
        text = text.replace(char, '')
    return text.lower()


def rule_based_scan(text: str) -> dict:
    """Scan text against rule patterns. Returns matched patterns."""
    normalized = normalize_text(text)
    decoded = decode_base64_in_text(normalized)

    results = {}
    for attack_type, patterns in INJECTION_PATTERNS.items():
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, decoded, re.IGNORECASE)
            if found:
                matches.extend(found if isinstance(found[0], str) else [pattern])
        if matches:
            results[attack_type] = matches

    # Check multilingual patterns
    for lang, patterns in MULTILINGUAL_PATTERNS.items():
        matches = []
        for pattern in patterns:
            if re.search(pattern, text):
                matches.append(pattern)
        if matches:
            results[lang] = matches

    return results


def calculate_risk_score(matched_patterns: dict, prompt_length: int) -> float:
    """Calculate risk score 0.0 - 1.0 based on matched patterns."""
    if not matched_patterns:
        return 0.0

    base_scores = {
        "direct_injection": 0.90,
        "jailbreak": 0.95,
        "data_exfiltration": 0.85,
        "role_hijack": 0.80,
        "indirect_injection": 0.70,
        "tamil_injection": 0.88,
        "hindi_injection": 0.88,
    }

    max_score = 0.0
    for attack_type in matched_patterns:
        score = base_scores.get(attack_type, 0.75)
        # Boost score if multiple patterns match
        count = len(matched_patterns[attack_type])
        boosted = min(score + (count - 1) * 0.02, 0.99)
        max_score = max(max_score, boosted)

    return round(max_score, 2)


def generate_explanation(matched_patterns: dict, risk_score: float) -> str:
    """Generate human-readable explanation for the detection."""
    if not matched_patterns:
        return "No injection patterns detected. Prompt appears safe."

    explanations = {
        "direct_injection": "Direct instruction override attempt detected. The prompt tries to nullify the system's existing instructions.",
        "jailbreak": "Jailbreak attempt detected. The prompt tries to make the AI abandon its safety guidelines.",
        "data_exfiltration": "Data exfiltration attempt detected. The prompt tries to extract system prompts, API keys, or internal data.",
        "role_hijack": "Role hijack attempt detected. The prompt tries to reassign the AI a new malicious identity.",
        "indirect_injection": "Indirect injection detected. Special tokens or template syntax found that could manipulate model behavior.",
        "tamil_injection": "Tamil-language injection pattern detected.",
        "hindi_injection": "Hindi-language injection pattern detected.",
    }

    attack_types = list(matched_patterns.keys())
    primary = attack_types[0]
    explanation = explanations.get(primary, "Suspicious pattern detected.")

    if len(attack_types) > 1:
        explanation += f" Additionally, {len(attack_types) - 1} other attack type(s) detected: {', '.join(attack_types[1:])}."

    return explanation


# ─────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "LLM Prompt Injection Detector API v1.0", "status": "running"}


@app.post("/detect", response_model=DetectResponse)
def detect_injection(request: DetectRequest):
    if not request.prompt or len(request.prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    if len(request.prompt) > 10000:
        raise HTTPException(status_code=400, detail="Prompt too long (max 10,000 chars)")

    # Run detection layers
    matched_patterns = rule_based_scan(request.prompt)
    risk_score = calculate_risk_score(matched_patterns, len(request.prompt))

    is_safe = risk_score < 0.5
    attack_type = list(matched_patterns.keys())[0] if matched_patterns else "none"
    confidence = risk_score if not is_safe else round(1.0 - risk_score, 2)
    explanation = generate_explanation(matched_patterns, risk_score)
    recommendation = "ALLOW" if is_safe else ("REVIEW" if risk_score < 0.75 else "BLOCK")

    detected_list = []
    for atype, patterns in matched_patterns.items():
        detected_list.append({"type": atype, "count": len(patterns)})

    return DetectResponse(
        is_safe=is_safe,
        risk_score=risk_score,
        attack_type=attack_type,
        confidence=confidence,
        explanation=explanation,
        recommendation=recommendation,
        detected_patterns=detected_list,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/stats")
def get_pattern_stats():
    """Return total number of detection rules loaded."""
    total = sum(len(v) for v in INJECTION_PATTERNS.values())
    return {
        "total_rules": total,
        "attack_categories": list(INJECTION_PATTERNS.keys()),
        "multilingual_support": ["tamil", "hindi"],
        "encoding_detection": ["base64", "unicode_normalization", "zero_width_chars"]
    }
