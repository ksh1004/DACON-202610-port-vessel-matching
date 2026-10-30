"""한글 선박명 ↔ 영문 선박명을 같은 '발음 키' 공간으로 옮긴다.

한국어 외래어 표기는 모음과 일부 자음(R, W, Y)을 크게 바꾸지만 자음 골격은 대체로 보존한다.
    모닝 → MNN        Morning → MNN   (R 뒤에 모음이 없으면 묵음, NG → N)
    갤럭시 → KLKS     Galaxy  → KLKS  (X → KS, 겹친 ㄹㄹ → L)
    엠에스 → MS       MS(약어) → MS   (대문자 약어는 알파벳 읽기 키로)
규칙만 사용하므로 학습 데이터가 필요 없고 행 독립 원칙에도 맞는다.
"""

from __future__ import annotations

import re
import unicodedata

# 초성 19개, 종성 28개 → 자음 계열
_INITIAL = ["K", "K", "N", "T", "T", "L", "M", "P", "P", "S", "S", "", "J", "J", "J", "K", "T", "P", "H"]
# 외래어 표기에서 받침 ㅅ 은 T 소리 (스프릿 = Spirit)
_FINAL = ["", "K", "K", "K", "N", "N", "N", "T", "L", "K", "M", "L", "L", "L", "P", "L",
          "M", "P", "P", "T", "T", "N", "J", "J", "K", "T", "P", "H"]
_SYLLABLE_RE = 0xB974  # '르': 비영어권 R 표기 (자카르타, 콘도르) → 영문 쪽 R 탈락과 맞추기 위해 생략

# 알파벳 이름을 한글로 읽었을 때의 자음 키 (에이, 비, 씨, 디 …)
_LETTER_READING = {
    "A": "", "B": "P", "C": "S", "D": "T", "E": "", "F": "P", "G": "J", "H": "J", "I": "",
    "J": "J", "K": "K", "L": "L", "M": "M", "N": "N", "O": "", "P": "P", "Q": "K", "R": "L",
    "S": "S", "T": "T", "U": "", "V": "P", "W": "TPL", "X": "KS", "Y": "", "Z": "JT",
}
_VOWELS = set("AEIOUY")


def _collapse(key: str) -> str:
    """연속된 같은 자음 계열을 하나로 (숫자는 그대로)."""
    out = []
    for ch in key:
        if not out or out[-1] != ch or ch.isdigit():
            out.append(ch)
    return "".join(out)


# ---- 한글 ------------------------------------------------------------------
_KO_NUMBER = re.compile(r"제\s*(\d+)\s*호?")


def normalize_hangul_name(name: str) -> str:
    """'제7하나' → ' 7 하나', 끝의 '호' 제거, 숫자와 한글 사이 띄우기."""
    name = unicodedata.normalize("NFKC", name)
    name = _KO_NUMBER.sub(r" \1 ", name)
    name = re.sub(r"(\d)\s*호", r"\1 ", name)
    name = re.sub(r"(?<=[가-힣])호(?=\s|$)", "", name.strip())
    name = re.sub(r"(?<=\d)(?=[가-힣A-Za-z])|(?<=[가-힣A-Za-z])(?=\d)", " ", name)
    return name


def _hangul_word_key(word: str) -> str:
    out = []
    for ch in word:
        code = ord(ch) - 0xAC00
        if ord(ch) == _SYLLABLE_RE:
            continue
        if 0 <= code < 11172:
            out.append(_INITIAL[code // 588])
            out.append(_FINAL[code % 28])
        elif ch.isdigit():
            out.append(ch)
        elif ch.isalpha():
            out.append(_english_word_key(ch.upper()))
    return _collapse("".join(out))


def hangul_key(name: object) -> str:
    if not isinstance(name, str):
        return ""
    words = re.split(r"[^0-9A-Za-z가-힣]+", normalize_hangul_name(name))
    return " ".join(k for k in (_hangul_word_key(w) for w in words if w) if k)


# ---- 영문 ------------------------------------------------------------------
# 순서가 중요하다: 앞 규칙의 결과에 뒤 규칙이 적용된다
_EN_RULES = [
    (re.compile(r"NG(?=[AEIOU])"), "NK"),  # Singapore → 싱가포르
    (re.compile(r"NG"), "N"),              # Morning → 모닝
    (re.compile(r"^GH"), "K"),
    (re.compile(r"GH"), ""),               # Highway → 하이웨이
    (re.compile(r"PH"), "P"),
    (re.compile(r"TH"), "T"),
    (re.compile(r"CK"), "K"),
    (re.compile(r"T(?=IO)"), "S"),         # Resolution → 리솔루션
    (re.compile(r"SH|SCH"), "S"),
    (re.compile(r"CH(?=EM|R|OR)"), "K"),   # Chemroad → 켐로드, Christ → 크리스, Chorus → 코러스
    (re.compile(r"^X(?=[IU])"), "S"),      # 중국어 병음: Xiamen → 샤먼, Xin → 신
    (re.compile(r"CH|TCH|DG"), "J"),
    (re.compile(r"ZH|^TS"), "J"),          # Shenzhen → 센젠, Tsurugi → 쯔루기
    (re.compile(r"KH"), "K"),
    (re.compile(r"(?<=[AEIOUY])SE$"), "J"),  # Sunrise → 선라이즈, Rose → 로즈 (모음 사이 S 전체에 적용하면 Pegasus 등이 깨짐)
    (re.compile(r"C(?=[EIY])"), "S"),
    (re.compile(r"G(?=[EIY])"), "J"),
    (re.compile(r"R(?![AEIOUY])"), ""),    # Star → 스타
    (re.compile(r"H(?![AEIOUY])"), ""),    # Sarah → 사라
    (re.compile(r"X"), "KS"),
    (re.compile(r"QU?"), "K"),
]
_EN_MAP = str.maketrans({
    "B": "P", "F": "P", "V": "P", "C": "K", "G": "K", "D": "T", "Z": "J", "R": "L",
    "A": None, "E": None, "I": None, "O": None, "U": None, "Y": None, "W": None,
})


def _english_word_key(word: str) -> str:
    for pattern, repl in _EN_RULES:
        word = pattern.sub(repl, word)
    return _collapse(word.translate(_EN_MAP))


def _is_acronym(token: str) -> bool:
    """원문이 대문자 2~5글자인 단어 (혼합 대소문자 이름 안에서만 판단)."""
    return 2 <= len(token) <= 5 and token.isalpha() and token.isupper()


def english_key(name: object) -> str:
    """카탈로그 영문명 → 발음 키. 대소문자가 섞인 이름의 대문자 약어는 알파벳 읽기로 바꾼다."""
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKC", name)
    tokens = re.findall(r"[A-Za-z]+|\d+", name)
    mixed_case = any(t.lower() != t and t.upper() != t for t in tokens)
    keys = []
    for i, token in enumerate(tokens):
        upper = token.upper()
        if upper == "NO" and i + 1 < len(tokens) and tokens[i + 1].isdigit():
            continue  # 'No. 7' 의 No 는 한글에서 '제'로 표기되거나 생략됨
        if token.isdigit():
            keys.append(token)
        elif mixed_case and _is_acronym(token):
            keys.append(_collapse("".join(_LETTER_READING[c] for c in upper)))
        else:
            keys.append(_english_word_key(upper))
    return " ".join(k for k in keys if k)
