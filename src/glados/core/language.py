"""French/English detection for bilingual mode: each user turn sets the reply language and voice."""

from __future__ import annotations

import re
import threading
from typing import Literal

Language = Literal["en", "fr"]
LANGUAGES: tuple[Language, ...] = ("en", "fr")

FRENCH_TURN_REMINDER = "Réponds en français, même si la demande, le contexte ou les outils sont en anglais."
ENGLISH_TURN_REMINDER = "Reply in English, even if the context, memory or tool results are in French."
TURN_REMINDERS: dict[str, str] = {"fr": FRENCH_TURN_REMINDER, "en": ENGLISH_TURN_REMINDER}

# Frequent words that are not also common in the other language ("a", "on", "son", "Glados"... are left out).
_FRENCH_WORDS = (
    "le la les un une des du de au aux et est sont suis es sommes êtes ce cette ces cet que qui quoi quel quelle"
    " quels quelles pas ne je tu il elle nous vous ils elles mon ma mes ton ta tes sa ses notre votre leur leurs"
    " pour avec dans sur sous mais ou où donc comment pourquoi quand combien oui ouais non merci bonjour salut"
    " peux peut veux veut fais fait faire dis dit dire très bien aussi encore toujours rien tout tous toute moi"
    " toi lui eux ça cela ceci voilà alors parce puis avoir être ai as avez ont était sera heure"
)
_ENGLISH_WORDS = (
    "the is are am was were be been and or but not no yes yeah what which who whom whose why how when where"
    " this that these those there here it its i you he she we they him her us them my your his our their"
    " to of for with from about into over can could would should will shall do does did don't doesn't didn't"
    " i'm it's you're what's that's let's have has had please thanks thank hello hi just also very much"
    " many some any all every tell show give get make go going want need know think time today tomorrow"
)
_FRENCH = frozenset(_FRENCH_WORDS.split())
_ENGLISH = frozenset(_ENGLISH_WORDS.split())
# Elisions and accented letters only French uses.
_FRENCH_MARKS = re.compile(r"\b(?:[cdjlmnst]|qu)['\u2019]\w|[éèêëàâîïôûùçœ]", re.IGNORECASE)
_WORD = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)?")


def language_scores(text: str) -> tuple[int, int]:
    """(French, English) evidence: distinctive words and French-only spelling."""
    words = [w.casefold().replace("\u2019", "'") for w in _WORD.findall(text)]
    french = sum(w in _FRENCH or w.split("'")[-1] in _FRENCH for w in words) + len(_FRENCH_MARKS.findall(text))
    english = sum(w in _ENGLISH for w in words)
    return french, english


def detect_language(text: str, default: Language, margin: int = 1) -> Language:
    """The language this text is clearly written in, else ``default`` ("OK", "Glados ?", names)."""
    french, english = language_scores(text)
    if french - english >= margin:
        return "fr"
    if english - french >= margin:
        return "en"
    return default


class LanguageState:
    """The conversation's current language: the user's latest clear utterance decides it."""

    def __init__(self, default: Language = "fr") -> None:
        self._lock = threading.Lock()
        self._current: Language = default

    @property
    def current(self) -> Language:
        with self._lock:
            return self._current

    def observe(self, text: str) -> Language:
        """Follow the user's language; an ambiguous utterance keeps the conversation's."""
        with self._lock:
            self._current = detect_language(text, self._current)
            return self._current
