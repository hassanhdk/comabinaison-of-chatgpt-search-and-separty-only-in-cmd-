#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sleepia_full_updated.py

Terminal Web QA + ML Assistant (Updated)

Features:
- Reads API keys from environment, .env, or config.json.
- Uses SerpAPI (serpapi package) for Google search results.
- Fetches pages, extracts text via BeautifulSoup.
- Ranks results using TF-IDF + sentence-transformers embeddings.
- Uses OpenAI ChatCompletion for final summarization (model configurable).
- Local caching of search results and fetched pages to speed repeated queries.
- Helpful CLI with debug/test utilities.

Security:
- DO NOT hardcode API keys into this file.
- Use a .env file (kept out of git) or environment variables.

How to run:
1) Install required packages:
   pip install google-search-results openai beautifulsoup4 requests sentence-transformers scikit-learn nltk python-dotenv tqdm

2) Create a .env in the same folder (or export env vars):
   SERPAPI_KEY=your_serpapi_key_here
   OPENAI_API_KEY=your_openai_api_key_here

3) Run:
   python sleepia_full_updated.py

This file includes extra helper functions for troubleshooting environment issues
and a small test runner to validate that dependencies and keys are available.

"""

import os

# --- مؤقت لتجربة التشغيل فقط ---
os.environ["SERPAPI_KEY"] = "fabcdd3ada31868f0fcc8ffe10ab57ac3b26fe167sc44b0eecbe41b6fd7f30ef0"
os.environ["OPENAI_API_KEY"] = "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"
import sys
import json
import time
import hashlib
import logging
import threading
from pathlib import Path
from queue import Queue
from typing import List, Dict, Any, Optional, Tuple

# third-party imports (may not exist yet on user's system)
try:
    import requests
    from bs4 import BeautifulSoup
    from tqdm import tqdm
except Exception as e:
    print("Some required packages are missing. Install requirements: pip install requests beautifulsoup4 tqdm")
    # allow script to continue to show better error messages later

# SerpAPI / google-search-results wrapper
# note: package installed via `pip install google-search-results` exposes `serpapi` module
try:
    from serpapi import GoogleSearch
except Exception:
    GoogleSearch = None

# OpenAI
try:
    import openai
except Exception:
    openai = None

# ML / NLP
try:
    from sentence_transformers import SentenceTransformer, util
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    SentenceTransformer = None
    util = None
    TfidfVectorizer = None
    cosine_similarity = None

# NLTK sentence tokenizer
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except Exception:
    try:
        import nltk
        nltk.download("punkt")
    except Exception:
        pass

# dotenv for .env file
try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except Exception:
    _DOTENV_AVAILABLE = False

# ---------- Configuration & Constants ----------

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/116.0 Safari/537.36"
)

CACHE_DIR = Path.home() / ".sleepia_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = os.getenv("SLEEP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")

# Model and limits (tunable)
EMBEDDING_MODEL_NAME = os.getenv("SLEEP_EMBED_MODEL", "all-MiniLM-L6-v2")
OPENAI_MODEL_FOR_SUMMARY = os.getenv("SLEEP_OPENAI_MODEL", "gpt-3.5-turbo")
MAX_SEARCH_RESULTS = int(os.getenv("SLEEP_MAX_SEARCH_RESULTS", "10"))
MAX_PAGES_TO_FETCH = int(os.getenv("SLEEP_MAX_PAGES_TO_FETCH", "6"))
MAX_SENTENCES_PER_PAGE = int(os.getenv("SLEEP_MAX_SENTENCES_PER_PAGE", "40"))

# ---------- Utilities ----------

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def cache_load(key: str) -> Optional[Any]:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def cache_save(key: str, data: Any) -> None:
    path = CACHE_DIR / f"{key}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        logging.exception("Failed writing cache %s", key)


# ---------- Environment & Config Loading ----------

def load_envs(verbose: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """Load SERPAPI_KEY and OPENAI_API_KEY from environment, .env, or config.json.

    Returns (serpapi_key, openai_key)
    """
    # 1) Load .env if python-dotenv available
    if _DOTENV_AVAILABLE:
        # load .env in script directory first
        project_env = Path(__file__).parent / ".env"
        if project_env.exists():
            load_dotenv(dotenv_path=str(project_env))
            if verbose:
                logging.info("Loaded .env from %s", project_env)
        else:
            # fallback to default load (system-wide)
            load_dotenv(override=False)

    # 2) Check environment variables
    serpapi_key = os.getenv("SERPAPI_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    # 3) Fallback to config.json in same folder as script
    if (not serpapi_key or not openai_key):
        cfg = Path(__file__).parent / "config.json"
        if cfg.exists():
            try:
                with open(cfg, "r", encoding="utf-8") as f:
                    parsed = json.load(f)
                    serpapi_key = serpapi_key or parsed.get("SERPAPI_KEY")
                    openai_key = openai_key or parsed.get("OPENAI_API_KEY")
                    if verbose:
                        logging.info("Loaded keys from config.json")
            except Exception:
                logging.exception("Failed to read config.json")

    return serpapi_key, openai_key


# ---------- Web Search via SerpAPI ----------

class WebSearcher:
    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("No SerpAPI key provided")
        if GoogleSearch is None:
            raise RuntimeError("serpapi package not installed. Run: pip install google-search-results")
        self.api_key = api_key
        self.client = GoogleSearch({"api_key": self.api_key})

    def search(self, query: str, num: int = MAX_SEARCH_RESULTS) -> List[Dict[str, Any]]:
        cache_key = "search_" + sha1(query + str(num))
        cached = cache_load(cache_key)
        if cached:
            logging.info("Using cached search results for query: %s", query)
            return cached

        params = {"engine": "google", "q": query, "num": num, "hl": "en"}
        try:
            raw = self.client.get_dict(params)
        except Exception as e:
            logging.exception("SerpAPI request failed")
            return []

        results: List[Dict[str, Any]] = []
        for r in raw.get("organic_results", [])[:num]:
            results.append({
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet"),
                "domain": r.get("displayed_link") or r.get("link")
            })
        cache_save(cache_key, results)
        return results


# ---------- Fetching & Parsing Pages ----------

def fetch_page(url: str, timeout: int = 10) -> Optional[str]:
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception:
        logging.warning("Failed to fetch page: %s", url)
        return None


def extract_text_from_html(html: str, max_sentences: int = MAX_SENTENCES_PER_PAGE) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    # Prefer article/main sections if present
    article = soup.find(["article", "main"]) or soup.find("div", {"role": "main"})
    texts: List[str] = []
    if article:
        for t in article.find_all(["p", "li", "h1", "h2", "h3"]):
            text = t.get_text(separator=" ", strip=True)
            if text:
                texts.append(text)
    else:
        for p in soup.find_all("p"):
            text = p.get_text(separator=" ", strip=True)
            if text:
                texts.append(text)
        if not texts:
            # ultimate fallback: grab body text
            body = soup.get_text(separator=" ", strip=True)
            texts = [body] if body else []

    joined = "\n".join(texts)
    try:
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(joined)
    except Exception:
        sents = [s.strip() for s in joined.split(".") if s.strip()]

    sents = sents[:max_sentences]
    return " ".join(sents)


def fetch_pages_concurrently(results: List[Dict[str, Any]], max_workers: int = 6) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    q: "Queue" = Queue()
    for r in results[:MAX_PAGES_TO_FETCH]:
        q.put(r)

    lock = threading.Lock()

    def worker():
        while not q.empty():
            try:
                r = q.get_nowait()
            except Exception:
                break
            link = r.get("link")
            key = "page_" + sha1(link)
            cached = cache_load(key)
            if cached:
                page_meta = cached
            else:
                html = fetch_page(link)
                text = extract_text_from_html(html) if html else ""
                page_meta = {
                    "title": r.get("title"),
                    "link": link,
                    "domain": r.get("domain"),
                    "snippet": r.get("snippet"),
                    "text": text,
                }
                cache_save(key, page_meta)
            with lock:
                out.append(page_meta)
            q.task_done()

    threads = []
    for _ in range(min(max_workers, q.qsize())):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)
    q.join()
    return out


# ---------- Ranking & ML ----------

class Ranker:
    def __init__(self):
        if TfidfVectorizer is None:
            logging.warning("scikit-learn not available: TF-IDF ranking will be disabled.")
            self.vectorizer = None
        else:
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

        if SentenceTransformer is None:
            logging.warning("sentence-transformers not available: embedding ranking will be disabled.")
            self.embed_model = None
        else:
            try:
                self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            except Exception:
                logging.exception("Failed to load sentence-transformers model")
                self.embed_model = None

    def rank_by_query(self, query: str, pages: List[Dict[str, Any]], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        docs = [p.get("text", "") or p.get("snippet", "") or "" for p in pages]
        scores = [0.0] * len(docs)

        if self.vectorizer and any(docs):
            try:
                tfidf = self.vectorizer.fit_transform(docs + [query])
                qv = tfidf[-1]
                doc_mat = tfidf[:-1]
                sims = cosine_similarity(doc_mat, qv)
                scores = [float(s[0]) for s in sims]
            except Exception:
                logging.exception("TF-IDF scoring failed")

        emb_scores = [0.0] * len(docs)
        if self.embed_model:
            try:
                query_emb = self.embed_model.encode(query, convert_to_tensor=True)
                doc_embs = self.embed_model.encode(docs, convert_to_tensor=True)
                sims = util.cos_sim(query_emb, doc_embs).cpu().numpy()[0]
                emb_scores = [float(s) for s in sims]
            except Exception:
                logging.exception("Embedding scoring failed")

        combined = []
        for i, p in enumerate(pages):
            combined_score = 0.6 * scores[i] + 0.4 * emb_scores[i]
            combined.append((p, combined_score))
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

    def extract_top_sentences(self, text: str, query: str, top_n: int = 5) -> List[str]:
        try:
            from nltk.tokenize import sent_tokenize
            sents = sent_tokenize(text)
        except Exception:
            sents = [s.strip() for s in text.split('.') if s.strip()]
        if not sents:
            return []
        q_terms = set(query.lower().split())
        scores = []
        for idx, s in enumerate(sents):
            s_terms = set(s.lower().split())
            overlap = len(q_terms & s_terms)
            score = overlap + (1.0 / (1 + idx))
            scores.append((s, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scores[:top_n]]


# ---------- OpenAI integration ----------

class OpenAISummarizer:
    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("No OpenAI API key provided")
        if openai is None:
            raise RuntimeError("openai package not installed. Run: pip install openai")
        openai.api_key = api_key

    def generate_answer(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        combined = []
        for i, c in enumerate(contexts, start=1):
            title = c.get("title") or ""
            link = c.get("link") or ""
            domain = c.get("domain") or ""
            snippet = c.get("snippet") or ""
            top_text = c.get("top_text") or c.get("text", "")
            block = (
                f"[{i}] Title: {title}\nDomain: {domain}\nURL: {link}\nSnippet: {snippet}\nContent: {top_text}\n---\n"
            )
            combined.append(block)
        context_text = "\n".join(combined)
        if len(context_text) > 25000:
            context_text = context_text[:25000] + "\n...[truncated]\n"

        system_prompt = (
            "You are a helpful assistant that answers user questions using only the provided sources. "
            "Be concise and include citations like [1], [2], ... pointing to the provided sources. "
            "If the sources do not contain the answer, say so and provide best-effort resources."
        )
        user_prompt = (
            f"Question: {question}\n\nSources:\n{context_text}\n\n"
            "Please provide:\n1) A concise answer (1-4 short paragraphs).\n2) A short list of cited sources by number and URL.\n3) A confidence note (high/medium/low)."
        )

        try:
            response = openai.ChatCompletion.create(
                model=OPENAI_MODEL_FOR_SUMMARY,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=700,
                temperature=0.15,
            )
            answer = response["choices"][0]["message"]["content"].strip()
            return answer
        except Exception:
            logging.exception("OpenAI call failed")
            return "[OpenAI error: failed to generate summary]"


# ---------- CLI & orchestration ----------

def build_contexts_from_ranked(ranked: List[Tuple[Dict[str, Any], float]], ranker: Ranker, query: str) -> List[Dict[str, Any]]:
    contexts = []
    for page, score in ranked:
        text = page.get("text", "")
        top_sents = ranker.extract_top_sentences(text or page.get("snippet", ""), query, top_n=6)
        top_text = " ".join(top_sents)
        contexts.append({
            "title": page.get("title"),
            "link": page.get("link"),
            "domain": page.get("domain"),
            "snippet": page.get("snippet"),
            "text": text,
            "top_text": top_text,
            "score": score,
        })
    return contexts


def pretty_print_sources(sources: List[Dict[str, Any]]) -> None:
    for i, s in enumerate(sources, start=1):
        print(f"[{i}] ({s.get('domain','?')}) {s.get('title','No title')}")
        print(f"     {s.get('link')}")
        snippet = s.get('snippet') or s.get('text','')
        if snippet:
            print(f"     → {snippet[:300].replace('\n',' ')}{'...' if len(snippet)>300 else ''}")
        print()


def run_interactive():
    serpapi_key, openai_key = load_envs(verbose=True)
    print("=== Web QA ML Assistant — الطرفي ===")
    print("أدخل سؤالك. اكتب 'exit' للخروج.")

    if not serpapi_key:
        print("خطأ: لم يتم العثور على SERPAPI_KEY. عيّن متغير البيئة SERPAPI_KEY أو أضفه إلى .env أو config.json.")
        print("انظر الوثائق. يمكنك استخدام test_env.py للتحقق من قراءة المفاتيح.")
        return
    if not openai_key:
        print("خطأ: لم يتم العثور على OPENAI_API_KEY. عيّن متغير البيئة OPENAI_API_KEY أو أضفه إلى .env أو config.json.")
        return

    try:
        searcher = WebSearcher(serpapi_key)
    except Exception as e:
        logging.exception("Failed to initialize WebSearcher")
        print("خطأ عند إعداد SerpAPI client:", e)
        return

    ranker = Ranker()
    try:
        summarizer = OpenAISummarizer(openai_key)
    except Exception as e:
        logging.exception("Failed to initialize OpenAI summarizer")
        print("خطأ عند إعداد OpenAI client:", e)
        return

    while True:
        try:
            query = input("\nسؤال: ").strip()
        except KeyboardInterrupt:
            print("\nباي 👋")
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("باي 👋")
            break

        print("🔎 البحث على Google... (قد يُستخدم الكاش)")
        results = searcher.search(query, num=MAX_SEARCH_RESULTS)
        if not results:
            print("لم تُرجع نتائج من محرك البحث.")
            continue

        pretty_print_sources(results[:6])

        print("🌐 جلب صفحات الويب (قد يستغرق بعض الوقت)...")
        pages = fetch_pages_concurrently(results)
        if not pages:
            print("لا يوجد صفحات مُستخرجة.")
            continue

        print("⚙️ ترتيب المصادر واستخراج المقتطفات ذات الصلة...")
        ranked = ranker.rank_by_query(query, pages, top_k=6)
        contexts = build_contexts_from_ranked(ranked, ranker, query)

        print("\nالمصادر الأعلى صلة:")
        for i, c in enumerate(contexts, 1):
            print(f"[{i}] ({c.get('domain')}) {c.get('title')}")
            preview = c.get("top_text") or c.get("snippet") or ""
            print("    " + (preview[:250] + "..." if len(preview) > 250 else preview))

        print("\n🧠 توليد الإجابة الملخصة باستخدام OpenAI...")
        answer = summarizer.generate_answer(query, contexts)

        print("\n" + ("=" * 60))
        print("💡 الإجابة:")
        print(answer)
        print(("=" * 60) + "\n")

        save = input("هل تريد حفظ هذه الجلسة (y/n)? ").strip().lower()
        if save.startswith("y"):
            key = "session_" + sha1(query + str(time.time()))
            session_data = {"query": query, "results": results, "contexts": contexts, "answer": answer, "timestamp": time.time()}
            cache_save(key, session_data)
            print(f"تم الحفظ في الكاش: {CACHE_DIR}/{key}.json")


# ---------- Small helpers & tests ----------

def print_setup_instructions():
    print("""
تعليمات سريعة قبل التشغيل:
1) ثبّت الحزم المطلوبة:
   pip install google-search-results openai beautifulsoup4 requests sentence-transformers scikit-learn nltk python-dotenv tqdm

2) أنشئ ملف .env داخل نفس مجلد هذا السكربت أو عيّن المتغيرات في بيئة النظام:
   SERPAPI_KEY=your_serpapi_key_here
   OPENAI_API_KEY=your_openai_api_key_here

3) شغّل السكربت:
   python sleepia_full_updated.py

ملاحظات:
- تأكد من عدم رفع ملف .env لمستودعات عامة.
- إذا رغبت، يمكن وضع المفاتيح داخل config.json (شكل JSON بسيط) كبديل.
""")


def test_environment():
    print("Running quick environment checks...\n")
    serpapi_key, openai_key = load_envs(verbose=False)
    print("python version:", sys.version.split()[0])
    print("requests installed:", 'requests' in sys.modules)
    print("serpapi installed:", GoogleSearch is not None)
    print("openai installed:", openai is not None)
    print("sentence-transformers installed:", SentenceTransformer is not None)
    print("scikit-learn available:", TfidfVectorizer is not None)
    print("python-dotenv available:", _DOTENV_AVAILABLE)
    print("\nSERPAPI_KEY present:", bool(serpapi_key))
    print("OPENAI_API_KEY present:", bool(openai_key))
    print("Cache dir:", CACHE_DIR)


# ---------- Entry point ----------

def main(argv: List[str]):
    if len(argv) > 1 and argv[1] in ("-h", "--help"):
        print(__doc__)
        print_setup_instructions()
        return
    if len(argv) > 1 and argv[1] == "--test-env":
        test_environment()
        return
    run_interactive()


if __name__ == "__main__":
    main(sys.argv)
