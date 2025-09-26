


import os, re, json, argparse, random
from typing import List, Dict
import numpy as np


SEED = 42
PARAPHRASE_MODEL = "eugenesiow/bart-paraphrase"
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MASK_MODELS = ["nlpaueb/bert-base-greek-uncased-v1", "bert-base-multilingual-cased"]

DEFAULT_TEXTS = {
    "text1": "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes.",
    "text2": "During our final discuss, I told him about the new submission — the one we were waiting since last autumn..."
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def sentence_split(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]

def load_texts(path: str = None) -> Dict[str, str]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_TEXTS


def heuristics_rewrite_sentence(s: str) -> str:
    s = re.sub(r"\bdiscuss\b", "discussion", s, flags=re.I)
    s = re.sub(r"\bupdates? was\b", "updates were", s, flags=re.I)
    s = re.sub(r"\bbit\b", "a bit", s, flags=re.I)
    s = re.sub(r"\bI am very appreciated\b", "I greatly appreciate", s, flags=re.I)
    s = re.sub(r"\bwith all safe and great\b", "safely and joyfully", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def pipeline_custom(text: str) -> str:
    return " ".join(heuristics_rewrite_sentence(s) for s in sentence_split(text))

def pipeline_transformer(text: str) -> str:
    from transformers import pipeline as hf_pipeline
    gen = hf_pipeline("text2text-generation", model=PARAPHRASE_MODEL)
    outs = gen([f"paraphrase: {s}" for s in sentence_split(text)], max_new_tokens=128, do_sample=False)
    return " ".join(o["generated_text"].strip() for o in outs)

def pipeline_languagetool(text: str) -> str:
    import time
    import language_tool_python
    from language_tool_python.utils import RateLimitError


    if os.getenv("USE_LANGUAGETOOL", "1") != "1":
        return text  # no-op


    try:
        tool = language_tool_python.LanguageToolPublicAPI("en-US")
    except Exception:

        return text


    for attempt in range(3):
        try:
            return tool.correct(text)
        except RateLimitError:

            time.sleep(3 * (attempt + 1))
        except Exception:
            break

    return text

def pipeline_paraphrase_lt(text: str) -> str:
    return pipeline_transformer(pipeline_languagetool(text))

PIPELINES = {
    "custom": pipeline_custom,
    "transformer": pipeline_transformer,
    "languagetool": pipeline_languagetool,
    "paraphrase_lt": pipeline_paraphrase_lt,
}


def sentence_embeddings(texts: List[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SBERT_MODEL)
    return np.array(model.encode(texts, normalize_embeddings=True))

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


ARTICLES_MASKED = [
    "ΣΥΓΚΥΡΙΟΤΗΤΑ — Άρθρο 1113. Κοινό πράγμα. — Αν η κυριότητα του [MASK] ανήκει σε περισσότερους [MASK] αδιαιρέτου κατ΄ιδανικά [MASK], εφαρμόζονται οι διατάξεις για την κοινωνία.",
    "Άρθρο 1114. Πραγματική δουλεία σε [MASK] η υπέρ του κοινού ακινήτου..."
]

def run_bonus(top_k: int = 1) -> Dict[str, List[Dict]]:
    from transformers import pipeline as hf_pipeline
    set_seed(SEED)
    results = {}
    for m in MASK_MODELS:
        fill = hf_pipeline("fill-mask", model=m)
        items = []
        for text in ARTICLES_MASKED:
            s = text
            tokens = []
            while "[MASK]" in s:
                out = fill(s, top_k=top_k)
                cand = out[0] if isinstance(out[0], dict) else out[0][0]
                tok = cand["token_str"]
                s = s.replace("[MASK]", tok, 1)
                tokens.append({"token": tok, "score": float(cand.get("score", 0.0))})
            items.append({"filled": s, "tokens": tokens})
        results[m] = items
    return results


def run_reconstruction(texts: Dict[str, str], pipelines: List[str]) -> Dict:
    out = {"pipelines": pipelines, "outputs": {}}
    for name in pipelines:
        fn = PIPELINES[name]
        out["outputs"][name] = {k: fn(v) for k, v in texts.items()}
    return out

def run_compare(texts: Dict[str, str], recon: Dict) -> List[Dict]:
    import pandas as pd
    rows = []
    for pipe, mapping in recon["outputs"].items():
        for key, recon_text in mapping.items():
            emb = sentence_embeddings([texts[key], recon_text])
            rows.append({"pipeline": pipe, "unit": key, "cosine": cosine(emb[0], emb[1])})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["reconstruct","compare","bonus","all"], default="all")
    parser.add_argument("--texts", type=str, help="JSON με text1,text2", default=None)
    parser.add_argument("--out", type=str, default="reconstruction.json")
    parser.add_argument("--csv", type=str, default="similarities.csv")
    parser.add_argument("--bonus-out", type=str, default="bonus.json")
    args = parser.parse_args()

    texts = load_texts(args.texts)

    if args.run in ("reconstruct","all"):
        recon = run_reconstruction(texts, ["custom","transformer","paraphrase_lt"])
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(recon, f, ensure_ascii=False, indent=2)
        print(f"[OK] Reconstruction -> {args.out}")

    if args.run in ("compare","all"):
        if not os.path.exists(args.out):
            print("Run reconstruct first!")
        else:
            with open(args.out, "r", encoding="utf-8") as f:
                recon = json.load(f)
            rows = run_compare(texts, recon)
            import pandas as pd
            pd.DataFrame(rows).to_csv(args.csv, index=False)
            print(f"[OK] Cosine similarities -> {args.csv}")

    if args.run in ("bonus","all"):
        bonus = run_bonus()
        with open(args.bonus_out, "w", encoding="utf-8") as f:
            json.dump(bonus, f, ensure_ascii=False, indent=2)
        print(f"[OK] Bonus results -> {args.bonus_out}")

if __name__ == "__main__":
    main()
