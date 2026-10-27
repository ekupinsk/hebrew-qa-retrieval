"""Hebrew question answering demo.

Three retrieval methods side by side: keyword search (BM25), an untrained embedding model, and our
system (hybrid search with a fine-tuned embedding model, then a fine-tuned reranker). The answer is
extracted from the top paragraphs by DictaBERT-HeQ, and, when a GPU is available, also written by DictaLM.
A second tab runs the same pipeline on a document the visitor uploads.
"""
import spaces   # must be imported before torch, so ZeroGPU can set up the GPU
import json, os, re, sys, time, traceback
import numpy as np, pandas as pd, torch, gradio as gr
from huggingface_hub import hf_hub_download
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering

# ---- Change these to your own Hugging Face username ----
USER = "ElyKok"
DATA_REPO = f"{USER}/hebrew-qa-retrieval-data"
FT_MODEL = f"{USER}/hebrew-qa-retriever-e5-large-nofilter"
BASE_MODEL = "intfloat/multilingual-e5-large"
RERANKER = f"{USER}/hebrew-qa-reranker"
QA_MODEL = "dicta-il/dictabert-heq"            # extracts the answer span from a paragraph (trained on HeQ)
LLM = "dicta-il/DictaLM-3.0-1.7B-Instruct"    # writes a full answer, only if the GPU works
TOP_K = 5          # results shown per column
CANDIDATES = 20    # hybrid search candidates passed to the reranker
CHUNK_WORDS = 150  # uploaded documents are split into passages of about this size, like HeQ paragraphs
MAX_WORDS = 30000  # about 60 pages; longer documents are cut, to keep indexing to a minute or two


def log(*args):
    # print to the Space logs right away (Logs -> Container)
    print("[app]", *args, flush=True)


log(f"torch {torch.__version__} | gradio {gr.__version__} | python {sys.version.split()[0]}")

# ---- Data ----
get = lambda f: hf_hub_download(DATA_REPO, f, repo_type="dataset")
corpus = pd.read_parquet(get("corpus.parquet"))
texts, titles = corpus["text"].tolist(), corpus["title"].tolist()
emb = {"base": np.load(get("emb_large.npy")).astype(np.float32),
       "ft": np.load(get("emb_finetuned_large_nofilter.npy")).astype(np.float32)}

examples = json.load(open(get("examples.json"), encoding="utf-8"))
# Keep the example questions neutral: skip any whose question or article title touches on religion
RELIGIOUS = ["ישו", "בשורה", "נצרות", "נוצרי", "כנסייה", "אפיפיור", "אלוהים", "תורה", "תלמוד", "הלכה",
             "רבי ", "קדוש", "דתי", " דת", "אמונה", "תפילה", "מקדש", "אסלאם", "מוסלמ", "מסגד", "קוראן",
             "תנ\"ך", "נביא"]
examples = [e for e in examples
            if not any(w in " " + e["question"] + " " + titles[e["pid"]] + " " for w in RELIGIOUS)]
gold = {e["question"]: e["pid"] for e in examples}

try:
    results = pd.read_csv("results_all.csv")   # summary of all experiments, uploaded next to app.py
except FileNotFoundError:
    log("results_all.csv not found next to app.py, using the result files from the dataset")
    results = pd.concat([pd.read_csv(get(f)) for f in ["results.csv", "ablation_results.csv", "results_reranker.csv"]])

# ---- Models: search, reranking and answer extraction run on the CPU ----
tokenize = lambda t: re.findall(r"\w+", t.lower())
bm25 = BM25Okapi([tokenize(t) for t in texts])
encoders = {"base": SentenceTransformer(BASE_MODEL, device="cpu"),
            "ft": SentenceTransformer(FT_MODEL, device="cpu")}
reranker = CrossEncoder(RERANKER, max_length=512, device="cpu")
qa_tok = AutoTokenizer.from_pretrained(QA_MODEL)
qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL).eval()

# The LLM is optional: it needs the GPU, and is skipped for the rest of the session if the GPU fails
tok = AutoTokenizer.from_pretrained(LLM)
llm = AutoModelForCausalLM.from_pretrained(LLM).to("cuda", dtype=torch.bfloat16)
llm_ok = True
log("All models loaded")


def to_ranks(scores):
    r = np.empty(len(scores))
    r[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
    return r


def top(scores):
    idx = np.argsort(-scores)[:TOP_K]
    return [(int(i), float(scores[i])) for i in idx]


def search(question):
    # Stage 1: keyword search, embeddings, and hybrid search (BM25 + fine-tuned model, RRF)
    bm = np.asarray(bm25.get_scores(tokenize(question)))
    dense = {m: emb[m] @ encoders[m].encode("query: " + question, normalize_embeddings=True)
             for m in ["base", "ft"]}
    hybrid = 1 / (60 + to_ranks(bm)) + 1 / (60 + to_ranks(dense["ft"]))
    candidates = [int(i) for i in np.argsort(-hybrid)[:CANDIDATES]]

    # Stage 2: the fine-tuned reranker reads each (question, paragraph) pair and scores it
    scores = np.asarray(reranker.predict([(question, texts[i]) for i in candidates], batch_size=CANDIDATES))
    ours = [(candidates[j], float(scores[j])) for j in np.argsort(-scores)[:TOP_K]]
    return top(bm), top(dense["base"]), ours


def find_span(question, context, max_answer_tokens=30):
    # DictaBERT-HeQ gives every token a score for "answer starts here" and "answer ends here".
    # The answer is the span (start <= end, within the paragraph) with the highest combined score.
    enc = qa_tok(question, context, truncation="only_second", max_length=512,
                 return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    in_context = [sid == 1 for sid in enc.sequence_ids(0)]
    with torch.no_grad():
        out = qa_model(**enc)
    start_p = torch.softmax(out.start_logits[0], -1).numpy()
    end_p = torch.softmax(out.end_logits[0], -1).numpy()
    best = (0.0, None, None)
    for s in np.argsort(-start_p)[:20]:
        if not in_context[s]:
            continue
        for e in np.argsort(-end_p)[:20]:
            if in_context[e] and s <= e < s + max_answer_tokens:
                score = float(start_p[s] * end_p[e])
                if score > best[0]:
                    best = (score, s, e)
    score, s, e = best
    if s is None:
        return None
    return {"answer": context[offsets[s][0]:offsets[e][1]].strip(), "score": score}


def extract_answer(question, hits, passages):
    # Stage 3a: DictaBERT-HeQ marks the answer span inside each of the top 3 paragraphs
    spans = [dict(a, pid=i) for i, _ in hits[:3] if (a := find_span(question, passages[i]))]
    best = max(spans, key=lambda a: a["score"], default=None)
    if best is None or best["score"] < 0.05:
        return None
    return best


@spaces.GPU(duration=60)
def write_answer(question, top_passages):
    # Stage 3b: DictaLM writes a full answer from the top 3 paragraphs
    context = "\n\n".join(f"[{n}] {p}" for n, p in enumerate(top_passages, 1))
    messages = [{"role": "user", "content":
                 "ענה על השאלה בעברית, במשפט אחד או שניים, רק על סמך הקטעים הבאים. "
                 "ציין בסוף את מספר הקטע שממנו לקחת את התשובה, למשל [1]. "
                 "אם התשובה לא מופיעה בקטעים, כתוב: \"לא מצאתי תשובה בקטעים\".\n\n"
                 f"קטעים:\n{context}\n\nשאלה: {question}"}]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                     enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    out = llm.generate(**inputs, max_new_tokens=200, do_sample=False)
    text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def render(hits, question):
    target = gold.get(question)
    out = []
    for rank, (i, s) in enumerate(hits, 1):
        mark = " ✅" if i == target else ""
        snippet = texts[i][:260] + ("..." if len(texts[i]) > 260 else "")
        out.append(f"**{rank}. {titles[i]}**{mark}  \n<small>ציון: {s:.4f}</small>  \n{snippet}")
    if target is not None and target not in [i for i, _ in hits]:
        out.append("❌ הפסקה הנכונה לא נמצאה ב-5 התוצאות הראשונות")
    return '<div dir="rtl">\n\n' + "\n\n---\n\n".join(out) + "\n\n</div>"


def answer_block(question, hits, passages, source_name):
    global llm_ok
    t = time.time()
    span = extract_answer(question, hits, passages)
    log(f"Answer extraction done in {time.time() - t:.1f}s")
    if span:
        md = (f"### תשובה: {span['answer']}\n\n"
              f"<small>חולצה מ{source_name(span['pid'])} על ידי DictaBERT-HeQ "
              f"(ביטחון {span['score']:.0%})</small>")
    else:
        md = "### לא נמצאה תשובה ודאית בקטעים שנמצאו"
    if llm_ok:
        t = time.time()
        try:
            written = write_answer(question, [passages[i] for i, _ in hits[:3]])
            md += f"\n\n**תשובה מנוסחת:** {written}\n\n<small>נוסחה על ידי DictaLM 3.0</small>"
            log(f"LLM answer done in {time.time() - t:.1f}s")
        except Exception:
            llm_ok = False   # don't keep trying a GPU that is not reachable
            log("LLM answer failed, skipping it from now on:\n" + traceback.format_exc())
    return f'<div dir="rtl">\n\n{md}\n\n</div>'


def run(question):
    question = (question or "").strip()
    if not question:
        return "", "", "", ""
    log(f"Question: {question}")
    t = time.time()
    bm, base, ours = search(question)
    log(f"Search and reranking done in {time.time() - t:.1f}s")
    answer = answer_block(question, ours, texts, lambda i: f'הפסקה "{titles[i]}"')
    return answer, render(bm, question), render(base, question), render(ours, question)


# ---- Your own document ----
def read_document(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        import docx
        d = docx.Document(path)
        parts = [p.text for p in d.paragraphs]
        for table in d.tables:              # text inside tables, row by row
            for row in table.rows:
                parts.append(" | ".join(c.text.strip() for c in row.cells))
        return "\n".join(parts)
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    raw = open(path, "rb").read()
    for enc in ["utf-8", "cp1255"]:       # cp1255: older Hebrew Windows text files
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="ignore")


def split_into_passages(text):
    # Join paragraphs into passages of about CHUNK_WORDS words; split very long paragraphs with overlap
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n|\n", text)]
    paragraphs = [p for p in paragraphs if p]
    passages, current = [], []
    for p in paragraphs:
        words = p.split()
        if len(words) > CHUNK_WORDS * 1.5:
            if current:
                passages.append(" ".join(current)); current = []
            step = CHUNK_WORDS - 30
            for start in range(0, len(words), step):
                passages.append(" ".join(words[start:start + CHUNK_WORDS]))
                if start + CHUNK_WORDS >= len(words):
                    break
        elif len(current) + len(words) > CHUNK_WORDS and current:
            passages.append(" ".join(current)); current = words
        else:
            current += words
    if current:
        passages.append(" ".join(current))
    return passages


def index_document(file, pasted):
    text = ""
    if file:
        try:
            text = read_document(file)
        except Exception as e:
            return None, f'<div dir="rtl">⚠️ לא הצלחתי לקרוא את הקובץ ({type(e).__name__}).</div>'
    if pasted and pasted.strip():
        text = (text + "\n" + pasted).strip()
    words = text.split()
    if not words:
        return None, '<div dir="rtl">⚠️ לא נמצא טקסט. העלו קובץ (Word, PDF או טקסט) או הדביקו טקסט.</div>'
    note = ""
    if len(words) > MAX_WORDS:
        text, note = " ".join(words[:MAX_WORDS]), f" (המסמך קוצר ל-{MAX_WORDS:,} המילים הראשונות)"
    t = time.time()
    passages = split_into_passages(text)
    # Vectors from both models: the untrained one only for the comparison column
    vecs = {m: np.asarray(encoders[m].encode(["passage: " + p for p in passages], batch_size=16,
                                             normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
            for m in ["base", "ft"]}
    doc = {"passages": passages, "emb": vecs, "bm25": BM25Okapi([tokenize(p) for p in passages])}
    log(f"Indexed an uploaded document: {len(passages)} passages in {time.time() - t:.1f}s")
    return doc, (f'<div dir="rtl">✅ המסמך מוכן: {len(passages)} קטעים, '
                 f'{min(len(words), MAX_WORDS):,} מילים{note}. אפשר לשאול.</div>')


def render_doc(hits, passages):
    out = [f"**{rank}. קטע {i + 1}**  \n<small>ציון: {s:.4f}</small>  \n"
           + passages[i][:300] + ("..." if len(passages[i]) > 300 else "")
           for rank, (i, s) in enumerate(hits, 1)]
    return '<div dir="rtl">\n\n' + "\n\n---\n\n".join(out) + "\n\n</div>"


def run_document(question, doc):
    question = (question or "").strip()
    if not question:
        return "", "", "", ""
    if not doc:
        return '<div dir="rtl">⚠️ קודם העלו מסמך ולחצו "עבד את המסמך".</div>', "", "", ""
    passages = doc["passages"]
    t = time.time()
    bm = np.asarray(doc["bm25"].get_scores(tokenize(question)))
    dense = {m: doc["emb"][m] @ encoders[m].encode("query: " + question, normalize_embeddings=True)
             for m in ["base", "ft"]}
    hybrid = 1 / (60 + to_ranks(bm)) + 1 / (60 + to_ranks(dense["ft"]))
    candidates = [int(i) for i in np.argsort(-hybrid)[:CANDIDATES]]
    scores = np.asarray(reranker.predict([(question, passages[i]) for i in candidates], batch_size=CANDIDATES))
    ours = [(candidates[j], float(scores[j])) for j in np.argsort(-scores)[:TOP_K]]
    log(f"Document search done in {time.time() - t:.1f}s")
    answer = answer_block(question, ours, passages, lambda i: f"קטע {i + 1}")
    return (answer, render_doc(top(bm), passages), render_doc(top(dense["base"]), passages),
            render_doc(ours, passages))


INTRO = """<div dir="rtl">

# חיפוש סמנטי בעברית ומערכת שאלה-תשובה

כתבו שאלה בעברית. המערכת מחפשת את הפסקה שעונה עליה מתוך כ-4,900 פסקאות של מאגר HeQ (ויקיפדיה העברית ו-Geektime),
בשלוש שיטות זו לצד זו. המערכת שלנו: חיפוש היברידי (מילים + מודל e5-large שאומן על שאלות בעברית) מביא 20 מועמדות,
Reranker מאומן מסדר אותן, ו-DictaBERT-HeQ מחלץ את התשובה מהפסקאות המובילות.

בשאלות לדוגמה, ✅ מסמן את הפסקה הנכונה. המערכת עונה רק על נושאים שמופיעים במאגר.

</div>"""

DOC_INTRO = """<div dir="rtl">

העלו מסמך משלכם (Word, PDF או קובץ טקסט), או הדביקו טקסט, ושאלו עליו שאלות. אותה מערכת בדיוק:
חיפוש היברידי, Reranker וחילוץ תשובה.

המסמך נשמר רק בזיכרון של החלון הזה, ונמחק כשסוגרים אותו. עד כ-60 עמודים. עיבוד המסמך לוקח מכמה שניות ועד כמה דקות.
המודלים אומנו על ויקיפדיה ועל כתבות טכנולוגיה, אז על סוגי טקסט אחרים הדיוק עשוי להיות נמוך יותר.
בקובצי PDF בעברית, חילוץ הטקסט לפעמים יוצא משובש. Word וטקסט רגיל עובדים טוב יותר.

</div>"""

with gr.Blocks(title="Hebrew QA Retrieval") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("מאגר HeQ"):
        with gr.Row():
            q = gr.Textbox(label="שאלה", placeholder="למשל: מי עמד בראש הוועדה שניסחה את מגילת העצמאות?", rtl=True, scale=5)
            btn = gr.Button("חפש", variant="primary", scale=1)
        gr.Examples([[e["question"]] for e in examples], inputs=q, label="שאלות לדוגמה מחלק הבדיקה")
        ans = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown('<div dir="rtl">\n\n### חיפוש מילים (BM25)\n\n</div>')
                o1 = gr.Markdown()
            with gr.Column():
                gr.Markdown('<div dir="rtl">\n\n### מודל בסיס, בלי אימון (e5-large)\n\n</div>')
                o2 = gr.Markdown()
            with gr.Column():
                gr.Markdown('<div dir="rtl">\n\n### המערכת שלנו (היברידי + Reranker)\n\n</div>')
                o3 = gr.Markdown()
        with gr.Accordion("תוצאות על חלק הבדיקה", open=False):
            gr.Dataframe(results)
        btn.click(run, q, [ans, o1, o2, o3])
        q.submit(run, q, [ans, o1, o2, o3])

    with gr.Tab("המסמך שלך"):
        gr.Markdown(DOC_INTRO)
        doc_state = gr.State(None)
        with gr.Row():
            doc_file = gr.File(label="קובץ", file_types=[".docx", ".pdf", ".txt"], type="filepath")
            doc_text = gr.Textbox(label="או הדביקו טקסט", lines=6, rtl=True)
        index_btn = gr.Button("עבד את המסמך", variant="secondary")
        doc_status = gr.Markdown()
        with gr.Row():
            dq = gr.Textbox(label="שאלה על המסמך", rtl=True, scale=5)
            dbtn = gr.Button("חפש", variant="primary", scale=1)
        d_ans = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown('<div dir="rtl">\n\n### חיפוש מילים (BM25)\n\n</div>')
                d1 = gr.Markdown()
            with gr.Column():
                gr.Markdown('<div dir="rtl">\n\n### מודל בסיס, בלי אימון (e5-large)\n\n</div>')
                d2 = gr.Markdown()
            with gr.Column():
                gr.Markdown('<div dir="rtl">\n\n### המערכת שלנו (היברידי + Reranker)\n\n</div>')
                d3 = gr.Markdown()
        index_btn.click(index_document, [doc_file, doc_text], [doc_state, doc_status])
        dbtn.click(run_document, [dq, doc_state], [d_ans, d1, d2, d3])
        dq.submit(run_document, [dq, doc_state], [d_ans, d1, d2, d3])

demo.launch(theme=gr.themes.Soft(), show_error=True, ssr_mode=False)
