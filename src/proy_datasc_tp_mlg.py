"""
Data Science III - NLP y Deep Learning  |  Proyecto integrador (AG News)
=========================================================================
UN solo script con TODAS las etapas del proyecto, una por modulo.
Cada etapa guarda sus resultados en  outputs/  y la siguiente los reutiliza.

    python proyecto_nlp.py m1        # Pre-entrega 1: pipeline PyTorch (train/val, Adam)
    python proyecto_nlp.py m2        # Pre-entrega 2: preprocesamiento + EDA + PDF
    python proyecto_nlp.py m3        # Pre-entrega 3: TF-IDF + clasificador (baseline)
    python proyecto_nlp.py m4        # Pre-entrega 4: DistilBERT + LoRA (requiere GPU)
    python proyecto_nlp.py m4-pdf    # PDF del checkpoint 4 (lee resultados de m3 y m4)
    python proyecto_nlp.py m5        # Informe final integrado (lee m2, m3, m4)

Opciones utiles:  --apellido Gomez --nombre Maria Lorena   (para el nombre de los PDF)

Reglas de oro que respeta el script:
  * El TEST (ag_news_test.csv) nunca se usa para entrenar ni para elegir hiperparametros.
  * TfidfVectorizer: fit solo sobre train, transform sobre test (sin data leakage).
  * Baseline (M3) y Transformer (M4) se evaluan sobre el MISMO test, con las mismas clases.
"""
import argparse
import html
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 0. Configuracion general
# --------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"                                  # CSV de la catedra
OUT = Path(os.environ.get("NLP_OUT", BASE / "outputs"))  # resultados
FIGS = OUT / "figs"
SEED = 42
DATASET = "ag_news"
MODEL_CKPT = "distilbert-base-uncased"                # modelo base del M4


def _mkdirs():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)


def cargar(dataset: str = DATASET):
    """Devuelve (train_df, test_df) con columnas text, label.
    Busca los CSV en data/, en la carpeta del script o en dataset/<nombre>/ (como cargar_dataset.py)."""
    for carpeta in (DATA, BASE, BASE / dataset):
        a, b = carpeta / f"{dataset}_train.csv", carpeta / f"{dataset}_test.csv"
        print(a, b)
        if a.exists() and b.exists():
            return pd.read_csv(a), pd.read_csv(b)
    raise FileNotFoundError("No encuentro ag_news_train.csv / ag_news_test.csv (ponelos en data/)")


def save_json(obj, name):
    _mkdirs()
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(name):
    p = OUT / name
    if not p.exists():
        raise FileNotFoundError(f"Falta {p}. Corre antes la etapa que lo genera.")
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 1. MODULO 2 - Preprocesamiento (se reutiliza en M3 y M5)
# --------------------------------------------------------------------------
_NUM_ENT = re.compile(r"#(\d+);")
# Firmas de agencia repetidas ('(AP) AP - ', 'NEW YORK (Reuters) - ') y pie bursatil: ruido que duplica n-gramas
AGENCY = re.compile(r"\((?:AP|Reuters|AFP|CP|Canadian Press)\)|\b(?:AP|Reuters|AFP|Canadian Press) - |"
                    r"\([^()]*Quote, Profile, Research\)")


def _numeric_entity(m):
    """AG News trae entidades rotas sin '&': '#39;' = apostrofe, '#36;' = $, '#151;' = guion largo."""
    n = int(m.group(1))
    if n < 128:
        return chr(n)
    try:
        return bytes([n]).decode("cp1252") if n < 256 else chr(n)
    except Exception:
        return " "


def clean_text(text: str) -> str:
    """Limpieza estructural con Regex: entidades HTML, tags, URLs y espacios."""
    t = html.unescape(str(text))             # '&lt;b&gt;' -> '<b>'
    t = _NUM_ENT.sub(_numeric_entity, t)     # '#39;' -> "'"
    t = re.sub(r"\b(quot|amp|nbsp|lt|gt);", lambda m: {"quot": '"', "amp": "&"}.get(m.group(1), " "), t)  # '&' perdido
    t = re.sub(r"<[^>]*>", " ", t)           # tags HTML
    t = re.sub(r"https?://\S+|\bwww\.?\S*", " ", t, flags=re.I)
    t = AGENCY.sub(" ", t)                   # firmas de agencia: '(AP)', 'Reuters - ', 'Quote, Profile, Research'
    return re.sub(r"\s+", " ", t).strip()


def _pre_lemma(text: str) -> str:
    """Normalizacion previa a la lematizacion: minusculas, siglas (U.S. -> us) y posesivos."""
    t = clean_text(text).lower()
    t = re.sub(r"\b(?:[a-z]\.){2,}", lambda m: m.group(0).replace(".", ""), t)
    return re.sub(r"'s\b", "", t)


def _lemma_fallback(w: str) -> str:
    """Reglas simples; SOLO se usa si spaCy no esta instalado (ver preprocess_corpus)."""
    if len(w) <= 3 or w.isdigit():
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith(("ss", "us", "is")):
        return w
    if w.endswith("s"):
        return w[:-1]
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    return w


_NLP = None
_LEMMATIZER = None


def _get_nlp():
    """Carga spaCy en_core_web_sm (python -m spacy download en_core_web_sm). None si no esta."""
    global _NLP, _LEMMATIZER
    if _LEMMATIZER is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            _LEMMATIZER = "spaCy en_core_web_sm"
        except Exception:
            _NLP, _LEMMATIZER = None, "reglas simples (fallback, spaCy no instalado)"
    return _NLP


def preprocess_corpus(texts):
    """Pipeline completo sobre una lista de textos -> lista de strings de lemas."""
    nlp = _get_nlp()
    pre = [_pre_lemma(t) for t in texts]
    if nlp is not None:
        out = []
        for doc in nlp.pipe(pre, batch_size=256):
            out.append(" ".join(t.lemma_.lower() for t in doc
                                if (t.is_alpha or t.like_num) and not t.is_space))
        return out
    return [" ".join(_lemma_fallback(w) for w in re.findall(r"[a-z]+|\d+", t)) for t in pre]


def preprocess_text(text: str) -> str:
    """Funcion pedida en la consigna: limpieza Regex + lowercase + lematizacion (spaCy)."""
    return preprocess_corpus([text])[0]


def load_prepped():
    """train/test con columna 'prep' (texto preprocesado). Cachea en outputs/ para no repetir."""
    _mkdirs()
    _get_nlp()
    cache = OUT / ("corpus_preprocesado_spacy.csv" if _NLP is not None else "corpus_preprocesado_reglas.csv")
    tr, te = cargar()
    if cache.exists():
        c = pd.read_csv(cache, keep_default_na=False)
        if len(c) == len(tr) + len(te):
            tr["prep"], te["prep"] = c["prep"][:len(tr)].values, c["prep"][len(tr):].values
            return tr, te
    tr["prep"], te["prep"] = preprocess_corpus(tr.text), preprocess_corpus(te.text)
    pd.concat([tr.assign(split="train"), te.assign(split="test")]).to_csv(
        cache, index=False, columns=["text", "prep", "label", "split"])
    return tr, te


# --------------------------------------------------------------------------
# 2. MODULO 2 - EDA tecnico
# --------------------------------------------------------------------------
def m2(args):
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    _mkdirs()
    tr, te = load_prepped()
    _get_nlp()
    st = {"lemmatizer": _LEMMATIZER, "n_train": len(tr), "n_test": len(te)}

    # Ruido: antes vs. despues de la limpieza
    pat = r"&\w+;|#\d+;|\b(?:quot|amp|nbsp|lt|gt);|<[^>]*>|https?://|www\."
    st["ruido_antes"] = int(tr.text.str.contains(pat, regex=True).sum())
    st["bylines_agencia"] = int(tr.text.map(lambda x: bool(AGENCY.search(html.unescape(x)))).sum())
    # Ruido residual: se busca en el texto YA limpio (antes de lematizar, para no confundir 'quoted'->'quot')
    st["ruido_despues"] = int(tr.text.map(lambda x: bool(re.search(pat, clean_text(x), flags=re.I))).sum())

    # Longitud en tokens y percentil 95 (sera el max_len)
    lens = tr.prep.str.split().str.len()
    st["len"] = {"mean": float(lens.mean()), "median": float(lens.median()),
                 "p95": int(np.ceil(np.percentile(lens, 95))), "max": int(lens.max()),
                 "min": int(lens.min())}
    plt.figure(figsize=(7, 4))
    plt.hist(lens, bins=40, color="#3b6fb6", edgecolor="white")
    plt.axvline(st["len"]["p95"], color="crimson", ls="--", label=f"percentil 95 = {st['len']['p95']}")
    plt.axvline(st["len"]["median"], color="black", ls=":", label=f"mediana = {st['len']['median']:.0f}")
    plt.xlabel("tokens por documento (tras preprocesamiento)"); plt.ylabel("documentos")
    plt.title("Distribucion de longitud (train)"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGS / "m2_hist_longitud.png", dpi=150); plt.close()

    # Distribucion de clases (train y test)
    dist = pd.DataFrame({"train": tr.label.value_counts(), "test": te.label.value_counts()}).sort_index()
    st["clases"] = {k: {"train": int(r.train), "test": int(r.test)} for k, r in dist.iterrows()}
    ax = dist.plot(kind="bar", figsize=(7, 4), color=["#3b6fb6", "#e08a2c"], rot=0)
    ax.set_ylabel("documentos"); ax.set_title("Documentos por categoria")
    for c in ax.containers:
        ax.bar_label(c, fontsize=8)
    plt.tight_layout(); plt.savefig(FIGS / "m2_clases.png", dpi=150); plt.close()

    # N-gramas con scikit-learn (sin stop-words para que sean informativos)
    def top_ngrams(n, k=20):
        cv = CountVectorizer(ngram_range=(n, n), stop_words="english")
        X = cv.fit_transform(tr.prep)
        f = np.asarray(X.sum(axis=0)).ravel()
        idx = f.argsort()[::-1][:k]
        v = cv.get_feature_names_out()
        return [[v[i], int(f[i])] for i in idx]
    st["bigramas"], st["trigramas"] = top_ngrams(2), top_ngrams(3)

    # Top 50 palabras (con stop-words) y cuantas lo son
    cnt = Counter(w for d in tr.prep for w in d.split())
    top50 = cnt.most_common(50)
    st["top50"] = [[w, c, w in ENGLISH_STOP_WORDS] for w, c in top50]
    st["top50_stop"] = int(sum(1 for w, _, s in st["top50"] if s))
    st["vocab"] = len(cnt)
    st["hapax"] = int(sum(1 for c in cnt.values() if c == 1))
    save_json(st, "m2_stats.json")
    print(f"[M2] lematizador: {_LEMMATIZER} | p95={st['len']['p95']} | "
          f"stop-words en top50: {st['top50_stop']} | ruido antes/despues: "
          f"{st['ruido_antes']}/{st['ruido_despues']}")
    m2_pdf(st, args)


# --------------------------------------------------------------------------
# 3. Utilidades de PDF (reportlab)
# --------------------------------------------------------------------------
def _pdf_tools():
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)
    ss = getSampleStyleSheet()
    S = {"t": ParagraphStyle("t", parent=ss["Title"], fontSize=18, spaceAfter=6),
         "h1": ParagraphStyle("h1", parent=ss["Heading2"], textColor=colors.HexColor("#1f3b66"), spaceBefore=10),
         "h2": ParagraphStyle("h2", parent=ss["Heading3"], spaceBefore=6),
         "b": ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13.5),
         "s": ParagraphStyle("s", parent=ss["BodyText"], fontSize=8, leading=10, textColor=colors.grey),
         "c": ParagraphStyle("c", parent=ss["BodyText"], fontSize=8, leading=10)}

    def P(txt, s="b"):
        return Paragraph(txt, S[s])

    def T(rows, widths=None, head=True, fs=8):
        data = [[Paragraph(escape(str(c)), S["c"]) if not isinstance(c, Paragraph) else c for c in r] for r in rows]
        t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
        st = [("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("FONTSIZE", (0, 0), (-1, -1), fs)]
        if head:
            st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f5")))
        t.setStyle(TableStyle(st))
        return t

    def I(path, w=14):
        from PIL import Image as PI
        iw, ih = PI.open(path).size
        return Image(str(path), width=w * cm, height=w * cm * ih / iw)

    def build(path, story, title):
        doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                                topMargin=1.8 * cm, bottomMargin=1.8 * cm, title=title)
        doc.build(story)
        print(f"[PDF] {path}")
    return P, T, I, build, Spacer, PageBreak, cm


def _name(args):
    return f"{args.apellido}{args.nombre}", f"{args.apellido}_{args.nombre}"


def _fmt(x):
    return f"{x:.4f}"


# --------------------------------------------------------------------------
# 4. MODULO 2 - Informe PDF del EDA
# --------------------------------------------------------------------------
def m2_pdf(st, args):
    P, T, I, build, Sp, PB, cm = _pdf_tools()
    L = st["len"]
    dist_txt = ", ".join(f"{k} {v['train']}" for k, v in st["clases"].items())
    # Conclusion (max 300 palabras) generada con los numeros reales
    concl = (
        f"El corpus tiene {st['n_train']} noticias de entrenamiento y {st['n_test']} de test, con 4 categorias "
        f"perfectamente balanceadas ({dist_txt}); por eso Accuracy y F1 seran equivalentes y no hace falta "
        f"ponderar la perdida ni remuestrear. El principal desafio fue el ruido de codificacion: "
        f"{st['ruido_antes']} documentos de train traian entidades HTML escapadas ('&amp;lt;b&amp;gt;'), "
        f"fragmentos rotos como '#39;' en lugar del apostrofe y URLs; ademas {st['bylines_agencia']} traian firmas de agencia repetidas. El pipeline las decodifica con Regex "
        f"(html.unescape y reemplazo de entidades numericas), elimina tags y URLs y normaliza espacios; tras la "
        f"limpieza quedan {st['ruido_despues']} documentos con ruido residual y los N-gramas ya no muestran artefactos. "
        f"Los textos son cortos (mediana {L['median']:.0f} tokens, media {L['mean']:.1f}); el percentil 95 es "
        f"{L['p95']} tokens, valor que se propone como max_len de referencia (en M4 se recalcula en subpalabras). "
        f"En el top 50 de palabras, {st['top50_stop']} son stop-words: confirma la cola dominada por palabras "
        f"funcionales, que en los modelos clasicos se filtran (TF-IDF con stop_words) y en el Transformer se "
        f"conservan. El vocabulario lematizado es de {st['vocab']} tipos, con {st['hapax']} que aparecen una sola "
        f"vez (cola larga), lo que justifica limitar max_features en TF-IDF y usar subpalabras en el Transformer. "
        f"Lematizador usado: {st['lemmatizer']}.")
    assert len(concl.split()) <= 300, len(concl.split())

    s = [P("EDA de texto: corpus preprocesado listo para modelado", "t"),
         P(f"Data Science III - Pre-entrega Modulo 2 | Dataset: AG News | {args.nombre} {args.apellido}", "s"),
         P("1. Resumen del pipeline de preprocesamiento", "h1"),
         P("La funcion <b>preprocess_text(text)</b> aplica, en este orden: (1) <b>limpieza Regex</b>: decodifica entidades HTML "
           "(&amp;lt;, &amp;gt;), repara entidades numericas sin '&amp;' (#39; = apostrofe, #36; = $, #151; = guion), elimina tags HTML y "
           "URLs y colapsa espacios; (2) <b>normalizacion</b>: minusculas, siglas (U.S. a us), posesivos 's, y descarte de "
           "puntuacion; (3) <b>lematizacion</b> con spaCy (en_core_web_sm), conservando tokens alfabeticos y numericos. "
           "Las stop-words no se eliminan aqui (se miden en el EDA y se filtran solo en los modelos clasicos). "
           "Se aplico a las 8.000 noticias de train y 2.000 de test, de forma identica. "
           f"<b>Lematizador de esta corrida:</b> {escape(st['lemmatizer'])}. "
           f"Ruido (entidades/tags/URLs) en train: {st['ruido_antes']} documentos antes, {st['ruido_despues']} despues. "
           f"Ademas se eliminan las firmas de agencia ('(AP) AP -', '(Reuters) -', 'Quote, Profile, Research'), presentes en "
           f"{st['bylines_agencia']} documentos: duplicaban n-gramas ('ap ap', 'reuters reuters') sin aportar contenido tematico.", "b"),
         P("2. Longitud de documentos y percentil 95", "h1"),
         I(FIGS / "m2_hist_longitud.png", 12),
         P(f"Tokens por documento: media {L['mean']:.1f}, mediana {L['median']:.0f}, min {L['min']}, max {L['max']}. "
           f"<b>Percentil 95 = {L['p95']} tokens</b> (max_len de referencia): cubre el 95% de las noticias sin truncar.", "b"),
         P("3. Distribucion de clases", "h1"),
         I(FIGS / "m2_clases.png", 11),
         P("Las cuatro clases estan perfectamente balanceadas en train (2.000 c/u) y en test (500 c/u). "
           "Consecuencia para M3 y M4: no se requiere class_weight; F1 macro = F1 weighted.", "b"),
         PB(),
         P("4. Top 20 bi-gramas y tri-gramas (scikit-learn, sin stop-words)", "h1"),
         T([["#", "Bi-grama", "Frec."]] + [[i + 1, a, b] for i, (a, b) in enumerate(st["bigramas"])], [1 * cm, 6 * cm, 2 * cm]),
         Sp(1, 8),
         T([["#", "Tri-grama", "Frec."]] + [[i + 1, a, b] for i, (a, b) in enumerate(st["trigramas"])], [1 * cm, 8 * cm, 2 * cm]),
         P("Interpretacion: los n-gramas son entidades y frases tipicas de prensa (organizaciones, agencias, "
           "cargos y eventos), sin tags HTML ni fragmentos como 'amp', 'lt' o '39'. "
           "Los bigramas reflejan los cuatro temas del corpus (mundo, deportes, negocios, ciencia/tecnologia).", "b"),
         PB(),
         P("5. Top 50 palabras mas frecuentes y stop-words", "h1")]
    rows = [["#", "Palabra", "Frec.", "Stop?", "#", "Palabra", "Frec.", "Stop?"]]
    half = 25
    for i in range(half):
        a, b = st["top50"][i], st["top50"][i + half]
        rows.append([i + 1, a[0], a[1], "si" if a[2] else "", i + 1 + half, b[0], b[1], "si" if b[2] else ""])
    s += [T(rows, [1 * cm, 3 * cm, 2 * cm, 1.5 * cm] * 2),
          P(f"De las 50 palabras mas frecuentes, <b>{st['top50_stop']} son stop-words</b> (lista ENGLISH_STOP_WORDS de scikit-learn). "
            f"Vocabulario: {st['vocab']} tipos; {st['hapax']} aparecen una sola vez.", "b"),
          P("6. Conclusion", "h1"), P(concl, "b")]
    ap, _ = _name(args)
    build(OUT / f"EDA_NLP_{ap}.pdf", s, "EDA NLP")


# --------------------------------------------------------------------------
# 5. MODULO 3 - TF-IDF + clasificador (baseline)
# --------------------------------------------------------------------------
def _metrics(y_true, y_pred, clases):
    from sklearn.metrics import classification_report, confusion_matrix
    rep = classification_report(y_true, y_pred, labels=clases, output_dict=True, digits=4)
    return {"report": rep, "accuracy": rep["accuracy"],
            "macro": rep["macro avg"], "weighted": rep["weighted avg"],
            "confusion": confusion_matrix(y_true, y_pred, labels=clases).tolist()}


def plot_cm(cm, clases, titulo, path):
    cm = np.array(cm)
    plt.figure(figsize=(5.2, 4.4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(clases)), clases, rotation=30); plt.yticks(range(len(clases)), clases)
    for i in range(len(clases)):
        for j in range(len(clases)):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    plt.xlabel("Prediccion"); plt.ylabel("Real"); plt.title(titulo); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def m3(args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    from sklearn.metrics import f1_score
    import joblib
    _mkdirs()
    tr, te = load_prepped()
    clases = sorted(tr.label.unique())

    # Validacion interna: SOLO sale del train. El test no se toca hasta el final.
    X_tr, X_va, y_tr, y_va = train_test_split(tr.prep, tr.label, test_size=0.2,
                                              stratify=tr.label, random_state=SEED)
    modelos = {"MultinomialNB": lambda: MultinomialNB(alpha=0.5),
               "LogisticRegression": lambda: LogisticRegression(C=10, max_iter=1000),
               "LinearSVC": lambda: LinearSVC(C=0.5, random_state=SEED)}
    grid = []
    for mf in (5000, 20000, None):
        for ng in ((1, 1), (1, 2)):
            vec = TfidfVectorizer(max_features=mf, ngram_range=ng, sublinear_tf=True,
                                  min_df=2, stop_words="english")
            A = vec.fit_transform(X_tr)          # fit SOLO en train
            B = vec.transform(X_va)              # transform en validacion
            for nom, mk in modelos.items():
                t0 = time.time(); clf = mk().fit(A, y_tr); dt = time.time() - t0
                f1 = f1_score(y_va, clf.predict(B), average="macro")
                grid.append({"max_features": str(mf), "ngram_range": str(ng), "modelo": nom,
                             "n_features": A.shape[1], "f1_macro_val": round(f1, 4), "fit_s": round(dt, 2)})
    g = pd.DataFrame(grid).sort_values("f1_macro_val", ascending=False)
    g.to_csv(OUT / "m3_grid.csv", index=False)
    best = g.iloc[0]
    print(g.head(8).to_string(index=False))

    # Modelo final: refit sobre TODO el train y evaluacion unica sobre test
    mf = None if best.max_features == "None" else int(best.max_features)
    ng = eval(best.ngram_range)
    vec = TfidfVectorizer(max_features=mf, ngram_range=ng, sublinear_tf=True, min_df=2, stop_words="english")
    t0 = time.time()
    Xtr = vec.fit_transform(tr.prep)             # fit_transform SOLO train
    clf = modelos[best.modelo]().fit(Xtr, tr.label)
    train_s = time.time() - t0
    Xte = vec.transform(te.prep)                 # solo transform en test
    t0 = time.time(); pred = clf.predict(Xte); infer_s = time.time() - t0
    met = _metrics(te.label, pred, clases)
    met.update({"modelo": best.modelo, "max_features": best.max_features, "ngram_range": best.ngram_range,
                "n_features": int(Xtr.shape[1]), "train_seconds": round(train_s, 2),
                "inference_seconds": round(infer_s, 3), "clases": clases,
                "lemmatizer": _LEMMATIZER or "ver M2", "preds": [str(p) for p in pred],
                "grid": g.head(12).to_dict("records")})

    # Terminos mas influyentes por clase (interpretabilidad)
    W = clf.coef_ if hasattr(clf, "coef_") else getattr(clf, "feature_log_prob_", None)   # NB: log-probabilidades
    if W is not None:
        v = vec.get_feature_names_out()
        met["top_terms"] = {c: [v[j] for j in np.argsort(W[i])[::-1][:10]] for i, c in enumerate(clases)}
    # Par de clases mas confundido
    cm = np.array(met["confusion"]); off = cm.copy(); np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    met["par_mas_confundido"] = [clases[i], clases[j], int(off[i, j])]
    plot_cm(met["confusion"], clases, f"TF-IDF + {best.modelo}", FIGS / "m3_confusion.png")
    save_json(met, "m3_metrics.json")
    joblib.dump({"vec": vec, "clf": clf}, OUT / "m3_model.joblib")
    print(classification_text(te.label, pred, clases))
    print(f"[M3] {best.modelo} | max_features={best.max_features} ngram={best.ngram_range} | "
          f"F1 macro test = {met['macro']['f1-score']:.4f} | par mas confundido: {met['par_mas_confundido']}")


def classification_text(y, p, clases):
    from sklearn.metrics import classification_report
    return classification_report(y, p, labels=clases, digits=4)


# --------------------------------------------------------------------------
# 6. MODULO 1 - Pipeline PyTorch (entrenamiento + validacion)
# --------------------------------------------------------------------------
def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def m1(args):
    """Clasificador MLP sobre vectores TF-IDF de AG News. Solo usa train (split train/val)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    _mkdirs()

    # 1) Entorno y reproducibilidad
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = get_device()
    print(f"[M1] PyTorch {torch.__version__} | dispositivo: {device}")

    # 2) Datos: split train/validacion estratificado (el test no se usa en M1)
    tr, _ = cargar()
    clases = sorted(tr.label.unique()); c2i = {c: i for i, c in enumerate(clases)}
    t_tr, t_va, y_tr, y_va = train_test_split(tr.text, tr.label.map(c2i).values, test_size=0.2,
                                              stratify=tr.label, random_state=SEED)
    vec = TfidfVectorizer(max_features=5000, stop_words="english", sublinear_tf=True)
    X_tr = torch.tensor(vec.fit_transform(t_tr).toarray(), dtype=torch.float32)   # fit solo train
    X_va = torch.tensor(vec.transform(t_va).toarray(), dtype=torch.float32)
    y_tr, y_va = torch.tensor(y_tr, dtype=torch.long), torch.tensor(y_va, dtype=torch.long)
    dl_tr = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch, shuffle=True)
    dl_va = DataLoader(TensorDataset(X_va, y_va), batch_size=256)

    # 3) Arquitectura base (nn.Module simple): MLP de una capa oculta con ReLU
    class MLPClassifier(nn.Module):
        def __init__(self, d_in, d_hidden, n_classes, p=0.3):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(p),
                                     nn.Linear(d_hidden, n_classes))   # logits (CrossEntropy aplica softmax)

        def forward(self, x):
            return self.net(x)

    model = MLPClassifier(X_tr.shape[1], 256, len(clases)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4) Ciclos de entrenamiento y validacion
    def train_epoch():
        model.train(); tot, ok, n = 0.0, 0, 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()                 # limpiar gradientes acumulados
            out = model(xb)                       # forward
            loss = criterion(out, yb)             # perdida
            loss.backward()                       # backward (autograd)
            optimizer.step()                      # actualizacion Adam
            tot += loss.item() * len(yb); ok += (out.argmax(1) == yb).sum().item(); n += len(yb)
        return tot / n, ok / n

    @torch.no_grad()
    def evaluate(dl):
        model.eval(); tot, ok, n = 0.0, 0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb); tot += criterion(out, yb).item() * len(yb)
            ok += (out.argmax(1) == yb).sum().item(); n += len(yb)
        return tot / n, ok / n

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for ep in range(1, args.epochs + 1):
        a, b = train_epoch(); c, d = evaluate(dl_va)
        for k, v in zip(hist, (a, b, c, d)):
            hist[k].append(round(v, 5))
        print(f"Epoca {ep:02d}/{args.epochs} | loss {a:.4f} acc {b:.4f} | val_loss {c:.4f} val_acc {d:.4f}")

    # 5) Curvas y registro
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val"); ax[0].set_title("Loss")
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val"); ax[1].set_title("Accuracy")
    for a_ in ax:
        a_.set_xlabel("epoca"); a_.legend()
    plt.tight_layout(); plt.savefig(FIGS / "m1_curvas.png", dpi=150); plt.close()
    save_json({"torch": torch.__version__, "device": str(device), "lr": args.lr, "epochs": args.epochs,
               "batch": args.batch, "history": hist}, "m1_history.json")


# --------------------------------------------------------------------------
# 7. MODULO 4 - Fine-tuning con LoRA (DistilBERT) -> correr en GPU (Colab/Kaggle)
# --------------------------------------------------------------------------
def m4(args):
    import inspect
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from sklearn.model_selection import train_test_split
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments, set_seed)
    from sklearn.metrics import f1_score
    _mkdirs()
    set_seed(SEED)
    cuda = torch.cuda.is_available()
    print(f"[M4] cuda disponible: {cuda}" + (f" | GPU: {torch.cuda.get_device_name(0)}" if cuda else ""))
    if not cuda:
        print("ATENCION: sin GPU el entrenamiento sera muy lento. Usa Colab/Kaggle (Runtime > GPU).")

    # Datos: el Transformer recibe texto crudo (puntuacion y stop-words aportan contexto);
    # la tokenizacion en subpalabras la hace su propio tokenizer.
    tr, te = cargar()
    clases = sorted(tr.label.unique()); c2i = {c: i for i, c in enumerate(clases)}
    tr["labels"], te["labels"] = tr.label.map(c2i), te.label.map(c2i)
    tr["text"], te["text"] = tr.text.map(clean_text), te.text.map(clean_text)   # solo limpieza estructural
    t_tr, t_va = train_test_split(tr, test_size=0.1, stratify=tr.label, random_state=SEED)

    tok = AutoTokenizer.from_pretrained(MODEL_CKPT)
    lens = [len(x) for x in tok(tr.text.tolist(), add_special_tokens=True)["input_ids"]]
    p95 = int(np.ceil(np.percentile(lens, 95)))
    max_len = min(128, p95)
    print(f"[M4] percentil 95 en subpalabras = {p95} -> max_length = {max_len}")

    def to_ds(df):
        ds = Dataset.from_pandas(df[["text", "labels"]].reset_index(drop=True))
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len), batched=True,
                      remove_columns=["text"])
    ds_tr, ds_va, ds_te = to_ds(t_tr), to_ds(t_va), to_ds(te)

    # Modelo base + LoRA
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, num_labels=len(clases),
                                                              id2label=dict(enumerate(clases)), label2id=c2i)
    total_base = sum(p.numel() for p in base.parameters())
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.r, lora_alpha=args.alpha, lora_dropout=0.1,
                     target_modules=["q_lin", "v_lin"],                 # proyecciones query y value de la atencion
                     modules_to_save=["pre_classifier", "classifier"])  # cabeza de clasificacion entrenable
    model = get_peft_model(base, cfg)
    trainable, total = model.get_nb_trainable_parameters()
    model.print_trainable_parameters()

    def compute_metrics(p):
        pr = np.argmax(p.predictions, axis=-1)
        return {"accuracy": float((pr == p.label_ids).mean()),
                "f1_macro": float(f1_score(p.label_ids, pr, average="macro"))}

    kw = dict(output_dir=str(OUT / "lora_runs"), learning_rate=args.lr, num_train_epochs=args.epochs,
              per_device_train_batch_size=32, per_device_eval_batch_size=128, weight_decay=0.01,
              save_strategy="epoch", logging_steps=50, load_best_model_at_end=True,
              metric_for_best_model="f1_macro", fp16=cuda, report_to="none", seed=SEED, save_total_limit=1)
    sig = inspect.signature(TrainingArguments.__init__).parameters
    kw["eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"] = "epoch"
    trainer = Trainer(model=model, args=TrainingArguments(**kw), train_dataset=ds_tr, eval_dataset=ds_va,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
    t0 = time.time(); trainer.train(); train_s = time.time() - t0
    log = trainer.state.log_history
    tr_loss = [x["loss"] for x in log if "loss" in x]
    ev = [x for x in log if "eval_loss" in x]

    # Evaluacion unica sobre el MISMO test que el baseline
    t0 = time.time(); out = trainer.predict(ds_te); infer_s = time.time() - t0
    pred = [clases[i] for i in np.argmax(out.predictions, axis=-1)]
    met = _metrics(te.label, pred, clases)
    met.update({"modelo": MODEL_CKPT, "r": args.r, "alpha": args.alpha, "lr": args.lr, "epochs": args.epochs,
                "target_modules": cfg.target_modules, "max_length": max_len, "p95_subword": p95,
                "params_total_base": int(total_base), "params_total_peft": int(total),
                "params_trainable": int(trainable), "pct_trainable": 100 * trainable / total,
                "train_seconds": round(train_s, 1), "inference_seconds": round(infer_s, 2),
                "final_train_loss": tr_loss[-1] if tr_loss else None,
                "final_val_loss": ev[-1]["eval_loss"] if ev else None,
                "val_history": [{"epoch": x["epoch"], "eval_loss": x["eval_loss"], "f1_macro": x["eval_f1_macro"]} for x in ev],
                "gpu": torch.cuda.get_device_name(0) if cuda else "CPU", "torch": torch.__version__,
                "clases": clases, "preds": pred})
    cm = np.array(met["confusion"]); off = cm.copy(); np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    met["par_mas_confundido"] = [clases[i], clases[j], int(off[i, j])]
    plot_cm(met["confusion"], clases, "DistilBERT + LoRA", FIGS / "m4_confusion.png")
    model.save_pretrained(OUT / "lora_adapter")            # solo el adaptador (pocos MB)
    save_json(met, "m4_metrics.json")
    print(classification_text(te.label, pred, clases))
    print(f"[M4] F1 macro test = {met['macro']['f1-score']:.4f} | entrenables {trainable:,} de {total:,} "
          f"({met['pct_trainable']:.2f}%) | {train_s/60:.1f} min")


# --------------------------------------------------------------------------
# 8. Informes M4 y M5 (leen los JSON generados por las etapas anteriores)
# --------------------------------------------------------------------------
def _cmp_rows(m3d, m4d):
    rows = [["Metrica", "A: TF-IDF + " + m3d["modelo"], "B: DistilBERT + LoRA"]]
    for lab, k in (("Precision (macro)", "precision"), ("Recall (macro)", "recall"), ("F1 (macro)", "f1-score")):
        rows.append([lab, _fmt(m3d["macro"][k]), _fmt(m4d["macro"][k])])
    for lab, k in (("Precision (weighted)", "precision"), ("Recall (weighted)", "recall"), ("F1 (weighted)", "f1-score")):
        rows.append([lab, _fmt(m3d["weighted"][k]), _fmt(m4d["weighted"][k])])
    rows.append(["Accuracy", _fmt(m3d["accuracy"]), _fmt(m4d["accuracy"])])
    return rows


def _verdict(m3d, m4d):
    d = m4d["macro"]["f1-score"] - m3d["macro"]["f1-score"]
    ratio = m4d["train_seconds"] / max(m3d["train_seconds"], 0.01)
    if d >= 0.02:
        v = "justifica el costo adicional si la precision es prioritaria"
    elif d >= 0.005:
        v = "aporta una mejora moderada: el costo adicional se justifica solo si cada punto de F1 es valioso para el negocio"
    else:
        v = "no justifica por si sola el costo adicional: el baseline clasico ofrece un rendimiento comparable con una fraccion del costo"
    return d, ratio, v


def _fix_examples(m3d, n=5):
    tr, te = cargar()
    y, a, b = te.label.values, np.array(m3d["preds"]), np.array(load_json("m4_metrics.json")["preds"])
    idx = np.where((a != y) & (b == y))[0][:n]
    return [[te.text[i][:220], y[i], a[i], b[i]] for i in idx], int(((a != y) & (b == y)).sum()), int(((a == y) & (b != y)).sum())


def m4_pdf(args):
    P, T, I, build, Sp, PB, cm = _pdf_tools()
    a, b = load_json("m3_metrics.json"), load_json("m4_metrics.json")
    d, ratio, verdict = _verdict(a, b)
    s = [P("Checkpoint 4: clasificacion con fine-tuning eficiente (LoRA)", "t"),
         P(f"Data Science III | {args.nombre} {args.apellido} | Dataset: AG News (4 clases)", "s"),
         P("1. Resumen de arquitectura", "h1"),
         P(f"<b>Modelo base:</b> {b['modelo']} (Hugging Face Hub), Transformer encoder de 6 capas, 12 cabezas de atencion, "
           f"768 dimensiones, ~66 M parametros. <b>Justificacion:</b> el corpus esta en ingles (modelo uncased en ingles, vocabulario WordPiece de ~30k), "
           f"las noticias son cortas (percentil 95 = {b['p95_subword']} subpalabras, max_length = {b['max_length']}) y DistilBERT retiene ~97% del desempeno de BERT-base "
           f"con 40% menos parametros y mayor velocidad, lo que entra en una GPU gratuita (Colab/Kaggle: {escape(str(b['gpu']))}). "
           "El modelo recibe texto con limpieza estructural (sin lematizar), porque el tokenizer WordPiece y la atencion aprovechan "
           "la puntuacion, las formas flexionadas y las stop-words.")]
    s += [P("2. Configuracion PEFT / LoRA", "h1"),
          T([["Hiperparametro", "Valor", "Argumento"],
             ["rango r", b["r"], "Tarea simple (4 clases tematicas): un rango bajo captura la adaptacion necesaria con pocos parametros."],
             ["alpha", b["alpha"], f"Factor de escala alpha/r = {b['alpha']/b['r']:.1f}; criterio habitual alpha = 2r."],
             ["target_modules", ", ".join(b["target_modules"]), "Proyecciones query y value de la auto-atencion (q_lin, v_lin en DistilBERT)."],
             ["modules_to_save", "pre_classifier, classifier", "Cabeza de clasificacion nueva: se entrena completa."],
             ["lora_dropout", "0.1", "Regularizacion de los adaptadores."],
             ["learning rate / epocas / batch", f"{b['lr']} / {b['epochs']} / 32", "LR mayor que en full fine-tuning (2e-5) porque solo se entrenan los adaptadores."]],
            [3.5 * cm, 3.5 * cm, 10 * cm]),
          P("3. Parametros", "h1"),
          T([["Parametros del modelo original", "Parametros entrenables (LoRA + cabeza)", "% entrenable"],
             [f"{b['params_total_base']:,}", f"{b['params_trainable']:,}", f"{b['pct_trainable']:.2f}%"]], [5.5 * cm, 6.5 * cm, 4 * cm]),
          P("4. Entrenamiento", "h1"),
          T([["Epoca", "Val loss", "Val F1 macro"]] + [[round(x["epoch"]), _fmt(x["eval_loss"]), _fmt(x["f1_macro"])] for x in b["val_history"]],
            [3 * cm, 4 * cm, 4 * cm]),
          P(f"Perdida final de entrenamiento: {b['final_train_loss']:.4f} | perdida final de validacion: {b['final_val_loss']:.4f}. "
            f"Tiempo de entrenamiento: {b['train_seconds']/60:.1f} min en {escape(str(b['gpu']))}. "
            "Validacion = 10% estratificado del train; el test no se uso para decidir nada.", "b"),
          PB(),
          P("5. Resultados y comparativa (mismo ag_news_test.csv, 2.000 noticias)", "h1"),
          T(_cmp_rows(a, b), [5 * cm, 5.5 * cm, 5.5 * cm]),
          P("Las clases estan balanceadas, por lo que macro y weighted coinciden. "
            f"Eficiencia: TF-IDF entreno en {a['train_seconds']:.1f} s (CPU); LoRA en {b['train_seconds']:.0f} s (GPU), "
            f"unas {ratio:.0f} veces mas tiempo.", "b"),
          I(FIGS / "m3_confusion.png", 7.5), I(FIGS / "m4_confusion.png", 7.5),
          P("6. Conclusion tecnica", "h1"),
          P(f"La diferencia de F1 macro del Transformer respecto del baseline es {d:+.4f}. En este corpus, el Transformer + LoRA {verdict}. "
            f"La confusion principal del baseline es {a['par_mas_confundido'][0]} vs {a['par_mas_confundido'][1]} "
            f"({a['par_mas_confundido'][2]} casos) y la del Transformer es {b['par_mas_confundido'][0]} vs {b['par_mas_confundido'][1]} "
            f"({b['par_mas_confundido'][2]} casos). LoRA permite adaptar el modelo entrenando solo {b['pct_trainable']:.2f}% de los parametros y guardar un adaptador liviano.", "b")]
    _, ap = _name(args)
    build(OUT / f"{ap}_Checkpoint_NLP3.pdf", s, "Checkpoint NLP 4")


def m5(args):
    P, T, I, build, Sp, PB, cm = _pdf_tools()
    st, a, b = load_json("m2_stats.json"), load_json("m3_metrics.json"), load_json("m4_metrics.json")
    d, ratio, verdict = _verdict(a, b)
    ej, solo_b, solo_a = _fix_examples(a)
    L = st["len"]
    s = [P("Proyecto final: clasificacion de noticias con TF-IDF vs. Transformer + LoRA", "t"),
         P(f"Data Science III - Capstone Report | {args.nombre} {args.apellido}", "s"),
         P("1. Definicion del problema", "h1"),
         P(f"Objetivo: clasificar noticias en inglesas en 4 temas (World, Sports, Business, Sci_Tech) con el dataset AG News de la catedra "
           f"({st['n_train']} noticias de train, {st['n_test']} de test, clases perfectamente balanceadas). El caso de negocio es el enrutamiento automatico "
           "de contenido editorial. La pregunta que ordena el informe: <b>cuanto aporta, y a que costo, un Transformer adaptado con LoRA "
           "frente a un baseline clasico bien construido</b>, evaluados sobre exactamente el mismo test.", "b"),
         P("2. Pipeline de preprocesamiento", "h1"),
         P(f"El EDA detecto ruido de codificacion en {st['ruido_antes']} documentos de train (entidades HTML escapadas, '#39;', URLs). "
           "<b>Baseline:</b> limpieza Regex, minusculas, lematizacion con spaCy "
           f"({escape(st['lemmatizer'])}), y TF-IDF de scikit-learn con stop-words en ingles. La longitud mediana es {L['median']:.0f} tokens y el "
           f"percentil 95 es {L['p95']} (palabras lematizadas). <b>Transformer:</b> solo limpieza estructural (sin lematizar ni quitar stop-words) y tokenizacion "
           f"WordPiece de {b['modelo']} con truncation a max_length = {b['max_length']} (percentil 95 en subpalabras = {b['p95_subword']}), "
           "padding dinamico por batch y tokens especiales [CLS]/[SEP]. La diferencia es deliberada: las representaciones esparsas se benefician de "
           "reducir variabilidad lexica, mientras que la atencion contextual aprovecha el texto casi intacto.", "b"),
         I(FIGS / "m2_hist_longitud.png", 10),
         P("3. Baseline (Modulo 3): TF-IDF + clasificador", "h1"),
         P(f"Se exploraron max_features (5.000, 20.000, sin limite), ngram_range ((1,1), (1,2)) y tres clasificadores (Naive Bayes, Regresion Logistica, SVM lineal) "
           f"sobre una validacion interna del train (20%). Mejor configuracion: <b>{a['modelo']}</b>, max_features = {a['max_features']}, ngram_range = {a['ngram_range']} "
           f"({a['n_features']:,} features). El vectorizador se ajusto solo con train (sin data leakage). "
           f"F1 macro en test = <b>{_fmt(a['macro']['f1-score'])}</b>, accuracy = {_fmt(a['accuracy'])}.", "b"),
         T([["Configuracion (validacion)", "Modelo", "F1 macro val."]] +
           [[f"mf={r['max_features']}, ng={r['ngram_range']}", r["modelo"], r["f1_macro_val"]] for r in a["grid"][:6]], [7 * cm, 5 * cm, 3.5 * cm]),
         PB(),
         P("4. Solucion avanzada (Modulo 4): DistilBERT + LoRA", "h1"),
         P(f"Se adapto {b['modelo']} (encoder de 6 capas y 12 cabezas de atencion) con LoRA: en lugar de actualizar todas las matrices mediante backpropagation, "
           f"se congelan los pesos y se entrenan matrices de bajo rango (r = {b['r']}, alpha = {b['alpha']}) en las proyecciones query y value de la auto-atencion, "
           f"mas la cabeza de clasificacion. Entrenables: {b['params_trainable']:,} de {b['params_total_peft']:,} parametros (<b>{b['pct_trainable']:.2f}%</b>). "
           f"Ciclo en PyTorch (Trainer de Hugging Face, optimizador AdamW, lr = {b['lr']}, {b['epochs']} epocas, batch 32, "
           f"{'precision mixta fp16, ' if 'cuda' in str(b['gpu']).lower() or b['gpu']!='CPU' else ''}semilla {SEED}). "
           f"Perdida final: train {b['final_train_loss']:.4f}, validacion {b['final_val_loss']:.4f}. Tiempo de entrenamiento: {b['train_seconds']/60:.1f} min en {escape(str(b['gpu']))}.", "b"),
         P("5. Evaluacion comparativa", "h1"),
         T(_cmp_rows(a, b), [5 * cm, 5.5 * cm, 5.5 * cm]),
         P("Ambos modelos se evaluaron sobre el mismo ag_news_test.csv (2.000 noticias, 500 por clase). Al estar las clases balanceadas, "
           "F1 macro y weighted coinciden y Accuracy es una metrica valida.", "b"),
         Sp(1, 6), I(FIGS / "m3_confusion.png", 7.2), I(FIGS / "m4_confusion.png", 7.2),
         P(f"<b>Eficiencia.</b> TF-IDF: {a['train_seconds']:.1f} s de entrenamiento en CPU y {a['inference_seconds']:.2f} s de inferencia sobre el test. "
           f"LoRA: {b['train_seconds']:.0f} s de entrenamiento en GPU y {b['inference_seconds']:.1f} s de inferencia (~{ratio:.0f}x el tiempo de entrenamiento). "
           f"Mejora de F1 macro: {d:+.4f}.", "b"),
         P("Analisis de errores e interpretabilidad", "h2"),
         P(f"El Transformer acierta {solo_b} noticias que el baseline falla; el baseline acierta {solo_a} que el Transformer falla. "
           f"Confusion principal del baseline: {a['par_mas_confundido'][0]} vs {a['par_mas_confundido'][1]}; del Transformer: "
           f"{b['par_mas_confundido'][0]} vs {b['par_mas_confundido'][1]}. Terminos mas influyentes del baseline por clase: "
           + "; ".join(f"{k}: {', '.join(v[:5])}" for k, v in a.get("top_terms", {}).items()) +
           ". Ejemplos corregidos por el Transformer (no se inspeccionaron pesos de atencion; la comparacion es por casos):", "b"),
         T([["Texto (truncado)", "Real", "TF-IDF", "LoRA"]] + ej, [9.5 * cm, 2 * cm, 2.2 * cm, 2.2 * cm]),
         P("6. Conclusiones, limitaciones y mejoras", "h1"),
         P(f"<b>Hallazgos.</b> Sobre AG News, el baseline TF-IDF + {a['modelo']} alcanza F1 macro {_fmt(a['macro']['f1-score'])} y el Transformer + LoRA "
           f"{_fmt(b['macro']['f1-score'])} ({d:+.4f}); con {b['pct_trainable']:.2f}% de parametros entrenables, el Transformer {verdict}. "
           "<b>Por que:</b> la clasificacion tematica de noticias depende en gran parte de vocabulario distintivo (nombres de empresas, deportes, paises), "
           "que TF-IDF captura bien; el Transformer suma semantica contextual y robustez ante sinonimos y vocabulario no visto (subpalabras), "
           "lo que ayuda en textos ambiguos entre Business y Sci_Tech o World y Business. "
           "<b>Limitaciones.</b> Un solo corpus en ingles, una semilla y un solo modelo base; textos cortos; no se midio variabilidad entre corridas; "
           "no se analizaron pesos de atencion. <b>Mejoras.</b> Probar r y modulos objetivo mayores, varias semillas con intervalos de confianza, "
           "RoBERTa/BERT-base, y analisis de interpretabilidad (SHAP o atencion).", "b"),
         P("Referencias", "h2"),
         P("Zhang, X., Zhao, J., LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. | "
           "Vaswani et al. (2017). Attention Is All You Need. | Sanh et al. (2019). DistilBERT. | "
           "Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. | Pedregosa et al. (2011). Scikit-learn.", "c")]
    _, ap = _name(args)
    build(OUT / f"NLP_Capstone_{ap}.pdf", s, "NLP Capstone")


# --------------------------------------------------------------------------
# 9. CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapa", choices=["m1", "m2", "m3", "m4", "m4-pdf", "m5"])
    ap.add_argument("--apellido", default="Apellido")
    ap.add_argument("--nombre", default="Nombre")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    a = ap.parse_args()
    if a.etapa == "m1":
        a.epochs, a.lr = a.epochs or 15, a.lr or 1e-3
        m1(a)
    elif a.etapa == "m2":
        m2(a)
    elif a.etapa == "m3":
        m3(a)
    elif a.etapa == "m4":
        a.epochs, a.lr = a.epochs or 3, a.lr or 2e-4
        m4(a)
    elif a.etapa == "m4-pdf":
        m4_pdf(a)
    else:
        m5(a)


if __name__ == "__main__":
    main()
