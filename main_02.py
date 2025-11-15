from pathlib import Path
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# -------- Configuration --------
DATA_DIR = Path(__file__).parent / "dataset"
STOPWORDS_FILE = DATA_DIR / "english.stop.txt"
TRAIN_FILE = DATA_DIR / "training.xlsx"   # rename original to this
TEST_FILE = DATA_DIR / "testing.xlsx"     # rename original to this
FREQ_THRESHOLD = 5
CHART_FILE = Path("final_precision_chart_with_average.png")
SIM_MATRIX_FILE = Path("similarity_matrix.npy")
REPORT_FILE = Path("classification_report.txt")

lemmatizer = WordNetLemmatizer()

def load_stopwords(path: Path) -> set:
    if not path.exists():
        raise FileNotFoundError(f"Stopwords file missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

def safe_read_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Excel file missing: {path}")
    return pd.read_excel(path)

def preprocess_text(row: pd.Series, stop_words: set) -> str:
    title = str(row.get("core_video_video_title", ""))  # adapt if column names differ
    tags = str(row.get("core_video_video_tag", ""))
    text = f"{title} {tags}".lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w and w not in stop_words]
    # Lemmatize
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def build_category_docs(df: pd.DataFrame) -> pd.Series:
    return df.groupby("categorization")["cleaned_text"].apply(lambda s: " ".join(s))

def extract_cdts(category_super_docs: pd.Series, threshold: int) -> list:
    master = set()
    for text in category_super_docs:
        vec = CountVectorizer(min_df=1)
        counts = vec.fit_transform([text]).toarray()[0]
        for word, count in zip(vec.get_feature_names_out(), counts):
            if count > threshold:
                master.add(word)
    return sorted(master)

def vectorize_categories(category_super_docs: pd.Series, vocabulary: list):
    vec = TfidfVectorizer(vocabulary=vocabulary)
    mat = vec.fit_transform(category_super_docs)
    return vec, mat

def vectorize_tests(df_test: pd.DataFrame, vectorizer: TfidfVectorizer):
    return vectorizer.transform(df_test["cleaned_text"])

def classify(test_vectors, category_vectors, category_names):
    sim = cosine_similarity(test_vectors, category_vectors)
    pred_idx = np.argmax(sim, axis=1)
    preds = [category_names[i] for i in pred_idx]
    return sim, preds

def plot_precision(report_dict: dict, outfile: Path):
    report_df = pd.DataFrame(report_dict).transpose()
    cat_df = report_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
    macro_precision = report_df.get("macro avg", {}).get("precision", 0.0)
    plot_data = cat_df["precision"].copy()
    plot_data.loc["Macro Average"] = macro_precision

    plt.figure(figsize=(14, 8))
    colors = ["C0"] * (len(plot_data) - 1) + ["C2"]
    plt.bar(plot_data.index, plot_data.values, color=colors)
    plt.title("Precision per Category and Macro Average", fontsize=16)
    plt.xlabel("Video Category", fontsize=12)
    plt.ylabel("Precision Score", fontsize=12)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(outfile)
    print(f"Saved bar chart: {outfile}")

def main():
    print("Loading stopwords...")
    stop_words = load_stopwords(STOPWORDS_FILE)
    print(f"Stopwords loaded: {len(stop_words)}")

    print("Loading training data...")
    df_train = safe_read_excel(TRAIN_FILE)
    print(f"Training rows: {len(df_train)}")

    print("Preprocessing training data...")
    df_train["cleaned_text"] = df_train.apply(preprocess_text, axis=1, stop_words=stop_words)

    print("Building category super-documents...")
    category_super_docs = build_category_docs(df_train)
    print(f"Categories: {len(category_super_docs)}")

    print(f"Extracting CDTs (threshold={FREQ_THRESHOLD})...")
    master_cdts = extract_cdts(category_super_docs, FREQ_THRESHOLD)
    print(f"Total unique CDTs: {len(master_cdts)}")

    print("Vectorizing category documents...")
    category_vectorizer, category_vectors = vectorize_categories(category_super_docs, master_cdts)
    print(f"Category vector matrix shape: {category_vectors.shape}")

    print("Loading test data...")
    df_test = safe_read_excel(TEST_FILE)
    print(f"Test rows: {len(df_test)}")

    print("Preprocessing test data...")
    df_test["cleaned_text"] = df_test.apply(preprocess_text, axis=1, stop_words=stop_words)

    print("Vectorizing test videos...")
    test_vectors = vectorize_tests(df_test, category_vectorizer)
    print(f"Test vector matrix shape: {test_vectors.shape}")

    print("Computing similarity and predictions...")
    sim_matrix, y_pred = classify(test_vectors, category_vectors, category_super_docs.index)
    np.save(SIM_MATRIX_FILE, sim_matrix)
    print(f"Saved similarity matrix: {SIM_MATRIX_FILE}")

    y_true = df_test["categorization"]
    report_txt = classification_report(y_true, y_pred, zero_division=0)
    print("--- Classification Report ---")
    print(report_txt)
    with REPORT_FILE.open("w", encoding="utf-8") as f:
        f.write(report_txt)
    print(f"Saved report: {REPORT_FILE}")

    report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    print("Plotting precision chart...")
    plot_precision(report_dict, CHART_FILE)

    print("Done.")

if __name__ == "__main__":
    main()