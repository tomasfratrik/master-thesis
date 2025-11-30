# index.py
import faiss, numpy as np, json
from config import CLS_EMB_NPY, CLS_META_JSON, FAISS_INDEX

def main():
    xb = np.load(CLS_EMB_NPY).astype("float32")  # [C, D], unit-normal
    d = xb.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine similarity via inner product on normalized vectors
    index.add(xb)
    faiss.write_index(index, str(FAISS_INDEX))
    print(f"Indexed {xb.shape[0]} class vectors of dim {d} -> {FAISS_INDEX}")

    with open(CLS_META_JSON) as f:
        meta = json.load(f)
    print(f"Classes available: {len(meta)}")

if __name__ == "__main__":
    main()

