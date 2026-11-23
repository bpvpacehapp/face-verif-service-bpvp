import numpy as np
import psycopg

from app.config import DATABASE_URL, EMBEDDING_DIM, MODEL_NAME


def get_connection():
    return psycopg.connect(DATABASE_URL)


def save_embedding(
    user_id: str,
    embedding: np.ndarray,
    source_url: str | None = None,
    det_confidence: float | None = None,
    model_name: str = MODEL_NAME,
) -> int:
    """Simpan satu embedding. Jika (user_id, source_url) sudah ada, diperbarui."""
    vector = np.asarray(embedding, dtype=np.float32).flatten()
    if vector.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Dimensi embedding harus {EMBEDDING_DIM}, dapat {vector.shape[0]}")

    sql = """
        INSERT INTO face_embeddings
            (user_id, embedding, model_name, source_url, det_confidence)
        VALUES (%s, %s::real[], %s, %s, %s)
        ON CONFLICT (user_id, source_url) DO UPDATE
            SET embedding = EXCLUDED.embedding,
                det_confidence = EXCLUDED.det_confidence,
                is_active = TRUE
        RETURNING id
    """
    with get_connection() as conn:
        row = conn.execute(
            sql, (user_id, vector.tolist(), model_name, source_url, det_confidence)
        ).fetchone()
        return row[0]


def get_embeddings(user_id: str, model_name: str = MODEL_NAME) -> np.ndarray:
    """Ambil semua embedding aktif milik satu user. Hasil: array (n, 512)."""
    sql = """
        SELECT embedding FROM face_embeddings
        WHERE user_id = %s AND model_name = %s AND is_active = TRUE
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (user_id, model_name)).fetchall()
    if not rows:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    return np.array([r[0] for r in rows], dtype=np.float32)


def delete_user_embeddings(user_id: str) -> int:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM face_embeddings WHERE user_id = %s", (user_id,))
        return cur.rowcount