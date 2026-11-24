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
    attendance_date: str | None = None,
    model_name: str = MODEL_NAME,
) -> int:
    """Simpan satu embedding ke dataset pegawai (jika source_url sama, diperbarui)."""
    vector = np.asarray(embedding, dtype=np.float32).flatten()
    if vector.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Dimensi embedding harus {EMBEDDING_DIM}, dapat {vector.shape[0]}")

    sql = """
        INSERT INTO face_embeddings
            (user_id, embedding, model_name, source_url, det_confidence, attendance_date)
        VALUES (
            %s, %s::real[], %s, %s, %s,
            COALESCE(%s::timestamptz, now())
        )
        ON CONFLICT (user_id, source_url) DO UPDATE
            SET embedding = EXCLUDED.embedding,
                det_confidence = EXCLUDED.det_confidence,
                attendance_date = EXCLUDED.attendance_date,
                is_active = TRUE
        RETURNING id
    """
    with get_connection() as conn:
        row = conn.execute(
            sql, (user_id, vector.tolist(), model_name, source_url, det_confidence, attendance_date)
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


def count_user_embeddings(user_id: str, model_name: str = MODEL_NAME) -> int:
    """Hitung berapa banyak foto referensi aktif yang dimiliki user."""
    sql = """
        SELECT COUNT(*) FROM face_embeddings
        WHERE user_id = %s AND model_name = %s AND is_active = TRUE
    """
    with get_connection() as conn:
        row = conn.execute(sql, (user_id, model_name)).fetchone()
        return row[0] if row else 0


def save_attendance_record(
    user_id: str,
    verification_status: str,
    process_status: str = "DONE",
    attendance_date: str | None = None,
    photo_url: str | None = None,
    is_spoof: bool = False,
    similarity_percentage: str | None = None,
    similarity_score: float | None = None,
    execution_time_seconds: float | None = None,
    notes: str | None = None,
) -> int:
    """Simpan riwayat presensi yang diaudit ke tabel attendance_verifications."""
    sql = """
        INSERT INTO attendance_verifications (
            user_id, attendance_date, photo_url, process_status,
            is_spoof, similarity_percentage, similarity_score,
            verification_status, execution_time_seconds, notes
        )
        VALUES (
            %s, COALESCE(%s::timestamptz, now()), %s, %s,
            %s, %s, %s,
            %s, %s, %s
        )
        RETURNING id
    """
    with get_connection() as conn:
        row = conn.execute(
            sql,
            (
                user_id,
                attendance_date,
                photo_url,
                process_status,
                is_spoof,
                similarity_percentage,
                similarity_score,
                verification_status,
                execution_time_seconds,
                notes,
            ),
        ).fetchone()
        return row[0]


def get_attendance_records(
    user_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Ambil list riwayat presensi untuk tabel Admin Panel."""
    sql = """
        SELECT
            id, user_id, attendance_date, photo_url,
            process_status, is_spoof, similarity_percentage,
            similarity_score, verification_status,
            execution_time_seconds, notes, created_at
        FROM attendance_verifications
    """
    params = []
    if user_id:
        sql += " WHERE user_id = %s "
        params.append(user_id)

    sql += " ORDER BY attendance_date DESC, id DESC LIMIT %s OFFSET %s "
    params.extend([limit, offset])

    with get_connection() as conn:
        conn.row_factory = psycopg.rows.dict_row
        rows = conn.execute(sql, params).fetchall()
        return rows


def get_user_photo_history(user_id: str) -> list[dict]:
    """
    Ambil list seluruh foto presensi pegawai berdasarkan user_id dan tanggal upload.
    Digunakan untuk tombol Action Detail di Admin Panel.
    """
    sql = """
        SELECT
            id,
            user_id,
            source_url,
            det_confidence,
            attendance_date,
            created_at
        FROM face_embeddings
        WHERE user_id = %s AND is_active = TRUE
        ORDER BY attendance_date DESC
    """
    with get_connection() as conn:
        conn.row_factory = psycopg.rows.dict_row
        rows = conn.execute(sql, (user_id,)).fetchall()
        return rows