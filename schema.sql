-- ============================================================
-- Schema: face_detection_bpvp
-- Layanan Verifikasi Wajah Internal Admin BPVP
-- ============================================================

-- 1. Tabel dataset embedding referensi pegawai (Incremental Learning)
CREATE TABLE IF NOT EXISTS face_embeddings (
    id              BIGSERIAL PRIMARY KEY,
    user_id         VARCHAR(100) NOT NULL,
    embedding       REAL[] NOT NULL,
    model_name      VARCHAR(50) NOT NULL DEFAULT 'vggface2',
    source_url      TEXT,
    det_confidence  REAL,
    attendance_date TIMESTAMPTZ NOT NULL DEFAULT now(), -- tanggal foto diupload
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_embedding_dim CHECK (array_length(embedding, 1) = 512),
    CONSTRAINT uq_user_source UNIQUE (user_id, source_url)
);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_user_id ON face_embeddings (user_id);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_date ON face_embeddings (attendance_date);

-- 2. Tabel Riwayat Verifikasi Presensi Pegawai (Admin Panel Audit)
CREATE TABLE IF NOT EXISTS attendance_verifications (
    id                     BIGSERIAL PRIMARY KEY,
    user_id                VARCHAR(100) NOT NULL,
    attendance_date        TIMESTAMPTZ NOT NULL DEFAULT now(), -- tanggal absen diupload
    photo_url              TEXT,
    process_status         VARCHAR(20) NOT NULL DEFAULT 'DONE', -- 'NO_DATASET', 'ON_PROCESS', 'DONE', 'FAILED'
    is_spoof               BOOLEAN NOT NULL DEFAULT FALSE,     -- True jika terindikasi layar/kertas
    similarity_percentage  VARCHAR(10),                        -- contoh: '84.5%'
    similarity_score       REAL,                               -- contoh: 0.8450
    verification_status    VARCHAR(30) NOT NULL,               -- 'VERIFIED', 'NEEDS_REVIEW', 'NO_DATASET', 'FAILED'
    execution_time_seconds REAL,                               -- waktu proses mesin (detik)
    notes                  TEXT,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_att_verif_user_id ON attendance_verifications (user_id);
CREATE INDEX IF NOT EXISTS idx_att_verif_date ON attendance_verifications (attendance_date);
CREATE INDEX IF NOT EXISTS idx_att_verif_status ON attendance_verifications (verification_status);

-- 3. Hak akses user aplikasi presensi_app
GRANT USAGE ON SCHEMA public TO presensi_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON face_embeddings TO presensi_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON attendance_verifications TO presensi_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO presensi_app;
