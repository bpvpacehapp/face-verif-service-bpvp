-- 1. Create User untuk aplikasi
 create user presensi_app with password 'bpvp123';
 grant connect on database face_detection_bpvp to presensi_app;

-- 2. Tabel embedding wajah
create table if not exists face_embeddings (
	id BIGSERIAL primary key,
	user_id VARCHAR(100) not null,
	embedding real[] not null,
	model_name VARCHAR(50) not null default 'facenet12',
	source_url text,
	det_confidence real,
	is_active boolean not null default true,
	created_at TIMESTAMPTZ not null default now(),
	
constraint chk_embedding_dim check (array_length(embedding, 1)= 512),
constraint uq_user_source unique (user_id, source_url)
);

create index if not exists idx_face_embeddings_user_id on face_embeddings (user_id);

-- 3. Hak akses user aplikasi
grant usage on schema public to presensi_app;
grant select, insert, update, delete on face_embeddings to presensi_app;
grant usage, select on all sequences in schema public to presensi_app;
