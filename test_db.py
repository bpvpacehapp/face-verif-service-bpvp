import numpy as np

from app.db import save_embedding, get_embeddings, delete_user_embeddings

USER_ID = "TEST-12345"

# Embedding palsu: acak, lalu L2-normalize seperti yang akan kita lakukan nanti
fake = np.random.rand(512).astype(np.float32)
fake /= np.linalg.norm(fake)

print("1. Menyimpan embedding...")
new_id = save_embedding(USER_ID, fake, source_url="https://example.com/foto-1.jpg", det_confidence=0.95)
print(f"   Berhasil, id = {new_id}")

print("2. Membaca kembali...")
result = get_embeddings(USER_ID)
print(f"   Bentuk data: {result.shape}")

assert result.shape == (1, 512), "Bentuk data tidak sesuai"
assert np.allclose(result[0], fake, atol=1e-6), "Isi vektor berbeda dari yang disimpan"
print("   Isi vektor cocok dengan yang disimpan")

print("3. Membersihkan data tes...")
print(f"   {delete_user_embeddings(USER_ID)} baris dihapus")

print("\nSemua tes lolos. Koneksi Python ke database berfungsi.")