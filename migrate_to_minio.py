import os
import glob
from minio import Minio

# Cấu hình MinIO
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "openbctc_admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "openbctc_secret")
MINIO_BUCKET = "openbctc"

def migrate():
    print("Bat dau qua trinh Migrate du lieu tu Local (data/) len MinIO...")
    
    # 1. Khởi tạo MinIO Client
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    
    # 2. Tạo Bucket nếu chưa có
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)
        print(f"Da tao bucket: {MINIO_BUCKET}")
    else:
        print(f"Bucket '{MINIO_BUCKET}' da ton tai.")

    # 3. Quét thư mục data/
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("Khong tim thay thu muc data/ local. Khong co gi de Migrate.")
        return

    # 4. Upload các file PDF
    pdf_files = glob.glob(f"{data_dir}/**/*.pdf", recursive=True)
    if not pdf_files:
        print("Khong tim thay file PDF nao trong data/")
        
    for file_path in pdf_files:
        # Ví dụ: data/VNM_2025/VNM_2025.pdf -> VNM_2025/VNM_2025.pdf
        object_name = os.path.relpath(file_path, data_dir).replace("\\", "/")
        print(f"Dang upload: {object_name}...")
        
        client.fput_object(
            MINIO_BUCKET, 
            object_name, 
            file_path,
            content_type="application/pdf"
        )
        print(f"Da upload thanh cong: {object_name}")

    print("Hoan tat Migrate! Bay gio he thong da san sang chay Cloud-Native voi MinIO.")

if __name__ == "__main__":
    migrate()
