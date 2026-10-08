"""ingestion/parsers/json_block_loader.py — Trình đọc và chuẩn hoá JSON blocks từ OpenBCTC."""

import json
from pathlib import Path
from typing import Any

from src.models.block import BlockMetadata, JSONBlock


class JSONBlockLoader:
    """Loader nạp và validate các JSON blocks từ thư mục cache hoặc file đơn lẻ."""

    @staticmethod
    def load_from_file(file_path: Path | str) -> list[JSONBlock]:
        """Đọc và validate danh sách blocks từ 1 file JSON đơn lẻ."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            # Nếu file chứa 1 object đơn
            if isinstance(raw_data, dict):
                raw_data = [raw_data]
            else:
                raise ValueError(f"Dữ liệu JSON trong {path} phải là mảng các blocks, nhận: {type(raw_data)}")

        blocks: list[JSONBlock] = []
        for idx, item in enumerate(raw_data):
            try:
                block = JSONBlock.model_validate(item)
                blocks.append(block)
            except Exception as e:
                # Log hoặc bỏ qua block lỗi nhẹ kèm cảnh báo
                raise ValueError(f"Lỗi validate block thứ {idx} trong {path}: {e}") from e

        return blocks

    @staticmethod
    def load_from_dir(dir_path: Path | str, pattern: str = "*.json") -> list[JSONBlock]:
        """Quét và nạp tất cả các file JSON blocks từ một thư mục (ví dụ: data/VNM_2025/cache/notes)."""
        folder = Path(dir_path)
        if not folder.is_dir():
            raise NotADirectoryError(f"Thư mục không tồn tại: {folder}")

        all_blocks: list[JSONBlock] = []
        # Sắp xếp theo tên file để đảm bảo thứ tự trang tăng dần (page_13, page_14...)
        json_files = sorted(folder.glob(pattern), key=lambda p: p.name)

        for json_file in json_files:
            file_blocks = JSONBlockLoader.load_from_file(json_file)
            all_blocks.extend(file_blocks)

        return all_blocks
