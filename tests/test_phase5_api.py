from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import pdf


def test_pdf_bbox_route_returns_bbox_payload():
    app = FastAPI()
    app.include_router(pdf.router, prefix="/api/v1")
    client = TestClient(app)

    response = client.get(
        "/api/v1/pdf/bbox",
        params={
            "company": "VNM",
            "year": 2025,
            "page": 12,
            "bbox": "0.142,0.098,0.915,0.133",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["company"] == "VNM"
    assert payload["year"] == 2025
    assert payload["page"] == 12
    assert payload["bbox"] == [0.142, 0.098, 0.915, 0.133]


def test_pdf_bbox_route_rejects_invalid_bbox():
    app = FastAPI()
    app.include_router(pdf.router, prefix="/api/v1")
    client = TestClient(app)

    response = client.get(
        "/api/v1/pdf/bbox",
        params={
            "company": "VNM",
            "year": 2025,
            "page": 12,
            "bbox": "0.142,0.098,0.915",
        },
    )

    assert response.status_code == 400
