import io
import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from datacleaner.api.main import create_app
from datacleaner.api.settings import Settings
from datacleaner.io import FileFormat, read_bytes, to_bytes

CONFIG = json.dumps({"target": "Exam_Score", "feature_source": "Exam_Score"})


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(Settings(json_logs=False))) as c:
        yield c


@pytest.fixture
def csv_bytes(sample_path: Path) -> bytes:
    return sample_path.read_bytes()


def upload(content: bytes, name: str = "sample.csv") -> dict[str, tuple[str, io.BytesIO]]:
    return {"file": (name, io.BytesIO(content))}


class TestMeta:
    def test_healthz(self, client: TestClient) -> None:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_request_id_is_generated_and_echoed(self, client: TestClient) -> None:
        assert client.get("/healthz").headers["X-Request-ID"]
        echoed = client.get("/healthz", headers={"X-Request-ID": "abc"})
        assert echoed.headers["X-Request-ID"] == "abc"

    def test_default_config(self, client: TestClient) -> None:
        body = client.get("/v1/config/default").json()
        assert body["handle_missing"] is True
        assert body["target"] is None

    def test_openapi_schema_available(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        assert {"/v1/clean", "/v1/profile", "/healthz"} <= set(paths)


class TestProfile:
    def test_profile_sample(self, client: TestClient, csv_bytes: bytes) -> None:
        response = client.post("/v1/profile", files=upload(csv_bytes))
        assert response.status_code == 200

        body = response.json()
        profile = body["profile"]
        assert profile["rows"] == 19
        assert profile["duplicate_rows"] == 1
        assert profile["id_columns"] == ["Student_ID"]
        assert "Study_Hours" in profile["numeric_columns"]
        assert body["preview"]["total_rows"] == 19

        hist = next(c for c in profile["column_profiles"] if c["name"] == "Age")["histogram"]
        assert sum(hist["counts"]) == 19

    def test_preview_rows_limit(self, client: TestClient, csv_bytes: bytes) -> None:
        body = client.post("/v1/profile?preview_rows=5", files=upload(csv_bytes)).json()
        assert len(body["preview"]["rows"]) == 5
        assert body["preview"]["truncated"] is True

    def test_preview_rows_validated(self, client: TestClient, csv_bytes: bytes) -> None:
        response = client.post("/v1/profile?preview_rows=0", files=upload(csv_bytes))
        assert response.status_code == 422


class TestClean:
    def test_clean_sample(self, client: TestClient, csv_bytes: bytes) -> None:
        response = client.post("/v1/clean", files=upload(csv_bytes), data={"config": CONFIG})
        assert response.status_code == 200

        body = response.json()
        report = body["report"]
        assert report["duplicates_removed"] == 1
        assert report["missing_fixed"] == 4
        assert "Study_Hours" in report["outliers"]
        assert sorted(report["dropped_from_ml"]) == ["Exam_Score_Category", "Student_ID"]
        assert "Student_ID" not in body["ml"]["columns"]
        assert body["cleaned"]["total_rows"] == 18
        assert "Study_Hours" in body["outlier_rows"]

    def test_defaults_when_config_omitted(self, client: TestClient, csv_bytes: bytes) -> None:
        response = client.post("/v1/clean", files=upload(csv_bytes))
        assert response.status_code == 200
        assert response.json()["report"]["created_features"] == []

    def test_nan_becomes_null_in_json(self, client: TestClient, csv_bytes: bytes) -> None:
        config = json.dumps({"handle_missing": False})
        body = client.post("/v1/clean", files=upload(csv_bytes), data={"config": config}).json()
        assert any(None in row for row in body["cleaned"]["rows"])

    @pytest.mark.parametrize("fmt", [FileFormat.XLSX, FileFormat.PARQUET])
    def test_other_formats_match_csv(
        self, client: TestClient, csv_bytes: bytes, sample_df: pd.DataFrame, fmt: FileFormat
    ) -> None:
        csv_report = client.post("/v1/clean", files=upload(csv_bytes)).json()["report"]
        other = client.post(
            "/v1/clean", files=upload(to_bytes(sample_df, fmt), f"s.{fmt.value}")
        ).json()["report"]

        assert other["duplicates_removed"] == csv_report["duplicates_removed"]
        assert other["missing_fixed"] == csv_report["missing_fixed"]


class TestDownload:
    @pytest.mark.parametrize("fmt", list(FileFormat))
    def test_download_formats(self, client: TestClient, csv_bytes: bytes, fmt: FileFormat) -> None:
        response = client.post(
            f"/v1/clean/download?dataset=cleaned&format={fmt.value}",
            files=upload(csv_bytes),
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(fmt.media_type)
        assert f"cleaned_dataset.{fmt.value}" in response.headers["content-disposition"]
        assert len(read_bytes(response.content, f"x.{fmt.value}")) == 18

    def test_download_ml_dataset(self, client: TestClient, csv_bytes: bytes) -> None:
        response = client.post(
            "/v1/clean/download?dataset=ml",
            files=upload(csv_bytes),
            data={"config": CONFIG},
        )
        columns = pd.read_csv(io.BytesIO(response.content)).columns
        assert "Student_ID" not in columns
        assert "Gender_Male" in columns

    def test_invalid_dataset_param(self, client: TestClient, csv_bytes: bytes) -> None:
        response = client.post("/v1/clean/download?dataset=nope", files=upload(csv_bytes))
        assert response.status_code == 422


class TestErrors:
    def assert_problem(self, response, status: int) -> dict:  # type: ignore[no-untyped-def]
        assert response.status_code == status
        assert response.headers["content-type"].startswith("application/problem+json")
        body = response.json()
        assert body["status"] == status
        assert body["request_id"]
        return body

    def test_unsupported_extension(self, client: TestClient) -> None:
        response = client.post("/v1/clean", files=upload(b"x", "data.txt"))
        assert "Unsupported" in self.assert_problem(response, 400)["detail"]

    def test_empty_file(self, client: TestClient) -> None:
        self.assert_problem(client.post("/v1/clean", files=upload(b"", "a.csv")), 400)

    def test_garbage_parquet(self, client: TestClient) -> None:
        self.assert_problem(client.post("/v1/clean", files=upload(b"junk", "a.parquet")), 400)

    def test_missing_file_field(self, client: TestClient) -> None:
        self.assert_problem(client.post("/v1/clean"), 422)

    def test_bad_json_config(self, client: TestClient, csv_bytes: bytes) -> None:
        response = client.post("/v1/clean", files=upload(csv_bytes), data={"config": "{nope"})
        assert "not valid JSON" in self.assert_problem(response, 422)["detail"]

    def test_config_must_be_object(self, client: TestClient, csv_bytes: bytes) -> None:
        response = client.post("/v1/clean", files=upload(csv_bytes), data={"config": "[1]"})
        self.assert_problem(response, 422)

    def test_unknown_config_key(self, client: TestClient, csv_bytes: bytes) -> None:
        response = client.post("/v1/clean", files=upload(csv_bytes), data={"config": '{"x": 1}'})
        self.assert_problem(response, 422)

    def test_unknown_target_column(self, client: TestClient, csv_bytes: bytes) -> None:
        response = client.post(
            "/v1/clean", files=upload(csv_bytes), data={"config": '{"target": "nope"}'}
        )
        assert "nope" in self.assert_problem(response, 422)["detail"]

    def test_unknown_route_is_problem_json(self, client: TestClient) -> None:
        self.assert_problem(client.get("/nope"), 404)

    def test_upload_too_large(self, csv_bytes: bytes) -> None:
        small = Settings(max_upload_mb=1, json_logs=False)
        big = b"a\n" + b"1\n" * 700_000  # > 1 MB

        with TestClient(create_app(small)) as client:
            response = client.post("/v1/clean", files=upload(big))

        self.assert_problem(response, 413)

    def test_too_many_rows(self, csv_bytes: bytes) -> None:
        with TestClient(create_app(Settings(max_rows=5, json_logs=False))) as client:
            response = client.post("/v1/clean", files=upload(csv_bytes))

        assert "more than" in self.assert_problem(response, 400)["detail"]

    def test_unexpected_error_is_500_problem(
        self, client: TestClient, csv_bytes: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("secret internals")

        monkeypatch.setattr("datacleaner.api.routes.clean.Pipeline.run", boom)

        with TestClient(client.app, raise_server_exceptions=False) as quiet:
            response = quiet.post("/v1/clean", files=upload(csv_bytes))

        body = self.assert_problem(response, 500)
        assert "secret" not in body["detail"]
