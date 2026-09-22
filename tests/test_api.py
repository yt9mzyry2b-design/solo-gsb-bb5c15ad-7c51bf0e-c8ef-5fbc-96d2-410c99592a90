"""端到端 API 测试（FastAPI TestClient + 临时 SQLite）。"""
from __future__ import annotations

import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """每个用例使用独立 SQLite 文件重建应用。"""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_PATH", str(db_path))

    import app.config as config
    importlib.reload(config)

    import app.main as main
    importlib.reload(main)

    with TestClient(main.app) as c:
        yield c


# “十”：横 + 竖，画布 400x400
REF_SHI = [
    [[100.0, 200.0], [300.0, 200.0]],
    [[200.0, 100.0], [200.0, 300.0]],
]
PERFECT_SHI = REF_SHI


def _validate_payload(submission_id: str, strokes, **overrides):
    payload = {
        "submission_id": submission_id,
        "character": "十",
        "reference_strokes": REF_SHI,
        "strokes": strokes,
    }
    payload.update(overrides)
    return payload


# ------------------------------------------------------------------ 基础

def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_docs_available(client):
    r = client.get("/docs")
    assert r.status_code == 200
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert r.json()["info"]["title"] == "汉字笔顺轨迹校验 API"


def test_default_config(client):
    r = client.get("/v1/config")
    assert r.status_code == 200
    body = r.json()
    assert body["fork_extra_direction_deg"] == 20.0
    assert body["start_end_tolerance"] == 0.15


# ------------------------------------------------------------------ 校验通过

def test_perfect_writing_passes(client):
    r = client.post("/v1/validate", json=_validate_payload("sub-ok-1", PERFECT_SHI))
    assert r.status_code == 200
    body = r.json()
    assert body["passed"] is True
    assert body["score"] == 100.0
    assert body["first_failed_stroke"] is None
    assert body["deviation"] is None
    assert body["idempotent_replay"] is False
    assert body["expected_stroke_count"] == 2
    assert len(body["strokes"]) == 2


# ------------------------------------------------------------------ 各类偏差

def test_start_point_deviation(client):
    strokes = [[[160.0, 200.0], [300.0, 200.0]], REF_SHI[1]]
    body = client.post("/v1/validate", json=_validate_payload("sub-start", strokes)).json()
    assert body["passed"] is False
    assert body["first_failed_stroke"] == 0
    assert body["deviation"]["code"] == "START_POINT_DEVIATION"
    assert body["deviation"]["metrics"]["start_deviation"] > 0.15


def test_end_point_deviation(client):
    strokes = [[[100.0, 200.0], [300.0, 260.0]], REF_SHI[1]]
    body = client.post("/v1/validate", json=_validate_payload("sub-end", strokes)).json()
    assert body["deviation"]["code"] == "END_POINT_DEVIATION"
    assert body["first_failed_stroke"] == 0


def test_direction_deviation_with_custom_tolerances(client):
    # 放宽起止点、收紧方向容差， isolate 方向偏差
    strokes = [[[100.0, 200.0], [300.0, 250.0]], REF_SHI[1]]
    payload = _validate_payload(
        "sub-dir", strokes,
        tolerances={"start_end": 0.4, "direction_deg": 10.0},
    )
    body = client.post("/v1/validate", json=payload).json()
    assert body["deviation"]["code"] == "DIRECTION_DEVIATION"
    assert body["deviation"]["metrics"]["direction_deviation_deg"] > 10.0


def test_shape_deviation(client):
    strokes = [[[100.0, 200.0], [200.0, 290.0], [300.0, 200.0]], REF_SHI[1]]
    body = client.post("/v1/validate", json=_validate_payload("sub-shape", strokes)).json()
    assert body["deviation"]["code"] == "SHAPE_DEVIATION"


def test_wrong_order_detected(client):
    strokes = [REF_SHI[1], REF_SHI[0]]  # 先竖后横
    body = client.post("/v1/validate", json=_validate_payload("sub-order", strokes)).json()
    assert body["passed"] is False
    assert body["first_failed_stroke"] == 0
    assert body["deviation"]["code"] == "WRONG_ORDER"
    assert body["deviation"]["stroke_index"] == 0


def test_stroke_count_mismatch(client):
    body = client.post(
        "/v1/validate", json=_validate_payload("sub-count", [REF_SHI[0]])
    ).json()
    assert body["passed"] is False
    assert body["deviation"]["code"] == "STROKE_COUNT_MISMATCH"
    assert body["expected_stroke_count"] == 2
    assert body["actual_stroke_count"] == 1
    assert body["score"] == 50.0  # 唯一一笔满分 × 覆盖率 1/2


# ------------------------------------------------------------------ 分叉笔画容差

# 横折“┐”参考；用户拐角内切 110 单位
REF_ZHE = [[[100.0, 150.0], [300.0, 150.0], [300.0, 300.0]]]
USER_ZHE_SLOPPY = [[[100.0, 150.0], [300.0, 260.0], [300.0, 300.0]]]


def test_forked_stroke_default_tolerance_passes(client):
    payload = _validate_payload("sub-fork-1", USER_ZHE_SLOPPY, reference_strokes=REF_ZHE)
    body = client.post("/v1/validate", json=payload).json()
    assert body["passed"] is True
    assert body["strokes"][0]["metrics"]["is_forked"] is True


def test_forked_stroke_zero_extra_tolerance_fails(client):
    payload = _validate_payload(
        "sub-fork-2", USER_ZHE_SLOPPY,
        reference_strokes=REF_ZHE,
        tolerances={"fork_extra_direction_deg": 0.0, "fork_extra_shape": 0.0},
    )
    body = client.post("/v1/validate", json=payload).json()
    assert body["passed"] is False
    assert body["deviation"]["code"] == "DIRECTION_DEVIATION"
    assert body["deviation"]["metrics"]["is_forked"] is True


# ------------------------------------------------------------------ 幂等

def test_idempotent_replay(client):
    payload = _validate_payload("sub-idem", PERFECT_SHI)
    first = client.post("/v1/validate", json=payload).json()
    second = client.post("/v1/validate", json=payload).json()
    assert first["idempotent_replay"] is False
    assert second["idempotent_replay"] is True
    assert second["score"] == first["score"]
    assert second["created_at"] == first["created_at"]
    # 重放不生成新记录
    rec = client.get("/v1/submissions/sub-idem").json()
    assert rec["result"]["score"] == first["score"]


def test_same_id_different_payload_conflicts(client):
    client.post("/v1/validate", json=_validate_payload("sub-conflict", PERFECT_SHI))
    bad = _validate_payload("sub-conflict", [REF_SHI[0]])
    r = client.post("/v1/validate", json=bad)
    assert r.status_code == 409


def test_get_submission_404(client):
    assert client.get("/v1/submissions/nope").status_code == 404


# ------------------------------------------------------------------ 参考字注册

def test_register_and_validate_by_reference_id(client):
    r = client.put("/v1/references/十", json={
        "character": "十", "strokes": REF_SHI, "source": "test",
    })
    assert r.status_code == 200
    assert r.json()["stroke_count"] == 2

    got = client.get("/v1/references/十")
    assert got.status_code == 200
    assert got.json()["strokes"] == REF_SHI

    payload = _validate_payload("sub-ref-1", PERFECT_SHI)
    del payload["reference_strokes"]
    payload["reference_id"] = "十"
    body = client.post("/v1/validate", json=payload).json()
    assert body["passed"] is True


def test_validate_with_unregistered_reference_404(client):
    payload = _validate_payload("sub-ref-404", PERFECT_SHI)
    del payload["reference_strokes"]
    payload["reference_id"] = "永"
    assert client.post("/v1/validate", json=payload).status_code == 404


def test_reference_upsert_overwrites(client):
    client.put("/v1/references/十", json={"character": "十", "strokes": REF_SHI})
    new_ref = [REF_SHI[0]]
    r = client.put("/v1/references/十", json={"character": "十", "strokes": new_ref})
    assert r.json()["stroke_count"] == 1


# ------------------------------------------------------------------ 请求校验

def test_missing_reference_rejected(client):
    payload = _validate_payload("sub-bad-1", PERFECT_SHI)
    del payload["reference_strokes"]
    r = client.post("/v1/validate", json=payload)
    assert r.status_code == 422


def test_both_references_rejected(client):
    payload = _validate_payload("sub-bad-2", PERFECT_SHI, reference_id="十")
    r = client.post("/v1/validate", json=payload)
    assert r.status_code == 422


def test_degenerate_stroke_rejected(client):
    payload = _validate_payload("sub-bad-3", [[[1.0, 1.0]]])
    r = client.post("/v1/validate", json=payload)
    assert r.status_code == 422


# ------------------------------------------------------------------ 候选字识别

# “一”：单横
REF_YI = [[[100.0, 200.0], [300.0, 200.0]]]
# “二”：短横 + 长横
REF_ER = [
    [[120.0, 130.0], [280.0, 130.0]],
    [[100.0, 250.0], [300.0, 250.0]],
]


def _register_recognize_fixtures(client):
    client.put("/v1/references/一", json={"character": "一", "strokes": REF_YI})
    client.put("/v1/references/二", json={"character": "二", "strokes": REF_ER})
    client.put("/v1/references/十", json={"character": "十", "strokes": REF_SHI})


def test_recognize_empty_without_references(client):
    r = client.post("/v1/recognize", json={"strokes": REF_SHI, "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["candidates"] == []
    assert body["total_references"] == 0


def test_recognize_perfect_match_ranks_first(client):
    _register_recognize_fixtures(client)
    r = client.post("/v1/recognize", json={"strokes": REF_SHI, "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["total_references"] == 3
    assert len(body["candidates"]) == 3
    top = body["candidates"][0]
    assert top["character"] == "十"
    assert top["score"] == 100.0
    assert top["matched_stroke_count"] == 2
    # 笔画对应：用户笔 i ↔ 参考笔 i，代价为 0
    by_user = {c["user_stroke_index"]: c for c in top["correspondences"]}
    assert by_user[0]["reference_stroke_index"] == 0
    assert by_user[1]["reference_stroke_index"] == 1
    assert by_user[0]["cost"] == 0.0
    assert top["penalties"]["missing_count"] == 0
    assert top["penalties"]["extra_count"] == 0
    # 分数单调不升
    scores = [c["score"] for c in body["candidates"]]
    assert scores == sorted(scores, reverse=True)


def test_recognize_tolerates_wrong_order(client):
    _register_recognize_fixtures(client)
    # 先竖后横：与“十”内容相同、次序相反
    r = client.post(
        "/v1/recognize", json={"strokes": [REF_SHI[1], REF_SHI[0]], "top_k": 1}
    )
    top = r.json()["candidates"][0]
    assert top["character"] == "十"
    assert top["score"] == 100.0
    by_user = {c["user_stroke_index"]: c for c in top["correspondences"]}
    assert by_user[0]["reference_stroke_index"] == 1  # 用户竖 → 参考竖(1)
    assert by_user[1]["reference_stroke_index"] == 0  # 用户横 → 参考横(0)


def test_recognize_missing_stroke_penalized(client):
    _register_recognize_fixtures(client)
    # 只写一横：更接近“一”，对“十”则缺一笔
    r = client.post("/v1/recognize", json={"strokes": REF_YI, "top_k": 3})
    by_char = {c["character"]: c for c in r.json()["candidates"]}
    assert by_char["一"]["score"] == 100.0
    shi = by_char["十"]
    assert shi["penalties"]["missing_count"] == 1
    assert shi["penalties"]["missing_stroke_indices"] == [1]
    assert shi["penalties"]["missing_penalty_total"] == 1.0
    assert shi["matched_stroke_count"] == 1
    assert shi["score"] == 50.0
    assert by_char["一"]["score"] > shi["score"]


def test_recognize_extra_stroke_penalized(client):
    _register_recognize_fixtures(client)
    extra_dot = [[50.0, 60.0], [60.0, 50.0]]
    r = client.post(
        "/v1/recognize", json={"strokes": [*REF_SHI, extra_dot], "top_k": 3}
    )
    top = r.json()["candidates"][0]
    assert top["character"] == "十"
    p = top["penalties"]
    assert p["extra_count"] == 1
    assert p["extra_stroke_indices"] == [2]
    assert p["extra_penalty_total"] == 1.0
    # 两笔完美 + 一笔多笔：1 - 1/3
    assert top["score"] == 66.67
    extra_corr = [
        c for c in top["correspondences"] if c["user_stroke_index"] == 2
    ][0]
    assert extra_corr["reference_stroke_index"] is None


def test_recognize_top_k_truncates(client):
    _register_recognize_fixtures(client)
    r = client.post("/v1/recognize", json={"strokes": REF_SHI, "top_k": 2})
    body = r.json()
    assert body["top_k"] == 2
    assert len(body["candidates"]) == 2
    assert body["total_references"] == 3


def test_recognize_top_k_larger_than_catalog(client):
    _register_recognize_fixtures(client)
    r = client.post("/v1/recognize", json={"strokes": REF_SHI, "top_k": 10})
    assert len(r.json()["candidates"]) == 3


def test_recognize_invalid_top_k_422(client):
    _register_recognize_fixtures(client)
    assert client.post(
        "/v1/recognize", json={"strokes": REF_SHI, "top_k": 0}
    ).status_code == 422
    assert client.post(
        "/v1/recognize", json={"strokes": REF_SHI, "top_k": -1}
    ).status_code == 422
    assert client.post(
        "/v1/recognize", json={"strokes": REF_SHI, "top_k": 2.5}
    ).status_code == 422


def test_recognize_empty_strokes_422(client):
    _register_recognize_fixtures(client)
    r = client.post("/v1/recognize", json={"strokes": [], "top_k": 3})
    assert r.status_code == 422


def test_recognize_tie_break_is_stable(client):
    _register_recognize_fixtures(client)
    # 构造两个笔画完全相同的参考字：任何输入下二者分数都相同
    client.put(
        "/v1/references/乙",
        json={"character": "乙", "strokes": REF_SHI, "source": "dup"},
    )
    r1 = client.post("/v1/recognize", json={"strokes": REF_SHI, "top_k": 5})
    r2 = client.post("/v1/recognize", json={"strokes": REF_SHI, "top_k": 5})
    chars1 = [c["character"] for c in r1.json()["candidates"]]
    chars2 = [c["character"] for c in r2.json()["candidates"]]
    assert chars1 == chars2  # 重复请求顺序一致
    # 同分候选用字符稳定排序：码序 “乙” < “十”
    top_score = r1.json()["candidates"][0]["score"]
    tied = [c["character"] for c in r1.json()["candidates"]
            if c["score"] == top_score]
    assert tied == ["乙", "十"]
