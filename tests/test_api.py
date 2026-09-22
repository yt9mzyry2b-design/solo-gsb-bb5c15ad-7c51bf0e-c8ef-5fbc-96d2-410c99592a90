"""端到端 API 测试（FastAPI TestClient + 临时 SQLite）。"""
from __future__ import annotations

import importlib
import json
import os
import sqlite3

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


@pytest.fixture()
def client_db(tmp_path, monkeypatch):
    """同 client，额外暴露数据库路径，便于直接构造带固定时间戳的提交。"""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_PATH", str(db_path))

    import app.config as config
    importlib.reload(config)

    import app.main as main
    importlib.reload(main)

    with TestClient(main.app) as c:
        yield c, db_path


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


# ------------------------------------------------------------------ 练习质量趋势

def _seed_submission(db_path, sid, character, passed, score, created_at,
                     first_failed=None):
    """直接向 SQLite 写入带固定 UTC 时间戳的提交记录（绕过真实时间）。"""
    response = {
        "submission_id": sid,
        "character": character,
        "passed": passed,
        "score": score,
        "expected_stroke_count": 2,
        "actual_stroke_count": 2,
        "first_failed_stroke": first_failed,
        "deviation": None,
        "strokes": [],
        "idempotent_replay": False,
        "created_at": created_at,
    }
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO submissions (submission_id, request_hash, response,"
            " created_at) VALUES (?, ?, ?, ?)",
            (sid, f"hash-{sid}", json.dumps(response), created_at),
        )
        conn.commit()
    finally:
        conn.close()


# “十”：2 合格 + 2 失败，分数 100/100/80/50，分布在两个 UTC 日期
SHI_SEEDS = [
    ("s1", "十", True, 100.0, "2026-09-20T01:00:00+00:00", None),
    ("s2", "十", True, 100.0, "2026-09-20T08:30:00+00:00", None),
    ("s3", "十", False, 80.0, "2026-09-21T02:00:00+00:00", 0),
    ("s4", "十", False, 50.0, "2026-09-21T09:15:00+00:00", 1),
]
# 区间外（时间过晚）与其他字符的噪声数据
NOISE_SEEDS = [
    ("s-late", "十", True, 100.0, "2026-09-22T00:00:01+00:00", None),
    ("s-yi", "一", False, 10.0, "2026-09-20T12:00:00+00:00", 0),
]


def _seed_shi_fixture(db_path):
    for row in SHI_SEEDS + NOISE_SEEDS:
        _seed_submission(db_path, *row)


def _practice_url(**params):
    from urllib.parse import urlencode

    return "/v1/analytics/practice?" + urlencode(params)


def test_practice_empty_window_returns_zero_values(client_db):
    client, db_path = client_db
    _seed_submission(db_path, "only", "十", True, 90.0,
                     "2026-09-19T00:00:00+00:00")
    r = client.get(_practice_url(
        character="十",
        start="2026-09-20T00:00:00+00:00",
        end="2026-09-21T00:00:00+00:00",
    ))
    assert r.status_code == 200
    body = r.json()
    assert body["total_submissions"] == 0
    assert body["passed_submissions"] == 0
    assert body["pass_rate"] == 0.0
    assert body["average_score"] == 0.0
    assert body["p50_score"] == 0.0
    assert body["p90_score"] == 0.0
    assert body["first_failed_stroke_distribution"] == []
    assert body["daily"] == []
    assert body["filter"] == {
        "character": "十",
        "start": "2026-09-20T00:00:00+00:00",
        "end": "2026-09-21T00:00:00+00:00",
        "passed": None,
    }
    assert body["generated_at"].endswith("+00:00")


def test_practice_aggregates_scores_distribution_and_daily(client_db):
    client, db_path = client_db
    _seed_shi_fixture(db_path)
    r = client.get(_practice_url(
        character="十",
        start="2026-09-20T00:00:00+00:00",
        end="2026-09-21T23:59:59+00:00",
    ))
    assert r.status_code == 200
    body = r.json()
    # 过滤：仅 4 条“十”，噪声（其他字符/区间外）被剔除
    assert body["total_submissions"] == 4
    assert body["passed_submissions"] == 2
    assert body["pass_rate"] == 0.5
    assert body["average_score"] == 82.5
    # 分数升序 [50, 80, 100, 100]：P50 秩 1.5、P90 秩 2.7（线性插值）
    assert body["p50_score"] == 90.0
    assert body["p90_score"] == 100.0
    assert body["first_failed_stroke_distribution"] == [
        {"stroke_index": 0, "count": 1},
        {"stroke_index": 1, "count": 1},
    ]
    assert body["daily"] == [
        {"date": "2026-09-20", "total": 2, "passed": 2,
         "pass_rate": 1.0, "average_score": 100.0},
        {"date": "2026-09-21", "total": 2, "passed": 0,
         "pass_rate": 0.0, "average_score": 65.0},
    ]


def test_practice_filter_passed_true(client_db):
    client, db_path = client_db
    _seed_shi_fixture(db_path)
    r = client.get(_practice_url(
        character="十",
        start="2026-09-20T00:00:00+00:00",
        end="2026-09-21T23:59:59+00:00",
        passed="true",
    ))
    body = r.json()
    assert body["filter"]["passed"] is True
    assert body["total_submissions"] == 2
    assert body["passed_submissions"] == 2
    assert body["pass_rate"] == 1.0
    assert body["average_score"] == 100.0
    assert body["p50_score"] == 100.0
    assert body["p90_score"] == 100.0
    assert body["first_failed_stroke_distribution"] == []
    assert [b["date"] for b in body["daily"]] == ["2026-09-20"]


def test_practice_filter_passed_false(client_db):
    client, db_path = client_db
    _seed_shi_fixture(db_path)
    r = client.get(_practice_url(
        character="十",
        start="2026-09-20T00:00:00+00:00",
        end="2026-09-21T23:59:59+00:00",
        passed="false",
    ))
    body = r.json()
    assert body["filter"]["passed"] is False
    assert body["total_submissions"] == 2
    assert body["passed_submissions"] == 0
    assert body["pass_rate"] == 0.0
    assert body["average_score"] == 65.0
    # [50, 80]：P50 秩 0.5 → 65；P90 秩 0.9 → 77
    assert body["p50_score"] == 65.0
    assert body["p90_score"] == 77.0
    assert body["first_failed_stroke_distribution"] == [
        {"stroke_index": 0, "count": 1},
        {"stroke_index": 1, "count": 1},
    ]
    assert [b["date"] for b in body["daily"]] == ["2026-09-21"]


def test_practice_closed_interval_includes_both_endpoints(client_db):
    client, db_path = client_db
    _seed_shi_fixture(db_path)
    r = client.get(_practice_url(
        character="十",
        start="2026-09-20T01:00:00+00:00",
        end="2026-09-21T02:00:00+00:00",
    ))
    body = r.json()
    # 恰好在起止边界上的 s1、s3 均计入；s4(09:15) 与 s-late 不计入
    assert body["total_submissions"] == 3
    assert [b["date"] for b in body["daily"]] == ["2026-09-20", "2026-09-21"]
    assert body["daily"][1]["total"] == 1


def test_practice_other_character_excluded(client_db):
    client, db_path = client_db
    _seed_shi_fixture(db_path)
    r = client.get(_practice_url(
        character="一",
        start="2026-09-01T00:00:00+00:00",
        end="2026-09-30T00:00:00+00:00",
    ))
    body = r.json()
    assert body["total_submissions"] == 1
    assert body["passed_submissions"] == 0
    assert body["average_score"] == 10.0
    assert body["first_failed_stroke_distribution"] == [
        {"stroke_index": 0, "count": 1}
    ]


def test_practice_timezone_offset_normalized_to_utc(client_db):
    client, db_path = client_db
    _seed_submission(db_path, "tz", "十", True, 70.0,
                     "2026-09-20T10:00:00+00:00")
    # 北京时间 18:00 (+08:00) == UTC 10:00；闭区间边界应命中
    r = client.get(_practice_url(
        character="十",
        start="2026-09-20T17:00:00+08:00",
        end="2026-09-20T18:00:00+08:00",
    ))
    assert r.status_code == 200
    body = r.json()
    assert body["total_submissions"] == 1
    assert body["average_score"] == 70.0
    # 筛选条件统一回显为 UTC
    assert body["filter"]["start"] == "2026-09-20T09:00:00+00:00"
    assert body["filter"]["end"] == "2026-09-20T10:00:00+00:00"
    # UTC 分桶（而非 +08:00 的 9 月 20 日以外日期；此处同日但验证桶键）
    assert body["daily"][0]["date"] == "2026-09-20"


def test_practice_naive_datetime_treated_as_utc(client_db):
    client, db_path = client_db
    _seed_submission(db_path, "naive", "十", True, 60.0,
                     "2026-09-20T10:00:00+00:00")
    r = client.get(_practice_url(
        character="十",
        start="2026-09-20T10:00:00",
        end="2026-09-20T10:00:00",
    ))
    assert r.status_code == 200
    assert r.json()["total_submissions"] == 1


def test_practice_daily_utc_bucketing(client_db):
    client, db_path = client_db
    # 北京时间 9 月 21 日 08:00 == UTC 9 月 21 日 00:00（闭区间起点）
    _seed_submission(db_path, "utc-a", "十", True, 90.0,
                     "2026-09-20T23:59:59+00:00")
    _seed_submission(db_path, "utc-b", "十", False, 40.0,
                     "2026-09-21T00:00:00+00:00")
    r = client.get(_practice_url(
        character="十",
        start="2026-09-20T00:00:00+00:00",
        end="2026-09-22T00:00:00+00:00",
    ))
    daily = r.json()["daily"]
    assert [b["date"] for b in daily] == ["2026-09-20", "2026-09-21"]
    assert daily[0]["total"] == 1
    assert daily[1]["total"] == 1


def test_practice_missing_params_422(client):
    r = client.get(_practice_url(character="十",
                                end="2026-09-21T00:00:00+00:00"))
    assert r.status_code == 422


def test_practice_bad_datetime_422(client):
    r = client.get(_practice_url(
        character="十", start="not-a-date",
        end="2026-09-21T00:00:00+00:00",
    ))
    assert r.status_code == 422


def test_practice_start_after_end_422(client):
    r = client.get(_practice_url(
        character="十",
        start="2026-09-21T10:00:00+00:00",
        end="2026-09-20T10:00:00+00:00",
    ))
    assert r.status_code == 422


def test_practice_character_must_be_single_char(client):
    r = client.get(_practice_url(
        character="汉字",
        start="2026-09-20T00:00:00+00:00",
        end="2026-09-21T00:00:00+00:00",
    ))
    assert r.status_code == 422


def test_practice_end_to_end_with_real_validate(client):
    """走真实 /v1/validate 落库，再用趋势接口聚合（时间窗覆盖当天）。"""
    client.post("/v1/validate", json=_validate_payload("a-ok", PERFECT_SHI))
    client.post("/v1/validate",
                json=_validate_payload("a-fail", [REF_SHI[0]]))
    r = client.get(_practice_url(
        character="十",
        start="2000-01-01T00:00:00+00:00",
        end="2100-01-01T00:00:00+00:00",
    ))
    assert r.status_code == 200
    body = r.json()
    assert body["total_submissions"] == 2
    assert body["passed_submissions"] == 1
    assert body["pass_rate"] == 0.5
    assert body["average_score"] == 75.0  # 100 与 50
    # 缺笔失败提交的首个失败笔画下标为 1（两笔参考中缺失的第 2 笔）
    assert body["first_failed_stroke_distribution"] == [
        {"stroke_index": 1, "count": 1}
    ]
    assert len(body["daily"]) == 1
