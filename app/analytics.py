"""书写练习质量趋势分析。

基于 :meth:`Database.snapshot_submission` 得到的单一一致性快照，在快照之上做
过滤与聚合：提交数、通过率、平均分、P50/P90 分数（显式线性插值）、
首个失败笔写下标分布与按 UTC 日期分桶的趋势。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

# 空结果（无匹配提交）时返回的零值统计
EMPTY_RATES = {"total": 0, "passed": 0, "pass_rate": 0.0,
               "average_score": 0.0, "p50_score": 0.0, "p90_score": 0.0}


def parse_created_at(raw: str) -> datetime:
    """解析落库时间戳（UTC ISO-8601，秒精度）；朴素时间戳按 UTC 处理。"""
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def percentile_linear(sorted_values: list[float], pct: float) -> float:
    """显式线性插值分位数（与 numpy 默认 'linear' 方法一致）。

    对升序样本 x_0..x_{n-1}，百分位 p（0..100）的秩为 h = p/100·(n-1)，
    取下界/上界样本线性插值：
        q = x_floor(h) + (h - floor(h)) · (x_ceil(h) - x_floor(h))
    即 p=0 取最小值、p=100 取最大值。调用前样本必须已升序排列且非空。
    """
    if not sorted_values:
        raise ValueError("空样本无法计算分位数")
    n = len(sorted_values)
    if n == 1:
        return round(float(sorted_values[0]), 2)
    rank = (pct / 100.0) * (n - 1)
    low = int(rank)
    high = min(low + 1, n - 1)
    frac = rank - low
    value = sorted_values[low] + frac * (sorted_values[high] - sorted_values[low])
    return round(float(value), 2)


def _score_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return dict(EMPTY_RATES)
    passed = sum(1 for r in records if r["response"]["passed"])
    scores = sorted(float(r["response"]["score"]) for r in records)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 2),
        "average_score": round(sum(scores) / total, 2),
        "p50_score": percentile_linear(scores, 50),
        "p90_score": percentile_linear(scores, 90),
    }


def _first_failure_distribution(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """首个失败笔写下标分布：仅统计失败提交，按下标升序。"""
    counts: dict[int, int] = {}
    for r in records:
        resp = r["response"]
        if not resp["passed"] and resp.get("first_failed_stroke") is not None:
            idx = int(resp["first_failed_stroke"])
            counts[idx] = counts.get(idx, 0) + 1
    return [
        {"stroke_index": idx, "count": counts[idx]}
        for idx in sorted(counts)
    ]


def _daily_buckets(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 UTC 日期分桶：只出现有提交的日期，日期升序。"""
    by_day: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        day = parse_created_at(r["created_at"]).date().isoformat()
        by_day.setdefault(day, []).append(r)
    buckets = []
    for day in sorted(by_day):
        stats = _score_stats(by_day[day])
        buckets.append({
            "date": day,
            "total": stats["total"],
            "passed": stats["passed"],
            "pass_rate": stats["pass_rate"],
            "average_score": stats["average_score"],
        })
    return buckets


def build_practice_report(
    snapshot: list[dict[str, Any]],
    character: str,
    start: datetime,
    end: datetime,
    passed: Optional[bool] = None,
    generated_at: Optional[datetime] = None,
) -> dict[str, Any]:
    """在已取得的一致性快照上过滤并聚合练习趋势。

    时间区间为**闭区间** [start, end]，起止时刻统一按 UTC 比较；
    ``passed`` 为 None 时不过滤合格状态。空结果返回零值统计与空数组。
    """
    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)

    filtered: list[dict[str, Any]] = []
    for r in snapshot:
        resp = r["response"]
        if resp.get("character") != character:
            continue
        created = parse_created_at(r["created_at"])
        if not (start_utc <= created <= end_utc):
            continue
        if passed is not None and bool(resp.get("passed")) != passed:
            continue
        filtered.append(r)

    stats = _score_stats(filtered)
    generated = (generated_at or datetime.now(timezone.utc))
    return {
        "filter": {
            "character": character,
            "start": start_utc.isoformat(timespec="seconds"),
            "end": end_utc.isoformat(timespec="seconds"),
            "passed": passed,
        },
        "total_submissions": stats["total"],
        "passed_submissions": stats["passed"],
        "pass_rate": stats["pass_rate"],
        "average_score": stats["average_score"],
        "p50_score": stats["p50_score"],
        "p90_score": stats["p90_score"],
        "first_failed_stroke_distribution": _first_failure_distribution(filtered),
        "daily": _daily_buckets(filtered),
        "generated_at": generated.astimezone(timezone.utc)
        .isoformat(timespec="seconds"),
    }
