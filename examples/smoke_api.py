"""Exercise a running API over real HTTP and save reviewable evidence."""
import argparse
import json
import math
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def check_api(base_url, request_path):
    example = json.loads(Path(request_path).read_text())
    cases = [
        ("health", "GET", "/health", None, 200),
        ("features", "GET", "/features", None, 200),
        ("openapi", "GET", "/openapi.json", None, 200),
        ("example", "POST", "/predict", example, 200),
        ("minimal", "POST", "/predict", {"features": {"First6MonthOil_BBL": 1000}}, 200),
        ("null_optional", "POST", "/predict", {"features": {"First6MonthOil_BBL": 1000, "County": None}}, 200),
        ("missing_required", "POST", "/predict", {"features": {}}, 422),
        ("negative", "POST", "/predict", {"features": {"First6MonthOil_BBL": -1}}, 422),
        ("boolean", "POST", "/predict", {"features": {"First6MonthOil_BBL": True}}, 422),
        ("numeric_string", "POST", "/predict", {"features": {"First6MonthOil_BBL": "1000"}}, 422),
        ("future", "POST", "/predict", {"features": {"First6MonthOil_BBL": 1000, "First12MonthOil_BBL": 2000}}, 422),
        ("unknown", "POST", "/predict", {"features": {"First6MonthOil_BBL": 1000, "unknown": 1}}, 422),
        ("wrong_category", "POST", "/predict", {"features": {"First6MonthOil_BBL": 1000, "County": 1}}, 422),
        ("extra_envelope", "POST", "/predict", {"features": {"First6MonthOil_BBL": 1000}, "unknown": 1}, 422),
    ]
    evidence = {"base_url": base_url, "checks": []}
    for name, method, path, payload, expected in cases:
        request = Request(base_url.rstrip("/") + path, method=method,
                          data=None if payload is None else json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        try:
            response = urlopen(request, timeout=30)
        except HTTPError as exc:
            response = exc
        with response:
            status, body = response.status, json.load(response)
        assert status == expected, (name, status, body)
        if path == "/predict" and status == 200:
            assert all(math.isfinite(value) for value in body.values()), (name, body)
            assert body["predicted_months_7_to_12_oil_bbl"] >= 0
            assert abs(body["predicted_12_month_cumulative_oil_bbl"] -
                       body["predicted_months_7_to_12_oil_bbl"] -
                       body["actual_first_6_month_oil_bbl"]) <= 0.011
        evidence["checks"].append({"name": name, "status": status, "response": body if name != "openapi" else {"title": body["info"]["title"]}})
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--request", type=Path, default=Path(__file__).with_name("request.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = check_api(args.url, args.request)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))
