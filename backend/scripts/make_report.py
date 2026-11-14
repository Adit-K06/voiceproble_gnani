"""Generate a self-contained HTML report from saved VoiceProbe results."""

from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

_BACKEND = Path(__file__).resolve().parent.parent
_OUTPUT = _BACKEND / "output"
_RESULTS = _OUTPUT / "results.json"
_AUDIO = _OUTPUT / "audio"
_REPORT = _OUTPUT / "report.html"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _number(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.3f}{suffix}"


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _condition_order(results: list[dict], config: dict) -> list[str]:
    present = {row.get("condition", "unknown") for row in results}
    ordered = [
        condition
        for condition in config.get("conditions", [])
        if condition in present
    ]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _metrics_by_condition(
    results: list[dict], condition_order: list[str]
) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        grouped[row.get("condition", "unknown")].append(row)

    metrics: dict[str, dict] = {}
    for condition in condition_order:
        rows = grouped[condition]
        wers = [float(row.get("wer", 0.0)) for row in rows]
        amount_flags = [
            row["amount_correct"]
            for row in rows
            if isinstance(row.get("amount_correct"), bool)
        ]
        latencies = [
            float(row["stt_latency_s"])
            for row in rows
            if row.get("stt_latency_s") is not None
        ]
        metrics[condition] = {
            "count": len(rows),
            "mean_wer": sum(wers) / len(wers) if wers else None,
            "amount_accuracy": (
                sum(amount_flags) / len(amount_flags) if amount_flags else None
            ),
            "amount_correct": sum(amount_flags),
            "amount_scored": len(amount_flags),
            "stt_p50": _percentile(latencies, 0.50),
            "stt_p95": _percentile(latencies, 0.95),
        }
    return metrics


def _audio_src(raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    basename = str(raw_path).replace("\\", "/").rsplit("/", 1)[-1]
    if basename in {"", ".", ".."}:
        return None
    candidate = _AUDIO / basename
    if not candidate.is_file():
        return None
    return "audio/" + quote(basename, safe="-_.")


def _audio_player(raw_path: str | None, label: str) -> str:
    source = _audio_src(raw_path)
    if source is None:
        return '<span class="muted">clip unavailable</span>'
    return (
        f'<audio controls preload="none" aria-label="{html.escape(label, quote=True)}" '
        f'src="{html.escape(source, quote=True)}"></audio>'
    )


def _condition_table(metrics: dict[str, dict]) -> str:
    rows = []
    for condition, metric in metrics.items():
        rows.append(
            "<tr>"
            f"<th scope=\"row\">{html.escape(condition)}</th>"
            f"<td>{metric['count']}</td>"
            f"<td>{_percent(metric['mean_wer'])}</td>"
            f"<td>{_percent(metric['amount_accuracy'])} "
            f"<span class=\"muted\">({metric['amount_correct']}/"
            f"{metric['amount_scored']})</span></td>"
            f"<td>{_number(metric['stt_p50'], ' s')} / "
            f"{_number(metric['stt_p95'], ' s')}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _sentence_table(results: list[dict]) -> str:
    rows = []
    for row in results:
        condition = str(row.get("condition", "unknown"))
        sentence_id = str(row.get("sentence_id", "unknown"))
        amount = row.get("amount_correct")
        amount_text = "Yes" if amount is True else "No" if amount is False else "n/a"
        amount_class = "good" if amount is True else "bad" if amount is False else "muted"
        reference = html.escape(str(row.get("reference", "")))
        transcript = html.escape(str(row.get("hypothesis", "")))
        rows.append(
            f'<tr data-condition="{html.escape(condition, quote=True)}">'
            f"<td><strong>{html.escape(sentence_id)}</strong>"
            f"<span class=\"subline\">{html.escape(condition)}</span></td>"
            f"<td class=\"transcript\">{reference}</td>"
            f"<td class=\"transcript\">{transcript}</td>"
            f"<td>{float(row.get('wer', 0.0)):.3f}</td>"
            f"<td class=\"{amount_class}\">{amount_text}</td>"
            f"<td>{_audio_player(row.get('original_wav_path'), 'Original audio')}</td>"
            f"<td>{_audio_player(row.get('degraded_wav_path'), 'Degraded audio')}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _chart(metrics: dict[str, dict]) -> str:
    conditions = list(metrics)
    width = max(740, 175 * len(conditions) + 100)
    height = 360
    left, right, top, bottom = 66, 24, 50, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    baseline = top + plot_height
    values = [
        value
        for metric in metrics.values()
        for value in (metric["mean_wer"], metric["amount_accuracy"])
        if value is not None
    ]
    scale_max = max(1.0, max(values, default=1.0))
    scale_max = math.ceil(scale_max * 4) / 4
    pieces = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Mean word error rate and amount accuracy by condition">',
        '<title>WER and amount accuracy by condition</title>',
    ]

    for tick in range(5):
        ratio = tick / 4
        y = baseline - ratio * plot_height
        label = f"{scale_max * ratio:.0%}"
        pieces.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
            'class="gridline"/>'
        )
        pieces.append(
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" '
            f'class="axis-label">{label}</text>'
        )

    group_width = plot_width / max(1, len(conditions))
    bar_width = min(38, group_width * 0.25)
    for index, condition in enumerate(conditions):
        metric = metrics[condition]
        center = left + group_width * (index + 0.5)
        bars = [
            ("WER", metric["mean_wer"], "#4b6fe0", center - bar_width - 4),
            (
                "Amount accuracy",
                metric["amount_accuracy"],
                "#159878",
                center + 4,
            ),
        ]
        for label, value, color, x in bars:
            if value is None:
                pieces.append(
                    f'<text x="{x + bar_width/2:.1f}" y="{baseline-8}" '
                    f'text-anchor="middle" class="bar-value">n/a</text>'
                )
                continue
            bar_height = max(1.5, plot_height * value / scale_max) if value > 0 else 0
            y = baseline - bar_height
            pieces.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" rx="5" fill="{color}">'
                f'<title>{html.escape(condition)} — {label}: {value:.1%}</title>'
                '</rect>'
            )
            pieces.append(
                f'<text x="{x + bar_width/2:.1f}" y="{max(top-2, y-8):.1f}" '
                f'text-anchor="middle" class="bar-value">{value:.0%}</text>'
            )
        pieces.append(
            f'<text x="{center:.1f}" y="{baseline+25}" text-anchor="middle" '
            f'class="condition-label">{html.escape(condition)}</text>'
        )

    pieces.extend(
        [
            f'<rect x="{left}" y="{height-24}" width="12" height="12" rx="3" '
            'fill="#4b6fe0"/><text x="84" y="height-14" class="legend-label">'
            'Mean WER</text>',
            f'<rect x="{left+125}" y="{height-24}" width="12" height="12" rx="3" '
            'fill="#159878"/><text x="{left+143}" y="{height-14}" '
            'class="legend-label">Amount accuracy</text>',
            "</svg>",
        ]
    )
    # SVG coordinates in the legend text need numeric values rather than literals.
    svg = "\n".join(pieces)
    svg = svg.replace('x="{left+143}"', f'x="{left+143}"')
    svg = svg.replace('y="height-14"', f'y="{height-14}"')
    return svg


def _insights(results: list[dict], metrics: dict[str, dict]) -> str:
    if not results:
        return "<li>No result rows were found in the saved file.</li>"

    mean_wer = sum(float(row.get("wer", 0.0)) for row in results) / len(results)
    amount_flags = [
        row["amount_correct"]
        for row in results
        if isinstance(row.get("amount_correct"), bool)
    ]
    amount_accuracy = (
        sum(amount_flags) / len(amount_flags) if amount_flags else None
    )
    conditions = list(metrics.items())
    best = min(
        ((name, item["mean_wer"]) for name, item in conditions if item["mean_wer"] is not None),
        key=lambda item: item[1],
        default=None,
    )
    worst = max(
        ((name, item["mean_wer"]) for name, item in conditions if item["mean_wer"] is not None),
        key=lambda item: item[1],
        default=None,
    )

    bullets = [
        f"{len(results)} saved sentence-condition cases have a mean WER of "
        f"{mean_wer:.1%}."
    ]
    if amount_accuracy is not None:
        bullets.append(
            f"Amount extraction was correct in {sum(amount_flags)}/{len(amount_flags)} "
            f"scored cases ({amount_accuracy:.1%})."
        )
    if best is not None and worst is not None:
        bullets.append(
            f"Mean WER ranged from {best[1]:.1%} in {html.escape(best[0])} "
            f"to {worst[1]:.1%} in {html.escape(worst[0])}."
        )

    high_wer_amount_correct = [
        row
        for row in results
        if float(row.get("wer", 0.0)) >= 0.25 and row.get("amount_correct") is True
    ]
    if high_wer_amount_correct:
        examples = "; ".join(
            f"{html.escape(str(row.get('sentence_id', 'unknown')))} / "
            f"{html.escape(str(row.get('condition', 'unknown')))} "
            f"(WER {float(row.get('wer', 0.0)):.1%})"
            for row in high_wer_amount_correct
        )
        bullets.append(
            "High WER (at least 25%) coexists with a correct amount in: "
            + examples
            + ". This shows text similarity and amount correctness measure different things."
        )
    else:
        bullets.append(
            "No saved case has WER of at least 25% while its amount is correct. "
            "WER and amount correctness are separate measures, so this can occur in other runs."
        )
    return "\n".join(f"<li>{bullet}</li>" for bullet in bullets)


def render_report(payload: dict) -> str:
    results = payload.get("results", [])
    config = payload.get("run_config", {})
    condition_order = _condition_order(results, config)
    metrics = _metrics_by_condition(results, condition_order)

    all_amount_flags = [
        row["amount_correct"]
        for row in results
        if isinstance(row.get("amount_correct"), bool)
    ]
    overall_amount_accuracy = (
        sum(all_amount_flags) / len(all_amount_flags) if all_amount_flags else None
    )
    mean_wer = (
        sum(float(row.get("wer", 0.0)) for row in results) / len(results)
        if results
        else None
    )

    condition_options = "".join(
        f'<option value="{html.escape(condition, quote=True)}">'
        f"{html.escape(condition)}</option>"
        for condition in condition_order
    )
    cards = (
        f'<div class="stat"><span>Sentence-condition cases</span>'
        f"<strong>{len(results)}</strong></div>"
        f'<div class="stat"><span>Conditions</span>'
        f"<strong>{len(condition_order)}</strong></div>"
        f'<div class="stat"><span>Mean WER</span>'
        f"<strong>{_percent(mean_wer)}</strong></div>"
        f'<div class="stat"><span>Amount accuracy</span>'
        f"<strong>{_percent(overall_amount_accuracy)}</strong></div>"
    )

    css = """
:root{color-scheme:light;--ink:#17243b;--muted:#65738b;--line:#dfe6ef;--paper:#fff;--bg:#f3f6fb;--navy:#142541;--blue:#4b6fe0;--green:#159878;--amber:#fff3d5;--amber-ink:#704b05;--red:#a93636}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
header{background:linear-gradient(120deg,#142541,#203c64);color:white;padding:38px max(24px,calc((100vw - 1280px)/2));}
h1{margin:0;font-size:clamp(27px,4vw,40px);letter-spacing:-.03em}
header p{color:#d4deeb;margin:8px 0 0}
main{max-width:1280px;margin:24px auto;padding:0 24px 52px}
.banner{padding:15px 18px;background:var(--amber);color:var(--amber-ink);border-left:5px solid #e0a228;border-radius:8px;font-weight:750;margin-bottom:22px}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:18px 0}
.stat,.panel{background:var(--paper);border:1px solid var(--line);border-radius:14px;box-shadow:0 4px 16px rgba(21,39,68,.04)}
.stat{padding:18px 20px}.stat span{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.07em}.stat strong{display:block;font-size:28px;margin-top:4px}
.panel{padding:22px;margin:18px 0 24px}
h2{margin:0 0 12px;font-size:22px;letter-spacing:-.02em}
.section-note,.muted{color:var(--muted);font-size:13px}
.chart-wrap{overflow-x:auto}.chart{width:100%;min-width:680px;height:auto}
.gridline{stroke:#e5eaf1;stroke-width:1}.axis-label,.condition-label,.legend-label,.bar-value{font:12px system-ui,sans-serif;fill:#52627a}.bar-value{font-weight:700;fill:#24344f}.condition-label{font-size:12px}
.table-wrap{overflow:auto}
table{border-collapse:collapse;width:100%;min-width:730px}
th,td{padding:12px 13px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
thead th{background:#f7f9fc;color:#4c5c75;font-size:12px;text-transform:uppercase;letter-spacing:.055em;position:sticky;top:0;z-index:1}
tbody tr:hover{background:#f8faff}
.numeric{font-variant-numeric:tabular-nums;white-space:nowrap}
.good{color:#087c5d;font-weight:750}.bad{color:var(--red);font-weight:750}
.transcript{min-width:220px;max-width:370px;overflow-wrap:anywhere}
.subline{display:block;color:var(--muted);font-size:12px}
audio{width:190px;height:36px}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:12px 0}
.controls input,.controls select{border:1px solid #ccd6e3;border-radius:8px;padding:9px 11px;font:inherit;background:white;color:var(--ink)}
.controls input{min-width:240px}
.insights{padding-left:22px;margin:8px 0}.insights li{margin:9px 0}
footer{color:var(--muted);font-size:12px;text-align:center;padding:12px}
@media(max-width:760px){main{padding:0 14px 32px}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}.panel{padding:16px}.stat{padding:14px}}
"""

    javascript = """
(() => {
  const condition = document.getElementById('condition-filter');
  const search = document.getElementById('row-search');
  const rows = [...document.querySelectorAll('#sentence-rows tr')];
  const update = () => {
    const selected = condition.value;
    const query = search.value.trim().toLocaleLowerCase();
    for (const row of rows) {
      const matchesCondition = !selected || row.dataset.condition === selected;
      const matchesText = !query || row.textContent.toLocaleLowerCase().includes(query);
      row.hidden = !(matchesCondition && matchesText);
    }
  };
  condition.addEventListener('change', update);
  search.addEventListener('input', update);
})();
"""

    page = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VoiceProbe results report</title>
<style>@@CSS</style>
</head>
<body>
<header>
  <h1>VoiceProbe results</h1>
  <p>Saved transcript and entity scoring by simulated phone condition</p>
</header>
<main>
  <div class="banner">Synthetic audio and simulated phone degradation: relative comparisons only, not real-world accuracy</div>
  <section class="stats" aria-label="Overall summary">@@CARDS</section>

  <section class="panel">
    <h2>Condition summary</h2>
    <p class="section-note">WER is the mean word error rate. Latency reports STT latency p50 / p95.</p>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Condition</th><th>Cases</th><th>Mean WER</th><th>Amount accuracy</th><th>STT latency p50 / p95</th></tr></thead>
        <tbody>@@CONDITION_ROWS</tbody>
      </table>
    </div>
  </section>

  <section class="panel">
    <h2>WER and amount accuracy</h2>
    <p class="section-note">The chart uses one shared percentage scale; WER may exceed 100% in some datasets.</p>
    <div class="chart-wrap">@@CHART</div>
  </section>

  <section class="panel">
    <h2>What this shows</h2>
    <ul class="insights">@@INSIGHTS</ul>
  </section>

  <section class="panel">
    <h2>Sentence results and audio</h2>
    <div class="controls">
      <label for="condition-filter">Condition</label>
      <select id="condition-filter"><option value="">All conditions</option>@@OPTIONS</select>
      <label for="row-search">Find text or sentence</label>
      <input id="row-search" type="search" placeholder="Search reference, transcript, ID…">
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Sentence / condition</th><th>Reference</th><th>Transcript</th><th>WER</th><th>Amount correct</th><th>Original audio</th><th>Degraded audio</th></tr></thead>
        <tbody id="sentence-rows">@@SENTENCE_ROWS</tbody>
      </table>
    </div>
    <p class="section-note">Audio sources are relative links to the existing <code>output/audio</code> clips.</p>
  </section>
  <footer>Generated from the saved results file. CSS, JavaScript, and chart SVG are embedded in this HTML.</footer>
</main>
<script>@@JAVASCRIPT</script>
</body>
</html>
"""
    replacements = {
        "@@CSS": css,
        "@@CARDS": cards,
        "@@CONDITION_ROWS": _condition_table(metrics),
        "@@CHART": _chart(metrics),
        "@@INSIGHTS": _insights(results, metrics),
        "@@OPTIONS": condition_options,
        "@@SENTENCE_ROWS": _sentence_table(results),
        "@@JAVASCRIPT": javascript,
    }
    for marker, value in replacements.items():
        page = page.replace(marker, value)
    return page


def main() -> None:
    payload = json.loads(_RESULTS.read_text(encoding="utf-8"))
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    _REPORT.write_text(render_report(payload), encoding="utf-8")
    print(f"Wrote {_REPORT}")


if __name__ == "__main__":
    main()
