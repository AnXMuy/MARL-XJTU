"""Validate the paired evaluation matrix and rebuild report tables and SVGs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from itertools import product
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("fixed-time", "max-pressure", "ippo", "mappo", "mappo-network")
COLORS = dict(zip(METHODS, ("#697386", "#137a55", "#2449aa", "#b64626", "#923b9d")))


def load(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_matrix(output, protocol, network=ROOT / "scenarios/network/grid4x4.net.xml"):
    rows, locals_ = [], []
    keys = {(profile, seed, method, init) for profile, seed, method in
            product(protocol["profiles"], protocol["test_demand_seeds"], METHODS)
            for init in (protocol["initialization_seeds"] if method in METHODS[2:] else [None])}
    actual_paths = set((output / "evaluation").glob("*/*/*/*/result.json"))
    paths = {output / f"evaluation/{profile}/{seed}/{method}/{init}/result.json": (profile, seed, method, init)
             for profile, seed, method, init in keys}
    require(actual_paths == paths.keys(), f"Incomplete matrix: {len(paths.keys()-actual_paths)} missing, "
            f"{len(actual_paths-paths.keys())} unexpected")
    for path, (profile, seed, method, init) in sorted(paths.items()):
        result = load(path)
        config, metrics = result["config"], result["metrics"]
        require((result["profile"], config["seed"], result["method"], result["training_seed"]) ==
                (profile, seed, method, init), f"Unexpected identity: {path}")
        route = ROOT / f"scenarios/routes/test_{profile}_{seed}.rou.xml"
        require(config["route_sha256"] == digest(route), f"Route changed: {path}")
        require(config["network_sha256"] == digest(network), f"Network changed: {path}")
        require(config["episode_seconds"] == protocol["episode_seconds"], f"Wrong horizon: {path}")
        if init is not None:
            checkpoint = output / f"models/{method}/{init}/checkpoint.pt"
            require(result["checkpoint_sha256"] == digest(checkpoint), f"Model changed: {path}")
        require(tuple(config[field] for field in ("delta_time_seconds", "yellow_time_seconds", "min_green_seconds",
                "all_red_seconds")) == (5, 3, 5, 0), f"Wrong control timing: {path}")
        require(metrics["scheduled_vehicles"] == metrics["completed_vehicles"] + metrics["unfinished_vehicles"]
                and metrics["scheduled_vehicles"] == metrics["departed_vehicles"] + metrics["not_inserted_vehicles"]
                and metrics["completed_vehicles"] <= metrics["departed_vehicles"], f"Inconsistent counts: {path}")
        require(len(load(path.parent / Path(result["trace_file"]).name)) == protocol["episode_seconds"], f"Wrong trace length: {path}")
        require((path.parent / Path(result["tripinfo_file"]).name).is_file(), f"Missing tripinfo: {path}")
        labels = dict(profile=profile, demand_seed=seed, method=method, initialization=init)
        numeric = {key: value for key, value in metrics.items() if not isinstance(value, dict)}
        require(all(value is None or math.isfinite(value) for value in numeric.values()), f"Nonfinite metric: {path}")
        rows.append(dict(**labels, route_sha256=config["route_sha256"], runtime_seconds=result["runtime_seconds"], **numeric))
        require(len(metrics["intersection_queue_integrals"]) == 16, f"Missing intersections: {path}")
        for signal, value in sorted(metrics["intersection_queue_integrals"].items()):
            locals_.append(dict(**labels, signal=signal, queue_vehicle_seconds=value))
    return rows, locals_


def grouped(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    return groups


def available_mean(values):
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def statistics(rows, protocol):
    metric_names = list(rows[0])[6:]
    by_seed = {key: {metric: available_mean([row[metric] for row in group]) for metric in metric_names}
               for key, group in grouped(rows, ("profile", "method", "demand_seed")).items()}
    seeds = protocol["test_demand_seeds"]
    primary = protocol["primary_metric"]
    generator = np.random.default_rng(20261122)
    indices = generator.integers(0, len(seeds), size=(protocol["bootstrap_resamples"], len(seeds)))
    summary, contrasts, initializations = [], [], []
    comparisons = [(method, baseline) for method in METHODS[2:] for baseline in METHODS[:2]]
    comparisons += [("max-pressure", "fixed-time"), ("ippo", "mappo"), ("mappo-network", "mappo")]
    for profile in protocol["profiles"]:
        for method in METHODS:
            for metric in metric_names:
                values = [by_seed[profile, method, seed][metric] for seed in seeds]
                present = [value for value in values if value is not None]
                summary.append(dict(profile=profile, method=method, metric=metric, n=len(present),
                                    mean=available_mean(values), sd=stdev(present) if len(present) > 1 else None))
        for method, reference in comparisons:
            paired = np.array([by_seed[profile, method, seed][primary] - by_seed[profile, reference, seed][primary]
                               for seed in seeds])
            lower, upper = np.quantile(paired[indices].mean(axis=1), [0.025, 0.975])
            contrasts.append(dict(profile=profile, method=method, reference=reference, n=len(seeds),
                                  difference=paired.mean(), ci_low=lower, ci_high=upper))
    for key, group in grouped(rows, ("profile", "method", "initialization")).items():
        if key[2] is not None:
            initializations.append(dict(zip(("profile", "method", "initialization"), key), n=len(group),
                                        **{metric: available_mean([row[metric] for row in group]) for metric in metric_names}))
    return summary, contrasts, initializations


def svg(path, title, width, height, parts):
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                    f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
                    f'<g font-family="sans-serif" font-size="13" fill="#192238">'
                    f'{text(24, 30, title, size=19)}{"".join(parts)}</g></svg>\n')


def text(x, y, label, *, size=13, color="#192238"):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{label}</text>'


def line(x1, y1, x2, y2, color):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.4"/>'


def figures(output, protocol, summary, locals_, training):
    primary = protocol["primary_metric"]
    profiles = list(protocol["profiles"])
    lookup = {(row["profile"], row["method"]): row for row in summary if row["metric"] == primary}
    parts = [text(24 + 155 * index, 407, method, color=COLORS[method]) for index, method in enumerate(METHODS)]
    top = max(row["mean"] + row["sd"] for row in lookup.values()) * 1.1
    for tick in range(6):
        pos_y = 350 - tick * 56
        parts.extend([line(68, pos_y, 790, pos_y, "#dbe2eb"), text(18, pos_y + 4, f"{top*tick/5:.0f}")])
    for index, profile in enumerate(profiles):
        left = 100 + 238 * index
        parts.append(text(left + 62, 377, profile))
        for method_index, method in enumerate(METHODS):
            row = lookup[profile, method]
            pos_x = left + method_index * 34
            scale = 280 / top
            height = scale * row["mean"]
            hi, lo = 350 - scale * (row["mean"] + row["sd"]), 350 - scale * max(0, row["mean"] - row["sd"])
            parts.extend([f'<rect x="{pos_x}" y="{350-height}" width="28" height="{height}" fill="{COLORS[method]}"/>',
                          line(pos_x+14, hi, pos_x+14, lo, "#172038"), line(pos_x+9, hi, pos_x+19, hi, "#172038"),
                          line(pos_x+9, lo, pos_x+19, lo, "#172038")])
    svg(output / "demand_performance.svg", "Restricted road waiting (s / scheduled vehicle; mean ± SD)", 820, 430, parts)
    for profile in profiles:
        cells = {key[1:]: mean(row["queue_vehicle_seconds"] for row in group) for key, group in
                 grouped([row for row in locals_ if row["profile"] == profile], ("profile", "method", "signal")).items()}
        upper = max(cells.values()) or 1
        parts = []
        for index, method in enumerate(METHODS):
            left = 20 + 170 * index
            parts.append(text(left, 65, method, color=COLORS[method]))
            for position, signal in enumerate(sorted(signal for name, signal in cells if name == method)):
                value = cells[method, signal]
                pos_x, pos_y = left + position // 4 * 39, 85 + (3 - position % 4) * 39
                shade = int(251 - 160 * value / upper)
                fill = f"rgb({shade},{min(255, shade+10)},{min(255, shade+30)})"
                parts.extend([f'<rect x="{pos_x}" y="{pos_y}" width="37" height="37" fill="{fill}"/>',
                              text(pos_x+4, pos_y+13, signal, size=10), text(pos_x+2, pos_y+28, f"{value:.0f}", size=10)])
        svg(output / f"intersection_{profile}.svg", f"{profile} | Mean intersection queue integral (vehicle-seconds)", 870, 260, parts)
    parts = []
    ymin, ymax = min(row["mean_reward"] for row in training), 0
    for index, method in enumerate(METHODS[2:]):
        left = 45 + 295 * index
        parts.append(text(left, 65, method, color=COLORS[method]))
        for tick in range(5):
            pos_y = 265 - 43 * tick
            parts.extend([line(left, pos_y, left+246, pos_y, "#dbe2eb"),
                          text(left-36, pos_y+4, f"{ymin+(ymax-ymin)*tick/4:.2f}", size=10)])
        for init_index, initialization in enumerate(protocol["initialization_seeds"]):
            series = [row for row in training if row["method"] == method and row["initialization"] == initialization]
            coordinates = " ".join(f"{left+246*row['episode']/max(1,len(series)-1):.2f},"
                                   f"{265-172*(row['mean_reward']-ymin)/(ymax-ymin or 1):.2f}" for row in series)
            parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{COLORS[method]}" '
                         f'stroke-width="1.7" stroke-opacity="{1-init_index*.20}" stroke-dasharray="{["0", "5 3", "2 3"][init_index]}"/>')
        parts.extend([text(left, 287, "1"), text(left+219, 287, str(protocol["training_episodes"]))])
    svg(output / "training_curves.svg", "Training reward by episode (solid: 11; dashed: 22; dotted: 33)", 910, 310, parts)


def plot_failure(output, rows):
    worst = max((row for row in rows if row["profile"] == "C3" and row["method"] in METHODS[2:]),
                key=lambda row: row["unfinished_vehicles"])
    seed = worst["demand_seed"]
    cases = [(worst["method"], worst["initialization"]), ("max-pressure", None)]
    traces = {}
    for method, initialization in cases:
        folder = output / f"evaluation/C3/{seed}/{method}/{initialization}"
        result = load(folder / "result.json")
        traces[method] = load(folder / Path(result["trace_file"]).name)
    y_max = max(sample[field] for trace in traces.values() for sample in trace
                for field in ("vehicles", "stopped", "pending")) or 1
    parts = []
    for index, field in enumerate(("vehicles", "stopped", "pending")):
        left = 47 + 290 * index
        parts.append(text(left, 64, field))
        for tick in range(6):
            pos_y = 270 - 36 * tick
            parts.extend([line(left, pos_y, left+240, pos_y, "#dbe2eb"),
                          text(left-39, pos_y+4, f"{y_max*tick/5:.0f}", size=10)])
        parts.append(line(left+120, 90, left+120, 270, "#94a3b8"))
        for method, trace in traces.items():
            coordinates = " ".join(f"{left+240*sample['second']/600:.2f},{270-180*sample[field]/y_max:.2f}" for sample in trace)
            parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{COLORS[method]}" stroke-width="1.7"/>')
        parts.extend([text(left, 292, "0"), text(left+105, 292, "300"), text(left+204, 292, "600 s")])
    parts.extend([text(45, 321, f"{worst['method']} (init {worst['initialization']})", color=COLORS[worst["method"]]),
                  text(265, 321, "max-pressure", color=COLORS["max-pressure"])])
    svg(output / "figures/failure_trace.svg", f"C3 maximum unfinished count among learned policies | demand seed {seed}",
        900, 342, parts)


def publish(output, protocol, rows, locals_, summary, contrasts):
    primary = protocol["primary_metric"]
    stats = {(row["profile"], row["method"], row["metric"]): row for row in summary}

    def formatted(profile, method, metric):
        row = stats[profile, method, metric]
        return f'{row["mean"]:.2f} ± {row["sd"]:.2f}' if row["sd"] is not None else "—"

    lines = ["# 正式评估：结果表与图", "", f"已核验 {len(rows)} 个评估 episode，覆盖三个档位、"
             f"每档 {len(protocol['test_demand_seeds'])} 个配对需求；学习方法每个需求先平均三个独立初始化。", "",
             "主指标为仿真窗口内的全网停止车秒 / 计划发车数（秒/车，越低越好）；未插入车辆的等待不计入。"
             "完成车辆的等待与行程指标有条件抽样偏差，须同时查看滞留和未插入。表格为需求均值 ± 样本标准差。", "",
             "| 档位 | 方法 | 主指标 | 完成车辆 | 未完成车辆 | 未插入车辆 |", "|---|---|---:|---:|---:|---:|"]
    for profile, method in product(protocol["profiles"], METHODS):
        cells = [formatted(profile, method, metric) for metric in
                 (primary, "completed_vehicles", "unfinished_vehicles", "not_inserted_vehicles")]
        lines.append("| " + " | ".join([profile, method, *cells]) + " |")
    lines += ["", "![需求负载与主指标](figures/demand_performance.svg)", "",
              "## 运营与局部指标", "", "吞吐量为 600 s 全窗口完成数折算为辆/小时，不代表稳态容量。"
              "最坏路口按每个 episode 的最大进口排队积分（车秒）汇总。", "",
              "| 档位 | 方法 | 完成车等待 s | 完成车旅行 s | 吞吐量 | 平均网络停车数 | 平均在网数 | 最坏路口车秒 | 切相次数 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for profile, method in product(protocol["profiles"], METHODS):
        cells = [formatted(profile, method, metric) for metric in ("completed_mean_waiting_time_seconds",
                 "completed_mean_travel_time_seconds", "throughput_vehicles_per_hour", "mean_network_stopped_vehicles",
                 "mean_network_vehicles", "worst_intersection_queue_integral", "phase_switches")]
        lines.append("| " + " | ".join([profile, method, *cells]) + " |")
    lines += ["", "## 配对差值与消融", "", "差值 = 方法 − 参照；负值表示主指标下降。"
              f"按需求 seed 成对重采样 {protocol['bootstrap_resamples']} 次（统计 seed 20261122），"
              "区间不覆盖训练种子总体的不确定性，也不作为多重显著性检验。", "",
              "| 档位 | 方法 | 参照 | 平均差值 | 95% CI |", "|---|---|---|---:|---:|"]
    for row in contrasts:
        lines.append(f'| {row["profile"]} | {row["method"]} | {row["reference"]} | {row["difference"]:+.2f} | '
                     f'[{row["ci_low"]:+.2f}, {row["ci_high"]:+.2f}] |')
    lines += ["", "## 失效与局部退化", ""]
    for method in METHODS:
        high = max((row for row in rows if row["profile"] == "C3" and row["method"] == method),
                   key=lambda row: row["unfinished_vehicles"])
        lines.append(f'- C3 {method} 最大积压：需求 {high["demand_seed"]}、初始化 {high["initialization"]}，'
                     f'未完成 {high["unfinished_vehicles"]}/{high["scheduled_vehicles"]}，'
                     f'期末仍在网 {high["final_network_vehicles"]}、未插入 {high["not_inserted_vehicles"]}。')
    lines += ["", "下图选取学习方法中未完成数最多的一次运行，与同一需求的 Max Pressure 对照；竖线为停止注入的时刻。"
              "这是按积压挑选的极端案例，不代表典型需求。", "", "![C3 极端积压轨迹](figures/failure_trace.svg)", ""]
    local_avg = {key: mean(row["queue_vehicle_seconds"] for row in group) for key, group in
                 grouped(locals_, ("profile", "method", "signal")).items()}
    for profile in protocol["profiles"]:
        method, signal, change = max(((method, signal, value-local_avg[profile, "max-pressure", signal])
                   for (level, method, signal), value in local_avg.items() if level == profile and method in METHODS[2:]),
                   key=lambda item: item[2])
        lines.append(f'- {profile} 学习方法相对 Max Pressure 的最大路口排队积分差：{method} / {signal}，{change:+.1f} 车秒。')
        lines += ["", f"![{profile} 路口平均排队](figures/intersection_{profile}.svg)", ""]
    lines += ["## 训练曲线", "", "连续各 episode 使用不同档位和需求；曲线仅用于诊断，不支持收敛或性能结论。", "",
              "![训练曲线](figures/training_curves.svg)", "", "完整数据、逐初始化结果和配对统计见 `tables/`。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/study-v1")
    parser.add_argument("--recorded-network", type=Path, default=ROOT / "scenarios/network/grid4x4.net.xml")
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = load(output / "protocol.json")
    rows, locals_ = read_matrix(output, protocol, args.recorded_network)
    summary, contrasts, initializations = statistics(rows, protocol)
    training = []
    for method, initialization in product(METHODS[2:], protocol["initialization_seeds"]):
        path = output / f"models/{method}/{initialization}/train.jsonl"
        records = [json.loads(record) for record in path.read_text().splitlines()]
        require([row["episode"] for row in records] == list(range(protocol["training_episodes"])), f"Invalid training log: {path}")
        for row in records:
            row["profile"] = re.search(r"train_(C\d+)_", row["route"])[1]
            training.append(dict(method=method, initialization=initialization, **row))
    (output / "tables").mkdir(exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    for name, data in [("episodes", rows), ("intersection_episodes", locals_), ("summary", summary),
                       ("paired_primary", contrasts), ("initializations", initializations), ("training", training)]:
        write_csv(output / f"tables/{name}.csv", data)
    figures(output / "figures", protocol, summary, locals_, training)
    plot_failure(output, rows)
    publish(output, protocol, rows, locals_, summary, contrasts)
    print(f"Validated {len(rows)} episodes, {len(training)} training records; wrote {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
