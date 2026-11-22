from __future__ import annotations

import argparse
import itertools
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

from run_experiment import NETWORK, ROOT, atomic_json, baseline_names, evaluation_tasks, sha
from report_utils import grouped, line, load, require, svg, text, write_csv


def assessment(delta, resamples, seed=20261122):
    delta = np.asarray(delta, dtype=float)
    require(delta.ndim == 2 and delta.size and np.isfinite(delta).all(), "Invalid paired matrix")
    initializations, demands = delta.shape
    rng = np.random.default_rng(seed)
    demand_draws = rng.integers(demands, size=(resamples, demands))
    init_draws = rng.integers(initializations, size=(resamples, initializations))
    paired = delta.mean(axis=0)
    conditional = np.quantile(paired[demand_draws].mean(axis=1), [.025, .975])
    crossed = np.quantile(delta[init_draws[:, :, None], demand_draws[:, None, :]].mean(axis=(1, 2)), [.025, .975])
    require(demands <= 16, "Exact sign-flip test is only intended for small demand sets")
    null = np.array(list(itertools.product([-1, 1], repeat=demands))) @ paired / demands
    return dict(delta=float(delta.mean()), demand_lo=float(conditional[0]), demand_hi=float(conditional[1]),
                crossed_lo=float(crossed[0]), crossed_hi=float(crossed[1]), n_demand=demands, n_init=initializations,
                conditional_signflip_p=float(np.mean(np.abs(null) >= abs(paired.mean()) - 1e-12)))


def holm(p_values):
    result = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        running = max(running, (len(p_values) - rank) * p_values[index])
        result[index] = min(1.0, running)
    return result


def path_for(task):
    output, _, method, init, _, _, group, profile, seed = task
    return output / f"evaluation/{group}/{profile}/{seed}/{method}/{init}/result.json"


def checked_matrix(output, protocol, tasks, network=NETWORK):
    expected = {path_for(task): task for task in tasks}
    actual = set((output / "evaluation").glob("*/*/*/*/*/result.json"))
    require(len(expected) == len(tasks) and expected.keys() == actual,
            f"Incomplete evaluation: {len(expected.keys()-actual)} missing, {len(actual-expected.keys())} unexpected")
    inputs = load(output / "inputs.json")
    require(inputs["network_sha256"] == sha(network), "Changed network")
    route_dir = output / "routes"
    require({path.name for path in route_dir.glob("*.xml")} == inputs["routes"].keys(), "Changed route set")
    route_counts = {}
    for route, digest in inputs["routes"].items():
        require(sha(route_dir / route) == digest, f"Changed route bytes: {route}")
        route_counts[route] = len(ET.parse(route_dir / route).getroot().findall("vehicle"))
    checkpoint_hashes, results, rows, locals_ = {}, {}, [], []
    backends = set()
    for path, (_, route, method, init, checkpoint, horizon, group, profile, seed) in sorted(expected.items()):
        result = load(path)
        backends.add(result.get("backend", "traci"))
        config, metrics = result["config"], result["metrics"]
        identity = group, profile, seed, method, init
        require(identity == (result["group"], result["profile"], config["seed"], result["method"], result["training_seed"]),
                f"Wrong identity: {path}")
        require((config["episode_seconds"], config["delta_time_seconds"], config["yellow_time_seconds"],
                 config["min_green_seconds"], config["all_red_seconds"], config["controller"]) == (horizon, 5, 3, 5, 0, method),
                f"Wrong settings: {path}")
        require(config["route_sha256"] == inputs["routes"][route.name] and config["network_sha256"] == inputs["network_sha256"],
                f"Wrong input hash: {path}")
        if checkpoint not in checkpoint_hashes:
            checkpoint_hashes[checkpoint] = sha(checkpoint) if checkpoint else None
        require(result["checkpoint_sha256"] == checkpoint_hashes[checkpoint], f"Wrong checkpoint: {path}")
        trace_path = path.parent / Path(result["trace_file"]).name
        trace = load(trace_path)
        count = metrics["scheduled_vehicles"]
        require(len(trace) == horizon and [item["second"] for item in trace] == list(range(1, horizon + 1)),
                f"Bad second-level trace: {path}")
        require(count == route_counts[route.name] == metrics["completed_vehicles"] + metrics["unfinished_vehicles"]
                == metrics["departed_vehicles"] + metrics["not_inserted_vehicles"] and
                0 <= metrics["completed_vehicles"] <= metrics["departed_vehicles"], f"Invalid vehicle counts: {path}")
        stopped = sum(item["stopped"] for item in trace)
        require(stopped == metrics["total_stopped_vehicle_seconds"] and
                math.isclose(stopped / count, metrics[protocol["primary_metric"]]), f"Invalid waiting total: {path}")
        pending = sum(item["pending"] for item in trace)
        require(pending == metrics["pending_vehicle_seconds"] and trace[-1]["vehicles"] == metrics["final_network_vehicles"],
                f"Invalid pending/final counts: {path}")
        trip_path = path.parent / Path(result["tripinfo_file"]).name
        trips = ET.parse(trip_path).getroot().findall("tripinfo")
        require(len(trips) == metrics["completed_vehicles"], f"Invalid trip file: {path}")
        queues = metrics["intersection_queue_integrals"]
        require(len(queues) == 16 and all(value >= 0 for value in queues.values()), f"Bad local metrics: {path}")
        values = np.array(list(queues.values()))
        labels = dict(group=group, profile=profile, demand_seed=seed, method=method, initialization=init)
        numeric = {key: value for key, value in metrics.items() if not isinstance(value, dict)}
        require(all(value is None or math.isfinite(value) for value in numeric.values()), f"Nonfinite metric: {path}")
        rows.append(dict(**labels, backend=result.get("backend", "traci"), route_sha256=config["route_sha256"],
                         runtime_seconds=result["runtime_seconds"], **numeric,
                         completion_fraction=metrics["completed_vehicles"]/count, local_queue_median=float(np.median(values)),
                         stopped_plus_pending_seconds_per_scheduled=(stopped+pending)/count,
                         local_queue_p95=float(np.quantile(values, .95)), local_queue_jain=float(values.sum()**2 /
                         (len(values) * (values**2).sum())) if values.any() else None,
                         departed_minus_completions_and_final=metrics["departed_vehicles"]-len(trips)-trace[-1]["vehicles"],
                         completed_trips_with_teleport_marker=sum("teleport" in trip.get("vaporized", "").lower() for trip in trips)))
        for signal, queue in sorted(queues.items()):
            locals_.append(dict(**labels, signal=signal, queue_vehicle_seconds=queue,
                                switches=metrics["phase_switches_by_intersection"].get(signal, 0)))
        results[identity] = (path, result, checkpoint)
    require(len(backends) == 1, f"Evaluation matrix mixed backends: {backends}")
    return rows, locals_, results


def read_training(output, protocol):
    import torch

    rows = []
    inputs = load(output / "inputs.json")
    for method, init in itertools.product(protocol["methods"], protocol["initialization_seeds"]):
        directory = output / f"models/{method}/{init}"
        records = [json.loads(entry) for entry in (directory / "train.jsonl").read_text().splitlines()]
        require([row["episode"] for row in records] == list(range(protocol["training_episodes"])), f"Bad training log: {directory}")
        for budget in protocol["checkpoint_episodes"]:
            state = torch.load(directory / f"checkpoint_{budget}.pt", map_location="cpu", weights_only=False)
            expected = [f"train_{profile}_{seed}.rou.xml" for seed in protocol["train_demand_seeds"] for profile in protocol["profiles"]]
            config = state["config"]
            require(config["seed"] == init and config["algorithm"] == ("ippo" if method == "ippo" else "mappo") and
                    config["reward"] == ("network-queue" if method == "mappo-network" else "queue") and
                    state["model_spec"]["actor_observation"] == ("phase-only" if method == "mappo-phase" else "local") and
                    state["episode"] + 1 == budget and state["network_sha256"] == inputs["network_sha256"] and
                    [Path(route).name for route in state["routes"]] == expected and
                    set(state["scenario_hashes"].values()) == {inputs["routes"][route] for route in expected},
                    f"Bad training checkpoint: {directory}, episode {budget}")
        for record in records:
            require(all(not isinstance(value, float) or math.isfinite(value) for value in record.values()), f"Nonfinite training: {directory}")
            rows.append(dict(method=method, initialization=init, profile=Path(record["route"]).name.split("_")[1], **record))
    return rows


def summarize(rows, protocol, selection):
    figure_data, summaries, inits, contrasts = {}, [], [], []
    groups = grouped(rows, ("group", "profile", "method"))
    primary = protocol["primary_metric"]
    for (group, profile, method), episodes in sorted(groups.items()):
        labels = dict(group=group, profile=profile, method=method)
        by_demand = grouped(episodes, ("demand_seed",))
        by_init = grouped(episodes, ("initialization",))
        figure_data[group, profile, method] = np.array([[row[primary] for row in sorted(init_rows, key=lambda row: row["demand_seed"])]
                                                       for _, init_rows in sorted(by_init.items())])
        for (init,), init_rows in by_init.items():
            inits.append(dict(**labels, initialization=init, mean=float(np.mean([row[primary] for row in init_rows])),
                              mean_switches=float(np.mean([row["phase_switches"] for row in init_rows]))))
        for metric in list(episodes[0])[8:]:
            values = [[row[metric] for row in demand_rows if row[metric] is not None] for demand_rows in by_demand.values()]
            demands = [np.mean(value) for value in values if len(value)]
            init_values = [[row[metric] for row in init_rows if row[metric] is not None] for init_rows in by_init.values()]
            init_means = [np.mean(value) for value in init_values if value]
            summaries.append(dict(**labels, metric=metric, mean=float(np.mean(demands)) if demands else None,
                                  demand_sd=float(np.std(demands, ddof=1)) if len(demands)>1 else None, n_demand=len(demands),
                                  initialization_sd=float(np.std(init_means, ddof=1)) if len(init_means)>1 else None,
                                  n_init=len(by_init) if (None,) not in by_init else 0))

    def compare(family, profile, method, baseline, delta):
        contrasts.append(dict(family=family, profile=profile, method=method, reference=baseline,
                              **assessment(delta, protocol["bootstrap_resamples"]), holm_p=None))

    for profile in protocol["profiles"]:
        table = {method: figure_data["test", profile, method] for method in baseline_names(protocol) + protocol["methods"]}
        for method, baseline in [("mappo", selection[profile]["method"]), ("mappo", "max-pressure"), ("ippo", "mappo"),
                                 ("mappo-network", "mappo"), ("mappo-phase", "mappo")]:
            compare("primary", profile, method, baseline, table[method]-table[baseline])
        for method in ("ippo", "mappo"):
            budget = figure_data["budget", profile, method]
            compare("budget", profile, f"{method}@60", f"{method}@24", table[method][:, :budget.shape[1]]-budget)
    primary_rows = [row for row in contrasts if row["family"] == "primary"]
    for row, value in zip(primary_rows, holm([row["conditional_signflip_p"] for row in primary_rows])):
        row["holm_p"] = value
    for profile in ("burst", "hotspot", "sustained"):
        for method, baseline in itertools.product(("ippo", "mappo"), (selection["C2"]["method"], "max-pressure")):
            compare("ood", profile, method, baseline, figure_data["ood", profile, method]-figure_data["ood", profile, baseline])
    return summaries, inits, contrasts


def action_diagnostics(results, primary):
    import torch
    from traffic_signal_marl.agents.ppo import PPOController

    torch.set_num_threads(1)
    models, fixed_traces, diagnostics = {}, {}, []
    for (group, profile, seed, method, init), (path, result, checkpoint) in results.items():
        if group != "test" or checkpoint is None:
            continue
        if checkpoint not in models:
            models[checkpoint] = PPOController(checkpoint).model
        model = models[checkpoint]
        trace = load(path.parent / Path(result["action_trace_file"]).name)
        require([row["second"] for row in trace] == list(range(0, result["config"]["episode_seconds"], 5)) and
                all(row["agent_ids"] == trace[0]["agent_ids"] for row in trace) and len(set(trace[0]["agent_ids"])) == 16,
                f"Invalid action schedule: {path}")
        obs = torch.tensor([row["observations"] for row in trace], dtype=torch.float32)
        actions = torch.tensor([row["actions"] for row in trace])
        require(obs.shape == (120, 16, 11) and actions.shape == (120, 16) and torch.isfinite(obs).all(), f"Bad actions: {path}")
        with torch.inference_mode():
            logits = model.logits(obs)
            require(torch.equal(logits.argmax(-1), actions), f"Greedy replay mismatch: {path}")
            no_traffic = obs.clone()
            no_traffic[..., 3:] = 0
            shuffled = obs.clone()
            shuffled[..., 3:] = obs.roll(1, dims=1)[..., 3:]
            null_logits = model.logits(no_traffic)
            shuffled_logits = model.logits(shuffled)
            zero_flip = float((null_logits.argmax(-1) != actions).float().mean())
            shuffle_flip = float((shuffled_logits.argmax(-1) != actions).float().mean())
            margin_shift = float(((null_logits[..., 1]-null_logits[..., 0]) - (logits[..., 1]-logits[..., 0])).abs().mean())
            require(method != "mappo-phase" or (zero_flip == shuffle_flip == margin_shift == 0), f"Traffic leaked into phase actor: {path}")
            entropy = float(torch.distributions.Categorical(logits=logits).entropy().mean())
        fixed_path, fixed, _ = results[group, profile, seed, "fixed-10", None]
        if (profile, seed) not in fixed_traces:
            fixed_traces[profile, seed] = load(fixed_path.parent / Path(fixed["trace_file"]).name)
        limits = result["metrics"]
        same_trace = load(path.parent / Path(result["trace_file"]).name) == fixed_traces[profile, seed]
        diagnostics.append(dict(profile=profile, demand_seed=seed, method=method, initialization=init,
                                switch_request_fraction=float((actions != obs[..., :2].argmax(-1)).float().mean()),
                                legal_switch_request_fraction=float((actions[obs[..., 2] > .5] !=
                                obs[..., :2].argmax(-1)[obs[..., 2] > .5]).float().mean()) if (obs[..., 2] > .5).any() else None,
                                actual_switches=limits["phase_switches"], entropy=entropy, zero_traffic_flip=zero_flip,
                                shuffled_traffic_flip=shuffle_flip, zero_traffic_margin_shift=margin_shift,
                                exact_fixed10_primary=limits[primary] == fixed["metrics"][primary],
                                exact_fixed10_metrics=limits == fixed["metrics"], exact_fixed10_trace=same_trace,
                                exact_fixed10_joint=same_trace and limits == fixed["metrics"]))
    return diagnostics


def local_contrasts(locals_, protocol, selection):
    keys = ("profile", "demand_seed", "signal", "method", "initialization")
    cells = {tuple(row[key] for key in keys): row["queue_vehicle_seconds"] for row in locals_ if row["group"] == "test"}
    signals = sorted({row["signal"] for row in locals_})
    rows = []
    for profile, signal in itertools.product(protocol["profiles"], signals):
        for method in protocol["methods"]:
            for reference in (selection[profile]["method"], "max-pressure"):
                matrix = np.array([[cells[profile, seed, signal, method, init] - cells[profile, seed, signal, reference, None]
                                    for seed in protocol["test_demand_seeds"]] for init in protocol["initialization_seeds"]])
                rows.append(dict(profile=profile, signal=signal, method=method, reference=reference,
                                 mean_queue_delta=float(matrix.mean()), demand_sd=float(matrix.mean(0).std(ddof=1)),
                                 initialization_sd=float(matrix.mean(1).std(ddof=1)), fraction_worse=float((matrix > 0).mean()),
                                 fraction_better=float((matrix < 0).mean())))
    return rows


def make_figures(output, protocol, selection, summaries, inits, contrasts, training, locals_, diagnostics, results):
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    lookup = {(row["group"], row["profile"], row["method"], row["metric"]): row["mean"] for row in summaries}
    primary = protocol["primary_metric"]
    colors = ["#2563eb", "#b45309", "#047857", "#7e22ce"]
    fixed = [f"fixed-{period}" for period in protocol["fixed_periods"]]
    maximum = max(lookup["test", profile, method, primary] for profile in protocol["profiles"] for method in fixed)
    parts = []
    for index, (profile, chosen) in enumerate(selection.items()):
        color = colors[index]
        values = [lookup["test", profile, method, primary] for method in fixed]
        xs = [75 + 12 * period for period in protocol["fixed_periods"]]
        ys = [300 - 250 * value / maximum for value in values]
        for x_first, y_first, x_next, y_next in zip(xs, ys, xs[1:], ys[1:]):
            parts.append(line(x_first, y_first, x_next, y_next, color))
        parts.extend(f'<circle cx="{xx}" cy="{yy}" r="4" fill="{color}"/>' for xx, yy in zip(xs, ys))
        parts.append(text(40 + index*270, 355, f"{profile}: validation picked {chosen['method']}", color=color))
    parts += [text(65 + 12*period, 322, str(period)) for period in protocol["fixed_periods"]]
    parts += [text(360, 343, "Fixed duration (s)")]
    for tick in range(6):
        yy = 300-50*tick
        parts += [line(95, yy, 795, yy, "#e2e8f0"), text(25, yy+4, f"{maximum*tick/5:.0f}")]
    svg(figures / "fixed_sweep.svg", "Test duration sweep · stopped s / scheduled vehicle", 870, 377, parts)

    for family in ("primary", "budget", "ood"):
        selected = [row for row in contrasts if row["family"] == family]
        magnitude = max(1., max(max(abs(row["crossed_lo"]), abs(row["crossed_hi"])) for row in selected))
        height = 80 + 26*len(selected)
        parts = [line(560, 48, 560, height-24, "#94a3b8"), text(544, height-9, "0")]
        for index, row in enumerate(selected):
            yy = 62+26*index
            left, center, right = [560+240*row[key]/magnitude for key in ("crossed_lo", "delta", "crossed_hi")]
            parts += [text(20, yy+4, f"{row['profile']} · {row['method']} − {row['reference']}"),
                      line(left, yy, right, yy, colors[0]), f'<circle cx="{center}" cy="{yy}" r="4" fill="{colors[0]}"/>',
                      text(820, yy+4, f"{row['delta']:+.2f} [{row['crossed_lo']:+.2f}, {row['crossed_hi']:+.2f}]")]
        svg(figures / f"{family}_contrasts.svg", f"{family.capitalize()} · paired mean differences and crossed 95% intervals", 1040, height, parts)

    for metric in ("entropy", "value_loss", "mean_reward", "explained_variance_pre_update"):
        selected = grouped(training, ("profile", "method", "episode"))
        averages = {key: float(np.mean(values)) for key, rows in selected.items()
                    if (values := [row[metric] for row in rows if row[metric] is not None])}
        if not averages:
            continue
        parts = []
        for panel, profile in enumerate(protocol["profiles"]):
            subset = [value for (scenario, _, _), value in averages.items() if scenario == profile]
            low, high = min(subset), max(subset)
            span = max(high-low, 1e-9)
            origin = 32+310*panel
            parts += [text(origin, 54, f"{profile}: {low:.3g} to {high:.3g}"), line(origin, 244, origin+265, 244, "#94a3b8"),
                      text(origin, 262, "1"), text(origin+250, 262, "60")]
            for method, color in zip(protocol["methods"], colors):
                points = [(origin+265*episode/59, 238-164*(value-low)/span)
                          for (scenario, candidate, episode), value in sorted(averages.items()) if scenario == profile and candidate == method]
                parts.extend(line(*first, *second, color) for first, second in zip(points, points[1:]))
        parts += [text(30+220*index, 292, method, color=color) for index, (method, color) in enumerate(zip(protocol["methods"], colors))]
        svg(figures / f"training_{metric}.svg", f"Training {metric} · mean over 5 initializations · episode number", 960, 310, parts)

    init_rows = [row for row in inits if row["group"] == "test" and row["initialization"] is not None]
    parts = []
    for panel, profile in enumerate(protocol["profiles"]):
        subset = [row for row in init_rows if row["profile"] == profile]
        peak = max(row["mean"] for row in subset)
        origin = 33+310*panel
        parts += [text(origin, 53, f"{profile} · 0 to {peak:.1f}"), line(origin, 250, origin+265, 250, "#94a3b8")]
        for index, method in enumerate(protocol["methods"]):
            values = [row for row in subset if row["method"] == method]
            for offset, row in enumerate(values):
                xx, yy = origin+25+index*65+offset*5, 245-168*row["mean"]/peak
                parts.append(f'<circle cx="{xx}" cy="{yy}" r="3.5" fill="{colors[index]}"><title>seed {row["initialization"]}: {row["mean"]:.2f}</title></circle>')
    parts += [text(35+220*index, 284, method, color=color) for index, (method, color) in enumerate(zip(protocol["methods"], colors))]
    svg(figures / "initializations.svg", "Training-seed sensitivity · each point is a 10-demand mean", 960, 310, parts)

    high = list(protocol["profiles"])[-1]
    methods = list(dict.fromkeys([selection[high]["method"], "max-pressure", *protocol["methods"]]))
    filtered = [row for row in locals_ if row["group"] == "test" and row["profile"] == high and row["method"] in methods]
    local_means = {key: np.mean([row["queue_vehicle_seconds"] for row in group])
                   for key, group in grouped(filtered, ("method", "signal")).items()}
    peak = max(local_means.values())
    signals = sorted({row["signal"] for row in filtered})
    parts = []
    for index, method in enumerate(methods):
        origin = 25+165*index
        parts.append(text(origin, 60, method))
        for position, signal in enumerate(signals):
            value = local_means[method, signal]
            xx, yy = origin+34*(position//4), 75+34*(3-position%4)
            shade = int(245-155*value/max(1, peak))
            parts += [f'<rect x="{xx}" y="{yy}" width="32" height="32" fill="rgb(255,{shade},{shade})"><title>{value:.0f} stopped vehicle seconds</title></rect>',
                      text(xx+4, yy+20, signal, size=11)]
    parts.append(text(25, 239, f"Shared color scale: 0 to {peak:.0f} queue vehicle-seconds, 16 intersections"))
    svg(figures / "high_demand_local.svg", f"{high} · mean local queue integral", max(780, 25+165*len(methods)), 258, parts)

    worst = max((row for row in diagnostics if row["profile"] == high), key=lambda row: results[
        "test", high, row["demand_seed"], row["method"], row["initialization"]][1]["metrics"][primary])
    alternatives = [(worst["method"], worst["initialization"]), (selection[high]["method"], None), ("max-pressure", None)]
    traces = []
    for method, init in alternatives:
        path, result, _ = results["test", high, worst["demand_seed"], method, init]
        traces.append(load(path.parent / Path(result["trace_file"]).name))
    peak = max(row["stopped"] for trace in traces for row in trace)
    parts = [line(60, 285, 830, 285, "#94a3b8"), text(25, 50, f"{peak} stopped"), text(45, 306, "0 s"), text(800, 306, "600 s")]
    for index, (trace, (method, init)) in enumerate(zip(traces, alternatives)):
        points = " ".join(f'{60+770*row["second"]/600:.2f},{285-225*row["stopped"]/max(1, peak):.2f}' for row in trace)
        parts += [f'<polyline points="{points}" stroke="{colors[index]}" fill="none" stroke-width="1.7"/>',
                  text(35+285*index, 330, f"{method} · init {init}", color=colors[index])]
    svg(figures / "failure_trace.svg", f"{high} · worst learned run, demand {worst['demand_seed']} · stopped vehicles", 880, 350, parts)


def report(output, protocol, selection, rows, local_rows, local_deltas, summaries, contrasts, training, diagnostics):
    primary = protocol["primary_metric"]
    lookup = {(row["group"], row["profile"], row["method"], row["metric"]): row for row in summaries}
    method_order = baseline_names(protocol) + protocol["methods"]
    counts = {key[0]: len(group) for key, group in grouped(rows, ("group",)).items()}
    paragraphs = ["# 4×4 路网交通信号控制：实验结果", "",
        "**任务**：16 个路口每 5 秒决定保持还是请求切换绿灯。比较学习策略能否在相同需求和切相约束下，"
        "降低道路内停车秒数/计划车辆数。主对照是独立验证集选出的固定时制和等权车辆数 Max Pressure。", "",
        "## 哪些差异通过了显著性检验？", "",
        "下表列出全部 Holm 校正后 p < 0.05 的主比较。差值为左减右，单位秒/车，越低越好。"
        "p 值条件于实测训练种子，区间则同时重采样需求与训练初始化。", "",
        "| 工况 | 对比 | 差值 | 交叉 bootstrap 95% 区间 | Holm p |",
        "|---|---|---:|---:|---:|"]
    for row in contrasts:
        if row["family"] == "primary" and row["holm_p"] < 0.05:
            paragraphs.append(f"| {row['profile']} | {row['method']} − {row['reference']} | {row['delta']:+.2f} | "
                              f"[{row['crossed_lo']:+.2f}, {row['crossed_hi']:+.2f}] | {row['holm_p']:.4f} |")
    paragraphs += ["", "**能得出的结论**：局部奖励 MAPPO 的道路内停车指标低于本项目的 Max Pressure 实现，"
        "但相对验证集从六档选出的固定时制没有显著改进。低、中需求下，IPPO 和 MAPPO 的逐秒轨迹与固定时制一致"
        "（每档、每种方法各 50/50 次）；不能把相对 Max Pressure 的收益归因于学习出的交通协同。"
        "集中 critic 和 actor 交通观测也没有显示出显著增益。网络奖励提高了等待，但初始化敏感，C1 交叉区间仍包含零。", "",
        "![主比较差值与区间](figures/primary_contrasts.svg)", "",
        "## 实验范围与完整性", "",
        f"完整性校验通过：{len(training)} 个训练 episode，{len(rows)} 个评估 episode，{len(local_rows)} 条局部指标，"
        f"{len(diagnostics)} 个主测试评估通过逐动作 checkpoint 贪婪回放。评估分组：{counts}；后端：{rows[0]['backend']}。", "",
        "这是受前期预试启发的探索性实验，主比较和独立需求在汇总测试结果前冻结。训练/验证/测试/分布外使用互斥的需求种子；"
        "训练轮次 24 与 60 同轨迹存档，不按测试表现挑选权重。固定周期只在独立验证集选择。", "",
        "## 1. 主测试：固定周期扫参和三项消融", "",
        "主指标为 600 秒观察窗中道路内停车车辆秒数 ÷ 计划车辆数，越小越好；不含入网等待。"
        "每格为按初始化平均后、10 个需求种子的均值 ± 样本标准差。带 * 的固定周期由验证集选出，测试扫参仅作描述。", "",
        "| 方法 | C1 | C2 | C3 |", "|---|---:|---:|---:|"]
    for method in method_order:
        values = [lookup["test", profile, method, primary] for profile in protocol["profiles"]]
        cells = [f"{row['mean']:.2f} ± {row['demand_sd']:.2f}" + (" *" if selection[row["profile"]]["method"] == method else "") for row in values]
        paragraphs.append(f"| {method} | {' | '.join(cells)} |")
    for family, title in (("primary", "2. 主比较：成对差值"), ("budget", "3. 训练预算：60 轮减 24 轮"), ("ood", "4. 分布外压力：冻结策略")):
        paragraphs += ["", "## " + title, "",
            "负差值表示左侧方法等待更少。固定初始化条件下的 bootstrap 仅重采样需求；交叉 bootstrap 独立重采样初始化与需求，"
            "同一对比内保持配对。两个区间均为百分位 95% 区间，验证时制选择视为固定。少量初始化或需求下区间不稳定。", "",
            "| 工况 | 比较 | 差值 | 条件区间 | 交叉区间 | 需求数 | Holm p |",
            "|---|---|---:|---:|---:|---:|---:|"]
        for row in contrasts:
            if row["family"] == family:
                p_value = f"{row['holm_p']:.4f}" if row["holm_p"] is not None else "—"
                paragraphs.append(f"| {row['profile']} | {row['method']} − {row['reference']} | {row['delta']:+.2f} | "
                    f"[{row['demand_lo']:+.2f}, {row['demand_hi']:+.2f}] | [{row['crossed_lo']:+.2f}, {row['crossed_hi']:+.2f}] | "
                    f"{row['n_demand']} | {p_value} |")
        if family == "primary":
            paragraphs += ["", "Holm 校正只应用于 15 个主比较。精确符号翻转检验在固定初始化集合上对需求配对差做检验，"
                           "依赖零假设下差值的符号可交换性，不是跨训练种子显著性检验。"]
        if family == "budget":
            paragraphs += ["", "预算切片只用每档前五个测试需求；不能据此断言收敛。"]
        if family == "ood":
            paragraphs += ["", "脉冲：同规模车辆更集中地出发；热点：按经过 C2 的边加权重采样完整路线；持续：延长注入和观察窗。"
                           "三者不是同一强度轴，各工况与本工况的固定时制和 Max Pressure 比较。", "",
                           "| 工况 | 方法 | 主指标 | 完成率 | 未入网数 | 切相次数 |", "|---|---|---:|---:|---:|---:|"]
            for profile, method in itertools.product(("burst", "hotspot", "sustained"),
                                                     (selection["C2"]["method"], "max-pressure", "ippo", "mappo")):
                values = [lookup["ood", profile, method, metric]["mean"] for metric in
                          (primary, "completion_fraction", "not_inserted_vehicles", "phase_switches")]
                paragraphs.append(f"| {profile} | {method} | {values[0]:.2f} | {values[1]:.1%} | {values[2]:.1f} | {values[3]:.1f} |")
    paragraphs += ["", "## 5. 机制审计：是反应式控制还是快速轮转？", "",
        "| 工况 | 方法 | 与固定 10 秒完全一致（逐秒轨迹和全部流量指标） | 实际切相均值 | 清零交通特征动作翻转率 | 跨路口置换动作翻转率 | 平均策略熵 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for (profile, method), group in sorted(grouped(diagnostics, ("profile", "method")).items()):
        averages = [np.mean([row[metric] for row in group]) for metric in ("actual_switches", "zero_traffic_flip", "shuffled_traffic_flip", "entropy")]
        matches = sum(row["exact_fixed10_joint"] for row in group)
        paragraphs.append(f"| {profile} | {method} | {matches}/{len(group)} | {averages[0]:.1f} | {averages[1]:.2%} | {averages[2]:.2%} | {averages[3]:.3f} |")
    paragraphs += ["", "动作翻转是封闭回路轨迹上的离线反事实，不是部署后的闭环收益；清零/跨路口置换可能产生训练外输入。"
        "`mappo-phase` 只屏蔽 actor 流量字段，critic 仍看完整全局观测。匹配数不代表未见场景上的全局策略等价。", "",
        "各学习方法的跨初始化样本标准差（每个初始化先对 10 个需求求均值）：", "",
        "| 方法 | C1 | C2 | C3 |", "|---|---:|---:|---:|"]
    for method in protocol["methods"]:
        cells = [f"{lookup['test', profile, method, primary]['initialization_sd']:.2f}" for profile in protocol["profiles"]]
        paragraphs.append(f"| {method} | {' | '.join(cells)} |")
    paragraphs += ["",
        "## 6. 容量、切相与局部代价", "",
        "| 工况 | 方法 | 完成率 | 未入网车辆 | 未完成车辆 | 完成样本等待（秒） | 停车加待插入（秒/计划车辆） | 最坏路口队列积分 | 切相次数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for profile in protocol["profiles"]:
        for method in list(dict.fromkeys([selection[profile]["method"], "max-pressure", "hold", *protocol["methods"]])):
            values = [lookup["test", profile, method, metric]["mean"] for metric in ("completion_fraction", "not_inserted_vehicles",
                "unfinished_vehicles", "completed_mean_waiting_time_seconds", "stopped_plus_pending_seconds_per_scheduled",
                "worst_intersection_queue_integral", "phase_switches")]
            cells = [f"{values[0]:.1%}"] + [f"{value:.1f}" if value is not None else "—" for value in values[1:]]
            paragraphs.append(f"| {profile} | {method} | {' | '.join(cells)} |")
    worst = max((row for row in rows if row["group"] == "test" and row["initialization"] is not None), key=lambda row: row[primary])
    conservation = sum(row["departed_minus_completions_and_final"] != 0 for row in rows)
    paragraphs += ["", "C3 中相对验证选时制的平均局部队列代价（探索性，选最大/最小值不做显著性判断）：", "",
                   "| 方法 | 平均变差路口数 | 增量最大路口 | 增量最小路口 |", "|---|---:|---|---|"]
    for method in protocol["methods"]:
        selected = [row for row in local_deltas if row["method"] == method and row["profile"] == "C3"
                    and row["reference"] == selection["C3"]["method"]]
        largest, smallest = max(selected, key=lambda row: row["mean_queue_delta"]), min(selected, key=lambda row: row["mean_queue_delta"])
        paragraphs.append(f"| {method} | {sum(row['mean_queue_delta'] > 0 for row in selected)}/16 | "
                          f"{largest['signal']} / {largest['mean_queue_delta']:+.0f} 车秒 | "
                          f"{smallest['signal']} / {smallest['mean_queue_delta']:+.0f} 车秒 |")
    paragraphs += ["", f"主测试最大等待的学习策略复现点：{worst['profile']} / {worst['method']} / 初始化 {worst['initialization']} / "
        f"需求 {worst['demand_seed']}；主指标 {worst[primary]:.2f}，完成 {worst['completed_vehicles']}/{worst['scheduled_vehicles']}，"
        f"全网切相 {worst['phase_switches']} 次。完成样本等待可能低估积压策略的代价。"
        "‘停车加待插入’把观察窗内两个逐秒车数相加再除以计划车辆数，仅作敏感性描述，不替代主指标或完整旅行时间。",
        f"所有评估中有 {conservation}/{len(rows)} 个 episode 的‘入网 − 完成 − 末秒在网车辆’非零。"
        "该差额不是 teleport 事件数；SUMO-RL 1.4.5 默认 `time_to_teleport=-1`（关闭堵塞超时传送）。"
        "未记录全部 teleport 生命周期，即使差额为零也不能排除其他原因导致的传送。", "",
        "## 7. 数据与复现", "",
        "- `tables/episodes.csv`：完整评估矩阵；`intersections.csv`：队列积分、切相；`summary.csv`：各指标与跨需求标准差；"
        "`initializations.csv`：训练种子级均值；`contrasts.csv`：差值和区间；`local_contrasts.csv`：局部配对描述；"
        "`actions.csv`：机制审计；`training.csv`：PPO 诊断。",
        "- `figures/`：固定时制、置信区间、训练诊断、种子分布、最坏场景轨迹、C3 路口热图。",
        "- 原始评估、逐秒与动作轨迹、SUMO tripinfo 和 checkpoint 由实验命令重建，不收入 Git。",
        "- `inputs.json` 记录原始路网与需求哈希；`selection.json` 记录验证选时制。", "",
        "仅一个合成网格、两个许可绿灯相位；未测试事故、通信受限、真实车流、多路网或真实信号安全约束。"
        "共享参数 PPO 与集中 critic 实现是受 IPPO/MAPPO 启发的简化版本，不是论文完整实现。"
        "结论限于冻结预算和测试工况；未做等效性检验，不能把区间覆盖零解释为等效。", ""]
    (output / "REPORT.md").write_text("\n".join(paragraphs))


def main():
    parser = argparse.ArgumentParser(description="Audit raw SUMO results and rebuild the experiment report")
    parser.add_argument("--output", type=Path, default=ROOT / "results/experiment")
    parser.add_argument("--report-dir", type=Path, help="Write analysis products separately from the input evidence")
    parser.add_argument("--recorded-network", type=Path, default=NETWORK)
    args = parser.parse_args()
    output = args.output.resolve()
    report_dir = args.report_dir.resolve() if args.report_dir else output
    protocol = load(output / "protocol.json")
    selection = load(output / "selection.json")
    tasks = [task for stage in ("baselines", "evaluate", "ood") for task in evaluation_tasks(output, protocol, stage)]
    rows, locals_, results = checked_matrix(output, protocol, tasks, args.recorded_network)
    for profile, row in selection.items():
        scores = {method: float(np.mean([result[1]["metrics"][protocol["primary_metric"]]
                 for (group, scenario, _, candidate, _), result in results.items()
                 if (group, scenario, candidate) == ("validation", profile, method)])) for method in baseline_names(protocol)[:-2]}
        require(scores == row["validation_scores"] and row["method"] == min(scores, key=scores.get), f"Bad validation selection: {profile}")
    training = read_training(output, protocol)
    summaries, inits, contrasts = summarize(rows, protocol, selection)
    diagnostics = action_diagnostics(results, protocol["primary_metric"])
    local_deltas = local_contrasts(locals_, protocol, selection)
    tables = report_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name, records in dict(episodes=rows, intersections=locals_, training=training, summary=summaries,
                              initializations=inits, contrasts=contrasts, local_contrasts=local_deltas, actions=diagnostics).items():
        write_csv(tables / f"{name}.csv", records)
    make_figures(report_dir, protocol, selection, summaries, inits, contrasts, training, locals_, diagnostics, results)
    for figure in (report_dir / "figures").glob("*.svg"):
        ET.parse(figure)
    report(report_dir, protocol, selection, rows, locals_, local_deltas, summaries, contrasts, training, diagnostics)
    with zipfile.ZipFile(report_dir / "analysis-source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in ("src", "scripts", "configs", "tests", "docs"):
            for path in sorted((ROOT / directory).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(ROOT))
    atomic_json(report_dir / "analysis_manifest.json", {str(path.relative_to(report_dir)): sha(path) for path in
                sorted([report_dir / "REPORT.md", report_dir / "analysis-source.zip", *tables.glob("*.csv"), *(report_dir / "figures").glob("*.svg")])})
    print(f"Validated {len(rows)} evaluations, {len(training)} training episodes, and {len(diagnostics)} greedy action traces: {report_dir}")


if __name__ == "__main__":
    main()
