import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MODES = {
    "centralized": "Centralized",
    "splitfed": "SplitFed",
    "ring": "Ring Federated",
}


def project_root():
    return Path(__file__).resolve().parents[1]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")


def latest_run_id(root):
    logs_dir = root / "logs"
    if not logs_dir.exists():
        return None
    runs = [p.name for p in logs_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return max(runs) if runs else None


def mode_from_summary(run_dir):
    summary = run_dir / "monitor-summary.json"
    if not summary.exists():
        return None
    try:
        return load_json(summary).get("runtime", {}).get("mode")
    except (OSError, json.JSONDecodeError):
        return None


def metrics_info(metrics_path):
    metrics = load_json(metrics_path)
    rounds = metrics.get("rounds") or []
    test_acc = metrics.get("test_acc") or []
    test_loss = metrics.get("test_loss") or []
    return {
        "rounds": rounds,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "round_count": len(rounds),
    }


def find_latest_full_runs(root, required_rounds):
    selected = {}
    for metrics_path in sorted((root / "logs").glob("*/metrics.json"), reverse=True):
        run_dir = metrics_path.parent
        mode = mode_from_summary(run_dir)
        if mode not in MODES or mode in selected:
            continue
        info = metrics_info(metrics_path)
        if info["round_count"] >= required_rounds:
            selected[mode] = {
                "run_id": run_dir.name,
                "path": metrics_path,
                **info,
            }
    return selected


def make_config(root, mode, output_dir, rounds):
    if mode == "ring":
        cfg = load_json(root / "config" / "ring.json")
    else:
        cfg = load_json(root / "config" / "centralized.json")
        cfg["experiment"]["mode"] = mode

    cfg["experiment"]["global_epochs"] = rounds
    cfg["experiment"]["target_accuracy"] = 1.1
    cfg.setdefault("monitoring", {})["render_mode"] = "plain"

    if mode in {"centralized", "splitfed"}:
        cfg.setdefault("network", {})["server_timeout"] = max(float(cfg.get("network", {}).get("server_timeout", 15.0)), 300.0)

    path = output_dir / f"{mode}_10_rounds.json"
    write_json(path, cfg)
    return path


def run_command(cmd, cwd, env=None):
    print("Running:", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def run_training(root, modes, rounds):
    run_root = root / "runs" / "full_10_rounds" / time.strftime("%Y%m%d%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    produced = {}
    for mode in modes:
        config_path = make_config(root, mode, run_root, rounds)
        run_command([sys.executable, "cleanup.py", str(config_path)], root)
        before = latest_run_id(root)
        env = os.environ.copy()
        env.setdefault("FED_MONITOR_READY_TIMEOUT", "60")
        run_command([sys.executable, "manager.py", "--config", str(config_path)], root, env=env)
        after = latest_run_id(root)
        if after == before or after is None:
            raise RuntimeError(f"No new log run was created for mode={mode}")
        produced[mode] = after

    return produced


def plot_comparison(root, selected, rounds):
    import matplotlib.pyplot as plt

    output_dir = root / "Image"
    output_dir.mkdir(exist_ok=True)

    ordered_modes = ["centralized", "ring", "splitfed"]
    colors = {
        "centralized": "#1f77b4",
        "ring": "#2ca02c",
        "splitfed": "#d62728",
    }

    def draw(metric_key, ylabel, title, output_name):
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
        ax.set_facecolor("white")
        for mode in ordered_modes:
            item = selected[mode]
            x = item["rounds"][:rounds]
            y = item[metric_key][:rounds]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.2,
                markersize=5,
                label=f"{MODES[mode]} ({item['run_id']})",
                color=colors[mode],
            )
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, rounds + 1))
        if metric_key == "test_acc":
            ax.set_ylim(0.95, 1.0)
        else:
            ax.set_ylim(bottom=0)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
        ax.legend()
        fig.tight_layout()
        output_path = output_dir / output_name
        fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return output_path

    acc_path = draw(
        "test_acc",
        "Test Accuracy",
        "Test Accuracy Comparison (10 Rounds)",
        "test_accuracy_comparison_10rounds.png",
    )
    loss_path = draw(
        "test_loss",
        "Test Loss",
        "Test Loss Comparison (10 Rounds)",
        "test_loss_comparison_10rounds.png",
    )
    return acc_path, loss_path


def plot_communication(root, selected):
    import matplotlib.pyplot as plt

    output_dir = root / "Image"
    output_dir.mkdir(exist_ok=True)

    ordered_modes = ["centralized", "ring", "splitfed"]
    colors = {
        "centralized": "#1f77b4",
        "ring": "#2ca02c",
        "splitfed": "#d62728",
    }

    totals_mib = []
    messages = []
    labels = []
    for mode in ordered_modes:
        run_id = selected[mode]["run_id"]
        summary = load_json(root / "logs" / run_id / "monitor-summary.json")
        network = summary.get("network", {})
        totals_mib.append(float(network.get("bytes_sent", 0)) / (1024 * 1024))
        messages.append(int(network.get("messages_sent", 0)))
        labels.append(MODES[mode])

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("white")
    bars = ax.bar(
        labels,
        totals_mib,
        color=[colors[mode] for mode in ordered_modes],
        width=0.58,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Total Bytes Sent (MiB, log scale)")
    ax.set_title("Communication Overhead Comparison (10 Rounds)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.4)

    for bar, value, msg_count in zip(bars, totals_mib, messages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.08,
            f"{value:,.1f} MiB\n{msg_count:,} msgs",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    output_path = output_dir / "communication_overhead_comparison_10rounds.png"
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run and plot 10-round comparison for centralized, ring, and splitfed modes.")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--run", action="store_true", help="Run missing modes before plotting.")
    parser.add_argument("--force-run", action="store_true", help="Run all three modes before plotting.")
    args = parser.parse_args()

    root = project_root()
    modes = ["centralized", "ring", "splitfed"]
    selected = find_latest_full_runs(root, args.rounds)

    missing = [mode for mode in modes if mode not in selected]
    if args.force_run:
        run_training(root, modes, args.rounds)
    elif args.run and missing:
        run_training(root, missing, args.rounds)
    elif missing:
        names = ", ".join(missing)
        raise SystemExit(f"Missing {args.rounds}-round runs for: {names}. Re-run with --run to generate them.")

    selected = find_latest_full_runs(root, args.rounds)
    missing = [mode for mode in modes if mode not in selected]
    if missing:
        raise SystemExit(f"Still missing {args.rounds}-round runs for: {', '.join(missing)}")

    acc_path, loss_path = plot_comparison(root, selected, args.rounds)
    comm_path = plot_communication(root, selected)
    print("Saved:")
    print(acc_path)
    print(loss_path)
    print(comm_path)
    print("Sources:")
    for mode in modes:
        item = selected[mode]
        print(
            f"{MODES[mode]}: logs/{item['run_id']} "
            f"rounds={item['round_count']} final_acc={item['test_acc'][args.rounds - 1]} "
            f"final_loss={item['test_loss'][args.rounds - 1]}"
        )


if __name__ == "__main__":
    main()
