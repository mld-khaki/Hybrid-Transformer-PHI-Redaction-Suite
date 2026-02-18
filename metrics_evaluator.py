#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHI Redaction Evaluation Scanner (GUI)
=====================================

What it does
------------
Given folders of TSV files:
  - org/ : original (unredacted) files
  - red/ : gold redacted files (ground truth redaction)
Optionally:
  - pred/: model output redacted files (to evaluate vs gold)

It will:
  1) Pair files by relative path under org/ (recursive .tsv scan)
  2) Compute a char-level redaction mask via difflib alignment:
        mask[i] = 1 if character i in org was replaced/deleted in red/pred
  3) If pred provided: evaluate pred-mask vs gold-mask:
        - char precision, recall, F1
        - non-PHI retention (TN/(TN+FP))
        - false negatives count (missed gold redaction chars)
        - false positives count (extra redaction chars)
        - file-level counts: #files with any FN, etc.
  4) Write outputs:
        - summary.json
        - per_file.csv
        - per_file.jsonl (one JSON per file)
  5) Show results + progress in a GUI.

Notes
-----
- This assumes org/red/pred share the same *relative* path and filename.
- It is format-agnostic: it does not require knowing PHI contents, only org vs red/pred differences.
- Robust to line-ending differences (CRLF vs LF).
"""

import os
import sys
import csv
import json
import time
import threading
import queue
import traceback
import difflib
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk


# -----------------------------
# Utilities
# -----------------------------

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_text(s: str) -> str:
    # Minimal normalization: unify line endings only
    return s.replace("\r\n", "\n").replace("\r", "\n")


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return normalize_text(f.read())


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rel_from_root(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


def find_tsv_files(root: str) -> List[str]:
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".tsv"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def compute_redaction_mask(orig: str, redacted: str) -> List[int]:
    """
    Returns a list mask of length len(orig): 1 if orig char was replaced/deleted vs redacted.
    (Insertions in redacted don't map to orig chars and are ignored.)
    """
    orig = normalize_text(orig)
    red = normalize_text(redacted)
    sm = difflib.SequenceMatcher(a=orig, b=red, autojunk=False)
    mask = [0] * len(orig)
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            # mark affected orig region
            for i in range(i1, i2):
                if 0 <= i < len(mask):
                    mask[i] = 1
    return mask


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def char_metrics(gold: List[int], pred: List[int]) -> Dict[str, float]:
    """
    gold/pred are 0/1 masks of equal length.
    Returns precision, recall, f1, non_phi_retention, and counts.
    """
    if len(gold) != len(pred):
        raise ValueError("gold and pred mask lengths differ")

    tp = fp = fn = tn = 0
    for g, p in zip(gold, pred):
        if g == 1 and p == 1:
            tp += 1
        elif g == 0 and p == 1:
            fp += 1
        elif g == 1 and p == 0:
            fn += 1
        else:
            tn += 1

    precision = safe_div(tp, (tp + fp))
    recall = safe_div(tp, (tp + fn))
    f1 = safe_div(2 * precision * recall, (precision + recall))
    non_phi_retention = safe_div(tn, (tn + fp))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "non_phi_retention": non_phi_retention,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# -----------------------------
# Pairing & Evaluation
# -----------------------------

@dataclass
class FilePair:
    rel_path: str
    org_path: str
    red_path: str
    pred_path: Optional[str]


def build_pairs(org_dir: str, red_dir: str, pred_dir: Optional[str], log) -> List[FilePair]:
    org_files = find_tsv_files(org_dir)
    if not org_files:
        raise RuntimeError("No .tsv files found under org directory.")

    # Index red/pred by relative path
    red_map: Dict[str, str] = {}
    for p in find_tsv_files(red_dir):
        red_map[rel_from_root(p, red_dir)] = p

    pred_map: Dict[str, str] = {}
    if pred_dir:
        for p in find_tsv_files(pred_dir):
            pred_map[rel_from_root(p, pred_dir)] = p

    pairs: List[FilePair] = []
    missing_red = 0
    missing_pred = 0

    for org_path in org_files:
        rel = rel_from_root(org_path, org_dir)
        red_path = red_map.get(rel)
        if not red_path:
            missing_red += 1
            log(f"[WARN] Missing red file for: {rel}")
            continue

        pred_path = None
        if pred_dir:
            pred_path = pred_map.get(rel)
            if not pred_path:
                missing_pred += 1
                log(f"[WARN] Missing pred file for: {rel}")

        pairs.append(FilePair(rel_path=rel, org_path=org_path, red_path=red_path, pred_path=pred_path))

    log(f"[INFO] Found org files : {len(org_files)}")
    log(f"[INFO] Paired (org+red): {len(pairs)}")
    if missing_red:
        log(f"[INFO] Missing red   : {missing_red}")
    if pred_dir:
        log(f"[INFO] Missing pred  : {missing_pred}")

    if not pairs:
        raise RuntimeError("No paired files found (check org/red folder structure and filenames).")

    return pairs


def evaluate_pairs(pairs: List[FilePair], output_dir: str, log, stop_flag) -> Dict:
    ensure_dir(output_dir)
    per_file_csv = os.path.join(output_dir, "per_file.csv")
    per_file_jsonl = os.path.join(output_dir, "per_file.jsonl")
    summary_json = os.path.join(output_dir, "summary.json")

    has_pred = any(p.pred_path for p in pairs)

    # Aggregates
    agg = {
        "files_total_paired": len(pairs),
        "files_with_pred_present": 0,
        "files_missing_pred": 0,
        "total_chars": 0,
        "gold_redacted_chars": 0,
        "pred_redacted_chars": 0,
        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        "files_with_any_fn": 0,
        "files_with_any_fp": 0,
        "files_with_any_tp": 0,
        "created_at": now_ts(),
    }

    # CSV header
    fieldnames = [
        "rel_path",
        "chars",
        "gold_redacted_chars",
        "pred_redacted_chars",
        "tp", "fp", "fn", "tn",
        "precision", "recall", "f1", "non_phi_retention",
        "has_pred",
        "notes",
    ]

    with open(per_file_csv, "w", newline="", encoding="utf-8") as f_csv, \
         open(per_file_jsonl, "w", encoding="utf-8") as f_jsonl:

        w = csv.DictWriter(f_csv, fieldnames=fieldnames)
        w.writeheader()

        for idx, pair in enumerate(pairs, start=1):
            if stop_flag.is_set():
                log("[INFO] Stop requested. Exiting evaluation loop.")
                break

            notes = []
            has_this_pred = bool(pair.pred_path and os.path.isfile(pair.pred_path))
            if has_this_pred:
                agg["files_with_pred_present"] += 1
            else:
                agg["files_missing_pred"] += 1

            try:
                org_txt = read_text(pair.org_path)
                red_txt = read_text(pair.red_path)

                gold_mask = compute_redaction_mask(org_txt, red_txt)
                chars = len(gold_mask)
                gold_red_chars = sum(gold_mask)

                agg["total_chars"] += chars
                agg["gold_redacted_chars"] += gold_red_chars

                # If no pred folder or missing pred file, we still log dataset stats
                if not has_pred or not has_this_pred:
                    row = {
                        "rel_path": pair.rel_path,
                        "chars": chars,
                        "gold_redacted_chars": gold_red_chars,
                        "pred_redacted_chars": "",
                        "tp": "", "fp": "", "fn": "", "tn": "",
                        "precision": "", "recall": "", "f1": "", "non_phi_retention": "",
                        "has_pred": "yes" if has_this_pred else "no",
                        "notes": "no_pred_to_eval" if has_pred else "pred_not_provided",
                    }
                    w.writerow(row)
                    f_jsonl.write(json.dumps(row, ensure_ascii=True) + "\n")

                else:
                    pred_txt = read_text(pair.pred_path)  # type: ignore[arg-type]
                    pred_mask = compute_redaction_mask(org_txt, pred_txt)
                    if len(pred_mask) != len(gold_mask):
                        # In rare cases, normalization differences can cause mismatch;
                        # we can force equal length by truncation to min length (conservative).
                        m = min(len(pred_mask), len(gold_mask))
                        notes.append(f"mask_len_mismatch_truncated_to_{m}")
                        gold_mask = gold_mask[:m]
                        pred_mask = pred_mask[:m]
                        chars = m
                        gold_red_chars = sum(gold_mask)

                    pred_red_chars = sum(pred_mask)
                    agg["pred_redacted_chars"] += pred_red_chars

                    m = char_metrics(gold_mask, pred_mask)

                    agg["tp"] += int(m["tp"])
                    agg["fp"] += int(m["fp"])
                    agg["fn"] += int(m["fn"])
                    agg["tn"] += int(m["tn"])

                    if m["fn"] > 0:
                        agg["files_with_any_fn"] += 1
                    if m["fp"] > 0:
                        agg["files_with_any_fp"] += 1
                    if m["tp"] > 0:
                        agg["files_with_any_tp"] += 1

                    row = {
                        "rel_path": pair.rel_path,
                        "chars": chars,
                        "gold_redacted_chars": gold_red_chars,
                        "pred_redacted_chars": pred_red_chars,
                        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                        "precision": f"{m['precision']:.6f}",
                        "recall": f"{m['recall']:.6f}",
                        "f1": f"{m['f1']:.6f}",
                        "non_phi_retention": f"{m['non_phi_retention']:.6f}",
                        "has_pred": "yes",
                        "notes": ";".join(notes) if notes else "",
                    }
                    w.writerow(row)
                    f_jsonl.write(json.dumps(row, ensure_ascii=True) + "\n")

                if idx % 10 == 0 or idx == len(pairs):
                    log(f"[INFO] Processed {idx}/{len(pairs)} files...")

            except Exception as e:
                err = f"[ERROR] {pair.rel_path}: {e}"
                log(err)
                log(traceback.format_exc())
                row = {
                    "rel_path": pair.rel_path,
                    "chars": "",
                    "gold_redacted_chars": "",
                    "pred_redacted_chars": "",
                    "tp": "", "fp": "", "fn": "", "tn": "",
                    "precision": "", "recall": "", "f1": "", "non_phi_retention": "",
                    "has_pred": "yes" if has_this_pred else "no",
                    "notes": f"error:{e}",
                }
                w.writerow(row)
                f_jsonl.write(json.dumps(row, ensure_ascii=True) + "\n")

    # Final summary if pred exists (global metrics)
    summary = dict(agg)
    if agg["tp"] + agg["fp"] + agg["fn"] + agg["tn"] > 0:
        precision = safe_div(agg["tp"], (agg["tp"] + agg["fp"]))
        recall = safe_div(agg["tp"], (agg["tp"] + agg["fn"]))
        f1 = safe_div(2 * precision * recall, (precision + recall))
        non_phi_retention = safe_div(agg["tn"], (agg["tn"] + agg["fp"]))
        summary.update({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "non_phi_retention": non_phi_retention,
        })
    else:
        summary.update({
            "precision": None,
            "recall": None,
            "f1": None,
            "non_phi_retention": None,
        })

    # Add a friendly "FN per file" headline number
    summary["files_with_any_fn_rate"] = safe_div(summary["files_with_any_fn"], max(1, summary["files_with_pred_present"]))

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log(f"[INFO] Wrote: {per_file_csv}")
    log(f"[INFO] Wrote: {per_file_jsonl}")
    log(f"[INFO] Wrote: {summary_json}")

    return summary


# -----------------------------
# GUI
# -----------------------------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PHI Redaction Evaluation Scanner")
        self.root.geometry("980x680")

        self.org_dir = tk.StringVar()
        self.red_dir = tk.StringVar()
        self.pred_dir = tk.StringVar()
        self.output_dir = tk.StringVar()

        self.include_pred = tk.BooleanVar(value=True)

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

        self._build_ui()
        self._poll_log_queue()

    def log(self, msg: str):
        line = f"[{now_ts()}] {msg}"
        self.log_queue.put(line)

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # --- Paths frame
        frm = ttk.LabelFrame(self.root, text="Folders")
        frm.pack(fill="x", padx=12, pady=10)

        # org
        ttk.Label(frm, text="ORG folder (original TSVs):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.org_dir, width=80).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(frm, text="Browse", command=self._browse_org).grid(row=0, column=2, **pad)

        # red
        ttk.Label(frm, text="RED folder (gold redacted TSVs):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.red_dir, width=80).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(frm, text="Browse", command=self._browse_red).grid(row=1, column=2, **pad)

        # pred checkbox
        ttk.Checkbutton(frm, text="Evaluate PRED folder (optional)", variable=self.include_pred,
                        command=self._toggle_pred).grid(row=2, column=0, sticky="w", **pad)

        # pred
        ttk.Label(frm, text="PRED folder (model output TSVs):").grid(row=3, column=0, sticky="w", **pad)
        self.pred_entry = ttk.Entry(frm, textvariable=self.pred_dir, width=80)
        self.pred_entry.grid(row=3, column=1, sticky="we", **pad)
        self.pred_btn = ttk.Button(frm, text="Browse", command=self._browse_pred)
        self.pred_btn.grid(row=3, column=2, **pad)

        # output
        ttk.Label(frm, text="Output folder (reports):").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.output_dir, width=80).grid(row=4, column=1, sticky="we", **pad)
        ttk.Button(frm, text="Browse", command=self._browse_out).grid(row=4, column=2, **pad)

        frm.columnconfigure(1, weight=1)

        # --- Controls frame
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x", padx=12)

        self.run_btn = ttk.Button(ctrl, text="Run Scan", command=self._run)
        self.run_btn.pack(side="left", padx=6, pady=8)

        self.stop_btn = ttk.Button(ctrl, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6, pady=8)

        self.progress = ttk.Progressbar(ctrl, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=10, pady=8)

        # --- Results summary
        res = ttk.LabelFrame(self.root, text="Summary")
        res.pack(fill="x", padx=12, pady=8)

        self.summary_text = tk.Text(res, height=8, wrap="word")
        self.summary_text.pack(fill="both", expand=True, padx=10, pady=8)
        self.summary_text.configure(state="disabled")

        # --- Logs
        logf = ttk.LabelFrame(self.root, text="Logs")
        logf.pack(fill="both", expand=True, padx=12, pady=10)

        self.log_box = tk.Text(logf, height=18, wrap="none")
        self.log_box.pack(fill="both", expand=True, padx=10, pady=8)
        self.log_box.configure(state="disabled")

        self._toggle_pred()

    def _toggle_pred(self):
        enabled = self.include_pred.get()
        state = "normal" if enabled else "disabled"
        self.pred_entry.configure(state=state)
        self.pred_btn.configure(state=state)
        if not enabled:
            self.pred_dir.set("")

    def _browse_org(self):
        p = filedialog.askdirectory(title="Select ORG folder")
        if p:
            self.org_dir.set(p)

    def _browse_red(self):
        p = filedialog.askdirectory(title="Select RED folder")
        if p:
            self.red_dir.set(p)

    def _browse_pred(self):
        p = filedialog.askdirectory(title="Select PRED folder")
        if p:
            self.pred_dir.set(p)

    def _browse_out(self):
        p = filedialog.askdirectory(title="Select Output folder")
        if p:
            self.output_dir.set(p)

    def _set_summary(self, text: str):
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, text)
        self.summary_text.configure(state="disabled")

    def _append_log(self, line: str):
        self.log_box.configure(state="normal")
        self.log_box.insert(tk.END, line + "\n")
        self.log_box.see(tk.END)
        self.log_box.configure(state="disabled")

    def _poll_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                self._append_log(line)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_log_queue)

    def _stop(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_flag.set()
            self.log("[INFO] Stop requested...")

    def _run(self):
        org = self.org_dir.get().strip()
        red = self.red_dir.get().strip()
        pred = self.pred_dir.get().strip() if self.include_pred.get() else None
        outd = self.output_dir.get().strip()

        if not org or not os.path.isdir(org):
            messagebox.showerror("Error", "Please select a valid ORG folder.")
            return
        if not red or not os.path.isdir(red):
            messagebox.showerror("Error", "Please select a valid RED folder.")
            return
        if self.include_pred.get():
            if not pred or not os.path.isdir(pred):
                messagebox.showerror("Error", "Please select a valid PRED folder (or uncheck evaluation).")
                return
        if not outd:
            # Default to a timestamped folder next to org
            base = os.path.dirname(os.path.abspath(org))
            outd = os.path.join(base, f"phi_eval_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.output_dir.set(outd)
        ensure_dir(outd)

        # UI state
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.start(10)
        self.stop_flag.clear()
        self._set_summary("")

        def worker():
            try:
                self.log("[INFO] Building file pairs...")
                pairs = build_pairs(org, red, pred, self.log)
                self.log(f"[INFO] Starting evaluation. Output -> {outd}")

                summary = evaluate_pairs(pairs, outd, self.log, self.stop_flag)

                # Pretty summary
                lines = []
                lines.append(f"Reports written to: {outd}")
                lines.append("")
                lines.append(f"Files paired (org+red): {summary.get('files_total_paired')}")
                if pred:
                    lines.append(f"Files with pred present: {summary.get('files_with_pred_present')}")
                    lines.append(f"Files missing pred: {summary.get('files_missing_pred')}")
                lines.append(f"Total chars evaluated: {summary.get('total_chars')}")
                lines.append(f"Gold redacted chars: {summary.get('gold_redacted_chars')}")
                if pred:
                    lines.append(f"Pred redacted chars: {summary.get('pred_redacted_chars')}")
                    lines.append("")
                    lines.append("Global metrics (char-level):")
                    lines.append(f"  Precision         : {summary.get('precision'):.6f}" if summary.get("precision") is not None else "  Precision         : n/a")
                    lines.append(f"  Recall            : {summary.get('recall'):.6f}" if summary.get("recall") is not None else "  Recall            : n/a")
                    lines.append(f"  F1                : {summary.get('f1'):.6f}" if summary.get("f1") is not None else "  F1                : n/a")
                    lines.append(f"  Non-PHI retention : {summary.get('non_phi_retention'):.6f}" if summary.get("non_phi_retention") is not None else "  Non-PHI retention : n/a")
                    lines.append("")
                    lines.append("Error signal (file-level):")
                    lines.append(f"  Files with any FN : {summary.get('files_with_any_fn')} (rate={summary.get('files_with_any_fn_rate'):.4f})")
                    lines.append(f"  Files with any FP : {summary.get('files_with_any_fp')}")
                else:
                    lines.append("")
                    lines.append("Pred folder not provided -> dataset stats only (no precision/recall).")

                self.root.after(0, lambda: self._set_summary("\n".join(lines)))
                self.log("[INFO] Done.")
            except Exception as e:
                self.log(f"[ERROR] {e}")
                self.log(traceback.format_exc())
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, self._finish_run)

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _finish_run(self):
        self.progress.stop()
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")


def main():
    root = tk.Tk()
    # Make ttk look a bit nicer on Windows
    try:
        style = ttk.Style()
        if sys.platform.startswith("win"):
            style.theme_use("vista")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
