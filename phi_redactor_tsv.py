#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import subprocess
import tempfile
import shutil
import hashlib
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime


# ============================================================
# Utilities
# ============================================================

def file_hash(path):
    """
    Hash file content after normalizing line endings.
    Treats CRLF and LF as identical.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        data = f.read()

    # Normalize CRLF and CR → LF
    data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")

    h.update(data)
    return h.hexdigest()


def init_logger(output_dir):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        os.path.dirname(output_dir),
        f"redaction_log_{ts}.log"
    )

    def log(msg):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"Log file created at {log_path}")
    return log


# ============================================================
# Core processing logic
# ============================================================

def process_files(
    input_dir,
    output_dir,
    extension="tsv",
    recursive=False,
):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    log = init_logger(output_dir)

    log(f"Input directory : {input_dir}")
    log(f"Output directory: {output_dir}")
    log(f"Extension       : {extension}")
    log(f"Recursive       : {recursive}")

    pattern = f"**/*.{extension}" if recursive else f"*.{extension}"
    files = glob.glob(os.path.join(input_dir, pattern), recursive=recursive)

    if not files:
        log("No files found. Exiting.")
        return

    processed = 0
    skipped_uptodate = 0
    skipped_nochange = 0

    for idx, in_file in enumerate(files, 1):
        rel_path = os.path.relpath(in_file, input_dir)
        out_file = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        # Skip if output exists and is newer
        if os.path.exists(out_file):
            if os.path.getmtime(out_file) >= os.path.getmtime(in_file):
                log(f"SKIP_UPTODATE ({idx}/{len(files)}) {rel_path}")
                skipped_uptodate += 1
                continue

        log(f"PROCESS ({idx}/{len(files)}) {rel_path}")

        with tempfile.NamedTemporaryFile(
            suffix="." + extension,
            delete=False
        ) as tmp:
            tmp_path = tmp.name

        cmd = [
            "python",
            "phi_redactor.py",
            "predict",
            "--checkpoint", "./output/best",
            "--input_tsv", in_file,
            "--output_tsv", tmp_path,
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            log(f"ERROR processing {rel_path}: {e}")
            os.unlink(tmp_path)
            raise

        # Compare input and output
        if file_hash(in_file) == file_hash(tmp_path):
            log(f"NO_CHANGE {rel_path} (output not created)")
            os.unlink(tmp_path)
            skipped_nochange += 1
            continue

        shutil.move(tmp_path, out_file)
        processed += 1
        log(f"UPDATED {rel_path}")

    log("DONE")
    log(f"Processed      : {processed}")
    log(f"Skipped uptodate: {skipped_uptodate}")
    log(f"Skipped nochange: {skipped_nochange}")


# ============================================================
# GUI
# ============================================================

def launch_gui():
    root = tk.Tk()
    root.title("PHI Redaction Runner")
    root.geometry("520x260")
    root.resizable(False, False)

    input_dir = tk.StringVar()
    output_dir = tk.StringVar()
    extension = tk.StringVar(value="tsv")
    recursive = tk.BooleanVar(value=False)

    def browse_input():
        path = filedialog.askdirectory(title="Select Input Folder")
        if path:
            input_dir.set(path)

    def browse_output():
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            output_dir.set(path)

    def run():
        if not input_dir.get():
            messagebox.showerror("Error", "Please select an input folder.")
            return
        if not output_dir.get():
            messagebox.showerror("Error", "Please select an output folder.")
            return

        root.destroy()
        process_files(
            input_dir=input_dir.get(),
            output_dir=output_dir.get(),
            extension=extension.get().strip().lstrip("."),
            recursive=recursive.get(),
        )

    tk.Label(root, text="Input Folder").grid(row=0, column=0, padx=10, pady=10, sticky="w")
    tk.Entry(root, textvariable=input_dir, width=45).grid(row=0, column=1)
    tk.Button(root, text="Browse", command=browse_input).grid(row=0, column=2, padx=5)

    tk.Label(root, text="Output Folder").grid(row=1, column=0, padx=10, pady=10, sticky="w")
    tk.Entry(root, textvariable=output_dir, width=45).grid(row=1, column=1)
    tk.Button(root, text="Browse", command=browse_output).grid(row=1, column=2, padx=5)

    tk.Label(root, text="Extension").grid(row=2, column=0, padx=10, pady=10, sticky="w")
    tk.Entry(root, textvariable=extension, width=10).grid(row=2, column=1, sticky="w")

    tk.Checkbutton(
        root,
        text="Recursive search",
        variable=recursive
    ).grid(row=3, column=1, sticky="w", pady=10)

    tk.Button(
        root,
        text="Run Redaction",
        command=run,
        width=20
    ).grid(row=4, column=1, pady=20)

    root.mainloop()


# ============================================================
# CLI
# ============================================================

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--ext", default="tsv")
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


# ============================================================
# Entry point
# ============================================================

def main():
    if len(sys.argv) > 1:
        args = parse_cli()
        if not args.input or not args.output:
            print("[ERROR] --input and --output required in CLI mode")
            sys.exit(1)

        process_files(
            input_dir=args.input,
            output_dir=args.output,
            extension=args.ext,
            recursive=args.recursive,
        )
    else:
        launch_gui()


if __name__ == "__main__":
    main()
