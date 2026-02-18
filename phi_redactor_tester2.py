#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox


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

    print("[INFO] Input directory :", input_dir)
    print("[INFO] Output directory:", output_dir)
    print("[INFO] Extension       :", extension)
    print("[INFO] Recursive       :", recursive)

    if not os.path.isdir(input_dir):
        print("[ERROR] Input directory does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)

    pattern = f"**/*.{extension}" if recursive else f"*.{extension}"
    search_path = os.path.join(input_dir, pattern)

    files = glob.glob(search_path, recursive=recursive)

    if not files:
        print("[WARN] No files found.")
        return

    print(f"[INFO] Found {len(files)} file(s).")

    processed = 0
    skipped = 0

    for idx, in_file in enumerate(files, 1):
        rel_path = os.path.relpath(in_file, input_dir)
        out_file = os.path.join(output_dir, rel_path)

        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        # Skip if output exists and is newer
        if os.path.exists(out_file):
            if os.path.getmtime(out_file) >= os.path.getmtime(in_file):
                print(f"[SKIP] ({idx}/{len(files)}) {rel_path} (up-to-date)")
                skipped += 1
                continue

        print(f"[PROC] ({idx}/{len(files)}) {rel_path}")

        cmd = [
            "python",
            "phi_redactor.py",
            "predict",
            "--checkpoint", "./output/best",
            "--input_tsv", in_file,
            "--output_tsv", out_file,
        ]

        try:
            subprocess.run(cmd, check=True)
            processed += 1
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed processing {rel_path}")
            print(e)
            raise

    print("[DONE]")
    print(f"  Processed: {processed}")
    print(f"  Skipped  : {skipped}")


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

    # ---- Layout ----

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
    parser = argparse.ArgumentParser(
        description="Run PHI redaction on files with optional GUI fallback."
    )

    parser.add_argument("--input", help="Input directory")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--ext", default="tsv", help="File extension (default: tsv)")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Enable recursive search",
    )

    return parser.parse_args()


# ============================================================
# Entry point
# ============================================================

def main():
    # If ANY CLI args are provided → CLI mode
    if len(sys.argv) > 1:
        args = parse_cli()

        if not args.input or not args.output:
            print("[ERROR] --input and --output are required in CLI mode.")
            sys.exit(1)

        process_files(
            input_dir=args.input,
            output_dir=args.output,
            extension=args.ext,
            recursive=args.recursive,
        )
    else:
        # No args → GUI
        launch_gui()


if __name__ == "__main__":
    main()
