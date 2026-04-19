#!/usr/bin/env python3
"""Live stream and filter NPC conversation logs from conversation_logs_pretty.csv."""
import argparse
import csv
import os
import time


def normalize(value):
    return (value or "").strip().lower()


def row_matches(row, args):
    if args.source and normalize(row.get("source")) != normalize(args.source):
        return False
    if args.speaker and normalize(row.get("speaker")) != normalize(args.speaker):
        return False
    if args.target and normalize(row.get("target")) != normalize(args.target):
        return False
    if args.tone and normalize(row.get("tone")) != normalize(args.tone):
        return False
    if args.contains:
        text = normalize(row.get("text"))
        if normalize(args.contains) not in text:
            return False
    return True


def format_row(row):
    ts = row.get("timestamp", "")
    source = row.get("source", "")
    speaker = row.get("speaker", "")
    target = row.get("target", "")
    tone = row.get("tone", "")
    text = row.get("text", "")
    arrow = f"{speaker} -> {target}" if target else speaker
    return f"[{ts}] {source:10} {tone:10} {arrow:30} | {text}"


def read_existing_rows(file_path, args):
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            if row_matches(row, args):
                print(format_row(row))
    return fieldnames


def follow_rows(file_path, fieldnames, args):
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        # Seek to end for true live-follow mode
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.25)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                row = next(csv.DictReader([line], fieldnames=fieldnames))
            except Exception:
                continue
            if row_matches(row, args):
                print(format_row(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Stream and filter conversation logs.")
    parser.add_argument(
        "--file",
        default="conversation_logs_pretty.csv",
        help="CSV file to read (default: conversation_logs_pretty.csv)",
    )
    parser.add_argument("--source", help="Filter by source, e.g. npc_social or local")
    parser.add_argument("--speaker", help="Filter by speaker NPC name")
    parser.add_argument("--target", help="Filter by target name")
    parser.add_argument("--tone", help="Filter by tone, e.g. deep, humor, crime")
    parser.add_argument("--contains", help="Filter rows where text contains this phrase")
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Follow file for new rows in real time",
    )
    args = parser.parse_args()

    file_path = os.path.abspath(args.file)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    fieldnames = read_existing_rows(file_path, args)
    if not fieldnames:
        print("No header found in CSV.")
        return

    if args.follow:
        follow_rows(file_path, fieldnames, args)


if __name__ == "__main__":
    main()
