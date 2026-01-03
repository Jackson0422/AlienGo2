#!/usr/bin/env python3
"""
读取JSONL格式的终止日志工具
使用方法：python read_jsonl_log.py <日志文件路径>
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def read_jsonl_log(log_file):
    """读取JSONL格式的日志"""
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    return logs


def analyze_log(log_file):
    """分析日志并打印统计信息"""
    print(f"Analyzing: {log_file}")
    print("=" * 80)
    
    total_resets = 0
    term_counts = defaultdict(int)
    stage_resets = defaultdict(int)
    entries = 0
    
    with open(log_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            entries += 1
            total_resets += entry['num_resets']
            stage_resets[entry['stage']] += entry['num_resets']
            
            for term, count in entry.get('termination_stats', {}).items():
                term_counts[term] += count
    
    print(f"\nTotal log entries: {entries}")
    print(f"Total resets: {total_resets}")
    print(f"\nResets by training stage:")
    for stage in ['early', 'mid', 'late', 'final']:
        if stage in stage_resets:
            print(f"  {stage:8s}: {stage_resets[stage]:8d} resets")
    
    print(f"\nTermination reasons (sorted by count):")
    for term, count in sorted(term_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = 100.0 * count / total_resets if total_resets > 0 else 0
        print(f"  {term:20s}: {count:8d} occurrences ({percentage:5.1f}%)")
    
    print("=" * 80)


def tail_log(log_file, n=20):
    """显示最后N条日志"""
    print(f"Last {n} entries from: {log_file}")
    print("=" * 80)
    
    logs = read_jsonl_log(log_file)
    for entry in logs[-n:]:
        print(f"Step {entry['step']:8d} | Iter {entry['iteration']:5d} | "
              f"Progress {entry['progress']:5.1f}% | Stage: {entry['stage']:6s} | "
              f"Resets: {entry['num_resets']:4d}")
        if entry.get('termination_stats'):
            print(f"  Terminations: {entry['termination_stats']}")
    
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_jsonl_log.py <log_file.jsonl> [--tail N]")
        print("\nExample:")
        print("  python read_jsonl_log.py ~/termination_logs/terminations_20251220_183848.jsonl")
        print("  python read_jsonl_log.py ~/termination_logs/terminations_20251220_183848.jsonl --tail 50")
        sys.exit(1)
    
    log_file = Path(sys.argv[1])
    
    if not log_file.exists():
        print(f"Error: File not found: {log_file}")
        sys.exit(1)
    
    if len(sys.argv) > 2 and sys.argv[2] == '--tail':
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        tail_log(log_file, n)
    else:
        analyze_log(log_file)

