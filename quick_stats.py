#!/usr/bin/env python3
"""
Quick statistics from termination log using grep-based approach.
"""

import subprocess
import re

log_file = "/home/easyai/termination_logs/terminations_20251217_212737.json"

print("=" * 80)
print("Quick Statistics - Termination Log Analysis")
print("=" * 80)
print(f"File: {log_file}\n")

# Get file size
result = subprocess.run(['du', '-h', log_file], capture_output=True, text=True)
print(f"📁 File size: {result.stdout.split()[0]}")

# Count total records
result = subprocess.run(['grep', '-c', '"step":', log_file], capture_output=True, text=True)
total_records = int(result.stdout.strip())
print(f"📊 Total reset events: {total_records:,}")

# Get first and last step
result = subprocess.run(['grep', '-m', '1', '"step":', log_file], capture_output=True, text=True)
first_step = re.search(r'"step": (\d+)', result.stdout)
first_step = int(first_step.group(1)) if first_step else 0

result = subprocess.run(['bash', '-c', f'tac {log_file} | grep -m 1 \'"step":\''], capture_output=True, text=True)
last_step = re.search(r'"step": (\d+)', result.stdout)
last_step = int(last_step.group(1)) if last_step else 0

print(f"📈 Step range: {first_step:,} to {last_step:,}")

# Get first and last progress
result = subprocess.run(['grep', '-m', '1', '"training_progress":', log_file], capture_output=True, text=True)
first_prog = re.search(r'"training_progress": ([\d.]+)', result.stdout)
first_prog = float(first_prog.group(1)) if first_prog else 0

result = subprocess.run(['bash', '-c', f'tac {log_file} | grep -m 1 \'"training_progress":\''], capture_output=True, text=True)
last_prog = re.search(r'"training_progress": ([\d.]+)', result.stdout)
last_prog = float(last_prog.group(1)) if last_prog else 0

print(f"📊 Training progress: {first_prog:.2f}% to {last_prog:.2f}%")

# Count stages
print("\n🎯 Training Stage Distribution:")
for stage in ['early', 'mid', 'late', 'final']:
    result = subprocess.run(['grep', '-c', f'"training_stage": "{stage}"', log_file], capture_output=True, text=True)
    try:
        count = int(result.stdout.strip())
        pct = (count / total_records * 100) if total_records > 0 else 0
        print(f"  {stage.upper():8s}: {count:6,} records ({pct:5.1f}%)")
    except:
        print(f"  {stage.upper():8s}:      0 records (  0.0%)")

# Count termination reasons
print("\n⚠️  Termination Reasons:")
for reason in ['base_contact', 'upside_down', 'time_out']:
    result = subprocess.run(['grep', '-c', f'"{reason}":', log_file], capture_output=True, text=True)
    try:
        count = int(result.stdout.strip())
        pct = (count / total_records * 100) if total_records > 0 else 0
        print(f"  {reason:20s}: {count:6,} occurrences ({pct:5.1f}%)")
    except:
        print(f"  {reason:20s}:      0 occurrences (  0.0%)")

# Sample a few records to get num_resets
print("\n📋 Sampling records for detailed stats...")
result = subprocess.run(['bash', '-c', f'head -10000 {log_file} | grep \'"num_resets":\''], capture_output=True, text=True)
num_resets_match = re.search(r'"num_resets": (\d+)', result.stdout)
if num_resets_match:
    num_resets = int(num_resets_match.group(1))
    print(f"  Typical num_resets per event: {num_resets:,}")
    print(f"  Estimated total resets: {total_records * num_resets:,}")

# Check if all envs reset every time
print("\n🔍 First Record Sample:")
result = subprocess.run(['bash', '-c', f'head -100 {log_file}'], capture_output=True, text=True)
lines = result.stdout.split('\n')
for line in lines[:20]:
    if any(key in line for key in ['"step":', '"progress":', '"stage":', '"num_resets":']):
        print(f"  {line.strip()}")

print("\n" + "=" * 80)
print("✅ Analysis complete!")
print("\n💡 Key Findings:")
print(f"   - Training ran for {last_step:,} steps (progress: {last_prog:.1f}%)")
print(f"   - Logged {total_records:,} reset events")
if num_resets_match and num_resets == 4096:
    print(f"   - ALL {num_resets:,} environments reset at EVERY step!")
    print(f"   - ⚠️  This indicates a CRITICAL PROBLEM with training")
print("=" * 80)

