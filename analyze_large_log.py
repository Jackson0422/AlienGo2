#!/usr/bin/env python3
"""
Analyze large termination log file using streaming approach.
"""

import json
import sys
from collections import Counter, defaultdict

log_file = "/home/easyai/termination_logs/terminations_20251217_212737.json"

print("=" * 80)
print("Analyzing Termination Log (Streaming Mode)")
print("=" * 80)
print(f"File: {log_file}")
print("\nThis may take a few minutes for large files...\n")

# Statistics
total_records = 0
total_resets = 0
stage_counts = Counter()
reason_counts = Counter()
step_range = [float('inf'), float('-inf')]
progress_range = [float('inf'), float('-inf')]

# Sample records for detailed analysis
first_record = None
last_record = None
sample_records = []

# Read file line by line to avoid loading entire file
print("Reading and parsing log file...")

try:
    with open(log_file, 'r') as f:
        # Skip opening bracket
        f.readline()
        
        current_record = []
        in_record = False
        
        for line_num, line in enumerate(f, 1):
            if line_num % 10000000 == 0:
                print(f"  Processed {line_num:,} lines, {total_records:,} records...")
            
            stripped = line.strip()
            
            # Start of a record
            if stripped == '{':
                in_record = True
                current_record = [line]
                continue
            
            # End of a record
            if stripped.startswith('}'):
                current_record.append(line)
                in_record = False
                
                # Parse this record
                record_str = ''.join(current_record)
                try:
                    record = json.loads(record_str.rstrip(','))
                    
                    total_records += 1
                    total_resets += record.get('num_resets', 0)
                    
                    # Track first and last
                    if first_record is None:
                        first_record = record
                    last_record = record
                    
                    # Sample every 1000th record
                    if total_records % 1000 == 0:
                        sample_records.append(record)
                    
                    # Statistics
                    stage_counts[record.get('training_stage', 'unknown')] += 1
                    step_range[0] = min(step_range[0], record.get('step', 0))
                    step_range[1] = max(step_range[1], record.get('step', 0))
                    progress_range[0] = min(progress_range[0], record.get('training_progress', 0))
                    progress_range[1] = max(progress_range[1], record.get('training_progress', 0))
                    
                    # Count termination reasons
                    for reason in record.get('termination_reasons', {}).keys():
                        reason_counts[reason] += 1
                    
                except json.JSONDecodeError:
                    pass
                
                current_record = []
                continue
            
            # Middle of a record
            if in_record:
                current_record.append(line)
        
except KeyboardInterrupt:
    print("\n\nAnalysis interrupted by user.")
except Exception as e:
    print(f"\n\nError: {e}")

# Print results
print("\n" + "=" * 80)
print("ANALYSIS RESULTS")
print("=" * 80)

print(f"\n📊 Overall Statistics:")
print(f"  Total reset events logged: {total_records:,}")
print(f"  Total environment resets: {total_resets:,}")
print(f"  Average resets per event: {total_resets/total_records:.0f}" if total_records > 0 else "  N/A")

print(f"\n📈 Training Progress:")
print(f"  Step range: {step_range[0]:,} to {step_range[1]:,}")
print(f"  Progress range: {progress_range[0]:.2f}% to {progress_range[1]:.2f}%")

print(f"\n🎯 Training Stage Distribution:")
for stage in ['early', 'mid', 'late', 'final']:
    count = stage_counts[stage]
    pct = (count / total_records * 100) if total_records > 0 else 0
    print(f"  {stage.upper():8s}: {count:6,} records ({pct:5.1f}%)")

print(f"\n⚠️  Termination Reasons (occurrence in records):")
for reason, count in reason_counts.most_common():
    pct = (count / total_records * 100) if total_records > 0 else 0
    print(f"  {reason:20s}: {count:6,} records ({pct:5.1f}%)")

print(f"\n🔍 First Record:")
if first_record:
    print(f"  Step: {first_record.get('step')}")
    print(f"  Progress: {first_record.get('training_progress')}%")
    print(f"  Stage: {first_record.get('training_stage')}")
    print(f"  Resets: {first_record.get('num_resets')}")
    print(f"  Reasons: {list(first_record.get('termination_reasons', {}).keys())}")

print(f"\n🏁 Last Record:")
if last_record:
    print(f"  Step: {last_record.get('step')}")
    print(f"  Progress: {last_record.get('training_progress')}%")
    print(f"  Stage: {last_record.get('training_stage')}")
    print(f"  Resets: {last_record.get('num_resets')}")
    print(f"  Reasons: {list(last_record.get('termination_reasons', {}).keys())}")

# Analyze samples for detailed stats
if sample_records:
    print(f"\n📋 Sample Analysis ({len(sample_records)} samples):")
    
    # Calculate average number of envs per reason
    reason_env_counts = defaultdict(list)
    for record in sample_records:
        for reason, env_ids in record.get('termination_reasons', {}).items():
            reason_env_counts[reason].append(len(env_ids))
    
    print("\n  Average environments per termination reason:")
    for reason, counts in reason_env_counts.items():
        avg = sum(counts) / len(counts) if counts else 0
        print(f"    {reason:20s}: {avg:6.0f} environments")

print("\n" + "=" * 80)
print("\n✅ Analysis complete!")

