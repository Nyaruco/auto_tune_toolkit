#!/usr/bin/env python3
"""
合并多个JSON并导出为格式化的Excel文件（需要安装openpyxl）
pip install openpyxl
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

def extract_all_tests(json_files: List[str]) -> List[Dict[str, Any]]:
    """提取所有JSON文件的测试数据"""
    all_tests = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            tests = data.get("tests", [])
            for test in tests:
                test['source_file'] = os.path.basename(json_file)
                test['scene'] = data.get('scene', 'unknown')
            all_tests.extend(tests)
            print(f"Loaded {len(tests)} records from {os.path.basename(json_file)}")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    return all_tests

def create_excel_report(all_tests: List[Dict], output_file: str):
    """创建格式化的Excel报告"""
    if not HAS_OPENPYXL:
        print("Error: openpyxl not installed. Please run: pip install openpyxl")
        return
    
    wb = Workbook()
    
    # 按场景分sheet
    scenes = {}
    for test in all_tests:
        scene = test.get('scene', 'unknown')
        if scene not in scenes:
            scenes[scene] = []
        scenes[scene].append(test)
    
    # 定义样式
    header_font = Font(name='Arial', bold=True, size=11, color='FFFFFF')
    header_fill = PatternFill(start_color='2F5496', end_color='2F5496', fill_type='solid')
    satisfied_fill = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    # 定义列
    columns = [
        ('File', 'source_file', 40),
        ('Batch', 'batch', 8),
        ('Rate', 'rate', 10),
        ('Prompts', 'num_prompts', 10),
        ('Concurrency', 'max_concurrency', 12),
        ('TTFT P90(ms)', 'ttft_p90_ms', 15),
        ('TPOT P90(ms)', 'tpot_p90_ms', 15),
        ('E2EL P90(ms)', 'e2el_p90_ms', 15),
        ('Req Throughput', 'req_throughput', 15),
        ('Out Tok/s', 'out_tok_thr', 15),
        ('Total Tok/s', 'total_tok_thr', 15),
        ('Success', 'success_requests', 10),
        ('Failed', 'failed_requests', 10),
        ('Satisfied', 'satisfied', 10),
        ('TTFT Margin', 'ttft_margin', 15),
    ]
    
    for scene_name, scene_tests in scenes.items():
        ws = wb.create_sheet(title=scene_name[:31])  # Excel sheet名最长31字符
        
        # 写入表头
        for col_idx, (header, _, width) in enumerate(columns, 1):
            cell = ws.cell(row=1, column=col_idx, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border
            ws.column_dimensions[get_column_letter(col_idx)].width = width
        
        # 写入数据
        for row_idx, test in enumerate(scene_tests, 2):
            for col_idx, (_, key, _) in enumerate(columns, 1):
                value = test.get(key, '')
                cell = ws.cell(row=row_idx, column=col_idx, value=value)
                cell.border = border
                
                # 高亮满足条件的行
                if test.get('satisfied'):
                    cell.fill = satisfied_fill
                
                # 数字格式
                if isinstance(value, float):
                    cell.number_format = '#,##0.00'
        
        # 冻结首行
        ws.freeze_panes = 'A2'
        # 自动筛选
        ws.auto_filter.ref = ws.dimensions
    
    # 创建汇总sheet
    if len(scenes) > 1:
        summary_ws = wb.create_sheet(title='Summary', index=0)
        summary_ws.append(['Scene', 'Total Tests', 'Satisfied', 'Min TTFT P90', 'Max Throughput'])
        
        for scene_name, scene_tests in scenes.items():
            satisfied = sum(1 for t in scene_tests if t.get('satisfied'))
            min_ttft = min((t.get('ttft_p90_ms', float('inf')) for t in scene_tests), default=0)
            max_thr = max((t.get('req_throughput', 0) for t in scene_tests), default=0)
            summary_ws.append([scene_name, len(scene_tests), satisfied, min_ttft, max_thr])
    
    # 删除默认sheet
    if 'Sheet' in wb.sheetnames:
        del wb['Sheet']
    
    wb.save(output_file)
    print(f"\nExcel report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Merge JSON benchmark results to Excel')
    parser.add_argument('json_files', nargs='+', help='JSON files to process')
    parser.add_argument('-o', '--output', default='benchmark_report.xlsx', 
                       help='Output Excel file (default: benchmark_report.xlsx)')
    parser.add_argument('--csv', action='store_true', 
                       help='Also save as CSV')
    
    args = parser.parse_args()
    
    # 展开通配符
    expanded_files = []
    for pattern in args.json_files:
        matched = list(Path().glob(pattern))
        if matched:
            expanded_files.extend([str(p) for p in matched])
        elif os.path.exists(pattern):
            expanded_files.append(pattern)
    
    if not expanded_files:
        print("Error: No valid files found")
        return
    
    print(f"Processing {len(expanded_files)} files...")
    all_tests = extract_all_tests(expanded_files)
    
    if not all_tests:
        print("No test data found!")
        return
    
    # 生成Excel
    if HAS_OPENPYXL:
        create_excel_report(all_tests, args.output)
    else:
        print("openpyxl not available, falling back to CSV")
        args.csv = True
    
    # 同时保存CSV
    if args.csv:
        import csv
        csv_file = args.output.replace('.xlsx', '.csv')
        with open(csv_file, 'w', newline='') as f:
            if all_tests:
                writer = csv.DictWriter(f, fieldnames=all_tests[0].keys())
                writer.writeheader()
                writer.writerows(all_tests)
        print(f"CSV saved to: {csv_file}")

if __name__ == "__main__":
    main()