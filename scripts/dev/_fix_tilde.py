# 一次性修复: ASCII ~ 在 GFM 中触发删除线 -> 全角 ～ (跳过代码块)
import io

files = [
    'README.md', 'reports/final_report.md', 'reports/resume_bullets.md',
    'reports/data_quality.md', 'reports/wp1_acceptance.md',
    'reports/wp2_acceptance.md', 'reports/wp3_acceptance.md',
    'reports/wp4_acceptance.md', 'reports/wp5_acceptance.md',
]

for f in files:
    s = io.open(f, encoding='utf-8').read()
    out, in_code, n = [], False, 0
    for ln in s.split('\n'):
        if ln.lstrip().startswith('```'):
            in_code = not in_code
        if not in_code and '~' in ln:
            n += ln.count('~')
            ln = ln.replace('~', '～')
        out.append(ln)
    io.open(f, 'w', encoding='utf-8', newline='').write('\n'.join(out))
    print(f'{f}: {n}处替换')
