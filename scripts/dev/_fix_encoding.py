# 一次性修复: 混合编码 md (头部UTF-8 + 尾部GBK) -> 纯UTF-8
import io

def decode_mixed(raw: bytes) -> str:
    out, pos, enc = [], 0, 'utf-8'
    n = len(raw)
    while pos < n:
        try:
            out.append(raw[pos:].decode(enc))
            break
        except UnicodeDecodeError as e:
            p = pos + e.start
            if p > pos:
                out.append(raw[pos:p].decode(enc))
            enc = 'gbk' if enc == 'utf-8' else 'utf-8'
            pos = p
    return ''.join(out)

for f in ['docs/data_dictionary.md', 'docs/leakage_checklist.md',
          'docs/task_cards.md']:
    raw = open(f, 'rb').read()
    txt = decode_mixed(raw)
    bad = txt.count('�')
    io.open(f, 'w', encoding='utf-8', newline='').write(txt)
    print(f'{f}: len={len(txt)} replacement_chars={bad}')
