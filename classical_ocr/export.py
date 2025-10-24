def export_hocr_html(out_path, page_size, words_data):
    H, W = page_size
    html = []
    html.append("<!doctype html><html><head><meta charset='utf-8'>")
    html.append("<style>.word{position:absolute;border:1px solid #888;border-radius:3px;padding:0 2px;font:14px/1.3 monospace;}</style>")
    html.append("</head><body>")
    html.append(f"<div style='position:relative;width:{W}px;height:{H}px;background:#fff;'>")
    for w in words_data:
        x,y,ww,hh = w["box"]
        text = w["text"]
        html.append(f"<div class='word' style='left:{x}px;top:{y}px;width:{ww}px;height:{hh}px;' title='{text}'>{text}</div>")
    html.append("</div></body></html>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

def export_tsv(out_path, words_data):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("level\tleft\ttop\twidth\theight\ttext\tconf\n")
        for w in words_data:
            x,y,ww,hh = w["box"]
            f.write(f"word\t{x}\t{y}\t{ww}\t{hh}\t{w['text']}\t{w.get('conf',1.0):.3f}\n")
