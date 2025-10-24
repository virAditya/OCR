import os, cv2, math, textwrap, datetime
import numpy as np

class VisualReport:
    def __init__(self, out_dir="report", width=1024, font_scale=0.7, line_th=2):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.steps = []
        self.width = width
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.line_th = line_th

    def _resize(self, img, width=None):
        if width is None:
            width = self.width
        h, w = img.shape[:2]
        if w == width:
            return img
        scale = width / float(w)
        return cv2.resize(img, (width, int(h*scale)), interpolation=cv2.INTER_AREA)

    def _banner(self, title, w):
        banner = np.full((48, w, 3), (30, 30, 30), dtype=np.uint8)
        cv2.putText(banner, title, (12, 32), self.font, self.font_scale, (255, 255, 255), self.line_th, cv2.LINE_AA)
        return banner

    def _annotate(self, img, title):
        rgb = img.copy()
        if len(rgb.shape) == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        rgb = self._resize(rgb, self.width)
        return np.vstack([self._banner(title, rgb.shape[1]), rgb])

    def save_step(self, title, image_bgr_or_gray, caption=""):
        annotated = self._annotate(image_bgr_or_gray, title)
        fname = f"{len(self.steps):02d}_{title.replace(' ', '_')}.png"
        fpath = os.path.join(self.out_dir, fname)
        cv2.imwrite(fpath, annotated)
        self.steps.append({"title": title, "path": fname, "caption": caption})

    def save_contact_sheet(self, cols=3, gap=12, out_name="contact_sheet.png"):
        if not self.steps:
            return None
        thumbs = []
        thumb_w = self.width // cols - gap
        for s in self.steps:
            img = cv2.imread(os.path.join(self.out_dir, s["path"]))
            h, w = img.shape[:2]
            scale = thumb_w / float(w)
            thumbs.append(cv2.resize(img, (thumb_w, int(h*scale)), interpolation=cv2.INTER_AREA))
        rows = math.ceil(len(thumbs) / cols)
        max_h_per_row = []
        for r in range(rows):
            row_imgs = thumbs[r*cols:(r+1)*cols]
            max_h_per_row.append(max(im.shape[0] for im in row_imgs))
        grid_h = sum(max_h_per_row) + gap*(rows+1)
        grid_w = cols*thumb_w + gap*(cols+1)
        canvas = np.full((grid_h, grid_w, 3), 240, dtype=np.uint8)
        y = gap
        idx = 0
        for r in range(rows):
            x = gap
            for c in range(cols):
                if idx >= len(thumbs):
                    break
                im = thumbs[idx]
                ih, iw = im.shape[:2]
                canvas[y:y+ih, x:x+iw] = im
                x += thumb_w + gap
                idx += 1
            y += max_h_per_row[r] + gap
        out_path = os.path.join(self.out_dir, out_name)
        cv2.imwrite(out_path, canvas)
        return out_path

    def save_markdown(self, filename="report.md"):
        lines = []
        lines.append(f"# OCR Pipeline Report — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
        lines.append("")
        for s in self.steps:
            lines.append(f"## {s['title']}  ")
            if s["caption"]:
                lines.append(textwrap.fill(s["caption"], width=100) + "  ")
            lines.append(f"![{s['title']}]({s['path']})  ")
            lines.append("")
        md_path = os.path.join(self.out_dir, filename)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return md_path

    def save_html(self, filename="report.html"):
        html = []
        html.append("<!doctype html><html><head><meta charset='utf-8'>")
        html.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
        html.append("<style>body{font-family:system-ui,Arial,sans-serif;margin:24px;}h2{margin-top:32px;}img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:6px;}</style>")
        html.append("</head><body>")
        html.append(f"<h1>OCR Pipeline Report — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</h1>")
        for s in self.steps:
            html.append(f"<h2>{s['title']}</h2>")
            if s["caption"]:
                html.append(f"<p>{s['caption']}</p>")
            html.append(f"<img src='{s['path']}' alt='{s['title']}'/>")
        html.append("</body></html>")
        html_path = os.path.join(self.out_dir, filename)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        return html_path
