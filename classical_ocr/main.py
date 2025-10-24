import os, cv2, argparse, numpy as np

from .report import VisualReport
from .preprocess import to_gray_denoise, estimate_skew_correct, otsu_binarize, sauvola_binarize
from .layout import rlsa, project_valleys, group_into_words
from .utils import cc_boxes
from .detection import detect_text_regions_mser
from .features import hog_feature
from .classify import build_font_prototypes, knn_predict, correct_with_lexicon
from .export import export_hocr_html, export_tsv

def run_pipeline(args):
    report = VisualReport(out_dir=args.out, width=args.width)

    # 1) Ingest
    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"Cannot read input: {args.input}")
    report.save_step("Ingest", img, "Original input image read from disk.")

    # 2) Grayscale + denoise
    gray = to_gray_denoise(img)
    report.save_step("Grayscale + Denoise", gray, "Converted to grayscale and denoised for stable thresholding.")

    # 3) Skew detection/correction
    angle, edges, lines, gray_corr = estimate_skew_correct(gray)
    vis_skew = img.copy()
    if lines is not None:
        for l in lines[:200]:
            x1,y1,x2,y2 = l[0]
            cv2.line(vis_skew, (x1,y1), (x2,y2), (0,255,0), 2)
    report.save_step("Skew Detection", vis_skew, f"Detected line segments; estimated skew angle {angle:.2f}°.")
    if abs(angle) > 0.5:
        gray = gray_corr
        report.save_step("Deskewed", gray, "Image rotated to correct skew for better segmentation.")

    # 4) Binarization
    bw_otsu = otsu_binarize(gray)
    report.save_step("Otsu Binarization", bw_otsu, "Global thresholding for clean scans.")
    bw_sau = sauvola_binarize(gray, window=args.sauvola_window, k=args.sauvola_k)
    report.save_step("Sauvola Binarization", bw_sau, "Local thresholding for uneven illumination.")
    bw = bw_sau if args.mode == "scene" else bw_otsu

    words_boxes, glyph_boxes, words_data = [], [], []

    if args.mode == "document":
        # RLSA and lines
        bw_rlsa = rlsa(255 - bw, h_thresh=args.rlsa_h, v_thresh=args.rlsa_v)
        bw_rlsa = 255 - bw_rlsa
        report.save_step("RLSA Smearing", bw_rlsa, "Run-length smoothing to connect glyphs into words/lines.")
        proj, line_spans = project_valleys(bw, axis=0, min_run=10)
        vis_lines = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        for (y0,y1) in line_spans:
            cv2.rectangle(vis_lines, (5,y0), (vis_lines.shape[1]-5,y1), (0,0,255), 1)
        report.save_step("Line Segmentation", vis_lines, "Horizontal projection and whitespace to mark line bands.")

        # Words by grouping CCs
        all_boxes = cc_boxes(bw, min_w=3, min_h=8)
        vis_words = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        word_groups = group_into_words(all_boxes)
        for group in word_groups:
            xs = [b[0] for b in group]; ys = [b[1] for b in group]
            xe = [b[0]+b[2] for b in group]; ye = [b[1]+b[3] for b in group]
            x0,y0,x1,y1 = min(xs), min(ys), max(xe), max(ye)
            cv2.rectangle(vis_words, (x0,y0), (x1,y1), (0,255,0), 2)
            words_boxes.append((x0,y0,x1-x0,y1-y0))
        report.save_step("Word Grouping", vis_words, "Grouped connected components into word boxes.")

        # Glyph candidates
        glyph_boxes = cc_boxes(bw, min_w=3, min_h=8)
        vis_cc = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        for (x,y,w,h) in glyph_boxes[:1200]:
            cv2.rectangle(vis_cc, (x,y), (x+w,y+h), (255,0,0), 1)
        report.save_step("Glyph Candidates", vis_cc, "Connected components filtered by size as glyph candidates.")

    else:
        # Scene: MSER and grouping
        mser_boxes = detect_text_regions_mser(gray, min_area=args.mser_min_area, max_area=args.mser_max_area, delta=args.mser_delta)
        vis_mser = img.copy()
        for (x,y,w,h) in mser_boxes:
            cv2.rectangle(vis_mser, (x,y), (x+w,y+h), (0,255,255), 2)
        report.save_step("MSER Detections", vis_mser, "MSER-based text region proposals in natural scenes.")

        word_groups = group_into_words(mser_boxes, gap_factor=2.0)
        vis_words = img.copy()
        for group in word_groups:
            xs = [b[0] for b in group]; ys = [b[1] for b in group]
            xe = [b[0]+b[2] for b in group]; ye = [b[1]+b[3] for b in group]
            x0,y0,x1,y1 = min(xs), min(ys), max(xe), max(ye)
            cv2.rectangle(vis_words, (x0,y0), (x1,y1), (0,255,0), 2)
            words_boxes.append((x0,y0,x1-x0,y1-y0))
        report.save_step("Word Grouping", vis_words, "Grouped MSER components into word boxes.")

        # Glyphs via CC inside words
        glyph_boxes = []
        for (x,y,w,h) in words_boxes:
            sub = bw[y:y+h, x:x+w]
            ccs = cc_boxes(sub, min_w=3, min_h=8, max_w=w, max_h=h)
            for (cx,cy,cw,ch) in ccs:
                glyph_boxes.append((x+cx, y+cy, cw, ch))
        vis_cc = img.copy()
        for (x,y,w,h) in glyph_boxes[:1200]:
            cv2.rectangle(vis_cc, (x,y), (x+w,y+h), (255,0,0), 1)
        report.save_step("Glyph Candidates", vis_cc, "Connected components inside detected regions as glyph candidates.")

    # Features (HOG viz on one glyph)
    if glyph_boxes:
        gx, gy, gw, gh = glyph_boxes[0]
        sample = bw[gy:gy+gh, gx:gx+gw]
        _, hog_viz = hog_feature(sample)
        report.save_step("HOG Features", cv2.cvtColor(hog_viz, cv2.COLOR_GRAY2BGR), "Gradient structure visualization for a sample glyph.")
    else:
        report.save_step("HOG Features", img, "No glyph found for demo; showing original frame.")

    # Classifier
    X_train, Y_train = build_font_prototypes()

    # Recognize and decode per word
    words_data = []
    for (wx,wy,ww,wh) in words_boxes:
        g_in_word = [(x,y,w,h) for (x,y,w,h) in glyph_boxes if (x>=wx and y>=wy and x+w<=wx+ww and y+h<=wy+wh)]
        g_in_word = sorted(g_in_word, key=lambda b: b[0])
        chars, confs = [], []
        for (x,y,w,h) in g_in_word:
            crop = bw[y:y+h, x:x+w]
            if crop is None or crop.size == 0:
                continue
            feat, _ = hog_feature(crop)
            pred, conf = knn_predict(X_train, Y_train, feat, k=3)
            chars.append(pred); confs.append(conf)
        text = "".join(chars) if chars else ""
        avg_conf = float(np.mean(confs)) if confs else 0.0
        words_data.append({"box": (wx,wy,ww,wh), "text": text, "conf": avg_conf})

    lexicon = []
    if args.lexicon and os.path.exists(args.lexicon):
        with open(args.lexicon, "r", encoding="utf-8") as f:
            lexicon = [ln.strip() for ln in f if ln.strip()]

    corrected = []
    for w in words_data:
        cw, cconf = correct_with_lexicon(w["text"], lexicon, max_d=2)
        corrected.append({"box": w["box"], "text": cw, "conf": min(1.0, 0.5*w["conf"] + 0.5*cconf)})

    vis_pred = img.copy()
    for w in corrected:
        x,y,ww,hh = w["box"]
        cv2.rectangle(vis_pred, (x,y), (x+ww,y+hh), (0,255,0), 2)
        cv2.putText(vis_pred, f"{w['text']} ({w['conf']:.2f})", (x, max(0,y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
    report.save_step("Predictions", vis_pred, "Overlay predicted words with confidences on word boxes.")

    # Exports
    sheet = report.save_contact_sheet(cols=3, gap=12, out_name="contact_sheet.png")
    md_path = report.save_markdown("report.md")
    html_path = report.save_html("report.html")

    H, W = img.shape[:2]
    from .export import export_hocr_html, export_tsv
    export_hocr_html(os.path.join(args.out, "words.html"), (H,W), corrected)
    export_tsv(os.path.join(args.out, "words.tsv"), corrected)

    print("Saved artifacts:")
    print(f" - Contact sheet: {sheet}")
    print(f" - Markdown: {md_path}")
    print(f" - HTML: {html_path}")
    print(f" - hOCR-like HTML: {os.path.join(args.out, 'words.html')}")
    print(f" - TSV: {os.path.join(args.out, 'words.tsv')}")

def main():
    ap = argparse.ArgumentParser(description="Classical OCR pipeline (no deep learning)")
    ap.add_argument("--input", required=True, help="Path to input image")
    ap.add_argument("--mode", default="document", choices=["document","scene"], help="Input type")
    ap.add_argument("--out", default="ocr_report", help="Output directory for snapshots and exports")
    ap.add_argument("--width", type=int, default=1024, help="Snapshot width")
    # Sauvola
    ap.add_argument("--sauvola_window", type=int, default=25)
    ap.add_argument("--sauvola_k", type=float, default=0.34)
    # RLSA thresholds
    ap.add_argument("--rlsa_h", type=int, default=25)
    ap.add_argument("--rlsa_v", type=int, default=0)
    # MSER params
    ap.add_argument("--mser_min_area", type=int, default=60)
    ap.add_argument("--mser_max_area", type=int, default=10000)
    ap.add_argument("--mser_delta", type=int, default=5)
    # Lexicon path
    ap.add_argument("--lexicon", type=str, default="")
    args = ap.parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
