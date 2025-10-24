# OCR Pipeline Report — 2025-10-24 10:38  

## Ingest  
Original input image read from disk.  
![Ingest](00_Ingest.png)  

## Grayscale + Denoise  
Converted to grayscale and denoised for stable thresholding.  
![Grayscale + Denoise](01_Grayscale_+_Denoise.png)  

## Skew Detection  
Detected line segments; estimated skew angle 0.00°.  
![Skew Detection](02_Skew_Detection.png)  

## Otsu Binarization  
Global thresholding for clean scans.  
![Otsu Binarization](03_Otsu_Binarization.png)  

## Sauvola Binarization  
Local thresholding for uneven illumination.  
![Sauvola Binarization](04_Sauvola_Binarization.png)  

## MSER Detections  
MSER-based text region proposals in natural scenes.  
![MSER Detections](05_MSER_Detections.png)  

## Word Grouping  
Grouped MSER components into word boxes.  
![Word Grouping](06_Word_Grouping.png)  

## Glyph Candidates  
Connected components inside detected regions as glyph candidates.  
![Glyph Candidates](07_Glyph_Candidates.png)  

## HOG Features  
Gradient structure visualization for a sample glyph.  
![HOG Features](08_HOG_Features.png)  

## Predictions  
Overlay predicted words with confidences on word boxes.  
![Predictions](09_Predictions.png)  
