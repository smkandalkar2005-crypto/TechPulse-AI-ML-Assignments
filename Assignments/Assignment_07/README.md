# Assignment 7 — Pedestrian Detection using OpenCV (HOG + SVM)

**Tata Technologies TechPulse Applied AI/ML Laboratory**

## 1. Objective
Detect pedestrians in a computer vision scene using **OpenCV's Histogram of Oriented Gradients (HOG)** descriptor combined with a pre-trained **Linear Support Vector Machine (SVM)** detector (`cv2.HOGDescriptor_getDefaultPeopleDetector()`).

## 2. Methodology
1. **Sample Scene Loading:** Load a sample pedestrian scene (`pedestrians_sample.jpg`).
2. **HOG + SVM Configuration:** Configure `cv2.HOGDescriptor()` with `hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())`.
3. **Multi-Scale Detection:** Apply `hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)` to scan sliding windows across image resolution pyramid levels.
4. **Candidate vs. Confirmed Filtering:** Inspect raw candidate detection windows and filter out low-confidence background false positives (SVM weight $< 0.50$).
5. **Bounding Box Visualization:** Draw bounding boxes around confirmed pedestrian coordinates and visualize original vs. detection output using Matplotlib.

## 3. Actual Executed Results
All values below come directly from the executed notebook (`TTL_Assignment_07.ipynb`):

- **OpenCV Version:** 4.10.0
- **Input Image Resolution:** $576 \times 768 \times 3$ RGB pixels
- **Raw Candidate Detections:** **4 candidate boxes** (3 confirmed pedestrians + 1 background false positive with weight = 0.346).
- **Confirmed Pedestrian Detections (SVM Weight $\ge 0.50$):** **3 confirmed pedestrians** matching the 3 visible persons in the scene.
- **Detection Output:** Green bounding box coordinates and SVM confidence weights drawn around each confirmed pedestrian silhouette.

## 4. Key Concepts
- **HOG (Histogram of Oriented Gradients):** Captures edge structures and silhouette shapes by counting gradient direction histograms in localized image blocks.
- **Linear SVM:** Classifies HOG feature vectors into human vs. non-human categories based on pre-trained decision boundaries.
- **Multi-Scale Pyramid:** Allows detection of pedestrians at varying distances and scales within the camera field of view.

## 5. Limitations
- Classical HOG + SVM relies on hand-crafted gradient features and fixed aspect-ratio sliding windows.
- Sensitive to heavy occlusion, extreme lighting variations, and complex cluttered backgrounds compared to modern deep-learning detectors (e.g. YOLO, Faster R-CNN).

## 6. How to Run
1. Navigate to `Assignments/Assignment_07/`:
   ```bash
   cd Assignments/Assignment_07
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook TTL_Assignment_07.ipynb
   ```
3. Run all cells sequentially (**Cell -> Run All**).
