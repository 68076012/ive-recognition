# IVE Face Recognition — Detailed Explanation

_Updated on 2026-01-17_

เอกสารนี้อธิบาย **โค้ดและ flow การทำงานแบบละเอียด** ของ `ive_face_recognition.ipynb`
โดยอิงจากเวอร์ชันปัจจุบันที่รันแบบ **CPU-only** และมีระบบ **online gallery** ช่วย re-ID

---

## 1) ภาพรวม Flow ทั้งระบบ

1. **ติดตั้ง dependencies**  
   ติดตั้ง InsightFace, onnxruntime (CPU), OpenCV และไลบรารีที่ใช้
2. **ตั้งค่า path และ imports**  
   ตั้ง path ของชุดอ้างอิง (`ive_reference`) และโฟลเดอร์ผลลัพธ์ (`outputs`)
3. **โหลดโมเดล InsightFace**  
   ใช้โมเดล `buffalo_l` สำหรับ detection/recognition
4. **สร้าง gallery ของใบหน้า**  
   อ่านรูปอ้างอิง → สกัด embedding → เก็บเป็นหลาย embedding ต่อคน
5. **ประมวลผลวิดีโอทีละเฟรม**  
   ตรวจจับใบหน้า → แปลงเป็น embedding → เทียบกับ gallery  
   → ส่งเข้า tracker → วาดกรอบ+ชื่อ → เขียนออกวิดีโอ

---

## 2) Setup Dependencies

โค้ดในเซลล์แรกจะติดตั้งแพ็กเกจ (ถ้ายังไม่มี)

```25:59:ive_face_recognition.ipynb
# Setup: install deps (CPU-only), skip if already installed
requirements = [
    "insightface==0.7.3",
    "onnxruntime==1.17.1",
    "opencv-python==4.8.1.78",
    "scipy==1.10.1",
    "filterpy==1.4.5",
    "tqdm>=4.66.1",
]
```

**เหตุผล:**  
- ใช้ `onnxruntime` เวอร์ชัน CPU เพื่อให้รันเสถียรในเครื่องที่ไม่ใช้ GPU  
- `tqdm` สำหรับ progress bar  
- `filterpy` อาจมีใช้ในอนาคต (ตอนนี้ยังไม่ได้เรียก)

---

## 3) Imports & Basic Setup

```77:105:ive_face_recognition.ipynb
import os, math, time, glob, json, random
from pathlib import Path
from collections import defaultdict, Counter, deque
import numpy as np
import cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis
```

**สิ่งที่ทำ:**  
- เตรียมไลบรารีหลักสำหรับไฟล์, คำนวณ, OpenCV และ InsightFace  
- `defaultdict`, `deque` ช่วยจัดการ history ของ track

---

## 4) โหลดโมเดล InsightFace (CPU-only)

```167:190:ive_face_recognition.ipynb
def init_face_analysis():
    providers = ["CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=-1, det_size=(640, 640))
```

**คำอธิบาย**  
- `FaceAnalysis(name="buffalo_l")` โหลดโมเดลรวม 5 ส่วน:
  - `det_10g` (detection)
  - `w600k_r50` (recognition)
  - `1k3d68` และ `2d106det` (landmarks)
  - `genderage` (เพศ/อายุ)
- `ctx_id=-1` คือใช้ CPU  
- `det_size=(640,640)` ยิ่งใหญ่ยิ่งละเอียด แต่ช้าขึ้น

---

## 5) สร้าง Gallery จากรูปอ้างอิง

โค้ดส่วนนี้สร้างฐานข้อมูลใบหน้าของสมาชิกแต่ละคนจากรูปอ้างอิง

```239:267:ive_face_recognition.ipynb
def build_gallery(reference_root: Path, max_imgs_per_id: int = 50):
    ...
    embeds = np.stack(embeds, axis=0)
    # keep all normalized embeddings for better pose coverage
    embeds = np.stack([normalize_embedding(e) for e in embeds], axis=0)
    gallery[name] = embeds
```

**เหตุผลที่เปลี่ยนเป็น “หลาย embedding ต่อคน”:**
- ช่วยให้ match ได้แม่นขึ้นเมื่อมุมหน้าหรือแสงต่างจากรูปหลัก
- ใช้ max similarity แทนการ average

---

## 6) การจำใบหน้า (Matching)

โค้ดหลักของการจับคู่:

```297:317:ive_face_recognition.ipynb
def best_match(embedding, gallery, online_gallery=None, threshold=0.45):
    for name, ref_emb in gallery.items():
        if isinstance(ref_emb, np.ndarray) and ref_emb.ndim == 2:
            score = float(np.max(ref_emb @ embedding))
        else:
            score = cosine_sim(embedding, ref_emb)
        if online_gallery and name in online_gallery:
            online_mat = np.stack(list(online_gallery[name]), axis=0)
            score = max(score, float(np.max(online_mat @ embedding)))
```

### 6.1 ขั้นตอนละเอียด
1. **รับ embedding จากใบหน้าในเฟรมปัจจุบัน**  
   embedding ถูก normalize มาแล้ว (`normalize_embedding`)
2. **เทียบกับ gallery หลัก**  
   - ถ้า `ref_emb` เป็นเมทริกซ์ (หลาย embedding ต่อคน) → ใช้ `max(ref_emb @ embedding)`  
   - ถ้าเป็นเวกเตอร์เดี่ยว → ใช้ `cosine_sim`
3. **เทียบกับ online gallery (ถ้ามี)**  
   - เปรียบเทียบกับ embedding ล่าสุดที่สะสมระหว่างรัน  
   - เอาคะแนนที่สูงสุดระหว่าง gallery หลักกับ online gallery
4. **ตัดสินใจชื่อ**  
   - ถ้าคะแนนต่ำกว่า `threshold` → `"Unknown"`

### 6.2 เหตุผลที่ใช้ max similarity
- มุมหน้าแตกต่างกันมาก (หันข้าง/แสงเปลี่ยน)  
- การใช้ค่าเฉลี่ยอาจทำให้ similarity ลดลง  
- ใช้ **ค่าสูงสุด** ทำให้จับคู่กับมุมที่ “ใกล้ที่สุด” ได้ง่ายขึ้น

### 6.3 เกณฑ์สำคัญที่กระทบความแม่น
- `rec_thresh` สูงขึ้น → แม่นขึ้นแต่ Unknown มากขึ้น  
- `online_gallery` ช่วย re-ID เมื่อสลับกล้องหรือมุม

---

## 7) Tracking System

### 7.1 โครงสร้าง Track

```332:351:ive_face_recognition.ipynb
class Track:
    def __init__(..., bbox_momentum=0.6):
        self.bbox = bbox.astype(float)
        self.embedding = embedding
        self.name_history = deque([name], maxlen=10)
```

**บทบาทสำคัญ**
- `bbox_momentum` ทำ smoothing ให้กรอบนิ่งขึ้น  
  ```
  bbox = m * new + (1-m) * old
  ```
- `name_history` เก็บชื่อย้อนหลัง 10 เฟรม → ลดการกระพริบของ label  
- `embedding` มีการทำ EMA เพื่อให้ track ไม่สลับชื่อเร็ว

### 7.2 การจับคู่ track

```377:407:ive_face_recognition.ipynb
if det['sim'] < self.sim_update_thr:
    continue
...
if iou_score > self.iou_thr and sim_score > self.embed_thr:
    best_track.update(...)
```

**ขั้นตอนละเอียด**
1. **กรอง detection ที่คะแนนต่ำ**  
   ถ้า `det['sim'] < sim_update_thr` → ไม่เอามา match
2. **หาคู่ที่ดีที่สุดจาก track เดิม**  
   - คำนวณ **IoU** ของ bbox  
   - คำนวณ **cosine similarity** ของ embedding  
   - สร้างคะแนนรวม `score = iou_score + sim_score`
3. **ยืนยันการจับคู่**  
   ต้องผ่านทั้ง `iou_thr` และ `embed_thr`
4. **อัปเดตหรือสร้างใหม่**  
   - ถ้า match → `Track.update()`  
   - ถ้าไม่ match → สร้าง track ใหม่
5. **ลบ track ที่หายไป**  
   - ถ้า `lost > max_lost` จะถูกลบ (ลดกล่องค้าง)

### 7.3 ผลของพารามิเตอร์หลัก
- `tracker_iou` สูง → track เข้มขึ้น แต่หลุดง่ายเมื่อกล้องสลับ  
- `tracker_embed` สูง → กันผิดคน แต่บางมุมอาจหลุด  
- `max_lost` ต่ำ → กล่องหายเร็ว ลดการค้าง

---

## 8) Online Gallery (ช่วย re-ID หลังสลับกล้อง)

### 8.1 แนวคิด
Online gallery คือการ “จำ embedding ล่าสุด” จากวิดีโอเดียวกัน เพื่อช่วย re-ID หลังสลับกล้องหรือมุม

### 8.2 วิธีเก็บ

```483:490:ive_face_recognition.ipynb
if label != "Unknown" and trk.stable_sim >= rec_thresh:
    online_gallery[label].append(trk.embedding.copy())
```

**เงื่อนไขสำคัญ**
- ต้องมีชื่อที่ไม่ใช่ `Unknown`  
- ต้องมีความมั่นใจ `stable_sim >= rec_thresh`

### 8.3 วิธีนำไปใช้
เมื่อเรียก `best_match(...)`  
- จะเอาคะแนนจาก gallery หลัก + online gallery  
- ใช้ **คะแนนสูงสุด** เป็นตัวตัดสิน  
- ทำให้การกลับมาติดชื่อเดิมหลังสลับกล้องแม่นขึ้น

### 8.4 จุดที่ต้องระวัง
- ถ้า `rec_thresh` ต่ำเกินไป อาจเก็บ embedding ผิดคนเข้า online gallery  
- ถ้าสลับกล้องเร็วมาก ๆ ควรใช้ `max_lost` ต่ำเพื่อเคลียร์ track เก่าเร็ว

---

## 9) Main Pipeline: process_video

```436:503:ive_face_recognition.ipynb
def process_video(...):
    faces = app.get(frame)
    for f in faces:
        if f.det_score < det_thresh: continue
        emb = normalize_embedding(f.normed_embedding)
        name, sim = best_match(..., online_gallery=online_gallery)
        detections.append(...)
    tracks = tracker.step(detections)
    for trk in tracks:
        draw_label(...)
```

**Flow ภายใน**
1. อ่านเฟรม → detect faces  
2. สกัด embedding  
3. เทียบกับ gallery + online gallery  
4. ส่งเข้าตัว tracker  
5. วาดกล่องและเขียนลงวิดีโอ

---

## 10) พารามิเตอร์สำคัญ (ค่าปัจจุบัน)

```566:580:ive_face_recognition.ipynb
det_thresh=0.45
rec_thresh=0.40
tracker_iou=0.50
tracker_embed=0.36
max_lost=1
bbox_momentum=0.60
```

**ผลของแต่ละค่า**
- `det_thresh`: สูงขึ้น = ตรวจจับยากขึ้นแต่แม่น  
- `rec_thresh`: สูงขึ้น = มั่นใจมากขึ้นแต่ Unknown เยอะ  
- `tracker_iou`: สูงขึ้น = track ยากขึ้น  
- `tracker_embed`: สูงขึ้น = match ยากขึ้นแต่ผิดน้อย  
- `max_lost`: ต่ำ = ลบ track เร็ว (ลดกล่องค้าง)  
- `bbox_momentum`: สูง = กล่องนิ่งขึ้น แต่หน่วงมากขึ้น

---

## 11) สรุปเหตุผลที่ “แม่นขึ้น”

1. ใช้ **หลาย embedding ต่อคน** แทนการเฉลี่ย  
2. ใช้ **max similarity** → match มุมหน้าข้างได้ดีขึ้น  
3. มี **online gallery** ช่วย re-ID หลังสลับกล้อง  
4. ลด `max_lost` → ลดกล่องค้างผิดคน  
5. ใช้ smoothing (bbox + embedding) ทำให้ track นิ่งขึ้น

---

หากต้องการให้เอกสารนี้เพิ่มส่วน **diagram flow** หรืออธิบายเชิงคณิตศาสตร์ (cosine similarity, IoU) เพิ่มเติม บอกได้เลยครับ
