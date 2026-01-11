# คู่มือโครงการ IVE Face Recognition (InsightFace + Tracking)

เอกสารนี้สรุปแนวคิด โค้ด พารามิเตอร์หลัก และขั้นตอนใช้งานของโน้ตบุ๊ก `ive_face_recognition.ipynb` เพื่อให้ส่งออกเป็น PDF ได้ง่าย (เช่น ผ่าน VS Code / Cursor > Export to PDF หรือ `pandoc`).

## ภาพรวมสถาปัตยกรรม
- **โมเดล**: ใช้ InsightFace `buffalo_l` (ScrFD detector + ArcFace recognition) แบบ CPU-only เพื่อความเสถียร
- **ข้อมูลอ้างอิง**: โฟลเดอร์ `ive_reference/<Member_Name>/` เก็บภาพต่อบุคคล เพื่อนำมาทำค่าเฉลี่ย embedding เป็นแกลเลอรี
- **อินพุตวิดีโอ**: ใช้ไฟล์ท้องถิ่น `outputs/ive_interview_input.mp4` (หรือระบุเองผ่าน `custom_video_path`)
- **เอาต์พุต**: วิดีโอที่ใส่กรอบ/ชื่อใน `outputs/` เช่น `ive_quickcheck.mp4`, `ive_recognized.mp4`

## การเตรียมสภาพแวดล้อม
```bash
python -m venv .venv_ive
.venv_ive\Scripts\activate   # หรือ source .venv_ive/bin/activate บน mac/linux
pip install insightface==0.7.3 onnxruntime==1.17.1 opencv-python==4.8.1.78 scipy==1.10.1 filterpy==1.4.5 "tqdm>=4.66.1"
```
- ใช้ `onnxruntime` CPU ถ้าไม่มี GPU หรือไม่ต้องการ CUDA
- ตำแหน่งทำงาน: `project_recognition/`

## ลำดับการทำงานในโน้ตบุ๊ก
1) **ตั้งค่าพาธ**: ตรวจโฟลเดอร์ `ive_reference/`, `outputs/`; กำหนด `local_video_fallback = outputs/ive_interview_input.mp4`
2) **โหลดโมเดล**: `init_face_analysis()` ใช้ `FaceAnalysis` พร้อม `CPUExecutionProvider`
3) **สร้างแกลเลอรี**: `build_gallery(reference_root)`  
   - อ่านภาพต่อบุคคล → detect face → ดึง `normed_embedding` → เฉลี่ยแล้ว `normalize_embedding`
4) **เตรียมวิดีโออินพุต**: ใช้ `custom_video_path` ถ้ามี ไม่พบไฟล์จะ `raise FileNotFoundError`
5) **ประมวลผลวิดีโอ**: `process_video(...)`  
   - detect → embed → `best_match` กับแกลเลอรี → `SimpleTracker.step` จับคู่ IoU+embedding → วาดกรอบ/ชื่อ → เขียนเฟรม
6) **ทดสอบเร็ว**: `ive_quickcheck.mp4` (จำกัด `max_frames=200`)  
7) **รันเต็ม**: `ive_recognized.mp4`

## พารามิเตอร์สำคัญ (ปรับจูน)
- `build_gallery(max_imgs_per_id=50)`: จำกัดจำนวนภาพต่อคนเพื่อลดเวลาประมวลผล
- `best_match(threshold=0.38)`: ค่าความคล้ายขั้นต่ำ ถ้าต่ำกว่านี้ให้เป็น `"Unknown"`
- `process_video(...)`
  - `det_thresh=0.45`: ค่าความมั่นใจขั้นต่ำของตัวตรวจจับใบหน้า
  - `rec_thresh=0.40`: ค่า similarity ขั้นต่ำเวลาตัดสินชื่อ (ซิงก์กับ threshold ใน `best_match`)
  - `tracker_iou=0.45`: เกณฑ์ IoU สำหรับจับคู่บ็อกซ์กับ track เดิม
  - `tracker_embed=0.35`: เกณฑ์ cosine similarity ระหว่าง embedding ใหม่กับ track
  - `max_frames`: จำกัดจำนวนเฟรมเพื่อรันทดสอบเร็ว (ตั้ง `None` เพื่อรันทั้งคลิป)
  - `warmup`: เฟรมอุ่นเครื่อง (ยังไม่ใช้ในโค้ดหลัก แต่เตรียมให้ปรับได้)
- `SimpleTracker(max_lost=20)`: จำนวนเฟรมที่ track จะทนหายก่อนถูกลบ
- การแสดงผล: `draw_label` จะโชว์ `ชื่อ#track_id (score)` และใช้สีสุ่มต่อ track

## หลักการเบื้องหลัง
- **Normalization**: ทุก embedding ถูก `normalize_embedding` เพื่อใช้ cosine similarity ได้ถูกต้อง
- **การเฉลี่ย embedding**: ลด noise ต่อบุคคลด้วยค่าเฉลี่ยภาพหลายใบหน้าในแกลเลอรี
- **Tracking blend**: ใช้ IoU + similarity เพื่อรักษา identity ระหว่างเฟรม; ชื่อใช้ `stable_name` (mode ของประวัติ) ลดการสลับชื่อ
- **CPU-only**: ลดปัญหาไดรเวอร์ CUDA ไม่เสถียร; หากต้องการ GPU สามารถแก้ `init_face_analysis` ให้โหลด `CUDAExecutionProvider`

## แนวทางใช้งาน/ปรับแต่ง
- เพิ่ม/คัดกรองภาพใน `ive_reference/` เพื่อยกระดับความแม่นยำ (ภาพคมชัด มุมตรง แสงดี)
- ปรับ `det_thresh`, `rec_thresh`, `tracker_iou`, `tracker_embed` ตามความหนาแน่นใบหน้าและความต้องการความแม่น/ความต่อเนื่อง
- หากเฟรมใหญ่/จำนวนมาก → ลดความละเอียดวิดีโอหรือเพิ่ม `max_frames` สำหรับทดสอบก่อนรันเต็ม

## การส่งออกเป็น PDF
- เปิดไฟล์ `.md` นี้ใน VS Code / Cursor แล้วเลือก Export → PDF  
หรือใช้ `pandoc`:
```bash
pandoc FACE_RECOGNITION_GUIDE.md -o FACE_RECOGNITION_GUIDE.pdf
```
