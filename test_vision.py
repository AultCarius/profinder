import cv2
import time  # 引入计时模块

from modules.vision import detect_persons, crop_person, recognize_occupation

# ========== 1. 初始化摄像头并计时 ==========
start_time = time.time()
cap = cv2.VideoCapture(1 + cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
camera_init_time = time.time() - start_time
print(f"✅ 摄像头初始化耗时：{camera_init_time:.4f} 秒")

# ========== 2. 读取摄像头画面并计时 ==========
start_time = time.time()
ret, frame = cap.read()
cap.release()
frame_read_time = time.time() - start_time
print(f"✅ 摄像头读取单帧画面耗时：{frame_read_time:.4f} 秒")

if not ret:
    print("❌ 摄像头读取失败")
else:
    # ========== 3. 人物检测并计时 ==========
    start_time = time.time()
    boxes = detect_persons(frame)
    detect_time = time.time() - start_time
    print(f"✅ 人物检测耗时：{detect_time:.4f} 秒，共检测到 {len(boxes)} 个人")

    # ========== 4. 逐个人物处理（裁剪+职业识别）并计时 ==========
    total_crop_time = 0.0
    total_recognize_time = 0.0

    for i, box in enumerate(boxes):
        # 4.1 人像裁剪计时
        crop_start = time.time()
        crop = crop_person(frame, box)
        crop_time = time.time() - crop_start
        total_crop_time += crop_time

        # 4.2 职业识别计时
        recognize_start = time.time()
        occupation = recognize_occupation(crop)
        recognize_time = time.time() - recognize_start
        total_recognize_time += recognize_time

        # 打印单个人物的耗时和结果
        print(
            f"  🧑 人物 {i + 1}：裁剪耗时 {crop_time:.4f} 秒 | 职业识别耗时 {recognize_time:.4f} 秒 | 识别结果：{occupation}")

    # 打印累计耗时
    print(f"\n📊 累计统计：")
    print(f"   - 所有人物裁剪总耗时：{total_crop_time:.4f} 秒（平均 {total_crop_time / max(len(boxes), 1):.4f} 秒/人）")
    print(
        f"   - 所有人物职业识别总耗时：{total_recognize_time:.4f} 秒（平均 {total_recognize_time / max(len(boxes), 1):.4f} 秒/人）")

# ========== 5. 总耗时统计 ==========
total_program_time = camera_init_time + frame_read_time + (
    detect_time if 'detect_time' in locals() else 0) + total_crop_time + total_recognize_time
print(f"\n⏱️  程序总耗时：{total_program_time:.4f} 秒")