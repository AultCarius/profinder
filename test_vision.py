import cv2
import time
import os


def capture_photo(camera_id, save_path):
    """
    打开指定摄像头并拍摄一张照片
    :param camera_id: 摄像头编号（0/1）
    :param save_path: 照片保存路径
    """
    # 初始化摄像头（Windows 加 CAP_DSHOW 避免黑屏/卡顿）
    cap = cv2.VideoCapture(camera_id + cv2.CAP_DSHOW if os.name == 'nt' else camera_id)

    # 设置摄像头分辨率（可选，根据需要调整）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 给摄像头一点启动时间（避免第一张图模糊/黑屏）
    time.sleep(1)

    try:
        # 读取一帧画面
        ret, frame = cap.read()

        if not ret:
            print(f"❌ 摄像头 {camera_id} 读取画面失败！")
            return False

        # 保存照片
        cv2.imwrite(save_path, frame)
        print(f"✅ 摄像头 {camera_id} 拍摄成功！照片已保存至：{save_path}")
        return True

    except Exception as e:
        print(f"❌ 摄像头 {camera_id} 拍摄出错：{e}")
        return False

    finally:
        # 无论是否成功，都要释放摄像头
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 创建保存照片的文件夹（避免路径不存在）
    save_dir = "camera_photos"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 分别拍摄摄像头 0 和 1
    capture_photo(
        camera_id=0,
        save_path=os.path.join(save_dir, "camera_0_photo.jpg")
    )

    # 间隔1秒，避免摄像头占用冲突
    time.sleep(1)

    capture_photo(
        camera_id=1,
        save_path=os.path.join(save_dir, "camera_1_photo.jpg")
    )