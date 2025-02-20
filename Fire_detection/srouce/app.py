import base64
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import mysql.connector
import time
import torch
from flask import Flask, render_template, Response, jsonify
from datetime import datetime
from ultralytics import YOLO
from PIL import Image

#  Khởi tạo Flask Web Server
app = Flask(__name__)

# 🔧 Cấu hình MQTT
MQTT_BROKER = "192.168.1.74"
MQTT_PORT = 1883
MQTT_TOPIC = "img"

#  Kết nối MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",  #  Thay bằng user MySQL của bạn
    password="123",  #  Thay bằng mật khẩu MySQL
    database="fire_detection"
)
cursor = db.cursor()

#  Load mô hình YOLOv8 nhận diện đám cháy
model = YOLO('D:/Du_lieu_tong_hop/K57KMT/IOT/Nhan_dang_dam_chay/Nhan_dang_dam_chay/nhan_dang/model/best.pt')
model.eval()

#  Biến toàn cục lưu ảnh từ MQTT
image_data = {}
total_parts = None
latest_image = None


#  Hàm tự động sửa lỗi thiếu padding Base64
def fix_base64_padding(base64_string):
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += "=" * (4 - missing_padding)
    return base64_string


#  Hàm nhận diện đám cháy bằng YOLOv8
def detect_fire(image):
    """Nhận diện đám cháy với YOLOv8"""
    global latest_image

    results = model(image)  # Nhận diện ảnh

    #  Vẽ bounding box lên ảnh
    for result in results:
        image_cv = result.plot()  #  YOLOv8 dùng .plot() thay vì .render()

    #  Kiểm tra nếu có phát hiện đám cháy
    fire_detected = any(cls == 0 for _, _, _, _, _, cls in result.boxes.data.tolist())  #  Dùng `result.boxes.data`

    # 🖼 Chuyển ảnh về định dạng OpenCV để hiển thị trên web
    _, buffer = cv2.imencode('.jpg', image_cv)
    latest_image = buffer.tobytes()  #  Cập nhật biến `latest_image`

    return fire_detected

#  Hàm lưu dữ liệu vào MySQL khi phát hiện đám cháy
def save_fire_detection():
    """Lưu dữ liệu nhận diện đám cháy vào MySQL"""
    try:
        # 📌 Tạo kết nối MySQL mới mỗi lần truy vấn
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123",
            database="fire_detection"
        )

        thoi_gian = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        noi_dung = "Nhận diện được đám cháy"

        with db.cursor() as cursor:
            sql = "INSERT INTO fire_detection (thoi_gian, noi_dung) VALUES (%s, %s)"
            cursor.execute(sql, (thoi_gian, noi_dung))
            db.commit()  # 🔄 Đảm bảo lưu thay đổi vào MySQL

        print(f"🔥 Đã lưu vào MySQL: {thoi_gian} - {noi_dung}")

        db.close()  # 🔄 Đóng kết nối sau khi hoàn thành truy vấn

    except mysql.connector.Error as err:
        print(f"❌ Lỗi khi lưu vào MySQL: {err}")


#  Xử lý dữ liệu MQTT nhận được
def on_message(client, userdata, msg):
    """Nhận ảnh từ MQTT, ghép ảnh và chạy nhận diện"""
    global image_data, latest_image, total_parts

    message = msg.payload.decode()

    if message == "end":
        if total_parts is not None and len(image_data) == total_parts:
            try:
                #  Ghép các phần ảnh lại
                full_image_data = "".join(image_data[i] for i in sorted(image_data.keys()))
                full_image_data = fix_base64_padding(full_image_data)

                #  Giải mã ảnh
                image_bytes = base64.b64decode(full_image_data)
                np_arr = np.frombuffer(image_bytes, np.uint8)
                current_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if current_image is not None:
                    print(" 🔥 Ảnh nhận thành công! Chạy nhận diện đám cháy...")

                    #  Nhận diện đám cháy
                    fire_detected = detect_fire(current_image)

                    #  Nếu phát hiện cháy, lưu vào MySQL
                    if fire_detected:
                        save_fire_detection()
                        print(" Phát hiện đám cháy! Lưu vào MySQL.")

            except Exception as e:
                print(f" Lỗi khi xử lý ảnh: {e}")

        # 🧹 Xóa bộ nhớ sau khi xử lý xong
        image_data.clear()
        total_parts = None

    else:
        try:
            #  Ghép từng phần ảnh
            index, part = message.split(":", 1)
            part_index, total = map(int, index.split("/"))
            total_parts = total
            image_data[part_index] = part

        except Exception as e:
            print(f" Lỗi khi xử lý phần ảnh: {e}")


#  Kết nối MQTT
client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe(MQTT_TOPIC)
client.loop_start()


#  Route Flask hiển thị giao diện chính
@app.route('/')
def index():
    """Hiển thị trang web chính"""
    try:
        # 📌 Tạo kết nối MySQL mới mỗi lần truy vấn
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123",
            database="fire_detection"
        )

        with db.cursor() as cursor:
            cursor.execute("SELECT * FROM fire_detection ORDER BY stt DESC")
            data = cursor.fetchall()  # 🔄 Đọc toàn bộ dữ liệu

        db.close()  # 🔄 Đóng kết nối sau khi hoàn thành truy vấn

        return render_template('index.html', data=data)

    except mysql.connector.Error as err:
        print(f"❌ Lỗi khi lấy dữ liệu từ MySQL: {err}")
        return "Lỗi kết nối cơ sở dữ liệu!"
def generate():
    """Gửi ảnh liên tục đến trình duyệt"""
    global latest_image
    while True:
        if latest_image:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_image + b'\r\n')
        else:
            print("⚠️ Chưa có ảnh để hiển thị...")  # 📌 Debug nếu Flask không có ảnh
        time.sleep(0.1)  # 📌 Giảm tải CPU
#  Route Flask hiển thị ảnh nhận diện
@app.route('/esp_feed')
def esp_feed():
    """Truyền hình ảnh đã nhận diện lên web"""
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# 🌎 API trả về dữ liệu nhận diện đám cháy dưới dạng JSON
@app.route('/get_fire_detection')
def get_fire_detection():
    """API trả về dữ liệu nhận diện đám cháy"""
    try:
        # 📌 Tạo kết nối MySQL mới mỗi lần truy vấn
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123",
            database="fire_detection"
        )

        with db.cursor() as cursor:
            cursor.execute("SELECT * FROM fire_detection ORDER BY stt DESC")
            data = cursor.fetchall()  # 🔄 Đọc toàn bộ kết quả

        db.close()  # 🔄 Đóng kết nối sau khi hoàn thành truy vấn

        return jsonify(data)

    except mysql.connector.Error as err:
        print(f"❌ Lỗi khi lấy dữ liệu từ MySQL: {err}")
        return jsonify({"error": "Lỗi kết nối cơ sở dữ liệu!"})
@app.route('/get_chart_data')
def get_chart_data():
    """API trả về dữ liệu thống kê theo từng giờ cho tất cả các ngày"""
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123",
            database="fire_detection"
        )

        with db.cursor() as cursor:
            # 📌 Lấy dữ liệu theo từng giờ, không giới hạn ngày
            cursor.execute("""
                SELECT DATE_FORMAT(thoi_gian, '%Y-%m-%d %H:00:00') AS gio, COUNT(*) AS so_lan
                FROM fire_detection
                GROUP BY DATE_FORMAT(thoi_gian, '%Y-%m-%d %H:00:00')
                ORDER BY gio
            """)
            data = cursor.fetchall()

        db.close()

        if not data:
            return jsonify({"timestamps": [], "counts": []})  # 📌 Trả về mảng rỗng nếu không có dữ liệu

        timestamps = [str(row[0]) for row in data]  # Thời gian theo giờ (YYYY-MM-DD HH:00:00)
        counts = [row[1] for row in data]  # Số lần nhận diện mỗi giờ

        return jsonify({"timestamps": timestamps, "counts": counts})

    except mysql.connector.Error as err:
        print(f"❌ Lỗi khi lấy dữ liệu biểu đồ: {err}")
        return jsonify({"error": "Lỗi kết nối cơ sở dữ liệu!"})
# 🚀 Chạy Flask Server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
