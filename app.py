from flask import Flask, render_template, Response, request, jsonify
import time

app = Flask(__name__)
cam = None

def get_camera():
    global cam
    if cam is None:
        from camera import VideoCamera
        cam = VideoCamera()
        cam.start() # <--- IMPORTANT: STARTS THE BACKGROUND THREAD
        # Wait a moment for the thread to fill the buffer
        time.sleep(1.0) 
    return cam

@app.route('/')
def index():
    camera = get_camera()
    return render_template('index.html', hsv=camera.hsv_values, presets=camera.presets, port=camera.current_port)

def gen_main(camera):
    while True:
        # Just grab the bytes, don't trigger processing
        frame = camera.get_frame_bytes(feed_type='main')
        if frame:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        time.sleep(0.016) # Limit to ~60 FPS so we don't spam the browser

def gen_mask(camera):
    while True:
        frame = camera.get_frame_bytes(feed_type='mask')
        if frame:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        time.sleep(0.016)

@app.route('/video_feed_main')
def video_feed_main():
    return Response(gen_main(get_camera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_mask')
def video_feed_mask():
    return Response(gen_mask(get_camera()), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- API ENDPOINTS ---
@app.route('/toggle_tracking', methods=['POST'])
def toggle_tracking():
    state = request.json.get('active')
    get_camera().tracking_active = state
    return jsonify(success=True, state=state)

@app.route('/update_hsv', methods=['POST'])
def update_hsv():
    data = request.json
    get_camera().set_hsv(data['key'], int(data['value']))
    return jsonify(success=True)

@app.route('/set_port', methods=['POST'])
def set_port():
    port = request.json.get('port')
    success, msg = get_camera().connect_serial(port)
    return jsonify(success=success, msg=msg)

@app.route('/switch_mode', methods=['POST'])
def switch_mode():
    mode = request.json.get('mode')
    get_camera().mode = mode
    return jsonify(success=True, mode=mode)

@app.route('/preset', methods=['POST'])
def handle_preset():
    action = request.json.get('action')
    slot = int(request.json.get('slot'))
    name = request.json.get('name', '')
    cam = get_camera()
    if action == 'save':
        cam.save_preset(slot, name)
        return jsonify(success=True, presets=cam.presets)
    elif action == 'load':
        new_values = cam.load_preset(slot)
        return jsonify(success=True, values=new_values)
    return jsonify(success=False)

if __name__ == '__main__':
    # use_reloader=False is mandatory here because of the background thread
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)