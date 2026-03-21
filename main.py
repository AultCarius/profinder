from ui.app import app
from modules.asr import start_mic_stream, start_vad_thread

if __name__ == '__main__':
    start_mic_stream()
    start_vad_thread()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)