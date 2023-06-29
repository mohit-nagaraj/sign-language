from flask import Flask,render_template,url_for,flash,redirect,Response,request
from flask_sse import sse

import cv2,time

from main_model import big_one

cv_bro = big_one.SIGN_MODEL()


app = Flask(__name__)
app.config['REDIS_URL'] = 'redis://localhost:6379'
app.register_blueprint(sse,url_prefix='/stream')

camera_active = False
camera = None
text = ''
lp,mop = r'D:\keypoint_classifier_label.csv',r'D:\keypoint_classifier.tflite'

def start_camera():
    global camera_active, camera
    if not camera_active:
        camera = cv2.VideoCapture(0)
        camera_active = True

def stop_camera():
    global camera_active, camera
    if camera_active:
        camera.release()
        camera_active = False

def get_frame():
    global text
    while camera_active:
        success, frame = camera.read()
        if success:
            j,frame = cv_bro.model_runner(frame)
            if j!='':
                with app.app_context():
                    text = j
                    j = ''
                    sse.publish({'message':text},type='event_type')
            ret, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
             b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@app.route('/train_your_model',methods=['POST','GET'])
def train_your_model():
    if camera_active: stop_camera()
    return render_template('train_your_model.html')

@app.route('/')
@app.route('/home')
def home():
    if camera_active: stop_camera()
    return render_template('home.html')

@app.route('/live')
def live():
    global text
    if not cv_bro.real: cv_bro.reset(cv_bro.mop,cv_bro.lp)
    text = ''
    start_camera()
    return render_template('live.html',url=url_for('train_your_model'),namer='train your translater')


@app.route('/contact_us',methods=['POST','GET'])
def contact_us():
    if camera_active:stop_camera()
    return render_template('contact_us.html')

@app.route('/about_us')
def about_us():
    if camera_active: stop_camera()
    return render_template('about_us.html')

@app.route('/my_model',methods=['POST','GET'])
def my_model():
    lpp,mopp = request.files['label'],request.files['tflite']
    lpp.save(lp),mopp.save(mop)
    cv_bro.reset(mop,lp)
    return render_template('live.html',url=url_for('live'),namer='App')


#######################################################
#not visiting

@app.route('/video_feed')
def video_feed():
    if not camera_active: start_camera()
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)