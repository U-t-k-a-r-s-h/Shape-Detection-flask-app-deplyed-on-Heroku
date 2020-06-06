import csv
from flask import Flask, render_template, request, redirect, url_for, flash
import requests
from tensorflow.keras.models import load_model
import cv2
import os
import numpy as np
import imutils

class Pager(object):
    def __init__(self, count):
        self.count = count
        self.current = 0

    @property
    def next(self):
        n = self.current + 1
        if n > self.count-1:
            n -= self.count
        return n

    @property
    def prev(self):
        n = self.current - 1
        if n < 0 :
            n += self.count
        return n

def read_table(url):
    """Return a list of dict"""
    # r = requests.get(url)
    with open(url) as f:
        return [row for row in csv.DictReader(f.readlines())]


APPNAME = "Shape Detection"
STATIC_FOLDER = 'example'
TABLE_FILE = "example/fakecatalog.csv"

table = read_table(TABLE_FILE)
pager = Pager(len(table))

path ='example/images'

model = load_model('batch_norm_model2.h5')

IMG_SIZE=100

app = Flask(__name__, static_folder=STATIC_FOLDER)
app.config.update(
    APPNAME=APPNAME,
    )

@app.route('/')
def index():
    return redirect('/0')


@app.route('/<int:ind>/')
def image_view(ind=None):
    if ind >= pager.count:
        return render_template("404.html"), 404
    else:
        pager.current = ind
        table[pager.current]['detection']=str()
        table[pager.current]['confidence']=str()
        #print(table[1]['name'])
        return render_template(
            'imageview.html',
            index=ind,
            pager=pager,
            data=table[ind])

@app.route('/goto', methods=['POST', 'GET'])    
def goto():
    return redirect('/' + request.form['index'])

@app.route('/Architecture', methods=['POST', 'GET'])    
def Architecture():
    return render_template('Architecture.html')


@app.route('/detect', methods=['POST', 'GET'])
def detect():
    global model
    image_Data=[]
    CATAGORIES = ["square","triangle","star","circle"]
    texxt = str(table[pager.current]['name']+'.jpg')
    img1 = cv2.imread(os.path.join(path,texxt),cv2.IMREAD_GRAYSCALE)
    new_array2 = cv2.resize(img1,(100,100))
    img_array2 = cv2.threshold(new_array2,200,255,cv2.THRESH_BINARY)[1]
    image_Data.append([img_array2])
    x2=[]
    for feature in image_Data:
        x2.append(feature)
    x2= np.array(x2).reshape(-1,100,100,1) 
    x2=x2/255
    a=model.predict(x2)[0]
    b=int(np.where(a==a.max())[0])
    table[pager.current]['detection']=str(CATAGORIES[b])
    table[pager.current]['confidence']=float(a[b]*100)
    return render_template(
            'imageview.html',
            index=pager.current,
            pager=pager,
            data=table[pager.current])


@app.route('/detect_cv', methods=['POST', 'GET'])
def detect_cv():
    class ShapeDetector:
        def __init__(self):
            pass
        def detect1(self,c):
            shape = "unidentified"
            peri = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,0.04*peri,True)
            if len(approx)==3:
                shape = "triangle"
            elif len(approx)==4:
                (x,y,w,h) = cv2.boundingRect(approx)
                ar = w/float(h)
                shape = "square" if ar >=0.95 and ar<=1.05 else "rectangle"
            elif len(approx) == 10:
                shape = "star"
            else :
                shape = "circle"
            return shape
    texxt = str(table[pager.current]['name']+'.jpg')
    img3 = cv2.imread(os.path.join(path,texxt))
    img3 = cv2.resize(img3,(250,200))
    ratio = img3.shape[0] / float(img3.shape[0])
    gray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray ,(5,5),0)
    thresh = cv2.threshold(blurred,220,255,cv2.THRESH_BINARY)[1]
    thresh = cv2.Canny(thresh,0,255)
    cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sd = ShapeDetector()
    shape=sd.detect1(cnts[0])
    table[pager.current]['detection']=str(shape)
    table[pager.current]['confidence']='N/A'
    return render_template(
            'imageview.html',
            index=pager.current,
            pager=pager,
            data=table[pager.current])

if __name__ == '__main__':
    app.run(debug=False)
