from django.http.response import StreamingHttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.conf import settings
from django.urls import reverse_lazy, reverse
from .forms import CreateForm,SettingForm
from .owner import OwnerListView, OwnerDetailView, OwnerCreateView, OwnerUpdateView, OwnerDeleteView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse
from django.conf import settings
import cv2
# Create your views here.
from .models import Cam
#from .detect import Detect

import numpy as np

import os.path
from os import path

# This is a little complex because we need to detect when we are
# running in various configurations
flagW =dict()
flagW[0]=None

import datetime,pytz
import cv2
import time
from datetime import date
import numpy as np
import imutils




class YoloDetect():
    def __init__(self,url):
        self._config     = '/home/tan/Desktop/final_App/App/camera/model_yolov3tiny/yolov3-tiny.cfg'
        self._weights    = '/home/tan/Desktop/final_App/App/camera/model_yolov3tiny/yolov3-tiny_best.weights'
        self._classes    = '/home/tan/Desktop/final_App/App/camera/model_yolov3tiny/yolov3-tiny.names'
        self._net        = cv2.dnn.readNet(self._weights, self._config)
        self.url         = url
    def Read_Camera(self):
        return cv2.VideoCapture(self.url)

    def Read_ClassName(self):
        classes=None
        with open(self._classes, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def get_predict(self,image):
        scale = 0.00392
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        self._net.setInput(blob)
        return  self._net.forward(self.get_output_layers())

    def get_output_layers(self):
        net=self._net
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers
 
    def draw_prediction(self,COLORS,img,confidence ,class_id, x, y, x_plus_w, y_plus_h):
        label = str(self.Read_ClassName()[class_id])+"-"+str(round(confidence,2))
        color = COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 1)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img
        
    def Put_time_date_FPS(self,image,count,start):
        dtobj1          =   datetime.datetime.utcnow()   #utcnow class method
        dtobj3          =   dtobj1.replace(tzinfo=pytz.UTC) #replace method
        dtobj_hongkong  =   dtobj3.astimezone(pytz.timezone("Asia/Ho_Chi_Minh")) #astimezone method
        date_time       =   str(dtobj_hongkong)
        date            =   date_time.split(" ")[0]
        s               =   [i for i in date.split("-")]
        date            =   s[2]+"-"+s[1]+"-"+s[0]
        current_time    =   date_time.split(" ")[1].split("+")[0].split(".")[0]
        fps             =   int(count/(time.time()-start))
        text_time       =   str(current_time)
        text_day        =   str(date)
        fps_str         =   "FPS:  "+str(fps)
        cv2.putText(image,text_day, (10,30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0),2)
        cv2.putText(image,text_time, (10,70), 
            cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0),2)
        cv2.putText(image,fps_str, (1030,30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0),2)
    
    def get_boxes_classid_conf(self,image):
        outs        = self.get_predict(image)
        Width       = image.shape[1]
        Height      = image.shape[0]
        class_ids   = []
        confidences = []
        boxes       = []
        for out in outs:
            for detection in out:
                scores      = detection[5:]
                class_id    = np.argmax(scores)
                confidence = scores[class_id]
                if (confidence > 0.5):
                    center_x    = int(detection[0] * Width)
                    center_y    = int(detection[1] * Height)
                    w           = int(detection[2] * Width)
                    h           = int(detection[3] * Height)
                    x           = center_x - w / 2
                    y           = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])                  
        return boxes,class_ids,confidences  
        
from .models import Cam

def Detect(file,outname,x=None,y=None,w=None,h=None,pk=None,bs=None,flagW=None):
    flag=False
    x1,x2,y1,y2=None,None,None,None
    if(x!=None and y !=None and w != None and h != None):
        flag = True
        x1 = float(x)*5/3
        y1 = float(y)*5/3
        x2 = x1 + float(w)*5/3
        y2 = y1 + float(h)*5/3
    
    outname="image/"+outname+"-"+"show"+".png"
    Detector = YoloDetect(file)
    cap  = Detector.Read_Camera()
    classes = Detector.Read_ClassName()
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    start =time.time()
    count =0
    try:
        while (True): 
            # Doc frame
            ret,frame = cap.read()
            image = imutils.resize(frame, width=1200)
            count+=1
            if ret and count%bs==0:
                a = time.time()
                conf_threshold = 0.4
                nms_threshold = 0.4
                boxes,class_ids,confidences = Detector.get_boxes_classid_conf(image)
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
                if (flag == True):
                    start_point,end_point=(int(x1),int(y1)),(int(x2),int(y2))
                    cv2.rectangle(image,start_point,end_point,(0,0,255),4)
                Detector.Put_time_date_FPS(image,count,start)
                if indices == ():
                    flagW[pk]=0
                for i in indices:
                    i = i[0]
                    box = boxes[i]
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]
                    if (flag==True):
                        if(overlap([x1,y1,x2,y2],box)):
                            flagW[pk]=1
                        else:
                            flagW[pk]=0
                    if class_ids[i]==1:
                        Detector.draw_prediction((0, 255, 255),image,confidences[i],class_ids[i], round(x), round(y), round(x + w), round(y + h))
                    else:
                        Detector.draw_prediction((30,0,250),image,confidences[i],class_ids[i], round(x), round(y), round(x + w), round(y + h))
                cv2.imwrite(outname, image)
                print(time.time()-a)
                yield (b'--frame\r\n'
                    b'Content-Type: image/png\r\n\r\n' + open(outname, 'rb').read() + b'\r\n')
            
    except:
        print("ERR STREAM VIDEO")


def overlap(area,object):
    x1=object[0]-object[2]/2
    y1=object[1]-object[3]/2
    x2=object[0]+object[2]/2
    y2=object[1]+object[3]/2
    conf1 = area[0] < x2
    conf2 = area[2] > x1
    conf3 = area[1] < y2
    conf4 = area[3] > y1
    if conf1 and conf2 and conf3 and conf4:
        return True
    else:
        return False 






class CamListView(OwnerListView):
    model=Cam

class CamCreateView(LoginRequiredMixin, View):
    template_name = 'camera/cam_form.html'
    success_url = reverse_lazy('cams:all')

    def get(self, request, pk=None):
        form = CreateForm()
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        form = CreateForm(request.POST, request.FILES or None)
        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        # Add owner to the model before saving
        cam = form.save(commit=False)
        cam.owner = self.request.user
        cam.image_name = settings.MEDIA_ROOT+"/save/"+cam.picture.name
        cam.save()
        img = cv2.imread(cam.image_name)
        height, width, channels = img.shape
        cam.width =width
        cam.height = height
        cam.save()
        return redirect(self.success_url)

class CamDeleteView(OwnerDeleteView):
    model =Cam
    success_url = reverse_lazy('cams:all')
    template_name = "camera/cam_confirm_delete.html"
    def get(self,request,pk=None):
        return render(request, self.template_name)
    def post(self,request,pk):
        instance = Cam.objects.get(id=pk)
        file = settings.MEDIA_ROOT+"/"+instance.picture.name
        if path.exists(file):
            os.remove(file)
        ip = instance.ip
        name_stream =settings.STREAM_ROOT+"/"+str(instance.owner)+"-"+str(pk)+"-"+"show"+".png"
        if path.exists(name_stream):
            os.remove(name_stream)
        instance.delete()
        return redirect(self.success_url)

class CamUpdateView(LoginRequiredMixin, View):
    template_name = 'camera/cam_form.html'
    success_url = reverse_lazy('cams:all')

    def get(self, request, pk):
        cam = get_object_or_404(Cam, id=pk, owner=self.request.user)
        form = CreateForm(instance=cam)
        x = Cam.objects.get(id=pk)
        ctx = {'form': form,'cam':x}
        return render(request, self.template_name, ctx)

    def post(self, request, pk):
        cam = get_object_or_404(Cam, id=pk, owner=self.request.user)
        form = CreateForm(request.POST, request.FILES or None, instance=cam)
        x = Cam.objects.get(id=pk)
        if not form.is_valid():
            ctx = {'form': form,"cam":x}
            return render(request, self.template_name, ctx)
        cam = form.save(commit=False)
        temp = cam.picture.path
        cam.save()
        if cam.picture.name.split("/")[0]=='save':
            cam.image_name = settings.MEDIA_ROOT+"/"+cam.picture.name
        else:
            cam.image_name = settings.MEDIA_ROOT+"/save/"+cam.picture.name
        img = cv2.imread(cam.image_name)
        height, width, channels = img.shape
        cam.width =width
        cam.height = height
        cam.save()
        return redirect(self.success_url)

'''def stream_file(request, pk):
    cam = get_object_or_404(Cam, id=pk)
    response = HttpResponse()
    response['Content-Type'] = cam.content_type
    response['Content-Length'] = len(cam.picture)
    response.write(cam.picture)
    return response'''

class CamDetailView(OwnerDetailView):
    model= Cam
    success_url = reverse_lazy('cams:all')
    template_name = "camera/cam_detail.html"
    def get(self, request, pk) :
        cam = get_object_or_404(Cam,id=pk,owner=self.request.user)
        form =SettingForm(instance=cam)
        x = Cam.objects.get(id=pk)
        ctx ={'form': form,'cam' : x}
        return render(request, self.template_name, ctx)
        
    def post(self,request,pk):
        cam = get_object_or_404(Cam, id=pk, owner=self.request.user)
        form = SettingForm(request.POST, instance=cam)
        x = Cam.objects.get(id=pk)
        if not form.is_valid():
            ctx = {'form': form,'cam' : x}
            return render(request, self.template_name, ctx)
        cam = form.save(commit=False)
        cam.save()
        return redirect(self.success_url)

class MainViewCamera(LoginRequiredMixin,View):
    template_name = 'camera/cam_view.html'

    def get(self,request,pk):
        cam = get_object_or_404(Cam,id=pk,owner=self.request.user)
        x = Cam.objects.get(id=pk)
        ctx ={'cam' : x}
        return render(request,self.template_name,ctx)

def Stream_video(request,pk):
    cam = get_object_or_404(Cam, id=pk)
    ip = cam.ip
    x  = cam.x_pos
    y  = cam.y_pos
    w  = cam.w_pos
    h  = cam.h_pos
    bs = cam.bias
    if bs == None:
        bs = 5
    bs = int(bs)
    name =str(cam.owner)+"-"+str(pk)
    return (StreamingHttpResponse(Detect(ip,name,x,y,w,h,pk,bs,flagW), content_type='multipart/x-mixed-replace; boundary=frame'))

import time
import datetime
def stream(request,pk):
    def event_stream():
        while True:
            try:
                time.sleep(0.001)
                if flagW[pk]==1:
                    yield 'event: greet\ndata:Warning\n\n'
                    
                else:
                    yield 'event: greet\ndata:\n\n'
            except:
                pass
    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')
  
