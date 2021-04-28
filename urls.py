from django.urls import path,reverse_lazy
from . import views
from django.contrib.auth import views as auth_views

app_name="cams"
urlpatterns = [
    path('',views.CamListView.as_view(),name='all'),
    path('cam/create', views.CamCreateView.as_view(success_url=reverse_lazy('cams:all')), name='cam_create'),
    path('cam/<int:pk>', views.CamDetailView.as_view(success_url=reverse_lazy('cams:all')), name='cam_detail'),
    path('cam/<int:pk>/update',views.CamUpdateView.as_view(success_url=reverse_lazy('cams:all')), name='cam_update'),
    path('cam/<int:pk>/delete', views.CamDeleteView.as_view(success_url=reverse_lazy('cams:all')), name='cam_delete'),
    
    path('stream/<int:pk>',views.Stream_video,name='video_stream'),
    path('cam/show/stream/<int:pk>',views.stream,name='stream'),
    path('cam/show/<int:pk>',views.MainViewCamera.as_view(),name="cam_show")
]
