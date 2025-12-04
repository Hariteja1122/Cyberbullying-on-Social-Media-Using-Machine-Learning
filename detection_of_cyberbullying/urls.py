"""detection_of_cyberbullying URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path, re_path
from django.contrib import admin
from Remote_User import views as remoteuser
from detection_of_cyberbullying import settings
from Service_Provider import views as serviceprovider
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', remoteuser.login, name="login"),
    path('Register1/', remoteuser.Register1, name="Register1"),
    path('Search_DataSets/', remoteuser.Search_DataSets, name="Search_DataSets"),
    path('ViewYourProfile/', remoteuser.ViewYourProfile, name="ViewYourProfile"),
    path('Add_DataSet_Details/', remoteuser.Add_DataSet_Details, name="Add_DataSet_Details"),
    path('serviceproviderlogin/', serviceprovider.serviceproviderlogin, name="serviceproviderlogin"),
    path('View_Remote_Users/', serviceprovider.View_Remote_Users, name="View_Remote_Users"),
    re_path(r'^charts/(?P<chart_type>\w+)$', serviceprovider.charts, name="charts"),
    re_path(r'^charts1/(?P<chart_type>\w+)$', serviceprovider.charts1, name="charts1"),
    re_path(r'^likeschart/(?P<like_chart>\w+)$', serviceprovider.likeschart, name="likeschart"),
    path('Find_Cyberbullying_Prediction_Ratio/', serviceprovider.Find_Cyberbullying_Prediction_Ratio, name="Find_Cyberbullying_Prediction_Ratio"),
    path('train_model/', serviceprovider.train_model, name="train_model"),
    path('View_Cyberbullying_Predict_Type/', serviceprovider.View_Cyberbullying_Predict_Type, name="View_Cyberbullying_Predict_Type"),
    path('Download_Cyber_Bullying_Prediction/', serviceprovider.Download_Cyber_Bullying_Prediction, name="Download_Cyber_Bullying_Prediction"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
