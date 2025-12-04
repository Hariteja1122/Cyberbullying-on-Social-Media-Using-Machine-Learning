from django.conf import settings
from django.db.models import Count, Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import warnings
import re

warnings.filterwarnings('ignore')

# Create your views here.
from Remote_User.models import ClientRegister_Model, Tweet_Message_model, Tweet_Prediction_model, detection_ratio_model, detection_accuracy_model

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('Search_DataSets')
        except:
            pass

    return render(request,'RUser/login.html')

def Add_DataSet_Details(request):


    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Search_DataSets(request):
    context = {}
    if request.method == "POST":
        tweet_message = (request.POST.get('keyword') or "").strip()
        if not tweet_message:
            context['error'] = "Please provide a tweet message to analyse."
            return render(request, 'RUser/Search_DataSets.html', context)

        try:
            import os
            import pandas as pd
            from sklearn import svm
            from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import train_test_split
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", None) or str(exc)
            context['error'] = (
                f"Missing machine learning dependency: {missing}. "
                "Install the required package and try again."
            )
            return render(request, 'RUser/Search_DataSets.html', context)

        dataset_path = os.path.join(settings.BASE_DIR, "train_tweets.csv")
        df = pd.read_csv(dataset_path)

        def process_tweet(tweet):
            return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ", tweet.lower()).split())

        df['processed_tweets'] = df['tweet'].apply(process_tweet)
        cnt_non_fraud = df[df['label'] == 0]['processed_tweets'].count()
        df_class_fraud = df[df['label'] == 1]
        df_class_nonfraud = df[df['label'] == 0]
        df_class_fraud_oversample = df_class_fraud.sample(cnt_non_fraud, replace=True)
        df_oversampled = pd.concat([df_class_nonfraud, df_class_fraud_oversample], axis=0)

        X = df_oversampled['processed_tweets']
        y = df_oversampled['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=None)
        count_vect = CountVectorizer(stop_words='english')
        transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
        x_train_counts = count_vect.fit_transform(X_train)
        x_train_tfidf = transformer.fit_transform(x_train_counts)
        x_test_counts = count_vect.transform(X_test)
        x_test_tfidf = transformer.transform(x_test_counts)

        lin_clf = svm.LinearSVC()
        lin_clf.fit(x_train_tfidf, y_train)
        predict_svm = lin_clf.predict(x_test_tfidf)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        detection_accuracy_model.objects.create(names="SVM", ratio=svm_acc)

        review_data = [tweet_message]
        vector1 = count_vect.transform(review_data).toarray()
        predict_text = lin_clf.predict(vector1)

        pred = str(predict_text).replace("[", "").replace("]", "")

        try:
            prediction = int(pred)
        except ValueError:
            prediction = -1

        if prediction == 0:
            val = 'Non Offensive or Non Cyberbullying'
        elif prediction == 1:
            val = 'Offensive or Cyberbullying'
        else:
            val = 'Prediction unavailable'

        Tweet_Prediction_model.objects.create(Tweet_Message=tweet_message, Prediction_Type=val)
        context['objs'] = val
        return render(request, 'RUser/Search_DataSets.html', context)
    return render(request, 'RUser/Search_DataSets.html')



