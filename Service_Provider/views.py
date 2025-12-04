from django.conf import settings
from django.db.models import Avg, Count, Q
from django.http import HttpResponse
from django.shortcuts import render, redirect
import csv
import os
import re
import warnings

warnings.filterwarnings('ignore')

from Remote_User.models import (
    ClientRegister_Model,
    Tweet_Prediction_model,
    detection_accuracy_model,
    detection_ratio_model,
)


def serviceproviderlogin(request):
    if request.method == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password == "Admin":
            detection_accuracy_model.objects.all().delete()
            return redirect('View_Remote_Users')

    return render(request, 'SProvider/serviceproviderlogin.html')


def Find_Cyberbullying_Prediction_Ratio(request):
    detection_ratio_model.objects.all().delete()
    total_predictions = Tweet_Prediction_model.objects.count()
    if total_predictions:
        labels = [
            'Non Offensive or Non Cyberbullying',
            'Offensive or Cyberbullying',
        ]
        for label in labels:
            count = Tweet_Prediction_model.objects.filter(Prediction_Type=label).count()
            ratio = (count / total_predictions) * 100
            if ratio:
                detection_ratio_model.objects.create(names=label, ratio=ratio)

    obj = detection_ratio_model.objects.all()
    return render(request, 'SProvider/Find_Cyberbullying_Prediction_Ratio.html', {'objs': obj})


def View_Remote_Users(request):
    obj = ClientRegister_Model.objects.all()
    return render(request, 'SProvider/View_Remote_Users.html', {'objects': obj})


def ViewTrendings(request):
    topic = (
        Tweet_Prediction_model.objects.values('topics')
        .annotate(dcount=Count('topics'))
        .order_by('-dcount')
    )
    return render(request, 'SProvider/ViewTrendings.html', {'objects': topic})


def charts(request, chart_type):
    chart1 = detection_ratio_model.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request, 'SProvider/charts.html', {'form': chart1, 'chart_type': chart_type})


def charts1(request, chart_type):
    chart1 = detection_accuracy_model.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request, 'SProvider/charts1.html', {'form': chart1, 'chart_type': chart_type})


def View_Cyberbullying_Predict_Type(request):
    obj = Tweet_Prediction_model.objects.all()
    return render(request, 'SProvider/View_Cyberbullying_Predict_Type.html', {'list_objects': obj})


def likeschart(request, like_chart):
    charts = detection_accuracy_model.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request, 'SProvider/likeschart.html', {'form': charts, 'like_chart': like_chart})


def Download_Cyber_Bullying_Prediction(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="Cyberbullying_Predicted_DataSets.csv"'
    writer = csv.writer(response)
    writer.writerow(['Tweet_Message', 'Prediction_Type'])

    for record in Tweet_Prediction_model.objects.all():
        writer.writerow([record.Tweet_Message, record.Prediction_Type])

    return response


def train_model(request):
    detection_accuracy_model.objects.all().delete()
    context = {}

    try:
        import pandas as pd
        from sklearn import svm
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import MultinomialNB
    except ModuleNotFoundError as exc:
        missing = getattr(exc, 'name', None) or str(exc)
        context['error'] = (
            f"Missing machine learning dependency: {missing}. "
            'Install the required package and try again.'
        )
        return render(request, 'SProvider/train_model.html', context)

    dataset_path = os.path.join(settings.BASE_DIR, 'train_tweets.csv')
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        context['error'] = 'train_tweets.csv dataset not found.'
        return render(request, 'SProvider/train_model.html', context)

    def process_tweet(tweet):
        return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ", tweet.lower()).split())

    df['processed_tweets'] = df['tweet'].apply(process_tweet)
    cnt_non_fraud = df[df['label'] == 0]['processed_tweets'].count()
    if cnt_non_fraud == 0:
        context['error'] = 'Training data does not contain non-offensive tweets.'
        return render(request, 'SProvider/train_model.html', context)

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
    detection_accuracy_model.objects.create(names='SVM', ratio=svm_acc)
    svm_cm = confusion_matrix(y_test, predict_svm)

    logreg = LogisticRegression(random_state=42)
    logreg.fit(x_train_tfidf, y_train)
    predict_log = logreg.predict(x_test_tfidf)
    logistic = accuracy_score(y_test, predict_log) * 100
    detection_accuracy_model.objects.create(names='Logistic Regression', ratio=logistic)
    logistic_cm = confusion_matrix(y_test, predict_log)

    nb_clf = MultinomialNB()
    nb_clf.fit(x_train_tfidf, y_train)
    predict_nb = nb_clf.predict(x_test_tfidf)
    naivebayes = accuracy_score(y_test, predict_nb) * 100
    detection_accuracy_model.objects.create(names='Naive Bayes', ratio=naivebayes)
    nb_cm = confusion_matrix(y_test, predict_nb)

    test_dataset_path = os.path.join(settings.BASE_DIR, 'test_tweets.csv')
    try:
        df_test = pd.read_csv(test_dataset_path)
    except FileNotFoundError:
        context['error'] = 'test_tweets.csv dataset not found.'
        return render(request, 'SProvider/train_model.html', context)

    df_test['processed_tweets'] = df_test['tweet'].apply(process_tweet)
    test_counts = count_vect.transform(df_test['processed_tweets'])
    test_tfidf = transformer.transform(test_counts)
    df_test['predict_nb'] = nb_clf.predict(test_tfidf)
    df_test['predict_svm'] = lin_clf.predict(test_tfidf)
    predictions_path = os.path.join(settings.BASE_DIR, 'Predictions.csv')
    df_test.to_csv(predictions_path, index=False)

    obj = detection_accuracy_model.objects.all()
    context.update({
        'objs': obj,
        'svmcm': svm_cm,
        'lrcm': logistic_cm,
        'nbcm': nb_cm,
    })
    return render(request, 'SProvider/train_model.html', context)
