import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask, render_template,request, flash ,send_from_directory
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
app=Flask(__name__)
app.config['SECRET_KEY']="twitter"

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/about', methods=['POST','GET'])
def about():

    return render_template('about.html')


@app.route('/soilprediction', methods=['POST','GET'])
def soilprediction():
    global x_train, x_test, y_train, y_test, x, y

    if request.method == 'POST':
        df = pd.read_csv(r'datasets\data.csv')
        le = LabelEncoder()
        df['Output'] = le.fit_transform(df['Output'])
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=72)

        pH = float(request.form['pH'])
        EC = float(request.form['EC'])
        OC = float(request.form['OC'])
        OM = float(request.form['OM'])
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        Zn = float(request.form['Zn'])
        Fe = float(request.form['Fe'])
        Cu = float(request.form['Cu'])
        Mn = float(request.form['Mn'])
        Sand = float(request.form['Sand'])
        Silt = float(request.form['Silt'])
        Clay = float(request.form['Clay'])
        CaCO3 = float(request.form['CaCO3'])
        CEC = float(request.form['CEC'])

        PRED = [[pH, EC, OC, OM, N, P, K, Zn, Fe, Cu, Mn, Sand, Silt, Clay, CaCO3, CEC]]

        knn = RandomForestClassifier()
        knn.fit(x_train, y_train)
        xgp = np.array(knn.predict(PRED))

        if xgp == 0:
            flash(' This prediction result is : Non Fertile', "success")
        elif xgp == 1:
            flash(' This prediction result is : Fertile', "warning")
        return render_template('soilprediction.html')

    return render_template('soilprediction.html')



@app.route('/plantprediction', methods=['POST','GET'])
def plantprediction():

    return render_template('plantprediction.html')



@app.route('/croppredictiopn',methods=['POST','GET'])
def croppredictiopn():
    df = pd.read_csv("datasets/Crop_recommendation.csv")
    global x_trains, y_trains
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    x = df.drop(['label'], axis=1)
    y = df['label']

    x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, stratify=y, test_size=0.3, random_state=42)

    if request.method == "POST":
        f1 = float(request.form['N'])
        print(f1)
        f2 = float(request.form['P'])
        print(f2)
        f3 = float(request.form['K'])
        print(f3)
        f4 = float(request.form['temperature'])
        print(f4)
        f5 = float(request.form['humidity'])
        print(f5)
        f6 = float(request.form['ph'])
        print(f6)
        f7 = float(request.form['rainfall'])
        print(f7)

        li = [[f1, f2, f3, f4, f5, f6, f7]]
        print(li)

        logistic = RandomForestClassifier()
        logistic.fit(x_trains, y_trains)

        result = logistic.predict(li)
        result = result[0]
        if result == 0:
            flash('The Recommended Crop is predicted as Apple', "info")
        elif result == 1:
            flash('The Recommended Crop is predicted as Banana', "info")
        elif result == 2:
            flash('The Recommended Crop is predicted as Blackgram', "info")
        elif result == 3:
            flash('The Recommended Crop is predicted as Chickpea', "info")
        elif result == 4:
            flash('The Recommended Crop is predicted as Coconut', "info")
        elif result == 5:
            flash('The Recommended Crop is predicted as Coffee', "info")
        elif result == 6:
            flash('The Recommended Crop is predicted as Cotton', "info")
        elif result == 7:
            flash('The Recommended Crop is predicted as Grapes', "info")
        elif result == 8:
            flash('The Recommended Crop is predicted as Jute', "info")
        elif result == 9:
            flash('The Recommended Crop is predicted as Kidneybeans', "info")
        elif result == 10:
            flash('The Recommended Crop is predicted as Lentil', "info")
        elif result == 11:
           flash('The Recommended Crop is predicted as Maize', "info")
        elif result == 12:
            flash('The Recommended Crop is predicted as Mango', "info")
        elif result == 13:
            flash('The Recommended Crop is predicted as Mothbeans', "info")
        elif result == 14:
            flash('The Recommended Crop is predicted as Moongbeans', "info")
        elif result == 15:
            flash('The Recommended Crop is predicted as Muskmelon', "info")
        elif result == 16:
            flash('The Recommended Crop is predicted as Orange', "info")
        elif result == 17:
            flash('The Recommended Crop is predicted as Papaya', "info")
        elif result == 18:
            flash('The Recommended Crop is predicted as Pigeonpeas', "info")
        elif result == 19:
            flash('The Recommended Crop is predicted as Pomegranate', "info")
        elif result == 20:
            flash('The Recommended Crop is predicted as Rice', "info")
        elif result == 21:
            flash('The Recommended Crop is predicted as Watermelon', "info")

        return render_template('croppredictiopn.html', )
    return render_template('croppredictiopn.html')

@app.route('/result',methods=['POST','GET'])
def result():
    if request.method == 'POST':
        myfile = request.files['file']
        fn = myfile.filename
        mypath = os.path.join('images/', fn)
        myfile.save(mypath)

        print("{} is the file name", fn)
        print("Accept incoming file:", fn)
        print("Save it to:", mypath)
        Plants = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                  'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Potato___Early_blight',
                  'Potato___healthy', 'Potato___Late_blight', 'Tomato___Bacterial_spot', 'Tomato___healthy',
                  'Tomato___Late_blight', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']
        new_model=load_model("models/FinalModel.h5")
        test_image = image.load_img(mypath, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image /= 255
        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        preds = Plants[np.argmax(result)]

        if preds == "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
            flash("Foliar fungicides can be used to manage gray leaf spot outbreaks","warning")

        elif preds == "Corn_(maize)___Common_rust_":
            flash("Use resistant varieties like DHM 103, Ganga Safed - 2 and avoid sowing of suceptable varieties like DHM 105","warning")

        elif preds == "Corn_(maize)___healthy":
            flash("Plant is Good no treatment required","warning")

        elif preds == "Corn_(maize)___Northern_Leaf_Blight":
            flash("Integration of early sowing, seed treatment and foliar spray with Tilt 25 EC (propiconazole) was the best combination in controlling maydis leaf blight and increasing maize yield","warning")

        elif preds == "Potato___Early_blight":
            flash( "Mancozeb and chlorothalonil are perhaps the most frequently used protectant fungicides for early blight management","warning")

        elif preds == "Potato___healthy":
            flash("Plant is Good no treatment required","warning")

        elif preds == "Potato___Late_blight":
            flash("Effectively managed with prophylactic spray of mancozeb at 0.25% followed by cymoxanil+mancozeb or dimethomorph+mancozeb at 0.3% at the onset of disease and one more spray of mancozeb at 0.25% seven days","warning")

        elif preds == "Tomato___Bacterial_spot":
            flash("When possible, is the best way to avoid bacterial spot on tomato. Avoiding sprinkler irrigation and cull piles near greenhouse or field operations, and rotating with a nonhost crop also helps control the disease","warning")

        elif preds == "Tomato___healthy":
            flash("Plant is Good no treatment required","warning")

        elif preds == "Tomato___Late_blight":
            flash("Ungicides that contain maneb, mancozeb, chlorothanolil, or fixed copper can help protect plants from late tomato blight","warning")

        else:
            flash("Homemade Epsom salt mixture. Combine two tablespoons of Epsom salt with a gallon of water and spray the mixture on the plant","warning")

        return render_template("result.html",text=preds,image_name=fn)

    return render_template('result.html')

@app.route('/result/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)
if __name__=='__main__':
    app.run(debug=True)