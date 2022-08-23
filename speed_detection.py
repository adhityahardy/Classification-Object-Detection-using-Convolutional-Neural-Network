import os
import requests
# import SpeedRadar2
from flask import Flask, render_template, request, flash, url_for
from flask_mysqldb import (
    MySQL,
    MySQLdb,
)
# from tracker2 import *
from datetime import datetime
from requests import models
from werkzeug.utils import redirect, secure_filename

app = Flask(__name__)
url = "https://e27faf2cc66a42fbad2dce18747962e5.apig.cn-north-4.huaweicloudapis.com/v1/infers/49613a0c-7d1b-435a-940d-f22b317c9d6c"
headers = {
    "X-Apig-AppCode": "0546adf324844fb1a20df6bd01fdc504c48e44fb29f04e66a698799acde59573"
}

app.secret_key = "huawei"

app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = ""
app.config["MYSQL_DB"] = "test"
app.config["MYSQL_CURSORCLASS"] = "DictCursor"
mysql = MySQL(app)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET"])
def index():
    return render_template("Test_2.html")


@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == "POST":
        files = request.files.getlist("files[]")
        for file in files:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
    return redirect("/")


# @app.route("/recognize", methods=["POST", "GET"])
# def call_modelArts():
    cursor = mysql.connection.cursor()
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    now = datetime.now()
    if request.method == "POST":
        f = request.files["imgFilename"]
        filename = secure_filename(f.filename)
        print("recognize: " + f.filename)
        files = {"images": (f.filename, f.read(), f.content_type)}
        resp = requests.post(url, headers=headers, files=files)
        print("Result: " + resp.text)
        jsonResult = resp.json()
        result = jsonResult["predicted_label"]
        arScores = jsonResult["scores"]
        predicted_score = 0
        for score in arScores:
            if score[0] == result:
                predicted_score = float(score[1])
        print("Result: %s : predicted: %.2f" % (result, predicted_score))
        if resp.status_code == 200:
            strStatus = "Success"
        else:
            strStatus = "Failed"
        cur.execute(
            "INSERT INTO images (file_name, uploaded_on, result, scores) VALUES (%s, %s, %s, %s)",
            [filename, now, result, predicted_score],
        )
        mysql.connection.commit()
        cur.execute("""SELECT * FROM IMAGES""")
        hasil = cur.fetchall()
        cur.close()
        return render_template(
            "TestResult.html",
            weather=result,
            status=strStatus,
            score=predicted_score,
            hasil=hasil,
            filename=filename,
        )


@app.route("/hasil", methods=["POST", "GET"])
def display_image():
    cursor = mysql.connection.cursor()
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    now = datetime.now()
    cur.execute("""SELECT * FROM kendaraan_pelanggar""")
    kendaraan_pelanggar = cur.fetchall()
    cur.close()
    return render_template(
        "hasil.html",
        kendaraan_pelanggar=kendaraan_pelanggar,
    )

@app.route("/TestResult", methods=["POST", "GET"])
def display_image1():
    cursor = mysql.connection.cursor()
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    now = datetime.now()
    cur.execute("""SELECT * FROM kendaraan""")
    kendaraan = cur.fetchall()
    cur.close()
    return render_template(
        "TestResult.html",
        kendaraan=kendaraan,
    )


if __name__ == "__main__":
    app.run("0.0.0.0", port=8000, debug=True)
