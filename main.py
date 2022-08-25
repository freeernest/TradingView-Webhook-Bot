import json
import time

from flask import Flask, request

import config
from handler import *

app = Flask(__name__)


def get_timestamp():
    timestamp = time.strftime("%Y-%m-%d %X")
    return timestamp

@app.route("/")
def hello_world():
    return "Hello World!"

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        if request.method == "POST":
            data = request.get_json()
            key = data["key"]
            if key == config.sec_key:
                print(get_timestamp(), "Alert Received & Sent!")
                send_alert(data)
                return "Sent alert", 200

            else:
                print("[X]", get_timestamp(), "Alert Received & Refused! (Wrong Key)")
                return "Refused alert", 400

    except Exception as e:
        print("[X]", get_timestamp(), "Error:\n>", e)
        return str(e) + "test", 400


if __name__ == "__main__":
    from waitress import serve

    # app.run()
    serve(app, host="0.0.0.0", port=8080)
