from flask import Flask, jsonify, request
from flask_cors import CORS
import pymysql, os

app = Flask(__name__)
CORS(app)
DB = dict(host="localhost", user="root", password="2004", db="sentinel", charset='utf8mb4')

def db_conn():
    return pymysql.connect(**DB)

@app.route("/api/devices", methods=["GET"])
def list_devices():
    # minimal stub
    return jsonify([{"id":1,"name":"Camera 1","rtsp":"rtsp://..."}])

@app.route("/api/events", methods=["GET"])
def events():
    con = db_conn()
    cur = con.cursor(pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM detections ORDER BY timestamp DESC LIMIT 50")
    rows = cur.fetchall()
    cur.close(); con.close()
    return jsonify(rows)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)