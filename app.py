import os     # For file path operations
import io    # For in-memory file operations
import threading   # run amodel training in background
import sqlite3    # built-in database
import datetime
import json    # read & write json files
from flask import Flask, render_template, request, jsonify, send_file, abort
from model import train_model_background, extract_embedding_for_image, MODEL_PATH


APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "attendance.db")
DATASET_DIR = os.path.join(APP_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

TRAIN_STATUS_FILE = os.path.join(APP_DIR, "train_status.json")

app = Flask(__name__, static_folder="static", template_folder="templates")

# DB helpers
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        name TEXT NOT NULL,
        reg_no TEXT,
        batch TEXT,
        degree TEXT,
        created_at TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER, 
        name TEXT,
        module_code TEXT,
        timestamp TEXT
    )""")
    conn.commit()
    conn.close()

init_db()


# Train status helpers
def write_train_status(status_dict):
    with open(TRAIN_STATUS_FILE, "w") as f:
        json.dump(status_dict, f)

def read_train_status():
    if not os.path.exists(TRAIN_STATUS_FILE):
        return {"running": False, "progress": 0, "message": "Not trained"}
    with open(TRAIN_STATUS_FILE, "r") as f:
        return json.load(f)

# Ensure initial train status file exists
write_train_status({"running": False, "progress": 0, "message": "No traning yet."})



# Routes
@app.route("/")
def index():
    return render_template("index.html")

# Dashboard simple API for attendance status (last 30 days)
@app.route("/attendance_stats")
def attendance_stats():
    # Compute counts per day for the last 30 days without pandas
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Prepare date labels from oldest to newest
    last_30_dates = [
        (datetime.date.today() - datetime.timedelta(days=i))
        for i in range(29, -1, -1)
    ]
    date_labels = [d.strftime("%Y-%m-%d") for d in last_30_dates]

    # Query count per day in one go
    c.execute(
        """
        SELECT strftime('%Y-%m-%d', timestamp) as day, COUNT(*) as cnt
        FROM attendance
        WHERE date(timestamp) >= date('now','-29 day')
        GROUP BY day
        """
    )
    rows = c.fetchall()
    conn.close()

    day_to_count = {r[0]: int(r[1]) for r in rows}
    counts = [day_to_count.get(lbl, 0) for lbl in date_labels]
    pretty_labels = [d.strftime("%d-%b") for d in last_30_dates]
    return jsonify({"dates": pretty_labels, "counts": counts})




# ADD Student (from)
@app.route("/add_student", methods=["GET", "POST"])
def add_student():
    if request.method == "GET":
        return render_template("add_student.html")

    # POST: save student metadata and return student_id
    data = request.form
    name = data.get("name", "").strip()
    reg_no = data.get("reg_no", "").strip()
    batch = data.get("batch", "").strip()
    degree = data.get("degree", "").strip()
    

    if not name:
        return jsonify({"error": "name required"}), 400

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.datetime.now().isoformat()
    c.execute("INSERT INTO students (name, reg_no, batch, degree, created_at) VALUES (?, ?, ?, ?, ?)", (name, reg_no, batch, degree, now))
    sid = c.lastrowid
    conn.commit()
    conn.close()

    # create dataset folder for this student
    os.makedirs(os.path.join(DATASET_DIR, str(sid)), exist_ok=True)

    return jsonify({"student_id": sid})




# Upload Face images ( after capture )
@app.route("/upload_face", methods=["POST"])
def upload_face():
    student_id = request.form.get('student_id')
    if not student_id:
        return jsonify({"error": "student_id required"}), 400
    
    files = request.files.getlist("images[]")
    saved = 0
    folder = os.path.join(DATASET_DIR, student_id)
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    
    for f in files:
        try:
            fname = f"{datetime.datetime.now().timestamp():.6f}_{saved}.jpg"
            path = os.path.join(folder, fname)
            f.save(path)
            saved += 1
        except Exception as e:
            app.logger.error("saved error: %s", e)
    
    return jsonify({"saved": saved})





@app.route("/train_model", methods=["GET"])
def train_model_route():
    # if already running, respond accordingly
    status = read_train_status()
    if status.get("running"):
        return jsonify({"status":"already_running"}), 202
    
    # reset status
    write_train_status({"running": True, "progress": 0, "message": "Starting training"})
    
    # start background thread
    t = threading.Thread(target=train_model_background, args=(DATASET_DIR, lambda p,m: write_train_status({"running": True, "progress": p, "message": m})))
    t.daemon = True
    t.start()
    
    return jsonify({"status":"started"}), 202



# Train Progress
@app.route("/train_status", methods=["GET"])
def train_status():
    return jsonify(read_train_status())

# Mark Attendance
@app.route("/mark_attendance", methods=["GET"])
def mark_attendance_page():
    return render_template("mark_attendance.html")


# Recognize face endponint (POST image)
@app.route("/recognize_face", methods=["POST"])
def recognize_face():
    if "image" not in request.files:
        return jsonify({"recognized": False, "error":"no image"}), 400
    
    img_file = request.files["image"]
    module_code = request.form.get("module_code", "").strip()
    
    try:
        emb = extract_embedding_for_image(img_file.stream)
        if emb is None:
            return jsonify({"recognized": False, "error": "no face detected"}), 200
        
        # attempt prediction
        from model import load_model_if_exists, predict_with_model
        clf = load_model_if_exists()
        if clf is None:
            return jsonify({"recognized": False, "error":"model not trained"}), 200
        
        pred_label, conf = predict_with_model(clf, emb)
        
        # threshold confidence
        if conf < 0.5:
            return jsonify({"recognized": False, "confidence": float(conf)}), 200
        
        # find student name
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name FROM students WHERE id=?", (int(pred_label),))
        row = c.fetchone()
        name = row[0] if row else "Unknown"

        # duplicate check for same day and same module_code (if provided)
        ts = datetime.datetime.now().isoformat()
        if module_code:
            c.execute(
                """
                SELECT COUNT(*) FROM attendance
                WHERE student_id = ?
                  AND date(timestamp) = date('now')
                  AND module_code = ?
                """,
                (int(pred_label), module_code)
            )
            cnt = c.fetchone()[0]
            if cnt and cnt > 0:
                conn.close()
                return jsonify({"recognized": False, "error": "duplicate_today"}), 200

        # save attendance record with timestamp and module_code
        c.execute("INSERT INTO attendance (student_id, name, module_code, timestamp) VALUES (?, ?, ?, ?)", (int(pred_label), name, module_code or None, ts))
        conn.commit()
        conn.close()
        
        return jsonify({"recognized": True, "student_id": int(pred_label), "name": name, "confidence": float(conf)}), 200
    
    except Exception as e:
        app.logger.exception("recognize error")
        return jsonify({"recognized": False, "error": str(e)}), 500
    




# Attendance records & filters
@app.route("/attendance_record", methods=["GET"])
def attendance_record():
    period = request.args.get("period", "all")  # all, daily, weekly, monthly
    module_code = request.args.get("module_code", "").strip()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    q = "SELECT id, student_id, name, module_code, timestamp FROM attendance"
    clauses = []
    params = []
    if period == "daily":
        today = datetime.date.today().isoformat()
        clauses.append("date(timestamp) = ?")
        params.append(today)
    elif period == "weekly":
        start = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
        clauses.append("date(timestamp) >= ?")
        params.append(start)
    elif period == "monthly":
        start = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
        clauses.append("date(timestamp) >= ?")
        params.append(start)
    if module_code:
        clauses.append("module_code = ?")
        params.append(module_code)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY timestamp DESC LIMIT 5000"
    c.execute(q, tuple(params))
    rows = c.fetchall()

    # get list of distinct module codes for UI
    c.execute("SELECT DISTINCT module_code FROM attendance WHERE module_code IS NOT NULL AND module_code != '' ORDER BY module_code ASC")
    modules = [r[0] for r in c.fetchall()]
    conn.close()
    return render_template("attendance_record.html", records=rows, period=period, module_code=module_code, modules=modules)


# Download attendance as CSV
@app.route("/download_csv", methods=["GET"])
def download_csv():
    module_code = request.args.get("module_code", "").strip()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    q = "SELECT id, student_id, name, module_code, timestamp FROM attendance"
    params = []
    if module_code:
        q += " WHERE module_code = ?"
        params.append(module_code)
    q += " ORDER BY timestamp DESC"
    c.execute(q, tuple(params))
    rows = c.fetchall()
    conn.close()

    output = io.StringIO()
    output.write("id, student_id, name, module_code, date, timestamp\n")
    for r in rows:
        # r = (id, student_id, name, module_code, timestamp)
        date_only = r[4].split('T')[0] if isinstance(r[4], str) and 'T' in r[4] else r[4]
        output.write(f'{r[0]}, {r[1]}, {r[2]}, {r[3] or ""}, {date_only}, {r[4]}\n')

    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)
    fname = "attendance.csv" if not module_code else f"attendance_{module_code.replace(' ', '_')}.csv"
    return send_file(mem, as_attachment=True, download_name=fname, mimetype="text/csv")




# Student API for editing / listening
@app.route("/students", methods=["GET"])
def students_list():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, reg_no, batch, degree, created_at FROM students ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    data = [{"id": r[0], "name": r[1], "reg_no": r[2], "batch": r[3], "degree": r[4], "created_at": r[6]} for r in rows]
    return jsonify({"students": data})

@app.route("/students/<int:sid>", methods=["DELETE"])
def delete_student(sid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM students WHERE id = ?", (sid,))
    conn.commit()
    conn.close()

    # also delete dataset folder
    folder = os.path.join(DATASET_DIR, str(sid))
    if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder, ignore_errors=True)

    return jsonify({"deleted": True})

if __name__ == "__main__":
    app.run(debug=True)



