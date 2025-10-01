// camera_mark.js
const startMarkBtn = document.getElementById("startMarkBtn");
const stopMarkBtn = document.getElementById("stopMarkBtn");
const markVideo = document.getElementById("markVideo");
const markStatus = document.getElementById("markStatus");
const recognizedList = document.getElementById("recognizedList");
const moduleCodeSel = document.getElementById("moduleCode");
const moduleError = document.getElementById("moduleError");
const currentDateEl = document.getElementById("currentDate");

let markStream = null;
let markInterval = null;
let recognizedIds = new Set();

// set current date at load
if (currentDateEl) {
  const d = new Date();
  currentDateEl.textContent = d.toLocaleDateString();
}

startMarkBtn.addEventListener("click", async () => {
  moduleError.textContent = "";
  if (!moduleCodeSel.value) {
    moduleError.textContent = "Please select a Module Code";
    return;
  }
  startMarkBtn.disabled = true;
  stopMarkBtn.disabled = false;
  try {
    markStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    markVideo.srcObject = markStream;
    await markVideo.play();
    markStatus.innerText = "Scanning...";
    markInterval = setInterval(captureAndRecognize, 1200);
  } catch (err) {
    alert("Camera error: " + err.message);
    startMarkBtn.disabled = false;
    stopMarkBtn.disabled = true;
  }
});

stopMarkBtn.addEventListener("click", () => {
  if (markInterval) clearInterval(markInterval);
  if (markStream) markStream.getTracks().forEach(t => t.stop());
  startMarkBtn.disabled = false;
  stopMarkBtn.disabled = true;
  markStatus.innerText = "Stopped";
});

async function captureAndRecognize() {
  const canvas = document.createElement("canvas");
  canvas.width = markVideo.videoWidth || 640;
  canvas.height = markVideo.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(markVideo, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise(r => canvas.toBlob(r, "image/jpeg", 0.85));
  const fd = new FormData();
  fd.append("image", blob, "snap.jpg");
  if (moduleCodeSel && moduleCodeSel.value) {
    fd.append("module_code", moduleCodeSel.value);
  }
  try {
    const res = await fetch("/recognize_face", { method: "POST", body: fd });
    const j = await res.json();
    if (j.recognized) {
      markStatus.innerText = `Recognized: ${j.name} (conf ${Math.round(j.confidence*100)}%)`;
      if (!recognizedIds.has(j.student_id)) {
        recognizedIds.add(j.student_id);
        const li = document.createElement("li");
        li.className = "list-group-item";
        const timeStr = new Date().toLocaleTimeString();
        const modStr = moduleCodeSel && moduleCodeSel.value ? ` [${moduleCodeSel.value}]` : "";
        li.innerText = `${j.name}${modStr} — ${timeStr}`;
        recognizedList.prepend(li);
      }
    } else {
      if (j.error) {
        if (j.error === "duplicate_today") {
          moduleError.textContent = "Your attendance marked Previously";
        }
        markStatus.innerText = `Not recognized: ${j.error}`;
      }
      else markStatus.innerText = `Not recognized`;
    }
  } catch (err) {
    console.error(err);
  }
}