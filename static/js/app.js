/**
 * NeuralFace Studio Client Logic
 * Handles webcam streaming, 1:N identification, 1:1 verification,
 * gallery management, and hardware benchmarking.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Global State
  const state = {
    activeTab: "tab-camera",
    backend: "sface",
    streamActive: false,
    streamTimeout: null,
    videoElement: document.getElementById("webcam-video"),
    hudSvg: document.getElementById("stream-hud-svg"),
    canvasElement: document.getElementById("stream-canvas"),
    placeholder: document.getElementById("camera-placeholder"),
    lastCapturedBlob: null,
    identities: [],
    currentDetections: [],
    lastDetectionTime: 0
  };

  function escapeXml(unsafe) {
    if (unsafe === undefined || unsafe === null) return "";
    return unsafe.toString().replace(/[<>&'"]/g, c => {
      switch (c) {
        case '<': return '&lt;';
        case '>': return '&gt;';
        case '&': return '&amp;';
        case '\'': return '&apos;';
        case '"': return '&quot;';
      }
    });
  }

  // ==========================================
  // TOAST NOTIFICATIONS
  // ==========================================
  function showToast(message, type = "success") {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        ${type === "success" 
          ? '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline>'
          : '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>'}
      </svg>
      <span>${message}</span>
    `;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  // ==========================================
  // TAB NAVIGATION
  // ==========================================
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".studio-panel");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanels.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const activePanel = document.getElementById(targetTab);
      if (activePanel) activePanel.classList.add("active");

      state.activeTab = targetTab;
      if (targetTab === "tab-gallery") loadIdentities();
      if (targetTab === "tab-logs") loadLogs();
    });
  });

  // Global Backend Selector
  const backendSelect = document.getElementById("global-backend-select");
  if (backendSelect) {
    backendSelect.addEventListener("change", (e) => {
      state.backend = e.target.value;
      showToast(`Active backend switched to ${e.target.options[e.target.selectedIndex].text}`);
    });
  }

  // ==========================================
  // LIVE CAMERA STREAM
  // ==========================================
  const btnStartCam = document.getElementById("btn-start-camera");
  const btnStopCam = document.getElementById("btn-stop-camera");
  const btnCamEnroll = document.getElementById("btn-camera-enroll");
  const thresholdSlider = document.getElementById("camera-threshold-slider");
  const thresholdValDisplay = document.getElementById("camera-threshold-val");
  const latencyDisplay = document.getElementById("stream-latency-val");
  const facesCountDisplay = document.getElementById("stream-faces-count");
  const liveTargetsList = document.getElementById("live-targets-list");
  const camStatusBadge = document.getElementById("camera-status-badge");

  if (thresholdSlider) {
    thresholdSlider.addEventListener("input", (e) => {
      thresholdValDisplay.textContent = parseFloat(e.target.value).toFixed(3);
    });
  }

  btnStartCam.addEventListener("click", async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } }
      });
      state.videoElement.srcObject = stream;
      state.videoElement.style.display = "block";
      if (state.hudSvg) {
        state.hudSvg.style.display = "block";
        state.hudSvg.innerHTML = "";
      }
      state.placeholder.style.display = "none";
      btnStartCam.style.display = "none";
      btnStopCam.style.display = "inline-flex";
      btnCamEnroll.style.display = "inline-flex";
      camStatusBadge.textContent = "STREAMING LIVE";
      camStatusBadge.className = "badge badge-emerald";

      state.streamActive = true;
      state.currentDetections = [];
      state.lastDetectionTime = 0;

      try {
        await state.videoElement.play();
      } catch (e) {
        console.warn("video play error:", e);
      }

      startStreamProcessingLoop();
      showToast("Webcam feed initialized successfully");
    } catch (err) {
      console.warn("Could not access physical webcam:", err);
      showToast("Webcam access denied or unavailable: " + err.message, "error");
    }
  });

  btnStopCam.addEventListener("click", () => {
    stopCameraStream();
  });

  function stopCameraStream() {
    state.streamActive = false;
    if (state.streamTimeout) {
      clearTimeout(state.streamTimeout);
      state.streamTimeout = null;
    }
    if (state.videoElement.srcObject) {
      state.videoElement.srcObject.getTracks().forEach(t => t.stop());
      state.videoElement.srcObject = null;
    }
    state.videoElement.style.display = "none";
    if (state.hudSvg) {
      state.hudSvg.style.display = "none";
      state.hudSvg.innerHTML = "";
    }
    state.currentDetections = [];
    state.lastDetectionTime = 0;

    state.placeholder.style.display = "flex";
    btnStartCam.style.display = "inline-flex";
    btnStopCam.style.display = "none";
    btnCamEnroll.style.display = "none";
    camStatusBadge.textContent = "STANDBY";
    camStatusBadge.className = "badge badge-coral";
    if (latencyDisplay) latencyDisplay.innerHTML = `0.0<span class="metric-unit">ms</span>`;
    if (facesCountDisplay) facesCountDisplay.textContent = "0";
    if (liveTargetsList) {
      liveTargetsList.innerHTML = `<span style="font-size:0.85rem; color:var(--text-muted); font-style:italic;">No active stream</span>`;
    }
  }

  function renderSvgHUD(detections, vw, vh) {
    if (!state.hudSvg) return;
    if (!detections || detections.length === 0) {
      state.hudSvg.innerHTML = "";
      return;
    }

    state.hudSvg.setAttribute("viewBox", `0 0 ${vw} ${vh}`);

    let svgContent = `
      <defs>
        <filter id="glow-emerald" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="0" stdDeviation="3" flood-color="#10B981" flood-opacity="0.6"/>
        </filter>
        <filter id="glow-coral" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="0" stdDeviation="3" flood-color="#F43F5E" flood-opacity="0.6"/>
        </filter>
        <filter id="card-shadow" x="-10%" y="-10%" width="120%" height="120%">
          <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#000000" flood-opacity="0.75"/>
        </filter>
      </defs>
    `;

    detections.forEach((d) => {
      const [x, y, w, h] = d.bbox;
      const isIdentified = Boolean(d.is_identified);
      const color = isIdentified ? "#10B981" : "#F43F5E";
      const filterId = isIdentified ? "glow-emerald" : "glow-coral";
      const badgeBg = isIdentified ? "rgba(4, 30, 20, 0.90)" : "rgba(38, 8, 14, 0.90)";
      const badgeBorder = isIdentified ? "rgba(16, 185, 129, 0.65)" : "rgba(244, 63, 94, 0.65)";

      const cLen = Math.min(24, Math.max(12, Math.min(w, h) * 0.22));
      const cornersPath = `
        M ${x} ${y + cLen} L ${x} ${y} L ${x + cLen} ${y}
        M ${x + w - cLen} ${y} L ${x + w} ${y} L ${x + w} ${y + cLen}
        M ${x} ${y + h - cLen} L ${x} ${y + h} L ${x + cLen} ${y + h}
        M ${x + w - cLen} ${y + h} L ${x + w} ${y + h} L ${x + w} ${y + h - cLen}
      `;

      svgContent += `
        <g>
          <!-- Bounding Box -->
          <rect x="${x}" y="${y}" width="${w}" height="${h}"
                fill="${isIdentified ? 'rgba(16, 185, 129, 0.05)' : 'rgba(244, 63, 94, 0.05)'}"
                stroke="${color}" stroke-width="1.5" stroke-dasharray="6 4" rx="4" />
          <!-- Tactical Corner Accents -->
          <path d="${cornersPath}" fill="none" stroke="${color}" stroke-width="3.5"
                stroke-linecap="round" stroke-linejoin="round" filter="url(#${filterId})" />
      `;

      // Landmarks
      if (Array.isArray(d.landmarks) && d.landmarks.length > 0) {
        d.landmarks.forEach((pt) => {
          svgContent += `
            <circle cx="${pt[0]}" cy="${pt[1]}" r="3" fill="#FFFFFF" stroke="${color}" stroke-width="1.5" />
          `;
        });
      }

      // Floating Identity Card
      const labelName = isIdentified ? d.name : "Unknown Face";
      const labelConf = `${Math.round(d.confidence)}%`;
      const labelRole = isIdentified ? (d.role || "Member") : "Not Enrolled";

      const badgeWidth = Math.max(140, Math.min(220, labelName.length * 9 + 48));
      const badgeHeight = 44;

      let badgeX = Math.max(8, Math.min(x, vw - badgeWidth - 8));
      let badgeY = y - badgeHeight - 8;
      if (badgeY < 8) {
        badgeY = y + h + 8;
      }

      svgContent += `
          <g transform="translate(${badgeX}, ${badgeY})" filter="url(#card-shadow)">
            <rect width="${badgeWidth}" height="${badgeHeight}" rx="6"
                  fill="${badgeBg}" stroke="${badgeBorder}" stroke-width="1.5" />
            <rect x="0" y="0" width="4" height="${badgeHeight}" rx="2" fill="${color}" />
            <text x="12" y="17" fill="#FFFFFF" font-family="'Plus Jakarta Sans', system-ui, sans-serif" font-weight="700" font-size="13">
              ${escapeXml(labelName)}
            </text>
            <text x="${badgeWidth - 10}" y="17" text-anchor="end" fill="${color}" font-family="'JetBrains Mono', monospace" font-weight="700" font-size="12">
              ${escapeXml(labelConf)}
            </text>
            <text x="12" y="33" fill="rgba(255, 255, 255, 0.7)" font-family="'Plus Jakarta Sans', system-ui, sans-serif" font-size="10.5">
              ${escapeXml(labelRole)}
            </text>
          </g>
        </g>
      `;
    });

    state.hudSvg.innerHTML = svgContent;
  }

  function startStreamProcessingLoop() {
    const ctx = state.canvasElement.getContext("2d");
    let isProcessing = false;

    async function streamTick() {
      if (!state.streamActive) return;

      if (!isProcessing && state.videoElement.readyState >= 2 && state.videoElement.videoWidth > 0) {
        const vw = state.videoElement.videoWidth;
        const vh = state.videoElement.videoHeight;

        if (state.canvasElement.width !== vw || state.canvasElement.height !== vh) {
          state.canvasElement.width = vw;
          state.canvasElement.height = vh;
        }

        ctx.drawImage(state.videoElement, 0, 0, vw, vh);
        const frameB64 = state.canvasElement.toDataURL("image/jpeg", 0.7);

        isProcessing = true;
        try {
          const threshold = parseFloat(thresholdSlider.value);
          const res = await fetch("/api/stream_frame", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              image_b64: frameB64,
              threshold: threshold,
              include_annotated: false
            })
          });

          if (res.ok) {
            const data = await res.json();
            const detections = data.detections || [];

            state.currentDetections = detections;
            state.lastDetectionTime = Date.now();

            if (state.streamActive) {
              renderSvgHUD(detections, vw, vh);
            }

            if (latencyDisplay) latencyDisplay.innerHTML = `${data.latency_ms}<span class="metric-unit">ms</span>`;
            if (facesCountDisplay) facesCountDisplay.textContent = detections.length;

            // Render live target items in the telemetry sidebar
            if (liveTargetsList) {
              if (detections.length === 0) {
                liveTargetsList.innerHTML = `<span style="font-size:0.85rem; color:var(--text-muted); font-style:italic;">No faces in view</span>`;
              } else {
                liveTargetsList.innerHTML = detections.map(d => `
                  <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(255,255,255,0.03); padding:0.4rem 0.6rem; border-radius:var(--radius-sm); border-left:3px solid ${d.is_identified ? 'var(--accent-emerald)' : 'var(--accent-coral)'};">
                    <div>
                      <strong style="font-size:0.85rem;">${escapeXml(d.name)}</strong>
                      <div style="font-size:0.75rem; color:var(--text-muted);">${escapeXml(d.role)}</div>
                    </div>
                    <span class="badge ${d.is_identified ? 'badge-emerald' : 'badge-coral'}">${d.confidence}%</span>
                  </div>
                `).join("");
              }
            }
          }
        } catch (err) {
          console.error("Frame stream error:", err);
        } finally {
          isProcessing = false;
        }
      }

      if (state.streamActive) {
        state.streamTimeout = setTimeout(streamTick, 50);
      }
    }

    streamTick();
  }

  // Snapshot & Enroll from Camera
  btnCamEnroll.addEventListener("click", () => {
    if (!state.streamActive || !state.videoElement) return;
    if (state.videoElement.readyState < 2) return;
    const vw = state.videoElement.videoWidth;
    const vh = state.videoElement.videoHeight;
    state.canvasElement.width = vw;
    state.canvasElement.height = vh;
    const ctx = state.canvasElement.getContext("2d");
    ctx.drawImage(state.videoElement, 0, 0, vw, vh);
    const snapDataUrl = state.canvasElement.toDataURL("image/jpeg", 0.9);
    openEnrollModal(snapDataUrl);
  });

  // ==========================================
  // 1:N IDENTIFICATION
  // ==========================================
  const idDropzone = document.getElementById("identify-dropzone");
  const idFileInput = document.getElementById("identify-file-input");
  const idPreviewImg = document.getElementById("identify-preview-img");
  const btnRunIdentify = document.getElementById("btn-run-identify");
  const idAnnotatedWrapper = document.getElementById("identify-annotated-wrapper");
  const idResultAnnotatedImg = document.getElementById("identify-result-annotated-img");
  const idResultsList = document.getElementById("identify-results-list");

  let identifyImageFile = null;

  idDropzone.addEventListener("click", () => idFileInput.click());

  idDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    idDropzone.classList.add("dragover");
  });

  idDropzone.addEventListener("dragleave", () => idDropzone.classList.remove("dragover"));

  idDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    idDropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleIdentifyFile(e.dataTransfer.files[0]);
    }
  });

  idFileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleIdentifyFile(e.target.files[0]);
    }
  });

  function handleIdentifyFile(file) {
    identifyImageFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      idPreviewImg.src = e.target.result;
      idPreviewImg.style.display = "block";
      btnRunIdentify.disabled = false;
      idAnnotatedWrapper.style.display = "none";
    };
    reader.readAsDataURL(file);
  }

  btnRunIdentify.addEventListener("click", async () => {
    if (!identifyImageFile) return;

    btnRunIdentify.disabled = true;
    btnRunIdentify.innerHTML = `<svg class="spinner" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle></svg> Identifying...`;

    const formData = new FormData();
    formData.append("image", identifyImageFile);
    formData.append("threshold", "0.363");
    formData.append("backend", state.backend);

    try {
      const res = await fetch("/api/identify", {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      if (!res.ok) {
        showToast(data.message || "Identification failed", "error");
        return;
      }

      if (data.annotated_image) {
        idResultAnnotatedImg.src = data.annotated_image;
        idAnnotatedWrapper.style.display = "flex";
      }

      if (data.faces.length === 0) {
        idResultsList.innerHTML = `<div class="verdict-box verdict-no-match"><p>No faces detected in the image.</p></div>`;
      } else {
        idResultsList.innerHTML = data.faces.map(f => `
          <div class="glass-card" style="padding:1rem; border-left:4px solid ${f.is_identified ? 'var(--accent-emerald)' : 'var(--accent-coral)'};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <div>
                <strong style="font-size:1.05rem;">${f.name}</strong>
                <div style="font-size:0.8rem; color:var(--text-secondary);">${f.role}</div>
              </div>
              <span class="badge ${f.is_identified ? 'badge-emerald' : 'badge-coral'}">
                ${f.is_identified ? 'VERIFIED MATCH' : 'UNKNOWN'} (${f.confidence}%)
              </span>
            </div>

            <div style="display:flex; gap:1.25rem; font-family:var(--font-mono); font-size:0.8rem; color:var(--text-muted); margin-top:0.4rem;">
              <span>Cosine: <strong style="color:var(--text-primary);">${f.cosine_score}</strong></span>
              <span>L2 Dist: <strong style="color:var(--text-primary);">${f.euclidean_distance}</strong></span>
            </div>

            <!-- Top Candidates -->
            <div style="margin-top:0.6rem; border-top:1px solid var(--border-subtle); padding-top:0.5rem;">
              <span style="font-size:0.75rem; font-family:var(--font-mono); color:var(--text-muted);">CANDIDATE RANKINGS:</span>
              <div style="display:flex; flex-direction:column; gap:0.25rem; margin-top:0.3rem;">
                ${f.top_candidates.map((c, i) => `
                  <div style="display:flex; justify-content:space-between; font-size:0.78rem;">
                    <span>#${i+1} ${c.name}</span>
                    <span style="font-family:var(--font-mono); color:var(--accent-cyan);">${c.cosine}</span>
                  </div>
                `).join("")}
              </div>
            </div>
          </div>
        `).join("");
      }

      showToast(`Analyzed ${data.face_count} face(s) successfully`);
    } catch (err) {
      showToast("Error processing identification: " + err.message, "error");
    } finally {
      btnRunIdentify.disabled = false;
      btnRunIdentify.innerHTML = `<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg> Identify Faces in Image`;
    }
  });

  // ==========================================
  // 1:1 FACE VERIFICATION
  // ==========================================
  const verifyFile1 = document.getElementById("verify-file-1");
  const verifyFile2 = document.getElementById("verify-file-2");
  const verifyDrop1 = document.getElementById("verify-dropzone-1");
  const verifyDrop2 = document.getElementById("verify-dropzone-2");
  const verifyPrev1 = document.getElementById("verify-preview-1");
  const verifyPrev2 = document.getElementById("verify-preview-2");
  const btnRunVerify = document.getElementById("btn-run-verify");
  const verifySlider = document.getElementById("verify-threshold-slider");
  const verifyDisplay = document.getElementById("verify-threshold-display");
  const verdictCard = document.getElementById("verdict-card");
  const verdictTitle = document.getElementById("verdict-title");
  const verifyGaugeBar = document.getElementById("verify-gauge-bar");
  const verifyConfVal = document.getElementById("verify-confidence-val");
  const verifyCosVal = document.getElementById("verify-cosine-val");
  const verifyL2Val = document.getElementById("verify-l2-val");
  const verifyResultBox = document.getElementById("verify-result-box");

  let vFile1 = null, vFile2 = null;

  verifyDrop1.addEventListener("click", () => verifyFile1.click());
  verifyDrop2.addEventListener("click", () => verifyFile2.click());

  verifyFile1.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      vFile1 = e.target.files[0];
      const r = new FileReader();
      r.onload = (ev) => {
        verifyPrev1.src = ev.target.result;
        verifyPrev1.style.display = "block";
        checkVerifyReady();
      };
      r.readAsDataURL(vFile1);
    }
  });

  verifyFile2.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      vFile2 = e.target.files[0];
      const r = new FileReader();
      r.onload = (ev) => {
        verifyPrev2.src = ev.target.result;
        verifyPrev2.style.display = "block";
        checkVerifyReady();
      };
      r.readAsDataURL(vFile2);
    }
  });

  function checkVerifyReady() {
    btnRunVerify.disabled = !(vFile1 && vFile2);
  }

  if (verifySlider) {
    verifySlider.addEventListener("input", (e) => {
      verifyDisplay.textContent = parseFloat(e.target.value).toFixed(3);
    });
  }

  btnRunVerify.addEventListener("click", async () => {
    if (!vFile1 || !vFile2) return;

    btnRunVerify.disabled = true;
    btnRunVerify.innerHTML = `Verifying...`;

    const formData = new FormData();
    formData.append("image1", vFile1);
    formData.append("image2", vFile2);
    formData.append("threshold", verifySlider.value);
    formData.append("backend", state.backend);

    try {
      const res = await fetch("/api/verify", {
        method: "POST",
        body: formData
      });
      const data = await res.json();

      if (!res.ok) {
        showToast(data.message || "Verification failed", "error");
        return;
      }

      verifyResultBox.style.display = "block";
      verdictTitle.textContent = data.verdict;
      verifyCosVal.textContent = data.cosine_similarity.toFixed(4);
      verifyL2Val.textContent = data.euclidean_distance.toFixed(4);
      verifyConfVal.textContent = `${data.confidence.toFixed(0)}%`;

      // Gauge animation
      const circumference = 2 * Math.PI * 45; // 283
      const offset = circumference - (data.confidence / 100) * circumference;
      verifyGaugeBar.style.strokeDashoffset = offset;

      if (data.is_match) {
        verdictCard.className = "verdict-box verdict-match";
        verifyGaugeBar.style.stroke = "var(--accent-emerald)";
        showToast("Biometric Match Confirmed! Same individual.");
      } else {
        verdictCard.className = "verdict-box verdict-no-match";
        verifyGaugeBar.style.stroke = "var(--accent-coral)";
        showToast("No Match. Faces belong to different individuals.", "error");
      }
    } catch (err) {
      showToast("Error verifying faces: " + err.message, "error");
    } finally {
      btnRunVerify.disabled = false;
      btnRunVerify.innerHTML = `<svg viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"></polyline></svg> Verify Face Pair (1:1)`;
    }
  });

  // ==========================================
  // IDENTITY GALLERY
  // ==========================================
  const identitiesGrid = document.getElementById("identities-grid");
  const headerCount = document.getElementById("header-identities-count");

  async function loadIdentities() {
    try {
      const res = await fetch("/api/identities");
      const data = await res.json();
      state.identities = data.identities;
      if (headerCount) headerCount.textContent = data.count;

      if (data.identities.length === 0) {
        identitiesGrid.innerHTML = `
          <div style="grid-column: 1 / -1; text-align:center; padding:3rem; color:var(--text-muted);">
            No identities registered yet. Click "Enroll New Identity" to register faces.
          </div>
        `;
        return;
      }

      identitiesGrid.innerHTML = data.identities.map(id => `
        <div class="identity-card" id="card-${id.identity_id}">
          ${id.has_thumbnail 
            ? `<img class="identity-thumb" src="/data/thumbnails/${id.thumbnail_file}" alt="${id.name}">`
            : `<div class="identity-thumb-fallback">${id.name[0]}</div>`}
          
          <div class="identity-body">
            <h4 class="identity-name">${id.name}</h4>
            <span class="identity-role">${id.role}</span>
            <div class="identity-meta">
              <span>Samples: <strong>${id.sample_count}</strong></span>
              <button onclick="window.deleteIdentity('${id.identity_id}')" class="btn btn-danger" style="padding:0.25rem 0.55rem; font-size:0.75rem;">Delete</button>
            </div>
          </div>
        </div>
      `).join("");
    } catch (err) {
      console.error("Failed to load identities:", err);
    }
  }

  // Delete Identity Global Helper
  window.deleteIdentity = async function(id) {
    if (!confirm(`Are you sure you want to delete identity "${id}"?`)) return;
    try {
      const res = await fetch(`/api/identity/${id}`, { method: "DELETE" });
      if (res.ok) {
        showToast("Identity successfully deleted");
        loadIdentities();
      }
    } catch (err) {
      showToast("Error deleting identity: " + err.message, "error");
    }
  };

  // ==========================================
  // ENROLLMENT MODAL
  // ==========================================
  const enrollModal = document.getElementById("enroll-modal");
  const btnOpenEnroll = document.getElementById("btn-open-enroll-modal");
  const btnCloseEnroll = document.getElementById("btn-close-enroll-modal");
  const btnCancelEnroll = document.getElementById("btn-cancel-enroll");
  const btnSubmitEnroll = document.getElementById("btn-submit-enroll");
  const enrollDropzone = document.getElementById("enroll-dropzone");
  const enrollFileInput = document.getElementById("enroll-file-input");
  const enrollPreviewImg = document.getElementById("enroll-preview-img");
  const enrollNameInput = document.getElementById("enroll-name-input");
  const enrollRoleInput = document.getElementById("enroll-role-input");

  let enrollImgDataUrl = null;

  btnOpenEnroll.addEventListener("click", () => openEnrollModal());
  btnCloseEnroll.addEventListener("click", () => closeEnrollModal());
  btnCancelEnroll.addEventListener("click", () => closeEnrollModal());

  function openEnrollModal(prefilledDataUrl = null) {
    enrollModal.classList.add("open");
    enrollNameInput.value = "";
    enrollRoleInput.value = "";
    enrollImgDataUrl = prefilledDataUrl;

    if (prefilledDataUrl) {
      enrollPreviewImg.src = prefilledDataUrl;
      enrollPreviewImg.style.display = "block";
      document.getElementById("enroll-dropzone-text").style.display = "none";
    } else {
      enrollPreviewImg.style.display = "none";
      document.getElementById("enroll-dropzone-text").style.display = "block";
    }
  }

  function closeEnrollModal() {
    enrollModal.classList.remove("open");
  }

  enrollDropzone.addEventListener("click", () => enrollFileInput.click());

  enrollFileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      const r = new FileReader();
      r.onload = (ev) => {
        enrollImgDataUrl = ev.target.result;
        enrollPreviewImg.src = enrollImgDataUrl;
        enrollPreviewImg.style.display = "block";
        document.getElementById("enroll-dropzone-text").style.display = "none";
      };
      r.readAsDataURL(e.target.files[0]);
    }
  });

  btnSubmitEnroll.addEventListener("click", async () => {
    const name = enrollNameInput.value.trim();
    const role = enrollRoleInput.value.trim() || "Member";

    if (!name) {
      showToast("Please enter person's full name", "error");
      return;
    }
    if (!enrollImgDataUrl) {
      showToast("Please provide a portrait photograph", "error");
      return;
    }

    btnSubmitEnroll.disabled = true;
    btnSubmitEnroll.textContent = "Enrolling...";

    try {
      const res = await fetch("/api/enroll", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name,
          role: role,
          image_b64: enrollImgDataUrl,
          backend: state.backend
        })
      });

      const data = await res.json();
      if (!res.ok) {
        showToast(data.message || "Enrollment failed", "error");
        return;
      }

      showToast(`Enrolled ${name} successfully! (Confidence: ${data.confidence}%)`);
      closeEnrollModal();
      loadIdentities();
    } catch (err) {
      showToast("Enrollment failed: " + err.message, "error");
    } finally {
      btnSubmitEnroll.disabled = false;
      btnSubmitEnroll.textContent = "Register Identity";
    }
  });

  // ==========================================
  // HARDWARE BENCHMARKS
  // ==========================================
  const btnRunBenchmark = document.getElementById("btn-run-benchmark");
  const bDetFps = document.getElementById("bench-det-fps");
  const bDetMs = document.getElementById("bench-det-ms");
  const bSfaceFps = document.getElementById("bench-sface-fps");
  const bSfaceMs = document.getElementById("bench-sface-ms");
  const bPtFps = document.getElementById("bench-pt-fps");
  const bPtMs = document.getElementById("bench-pt-ms");
  const bDbMs = document.getElementById("bench-db-ms");

  btnRunBenchmark.addEventListener("click", async () => {
    btnRunBenchmark.disabled = true;
    btnRunBenchmark.innerHTML = `Benchmarking...`;

    try {
      const res = await fetch("/api/benchmark", { method: "POST" });
      const data = await res.json();

      if (res.ok) {
        bDetFps.innerHTML = `${data.detection_fps} <span class="metric-unit">FPS</span>`;
        bDetMs.textContent = `Latency: ${data.detection_ms} ms`;

        bSfaceFps.innerHTML = `${data.sface_fps} <span class="metric-unit">FPS</span>`;
        bSfaceMs.textContent = `Latency: ${data.sface_ms} ms`;

        bPtFps.innerHTML = `${data.pytorch_fps} <span class="metric-unit">FPS</span>`;
        bPtMs.textContent = `Latency: ${data.pytorch_ms} ms`;

        bDbMs.innerHTML = `${data.db_search_ms} <span class="metric-unit">ms</span>`;

        showToast(`Benchmark complete! End-to-end pipeline: ${data.pipeline_fps} FPS`);
      }
    } catch (err) {
      showToast("Benchmark error: " + err.message, "error");
    } finally {
      btnRunBenchmark.disabled = false;
      btnRunBenchmark.innerHTML = `<svg viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg> Execute System Benchmark`;
    }
  });

  // ==========================================
  // ATTENDANCE & AUDIT LOGS
  // ==========================================
  const logsTableBody = document.getElementById("logs-table-body");
  const btnRefreshLogs = document.getElementById("btn-refresh-logs");

  async function loadLogs() {
    try {
      const res = await fetch("/api/logs");
      const data = await res.json();

      if (data.logs.length === 0) {
        logsTableBody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:var(--text-muted); padding:2rem;">No recognition events recorded yet.</td></tr>`;
        return;
      }

      logsTableBody.innerHTML = data.logs.map(log => `
        <tr>
          <td style="font-family:var(--font-mono); font-size:0.8rem; color:var(--text-muted);">${log.timestamp}</td>
          <td><code>${log.identity_id}</code></td>
          <td><strong>${log.name}</strong></td>
          <td style="color:var(--text-secondary);">${log.role}</td>
          <td><span class="badge badge-emerald">${log.confidence}%</span></td>
          <td style="font-family:var(--font-mono); color:var(--accent-cyan);">${log.cosine_score}</td>
        </tr>
      `).join("");
    } catch (err) {
      console.error("Failed to load logs:", err);
    }
  }

  // ==========================================
  // GUIDED MULTI-SAMPLE ENROLLMENT STUDIO
  // ==========================================
  const guidedModal = document.getElementById("guided-enroll-modal");
  const btnOpenGuided = document.getElementById("btn-open-guided-enroll");
  const btnOpenGuidedGallery = document.getElementById("btn-open-guided-enroll-gallery");
  const btnCloseGuided = document.getElementById("btn-close-guided-modal");
  const btnCancelGuided = document.getElementById("btn-cancel-guided-modal");
  const guidedVideo = document.getElementById("guided-video");
  const guidedCanvas = document.getElementById("guided-canvas");
  const reticleOval = document.getElementById("guided-reticle-oval");
  const promptBanner = document.getElementById("guided-prompt-banner");
  const instructionText = document.getElementById("guided-instruction-text");
  const feedbackBanner = document.getElementById("guided-feedback-banner");
  const countdownBadge = document.getElementById("guided-countdown-badge");
  const flashOverlay = document.getElementById("guided-flash-overlay");
  const btnManualSnap = document.getElementById("btn-guided-manual-snap");
  const btnResetSamples = document.getElementById("btn-guided-reset-samples");
  const progressText = document.getElementById("guided-progress-text");
  const guidedNameInput = document.getElementById("guided-name-input");
  const guidedRoleInput = document.getElementById("guided-role-input");
  const btnSubmitGuided = document.getElementById("btn-submit-guided-enroll");

  const guidedSession = {
    stream: null,
    active: false,
    interval: null,
    currentStep: 0,
    steps: [
      { id: "center", label: "Center", prompt: "Step 1/5: Look Straight Ahead (Center)" },
      { id: "left", label: "Left", prompt: "Step 2/5: Turn Head Slightly Left (~15°)" },
      { id: "right", label: "Right", prompt: "Step 3/5: Turn Head Slightly Right (~15°)" },
      { id: "up", label: "Up", prompt: "Step 4/5: Tilt Head Slightly Upwards" },
      { id: "expression", label: "Smile", prompt: "Step 5/5: Smile or Natural Expression" }
    ],
    capturedSamples: [null, null, null, null, null],
    matchStartTime: null,
    holdDuration: 1000 // ms
  };

  if (btnOpenGuided) btnOpenGuided.addEventListener("click", () => openGuidedModal());
  if (btnOpenGuidedGallery) btnOpenGuidedGallery.addEventListener("click", () => openGuidedModal());
  if (btnCloseGuided) btnCloseGuided.addEventListener("click", () => closeGuidedModal());
  if (btnCancelGuided) btnCancelGuided.addEventListener("click", () => closeGuidedModal());

  async function openGuidedModal() {
    guidedModal.classList.add("open");
    resetGuidedSession();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } }
      });
      guidedSession.stream = stream;
      guidedVideo.srcObject = stream;
      guidedSession.active = true;
      startGuidedAnalysisLoop();
      showToast("Biometric calibration camera initialized");
    } catch (err) {
      console.warn("Could not access camera for guided enrollment:", err);
      showToast("Webcam unavailable: " + err.message, "error");
    }
  }

  function closeGuidedModal() {
    guidedSession.active = false;
    if (guidedSession.interval) clearInterval(guidedSession.interval);
    if (guidedSession.stream) {
      guidedSession.stream.getTracks().forEach(t => t.stop());
      guidedSession.stream = null;
    }
    guidedVideo.srcObject = null;
    guidedModal.classList.remove("open");
  }

  function resetGuidedSession() {
    guidedSession.currentStep = 0;
    guidedSession.capturedSamples = [null, null, null, null, null];
    guidedSession.matchStartTime = null;

    for (let i = 0; i < 5; i++) {
      const pill = document.getElementById(`pill-${i}`);
      const slot = document.getElementById(`slot-${i}`);
      if (pill) {
        pill.className = `step-pill ${i === 0 ? 'active' : ''}`;
      }
      if (slot) {
        slot.className = "filmstrip-slot";
        slot.innerHTML = `<span style="color:var(--text-muted); font-size:1.2rem;">${['🎯','⬅️','➡️','⬆️','😊'][i]}</span><div class="filmstrip-label">${i+1}. ${guidedSession.steps[i].label}</div>`;
      }
    }

    promptBanner.textContent = guidedSession.steps[0].prompt;
    instructionText.textContent = "Align your face inside the target oval";
    feedbackBanner.className = "feedback-banner";
    reticleOval.className.baseVal = "reticle-oval";
    countdownBadge.style.display = "none";
    progressText.textContent = "Samples Captured: 0 / 5";
    btnSubmitGuided.disabled = true;
  }

  if (btnResetSamples) {
    btnResetSamples.addEventListener("click", () => resetGuidedSession());
  }

  function startGuidedAnalysisLoop() {
    const ctx = guidedCanvas.getContext("2d");
    let isAnalyzing = false;

    guidedSession.interval = setInterval(async () => {
      if (!guidedSession.active || isAnalyzing) return;
      if (guidedSession.currentStep >= 5) return;
      if (guidedVideo.readyState < 2) return;

      const vw = guidedVideo.videoWidth || 640;
      const vh = guidedVideo.videoHeight || 480;
      guidedCanvas.width = vw;
      guidedCanvas.height = vh;

      // Draw mirrored frame
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(guidedVideo, -vw, 0, vw, vh);
      ctx.restore();

      const frameB64 = guidedCanvas.toDataURL("image/jpeg", 0.75);
      const targetStep = guidedSession.steps[guidedSession.currentStep];

      isAnalyzing = true;
      try {
        const res = await fetch("/api/analyze_pose", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_b64: frameB64,
            target_pose: targetStep.id
          })
        });

        if (res.ok) {
          const data = await res.json();
          updateGuidedHUD(data);
        }
      } catch (err) {
        console.error("Pose analyze error:", err);
      } finally {
        isAnalyzing = false;
      }
    }, 130);
  }

  function updateGuidedHUD(data) {
    if (guidedSession.currentStep >= 5) return;

    instructionText.textContent = data.instruction;

    if (data.pose_matched) {
      reticleOval.className.baseVal = "reticle-oval matched";
      feedbackBanner.className = "feedback-banner matched";

      const now = Date.now();
      if (!guidedSession.matchStartTime) {
        guidedSession.matchStartTime = now;
      }

      const elapsed = now - guidedSession.matchStartTime;
      const remainingSec = Math.max(0.1, (guidedSession.holdDuration - elapsed) / 1000).toFixed(1);

      countdownBadge.style.display = "flex";
      countdownBadge.textContent = `${remainingSec}s`;

      if (elapsed >= guidedSession.holdDuration) {
        // Trigger capture!
        triggerSampleCapture();
      }
    } else {
      reticleOval.className.baseVal = data.face_detected ? "reticle-oval adjusting" : "reticle-oval";
      feedbackBanner.className = data.face_detected ? "feedback-banner adjusting" : "feedback-banner";
      countdownBadge.style.display = "none";
      guidedSession.matchStartTime = null;
    }
  }

  function triggerSampleCapture() {
    const stepIdx = guidedSession.currentStep;
    if (stepIdx >= 5) return;

    // Flash animation
    flashOverlay.classList.add("flashing");
    setTimeout(() => flashOverlay.classList.remove("flashing"), 120);

    // Capture frame from canvas
    const sampleDataUrl = guidedCanvas.toDataURL("image/jpeg", 0.9);
    guidedSession.capturedSamples[stepIdx] = sampleDataUrl;

    // Update filmstrip slot
    const slot = document.getElementById(`slot-${stepIdx}`);
    if (slot) {
      slot.className = "filmstrip-slot captured";
      slot.innerHTML = `
        <img src="${sampleDataUrl}" alt="Sample ${stepIdx+1}">
        <div class="filmstrip-label">✓ ${guidedSession.steps[stepIdx].label}</div>
      `;
    }

    // Mark pill completed
    const pill = document.getElementById(`pill-${stepIdx}`);
    if (pill) pill.className = "step-pill completed";

    guidedSession.currentStep++;
    guidedSession.matchStartTime = null;
    countdownBadge.style.display = "none";

    progressText.textContent = `Samples Captured: ${guidedSession.currentStep} / 5`;
    showToast(`Captured feature angle ${guidedSession.currentStep}/5: ${guidedSession.steps[stepIdx].label}`);

    if (guidedSession.currentStep < 5) {
      const nextPill = document.getElementById(`pill-${guidedSession.currentStep}`);
      if (nextPill) nextPill.className = "step-pill active";
      promptBanner.textContent = guidedSession.steps[guidedSession.currentStep].prompt;
    } else {
      // Completed all 5 angles!
      promptBanner.textContent = "🎉 All 5 Feature Angles Captured!";
      instructionText.textContent = "Calibration complete. Enter name and register identity below.";
      feedbackBanner.className = "feedback-banner matched";
      reticleOval.className.baseVal = "reticle-oval matched";
      btnSubmitGuided.disabled = false;
      showToast("Multi-angle calibration complete! Ready to register.");
    }
  }

  // Manual Snap Override
  if (btnManualSnap) {
    btnManualSnap.addEventListener("click", () => {
      if (guidedSession.currentStep < 5) {
        triggerSampleCapture();
      }
    });
  }

  // Submit Multi-Sample Identity
  if (btnSubmitGuided) {
    btnSubmitGuided.addEventListener("click", async () => {
      const name = guidedNameInput.value.trim();
      const role = guidedRoleInput.value.trim() || "Member";

      if (!name) {
        showToast("Please enter identity full name", "error");
        guidedNameInput.focus();
        return;
      }

      const validSamples = guidedSession.capturedSamples.filter(s => s !== null);
      if (validSamples.length < 3) {
        showToast("At least 3 facial angle samples are required.", "error");
        return;
      }

      btnSubmitGuided.disabled = true;
      btnSubmitGuided.innerHTML = `Registering Multi-Sample Identity...`;

      try {
        const res = await fetch("/api/enroll_multisample", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: name,
            role: role,
            backend: state.backend,
            samples: validSamples,
            labels: guidedSession.steps.map(s => s.label)
          })
        });

        const data = await res.json();
        if (!res.ok) {
          showToast(data.message || "Enrollment failed", "error");
          return;
        }

        showToast(`Enrolled "${name}" with ${data.sample_count} multi-angle biometric samples!`);
        closeGuidedModal();
        loadIdentities();
      } catch (err) {
        showToast("Enrollment error: " + err.message, "error");
      } finally {
        btnSubmitGuided.disabled = false;
        btnSubmitGuided.innerHTML = `<svg viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"></polyline></svg> Register Complete Multi-Sample Identity`;
      }
    });
  }

  // Initial Load
  loadIdentities();
});
