import {
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ---- MediaPipe 랜드마크 인덱스 ----
const LM = {
  NOSE: 0,
  L_SHOULDER: 11, R_SHOULDER: 12,
  L_HIP: 23, R_HIP: 24,
  L_KNEE: 25, R_KNEE: 26,
  L_ANKLE: 27, R_ANKLE: 28,
  L_HEEL: 29, R_HEEL: 30,
  L_FOOT: 31, R_FOOT: 32,
};

const SKELETON = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28],
  [27, 29], [29, 31], [27, 31], [28, 30], [30, 32], [28, 32],
];

const SAMPLE_FPS = 12;

// ---- DOM ----
const $ = (id) => document.getElementById(id);
const screens = {
  upload: $("screen-upload"),
  analyzing: $("screen-analyzing"),
  result: $("screen-result"),
};
const modelStatus = $("model-status");
const dropZone = $("drop-zone");
const fileInput = $("file-input");
const probe = $("probe");
const player = $("player");
const overlay = $("overlay");
const errorBanner = $("error-banner");

let landmarker = null;
let frames = [];        // { t, lm } — 분석된 프레임별 랜드마크
let analysis = null;
let currentUrl = null;
let lastTs = 0;         // MediaPipe VIDEO 모드는 타임스탬프가 계속 증가해야 함 (영상 간에도)

// ---- 초기화 ----
async function init() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    landmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });
    modelStatus.textContent = "AI 분석 준비 완료";
    modelStatus.classList.remove("loading");
    modelStatus.classList.add("ready");
  } catch (e) {
    console.error(e);
    modelStatus.textContent = "모델 로딩 실패 — 인터넷 연결을 확인하고 새로고침해 주세요";
    modelStatus.classList.add("failed");
  }
}
init();

// ---- 업로드 ----
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) startAnalysis(fileInput.files[0]);
});
["dragover", "dragenter"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
  })
);
dropZone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("video/")) startAnalysis(f);
});

function showScreen(name) {
  Object.entries(screens).forEach(([k, el]) => (el.hidden = k !== name));
}

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.hidden = false;
  showScreen("upload");
}

// ---- 분석 파이프라인 ----
async function startAnalysis(file) {
  if (!landmarker) {
    showError("AI 모델이 아직 준비되지 않았어요. 상단 상태가 초록불이 된 뒤 다시 시도해 주세요.");
    return;
  }
  errorBanner.hidden = true;

  if (currentUrl) URL.revokeObjectURL(currentUrl);
  currentUrl = URL.createObjectURL(file);

  showScreen("analyzing");
  $("progress-bar").style.width = "0%";
  $("progress-label").textContent = "0%";

  try {
    await loadVideo(probe, currentUrl);
  } catch {
    showError("영상을 열 수 없어요. MP4, MOV, WebM 형식인지 확인해 주세요.");
    return;
  }

  const duration = probe.duration;
  if (!isFinite(duration) || duration < 1) {
    showError("영상 길이를 읽을 수 없어요. 다른 영상으로 시도해 주세요.");
    return;
  }
  if (duration > 180) {
    showError("영상이 3분을 넘어요. 세트 단위(90초 이내)로 잘라서 올려주세요.");
    return;
  }

  frames = [];
  const step = 1 / SAMPLE_FPS;
  let detected = 0;

  for (let t = 0; t < duration; t += step) {
    await seekTo(probe, Math.min(t, duration - 0.05));
    let result;
    try {
      lastTs = Math.max(lastTs + 1, lastTs + Math.round((1 / SAMPLE_FPS) * 1000));
      result = landmarker.detectForVideo(probe, lastTs);
    } catch (e) {
      console.warn("detect 실패 @", t, e);
      continue;
    }
    if (result.landmarks && result.landmarks.length > 0) {
      frames.push({ t, lm: result.landmarks[0] });
      detected++;
    }
    const pct = Math.min(99, Math.round((t / duration) * 100));
    $("progress-bar").style.width = pct + "%";
    $("progress-label").textContent = pct + "%";
  }

  const totalSamples = Math.ceil(duration / step);
  if (detected < totalSamples * 0.3 || detected < SAMPLE_FPS) {
    showError(
      "영상에서 사람을 충분히 인식하지 못했어요. 전신이 나오고 조명이 밝은 영상으로 다시 시도해 주세요."
    );
    return;
  }

  // 다리(무릎·발목)가 실제로 보이는지 확인 — 상반신만 나온 영상 거르기
  const legVis =
    frames.reduce((s, f) => {
      const v = (i) => f.lm[i].visibility ?? 1;
      return s + (v(LM.L_KNEE) + v(LM.R_KNEE) + v(LM.L_ANKLE) + v(LM.R_ANKLE)) / 4;
    }, 0) / frames.length;
  if (legVis < 0.5) {
    showError(
      "다리가 화면에 잘 안 보여요. 머리부터 발끝까지 전신이 나오도록 카메라를 뒤로 물려서 찍어주세요."
    );
    return;
  }

  $("analyze-caption").textContent = "자세를 판정하고 있어요";
  analysis = analyzeSquat(frames, probe.videoWidth, probe.videoHeight);
  window.__fc = analysis; // 판정 기준 튜닝용 디버그

  if (analysis.reps.length === 0) {
    showError(
      "스쿼트 동작(앉았다 일어나기)을 찾지 못했어요. 전신 측면 영상인지 확인해 주세요."
    );
    return;
  }

  renderResult(duration);
}

function loadVideo(video, url) {
  return new Promise((resolve, reject) => {
    video.src = url;
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject();
  });
}

function seekTo(video, t) {
  return new Promise((resolve) => {
    const done = () => {
      video.removeEventListener("seeked", done);
      resolve();
    };
    video.addEventListener("seeked", done);
    video.currentTime = t;
  });
}

// ---- 스쿼트 분석 ----
function px(lm, i, vw, vh) {
  return { x: lm[i].x * vw, y: lm[i].y * vh };
}

function angleAt(a, b, c) {
  // b를 꼭짓점으로 하는 각도 (deg)
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const m1 = Math.hypot(v1.x, v1.y);
  const m2 = Math.hypot(v2.x, v2.y);
  if (m1 === 0 || m2 === 0) return 180;
  return (Math.acos(Math.max(-1, Math.min(1, dot / (m1 * m2)))) * 180) / Math.PI;
}

function median3(arr) {
  return arr.map((v, i) => {
    if (i === 0 || i === arr.length - 1) return v;
    return [arr[i - 1], v, arr[i + 1]].sort((a, b) => a - b)[1];
  });
}

function analyzeSquat(frames, vw, vh) {
  // 어느 쪽(좌/우)이 카메라에 더 잘 보이는지 선택
  const vis = (f, i) => f.lm[i].visibility ?? 1;
  let leftVis = 0, rightVis = 0;
  frames.forEach((f) => {
    leftVis += vis(f, LM.L_HIP) + vis(f, LM.L_KNEE) + vis(f, LM.L_ANKLE);
    rightVis += vis(f, LM.R_HIP) + vis(f, LM.R_KNEE) + vis(f, LM.R_ANKLE);
  });
  const useLeft = leftVis >= rightVis;
  const S = useLeft
    ? { sh: LM.L_SHOULDER, hip: LM.L_HIP, knee: LM.L_KNEE, ankle: LM.L_ANKLE, foot: LM.L_FOOT }
    : { sh: LM.R_SHOULDER, hip: LM.R_HIP, knee: LM.R_KNEE, ankle: LM.R_ANKLE, foot: LM.R_FOOT };

  // 측면/정면 판별: 어깨 좌우 벌어짐이 몸통 길이 대비 크면 정면
  let shoulderSpread = 0, torsoLen = 0;
  frames.forEach((f) => {
    const ls = px(f.lm, LM.L_SHOULDER, vw, vh);
    const rs = px(f.lm, LM.R_SHOULDER, vw, vh);
    const lh = px(f.lm, LM.L_HIP, vw, vh);
    const rh = px(f.lm, LM.R_HIP, vw, vh);
    const midS = { x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2 };
    const midH = { x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2 };
    shoulderSpread += Math.abs(ls.x - rs.x);
    torsoLen += Math.hypot(midS.x - midH.x, midS.y - midH.y);
  });
  const frontView = shoulderSpread / frames.length > 0.55 * (torsoLen / frames.length);

  // 진행 방향 (코가 엉덩이보다 어느 쪽에 있나)
  let noseX = 0, hipX = 0;
  frames.forEach((f) => {
    noseX += f.lm[LM.NOSE].x;
    hipX += f.lm[S.hip].x;
  });
  const dir = noseX > hipX ? 1 : -1; // 1 = 오른쪽을 보고 있음

  // 프레임별 지표
  const metrics = frames.map((f) => {
    const sh = px(f.lm, S.sh, vw, vh);
    const hip = px(f.lm, S.hip, vw, vh);
    const knee = px(f.lm, S.knee, vw, vh);
    const ankle = px(f.lm, S.ankle, vw, vh);
    const foot = px(f.lm, S.foot, vw, vh);
    const kneeAngle = angleAt(hip, knee, ankle);
    // 상체 기울기: 수직 대비 (0 = 꼿꼿, 90 = 수평)
    const dx = sh.x - hip.x, dy = hip.y - sh.y;
    const torsoLean = (Math.atan2(Math.abs(dx), Math.max(dy, 1)) * 180) / Math.PI;
    const shin = Math.hypot(knee.x - ankle.x, knee.y - ankle.y) || 1;
    const kneeForward = ((knee.x - foot.x) * dir) / shin;
    const hipBelowKnee = f.lm[S.hip].y >= f.lm[S.knee].y - 0.01;
    return { t: f.t, kneeAngle, torsoLean, kneeForward, hipBelowKnee };
  });

  const kneeSeries = median3(metrics.map((m) => m.kneeAngle));

  // 반복(rep) 검출: 무릎 각도 상태 머신
  const reps = [];
  let state = "up";
  let startIdx = 0, minIdx = 0, minAngle = 180;
  kneeSeries.forEach((ang, i) => {
    if (state === "up") {
      if (ang < 150) {
        state = "down";
        startIdx = Math.max(0, i - 2);
        minAngle = ang;
        minIdx = i;
      }
    } else {
      if (ang < minAngle) {
        minAngle = ang;
        minIdx = i;
      }
      if (ang > 160) {
        if (minAngle < 130) {
          reps.push(buildRep(metrics, startIdx, minIdx, i, frontView));
        }
        state = "up";
      }
    }
  });
  // 영상이 최저점 근처에서 끝난 경우 마지막 rep 마무리
  if (state === "down" && minAngle < 130) {
    reps.push(buildRep(metrics, startIdx, minIdx, kneeSeries.length - 1, frontView));
  }

  const avgScore = reps.length
    ? Math.round(reps.reduce((s, r) => s + r.score, 0) / reps.length)
    : 0;

  return { reps, avgScore, frontView, useLeft, side: S, kneeSeries, metrics };
}

function buildRep(metrics, startIdx, minIdx, endIdx, frontView) {
  const bottom = metrics[minIdx];
  const duration = metrics[endIdx].t - metrics[startIdx].t;
  const issues = [];
  let score = 100;

  // 깊이
  let depth;
  if (bottom.kneeAngle <= 95 || bottom.hipBelowKnee) {
    depth = { label: "충분한 깊이", cls: "good" };
  } else if (bottom.kneeAngle <= 110) {
    depth = { label: "적정 깊이", cls: "ok" };
  } else if (bottom.kneeAngle <= 125) {
    depth = { label: "약간 얕음", cls: "warn" };
    issues.push("🔽 조금 더 깊이 앉아보세요. 허벅지가 바닥과 평행해질 때까지가 목표예요.");
    score -= 12;
  } else {
    depth = { label: "깊이 부족", cls: "bad" };
    issues.push("🔽 깊이가 많이 부족해요. 엉덩이를 뒤로 빼며 더 낮게 앉아보세요.");
    score -= 25;
  }

  // 상체 기울기 (최저점 기준)
  if (bottom.torsoLean > 55) {
    issues.push("🔺 최저점에서 상체가 많이 숙여졌어요. 가슴을 들고 시선은 정면을 유지해 보세요.");
    score -= 15;
  }

  // 무릎 전방 이동 (정면 영상이면 판정 생략)
  if (!frontView && bottom.kneeForward > 0.45) {
    issues.push("💡 무릎이 발끝을 크게 넘었어요. 통증이 없다면 괜찮지만, 엉덩이를 먼저 뒤로 빼는 느낌으로 시작해 보세요.");
    score -= 10;
  }

  // 템포
  if (duration < 1.2) {
    issues.push("⏱️ 템포가 빨라요. 내려갈 때 2초 정도로 천천히 통제해 보세요.");
    score -= 8;
  }

  score = Math.max(40, score);
  return {
    startT: metrics[startIdx].t,
    bottomT: bottom.t,
    endT: metrics[endIdx].t,
    minKneeAngle: Math.round(bottom.kneeAngle),
    torsoLean: Math.round(bottom.torsoLean),
    duration,
    depth,
    issues,
    score,
  };
}

// ---- 결과 렌더링 ----
function renderResult(duration) {
  showScreen("result");
  $("view-warning").hidden = !analysis.frontView;

  player.src = currentUrl;
  player.load();

  // 요약 카드
  const { reps, avgScore } = analysis;
  const grade =
    avgScore >= 90 ? "훌륭해요! 💪" :
    avgScore >= 75 ? "좋아요 👍" :
    avgScore >= 60 ? "괜찮아요, 조금만 다듬으면 돼요" :
    "개선 포인트가 보여요";

  const issueCount = {};
  reps.forEach((r) => r.issues.forEach((i) => (issueCount[i] = (issueCount[i] || 0) + 1)));
  const topIssues = Object.entries(issueCount)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);

  const avgDur = reps.reduce((s, r) => s + r.duration, 0) / reps.length;
  let lines = "";
  if (topIssues.length === 0) {
    lines = "<p>✅ 모든 회차에서 특별한 문제를 찾지 못했어요. 지금 폼을 유지하세요!</p>";
  } else {
    lines = topIssues
      .map(([msg, n]) => `<p>${msg} <span style="color:var(--muted)">(${n}/${reps.length}회)</span></p>`)
      .join("");
  }

  $("summary-card").innerHTML = `
    <div class="summary-top">
      <div class="score-ring" style="--pct:${avgScore}">
        <div class="num">${avgScore}</div>
        <div class="cap">평균 점수</div>
      </div>
      <div>
        <div class="summary-title">${grade}</div>
        <div class="summary-stats">스쿼트 ${reps.length}회 감지 · 평균 ${avgDur.toFixed(1)}초/회</div>
      </div>
    </div>
    <div class="summary-lines">${lines}</div>
  `;

  // 타임라인 마커
  const timeline = $("timeline");
  timeline.innerHTML = "";
  reps.forEach((r, i) => {
    const btn = document.createElement("button");
    btn.className = "rep-marker" + (r.issues.length ? " has-issue" : "");
    btn.style.left = `${(r.bottomT / duration) * 100}%`;
    btn.textContent = i + 1;
    btn.title = `${i + 1}회차로 이동`;
    btn.onclick = () => {
      player.currentTime = r.bottomT;
      player.pause();
    };
    timeline.appendChild(btn);
  });

  // 회차 카드
  $("rep-cards").innerHTML = reps
    .map(
      (r, i) => `
    <div class="rep-card" data-t="${r.bottomT}">
      <div class="rep-head">
        <span class="rep-num">${i + 1}회차</span>
        <span class="badge ${r.depth.cls}">${r.depth.label}</span>
        <span class="rep-score">${r.score}점</span>
      </div>
      <div class="rep-meta">최저점 무릎 각도 ${r.minKneeAngle}° · 상체 기울기 ${r.torsoLean}° · ${r.duration.toFixed(1)}초</div>
      ${
        r.issues.length
          ? `<div class="rep-issues">${r.issues.map((x) => `<div>${x}</div>`).join("")}</div>`
          : `<div class="rep-clean">✅ 좋은 자세예요</div>`
      }
    </div>`
    )
    .join("");
  document.querySelectorAll(".rep-card").forEach((card) => {
    card.onclick = () => {
      player.currentTime = parseFloat(card.dataset.t);
      player.pause();
    };
  });

  setupOverlay();
}

// ---- 스켈레톤 오버레이 ----
function setupOverlay() {
  const ctx = overlay.getContext("2d");

  function resize() {
    overlay.width = player.clientWidth * devicePixelRatio;
    overlay.height = player.clientHeight * devicePixelRatio;
  }

  function nearestFrame(t) {
    if (!frames.length) return null;
    let lo = 0, hi = frames.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (frames[mid].t < t) lo = mid + 1;
      else hi = mid;
    }
    const cand = [frames[Math.max(0, lo - 1)], frames[lo]];
    const best = Math.abs(cand[0].t - t) < Math.abs(cand[1].t - t) ? cand[0] : cand[1];
    return Math.abs(best.t - t) < 0.25 ? best : null;
  }

  function draw() {
    resize();
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const f = nearestFrame(player.currentTime);
    if (!f) return;

    // 영상이 letterbox 없이 꽉 차게 렌더링된다고 가정 (object-fit 기본 contain 대응)
    const vw = player.videoWidth, vh = player.videoHeight;
    const cw = overlay.width, ch = overlay.height;
    const scale = Math.min(cw / vw, ch / vh);
    const ox = (cw - vw * scale) / 2;
    const oy = (ch - vh * scale) / 2;
    const P = (i) => ({
      x: ox + f.lm[i].x * vw * scale,
      y: oy + f.lm[i].y * vh * scale,
    });

    ctx.lineWidth = 3 * devicePixelRatio;
    ctx.strokeStyle = "rgba(163, 230, 53, 0.9)";
    SKELETON.forEach(([a, b]) => {
      const pa = P(a), pb = P(b);
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    });
    ctx.fillStyle = "#a3e635";
    for (let i = 11; i < 33; i++) {
      const p = P(i);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4 * devicePixelRatio, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function loop() {
    draw();
    requestAnimationFrame(loop);
  }
  loop();
}

// ---- 리셋 ----
$("reset-btn").onclick = () => {
  player.pause();
  player.removeAttribute("src");
  fileInput.value = "";
  frames = [];
  analysis = null;
  errorBanner.hidden = true;
  showScreen("upload");
};
