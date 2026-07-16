import {
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ---- MediaPipe 랜드마크 인덱스 ----
const LM = {
  NOSE: 0,
  L_SHOULDER: 11, R_SHOULDER: 12,
  L_ELBOW: 13, R_ELBOW: 14,
  L_WRIST: 15, R_WRIST: 16,
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

// ---- 운동별 설정 ----
const EX_INFO = {
  squat: {
    name: "스쿼트",
    tips: [
      "<strong>측면(옆모습)</strong>에서 촬영해 주세요 — 깊이와 무릎 위치 판정에 필수예요.",
      "머리부터 발끝까지 <strong>전신</strong>이 화면에 들어오게 해주세요.",
      "카메라를 고정하고(삼각대·선반), 밝은 곳에서 찍어주세요.",
      "한 영상에 <strong>한 사람</strong>만 나오는 게 좋아요.",
    ],
    noMotionMsg: "스쿼트 동작(앉았다 일어나기)을 찾지 못했어요. 전신 측면 영상인지, 운동 선택이 맞는지 확인해 주세요.",
  },
  pushup: {
    name: "푸시업",
    tips: [
      "<strong>측면(옆모습)</strong>에서 촬영해 주세요 — 팔꿈치 각도와 몸통 일직선 판정에 필수예요.",
      "머리부터 발끝까지 <strong>전신</strong>이 화면에 들어오게 해주세요.",
      "카메라를 <strong>바닥 가까이 낮게</strong> 두면 훨씬 정확해요.",
      "한 영상에 <strong>한 사람</strong>만 나오는 게 좋아요.",
    ],
    noMotionMsg: "푸시업 동작(내려갔다 올라오기)을 찾지 못했어요. 전신 측면 영상인지, 운동 선택이 맞는지 확인해 주세요.",
  },
  plank: {
    name: "플랭크",
    tips: [
      "<strong>측면(옆모습)</strong>에서 촬영해 주세요 — 몸 일직선 판정에 필수예요.",
      "머리부터 발끝까지 <strong>전신</strong>이 화면에 들어오게 해주세요.",
      "카메라를 <strong>바닥 가까이 낮게</strong> 두면 훨씬 정확해요.",
      "<strong>10초 이상</strong> 유지하는 영상이 좋아요.",
    ],
    noMotionMsg: "플랭크 유지 구간(몸을 수평으로 버티기)을 찾지 못했어요. 전신 측면 영상인지, 운동 선택이 맞는지 확인해 주세요.",
  },
  bulgarian: {
    name: "불가리안 스플릿 스쿼트",
    tips: [
      "<strong>측면(옆모습)</strong>에서 촬영해 주세요 — 앞무릎 각도 판정에 필수예요.",
      "머리부터 <strong>뒷발(벤치 위)</strong>까지 전신이 화면에 들어오게 해주세요.",
      "덤벨은 몸 옆에 들면 인식에 방해되지 않아요.",
      "한 영상에 <strong>한쪽 다리 세트</strong>만 담으면 좌우 비교가 쉬워요.",
    ],
    noMotionMsg: "불가리안 스플릿 스쿼트 동작(앉았다 일어나기)을 찾지 못했어요. 전신 측면 영상인지, 운동 선택이 맞는지 확인해 주세요.",
  },
  rdl: {
    name: "루마니안 데드리프트",
    tips: [
      "<strong>측면(옆모습)</strong>에서 촬영해 주세요 — 힙힌지와 바 궤적 판정에 필수예요.",
      "머리부터 발끝까지 <strong>전신</strong>이 화면에 들어오게 해주세요.",
      "바벨·덤벨을 <strong>몸 앞에</strong> 들고 있는 표준 자세 기준이에요.",
      "카메라를 고정하고, 밝은 곳에서 찍어주세요.",
    ],
    noMotionMsg: "루마니안 데드리프트 동작(엉덩이 접었다 펴기)을 찾지 못했어요. 전신 측면 영상인지, 운동 선택이 맞는지 확인해 주세요.",
    note: "🔍 참고: AI는 등이 굽었는지 자체는 보지 못해요(관절을 직선으로만 인식). 대신 등 굽음의 전조인 \"바가 다리에서 멀어짐\"과 무릎·힙 각도를 판정해요.",
  },
  latpull: {
    name: "랫풀다운",
    tips: [
      "<strong>측면(옆모습)</strong>에서 촬영해 주세요 — 당김 범위와 반동 판정 기준이에요.",
      "<strong>상체와 팔 전체</strong>가 기구 프레임에 가리지 않게 찍어주세요.",
      "하체는 기구에 가려져도 괜찮아요 — 상체만 분석해요.",
      "카메라를 고정하고, 밝은 곳에서 찍어주세요.",
    ],
    noMotionMsg: "랫풀다운 동작(당겼다 올리기)을 찾지 못했어요. 상체 측면이 잘 보이는 영상인지, 운동 선택이 맞는지 확인해 주세요.",
    skipLegCheck: true,
    note: "🔍 랫풀다운 판정은 다른 운동보다 대략적이에요. 이유: ① 기구가 하체를 가려서 상체(당김 범위·반동)만 분석하고, ② 적정 당김 높이가 체형·그립 너비에 따라 달라서 각도 기준에 오차가 있어요. 점수는 참고용으로 보고, 반동 경고 위주로 활용해 주세요.",
  },
};

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
let exercise = "squat";

// ---- 운동 선택 ----
function renderTips() {
  $("tips-list").innerHTML = EX_INFO[exercise].tips.map((t) => `<li>${t}</li>`).join("");
}
document.querySelectorAll(".ex-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    exercise = btn.dataset.ex;
    document.querySelectorAll(".ex-btn").forEach((b) => b.classList.toggle("active", b === btn));
    renderTips();
    errorBanner.hidden = true;
  });
});
renderTips();

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
  // (랫풀다운은 기구가 하체를 가리므로 대신 팔 확인)
  const avgVis = (idxs) =>
    frames.reduce((s, f) => s + idxs.reduce((a, i) => a + (f.lm[i].visibility ?? 1), 0) / idxs.length, 0) /
    frames.length;
  if (EX_INFO[exercise].skipLegCheck) {
    const armVis = Math.max(
      avgVis([LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST]),
      avgVis([LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST])
    );
    if (armVis < 0.5) {
      showError("팔과 상체가 잘 안 보여요. 옆에서 상체 전체가 나오게 찍어주세요.");
      return;
    }
  } else {
    const legVis = avgVis([LM.L_KNEE, LM.R_KNEE, LM.L_ANKLE, LM.R_ANKLE]);
    if (legVis < 0.5) {
      showError(
        "다리가 화면에 잘 안 보여요. 머리부터 발끝까지 전신이 나오도록 카메라를 뒤로 물려서 찍어주세요."
      );
      return;
    }
  }

  $("analyze-caption").textContent = "자세를 판정하고 있어요";
  const vw = probe.videoWidth, vh = probe.videoHeight;
  const ANALYZERS = {
    squat: analyzeSquat,
    pushup: analyzePushup,
    plank: analyzePlank,
    bulgarian: analyzeBulgarian,
    rdl: analyzeRDL,
    latpull: analyzeLatpull,
  };
  analysis = ANALYZERS[exercise](frames, vw, vh);
  window.__fc = analysis; // 판정 기준 튜닝용 디버그

  if (analysis.notHorizontal) {
    showError(
      `몸이 수평인 자세를 찾지 못했어요. ${EX_INFO[exercise].name}은(는) 옆에서, 바닥과 전신이 나오게 찍어주세요. 운동 선택이 맞는지도 확인해 주세요!`
    );
    return;
  }
  const empty =
    (analysis.kind === "reps" && analysis.reps.length === 0) ||
    (analysis.kind === "hold" && !analysis.holdDur);
  if (empty) {
    showError(EX_INFO[exercise].noMotionMsg);
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

// ---- 공용 기하 도구 ----
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

// 어깨→발목 직선 대비 엉덩이의 처짐 정도.
// 양수 = 화면상 선보다 아래(엉덩이 처짐), 음수 = 위(엉덩이 솟음). 몸 길이로 정규화.
function sagRatio(shoulder, hip, ankle) {
  const len = Math.hypot(ankle.x - shoulder.x, ankle.y - shoulder.y) || 1;
  const cross =
    (ankle.x - shoulder.x) * (hip.y - shoulder.y) -
    (ankle.y - shoulder.y) * (hip.x - shoulder.x);
  const mult = ankle.x >= shoulder.x ? 1 : -1;
  return (cross / len / len) * mult;
}

// 어깨→발목 벡터가 수평에서 벗어난 각도 (0 = 완전 수평)
function bodyTilt(shoulder, ankle) {
  return (Math.atan2(Math.abs(ankle.y - shoulder.y), Math.abs(ankle.x - shoulder.x)) * 180) / Math.PI;
}

// 수치가 내려갔다 올라오는 반복(rep) 구간 검출 — 각도·높이 등 어떤 시계열이든 사용
function detectReps(series, { startBelow, endAbove, validBelow }) {
  const out = [];
  let state = "up", startIdx = 0, minIdx = 0, minVal = Infinity;
  series.forEach((v, i) => {
    if (state === "up") {
      if (v < startBelow) {
        state = "down";
        startIdx = Math.max(0, i - 2);
        minVal = v;
        minIdx = i;
      }
    } else {
      if (v < minVal) {
        minVal = v;
        minIdx = i;
      }
      if (v > endAbove) {
        if (minVal < validBelow) out.push({ startIdx, minIdx, endIdx: i });
        state = "up";
      }
    }
  });
  if (state === "down" && minVal < validBelow) {
    out.push({ startIdx, minIdx, endIdx: series.length - 1 });
  }
  return out;
}

// 좌/우 중 카메라에 더 잘 보이는 쪽의 관절 세트 선택
function pickSide(frames, idxList) {
  const vis = (f, i) => f.lm[i].visibility ?? 1;
  let left = 0, right = 0;
  frames.forEach((f) => {
    idxList.forEach(([l, r]) => {
      left += vis(f, l);
      right += vis(f, r);
    });
  });
  return left >= right;
}

// ---- 스쿼트 분석 ----
function analyzeSquat(frames, vw, vh) {
  const useLeft = pickSide(frames, [[LM.L_HIP, LM.R_HIP], [LM.L_KNEE, LM.R_KNEE], [LM.L_ANKLE, LM.R_ANKLE]]);
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
  const dir = noseX > hipX ? 1 : -1;

  const metrics = frames.map((f) => {
    const sh = px(f.lm, S.sh, vw, vh);
    const hip = px(f.lm, S.hip, vw, vh);
    const knee = px(f.lm, S.knee, vw, vh);
    const ankle = px(f.lm, S.ankle, vw, vh);
    const foot = px(f.lm, S.foot, vw, vh);
    const kneeAngle = angleAt(hip, knee, ankle);
    const dx = sh.x - hip.x, dy = hip.y - sh.y;
    const torsoLean = (Math.atan2(Math.abs(dx), Math.max(dy, 1)) * 180) / Math.PI;
    const shin = Math.hypot(knee.x - ankle.x, knee.y - ankle.y) || 1;
    const kneeForward = ((knee.x - foot.x) * dir) / shin;
    const hipBelowKnee = f.lm[S.hip].y >= f.lm[S.knee].y - 0.01;
    return { t: f.t, kneeAngle, torsoLean, kneeForward, hipBelowKnee };
  });

  const kneeSeries = median3(metrics.map((m) => m.kneeAngle));
  const reps = detectReps(kneeSeries, { startBelow: 150, endAbove: 160, validBelow: 130 }).map((r) =>
    buildSquatRep(metrics, r.startIdx, r.minIdx, r.endIdx, frontView)
  );

  const avgScore = reps.length
    ? Math.round(reps.reduce((s, r) => s + r.score, 0) / reps.length)
    : 0;
  const avgDur = reps.length ? reps.reduce((s, r) => s + r.duration, 0) / reps.length : 0;

  return {
    kind: "reps",
    reps,
    avgScore,
    frontView,
    statsText: `스쿼트 ${reps.length}회 감지 · 평균 ${avgDur.toFixed(1)}초/회`,
    kneeSeries,
    metrics,
  };
}

function buildSquatRep(metrics, startIdx, minIdx, endIdx, frontView) {
  const bottom = metrics[minIdx];
  const duration = metrics[endIdx].t - metrics[startIdx].t;
  const issues = [];
  let score = 100;

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

  if (bottom.torsoLean > 55) {
    issues.push("🔺 최저점에서 상체가 많이 숙여졌어요. 가슴을 들고 시선은 정면을 유지해 보세요.");
    score -= 15;
  }

  if (!frontView && bottom.kneeForward > 0.45) {
    issues.push("💡 무릎이 발끝을 크게 넘었어요. 통증이 없다면 괜찮지만, 엉덩이를 먼저 뒤로 빼는 느낌으로 시작해 보세요.");
    score -= 10;
  }

  if (duration < 1.2) {
    issues.push("⏱️ 템포가 빨라요. 내려갈 때 2초 정도로 천천히 통제해 보세요.");
    score -= 8;
  }

  score = Math.max(40, score);
  return {
    startT: metrics[startIdx].t,
    bottomT: bottom.t,
    endT: metrics[endIdx].t,
    duration,
    depth,
    issues,
    score,
    metaText: `최저점 무릎 각도 ${Math.round(bottom.kneeAngle)}° · 상체 기울기 ${Math.round(bottom.torsoLean)}° · ${duration.toFixed(1)}초`,
  };
}

// ---- 푸시업 분석 ----
function analyzePushup(frames, vw, vh) {
  const useLeft = pickSide(frames, [
    [LM.L_SHOULDER, LM.R_SHOULDER],
    [LM.L_ELBOW, LM.R_ELBOW],
    [LM.L_WRIST, LM.R_WRIST],
  ]);
  const S = useLeft
    ? { sh: LM.L_SHOULDER, el: LM.L_ELBOW, wr: LM.L_WRIST, hip: LM.L_HIP, ankle: LM.L_ANKLE }
    : { sh: LM.R_SHOULDER, el: LM.R_ELBOW, wr: LM.R_WRIST, hip: LM.R_HIP, ankle: LM.R_ANKLE };

  const metrics = frames.map((f) => {
    const sh = px(f.lm, S.sh, vw, vh);
    const el = px(f.lm, S.el, vw, vh);
    const wr = px(f.lm, S.wr, vw, vh);
    const hip = px(f.lm, S.hip, vw, vh);
    const ankle = px(f.lm, S.ankle, vw, vh);
    return {
      t: f.t,
      elbowAngle: angleAt(sh, el, wr),
      sag: sagRatio(sh, hip, ankle),
      tilt: bodyTilt(sh, ankle),
    };
  });

  // 몸이 수평인 프레임이 대부분이어야 푸시업 영상
  const horizPct = metrics.filter((m) => m.tilt < 40).length / metrics.length;
  if (horizPct < 0.5) return { kind: "reps", reps: [], notHorizontal: true };

  const elbowSeries = median3(metrics.map((m) => m.elbowAngle));
  const reps = detectReps(elbowSeries, { startBelow: 140, endAbove: 150, validBelow: 120 }).map((r) =>
    buildPushupRep(metrics, r.startIdx, r.minIdx, r.endIdx)
  );

  const avgScore = reps.length
    ? Math.round(reps.reduce((s, r) => s + r.score, 0) / reps.length)
    : 0;
  const avgDur = reps.length ? reps.reduce((s, r) => s + r.duration, 0) / reps.length : 0;

  return {
    kind: "reps",
    reps,
    avgScore,
    frontView: false,
    statsText: `푸시업 ${reps.length}회 감지 · 평균 ${avgDur.toFixed(1)}초/회`,
    elbowSeries,
    metrics,
  };
}

function buildPushupRep(metrics, startIdx, minIdx, endIdx) {
  const bottom = metrics[minIdx];
  const duration = metrics[endIdx].t - metrics[startIdx].t;
  const issues = [];
  let score = 100;

  let depth;
  if (bottom.elbowAngle <= 80) {
    depth = { label: "충분한 깊이", cls: "good" };
  } else if (bottom.elbowAngle <= 100) {
    depth = { label: "적정 깊이", cls: "ok" };
  } else if (bottom.elbowAngle <= 115) {
    depth = { label: "약간 얕음", cls: "warn" };
    issues.push("🔽 조금 더 내려가 보세요. 가슴이 바닥에 가까워질 때까지가 목표예요.");
    score -= 12;
  } else {
    depth = { label: "깊이 부족", cls: "bad" };
    issues.push("🔽 깊이가 많이 부족해요. 팔꿈치를 90도 이상 접으며 내려가 보세요.");
    score -= 25;
  }

  // 회차 구간 내 최대 처짐/솟음
  let maxSag = -1, maxPike = 1;
  for (let i = startIdx; i <= endIdx; i++) {
    if (metrics[i].sag > maxSag) maxSag = metrics[i].sag;
    if (metrics[i].sag < maxPike) maxPike = metrics[i].sag;
  }
  if (maxSag > 0.07) {
    issues.push("🔻 엉덩이가 아래로 처졌어요. 배와 엉덩이에 힘을 주고 몸을 일직선으로 유지해 보세요.");
    score -= 15;
  } else if (maxPike < -0.13) {
    issues.push("🔺 엉덩이가 위로 솟았어요. 엉덩이를 낮춰 어깨부터 발목까지 일직선을 만들어 보세요.");
    score -= 10;
  }

  if (duration < 1.0) {
    issues.push("⏱️ 템포가 빨라요. 내려갈 때 천천히 통제하면 효과가 커져요.");
    score -= 8;
  }

  score = Math.max(40, score);
  return {
    startT: metrics[startIdx].t,
    bottomT: bottom.t,
    endT: metrics[endIdx].t,
    duration,
    depth,
    issues,
    score,
    metaText: `최저점 팔꿈치 각도 ${Math.round(bottom.elbowAngle)}° · ${duration.toFixed(1)}초`,
  };
}

// ---- 플랭크 분석 ----
function analyzePlank(frames, vw, vh) {
  const useLeft = pickSide(frames, [
    [LM.L_SHOULDER, LM.R_SHOULDER],
    [LM.L_HIP, LM.R_HIP],
    [LM.L_ANKLE, LM.R_ANKLE],
  ]);
  const S = useLeft
    ? { sh: LM.L_SHOULDER, hip: LM.L_HIP, ankle: LM.L_ANKLE }
    : { sh: LM.R_SHOULDER, hip: LM.R_HIP, ankle: LM.R_ANKLE };

  const metrics = frames.map((f) => {
    const sh = px(f.lm, S.sh, vw, vh);
    const hip = px(f.lm, S.hip, vw, vh);
    const ankle = px(f.lm, S.ankle, vw, vh);
    return { t: f.t, sag: sagRatio(sh, hip, ankle), tilt: bodyTilt(sh, ankle) };
  });

  // 수평 유지 구간(최장 연속 구간, 짧은 끊김 허용) 찾기
  const horiz = metrics.map((m) => m.tilt < 35);
  let best = { start: -1, end: -1 };
  let runStart = -1, gap = 0;
  for (let i = 0; i <= horiz.length; i++) {
    if (i < horiz.length && horiz[i]) {
      if (runStart === -1) runStart = i;
      gap = 0;
    } else if (runStart !== -1) {
      gap++;
      if (gap > 3 || i === horiz.length) {
        const end = i - gap;
        if (end - runStart > best.end - best.start) best = { start: runStart, end };
        runStart = -1;
        gap = 0;
      }
    }
  }
  if (best.start === -1) return { kind: "hold", holdDur: 0, notHorizontal: true };

  const hold = metrics.slice(best.start, best.end + 1);
  const holdDur = hold[hold.length - 1].t - hold[0].t;
  if (holdDur < 3) return { kind: "hold", holdDur: 0 };

  // 프레임별 정렬 판정
  const SAG_TH = 0.06, PIKE_TH = -0.13;
  const cls = hold.map((m) => (m.sag > SAG_TH ? "sag" : m.sag < PIKE_TH ? "pike" : "good"));

  // 문제 구간(0.5초 이상 연속) 추출
  const segments = [];
  let segStart = -1, segType = null;
  for (let i = 0; i <= cls.length; i++) {
    const c = i < cls.length ? cls[i] : "good";
    if (c !== "good" && segType === null) {
      segStart = i;
      segType = c;
    } else if (segType !== null && c !== segType) {
      if (i - segStart >= Math.round(SAMPLE_FPS * 0.5)) {
        segments.push({ startT: hold[segStart].t, endT: hold[i - 1].t, type: segType });
      }
      segType = c !== "good" ? c : null;
      segStart = i;
    }
  }

  const badFrames = cls.filter((c) => c !== "good").length;
  const goodPct = Math.round(((cls.length - badFrames) / cls.length) * 100);
  const sagTime = (cls.filter((c) => c === "sag").length / SAMPLE_FPS);
  const pikeTime = (cls.filter((c) => c === "pike").length / SAMPLE_FPS);
  const score = Math.max(
    40,
    Math.min(100, Math.round(100 - (sagTime / holdDur) * 80 - (pikeTime / holdDur) * 50))
  );

  const lines = [];
  const sagSegs = segments.filter((s) => s.type === "sag");
  const pikeSegs = segments.filter((s) => s.type === "pike");
  if (sagSegs.length)
    lines.push(`🔻 엉덩이 처짐 ${sagSegs.length}구간 · 총 ${sagTime.toFixed(1)}초 — 배꼽을 등 쪽으로 당기고 엉덩이에 힘을 주세요.`);
  if (pikeSegs.length)
    lines.push(`🔺 엉덩이 솟음 ${pikeSegs.length}구간 · 총 ${pikeTime.toFixed(1)}초 — 엉덩이를 낮춰 어깨~발목 일직선을 만들어 보세요.`);
  if (!lines.length) lines.push("✅ 유지 시간 내내 몸 정렬이 훌륭해요. 지금 폼을 유지하세요!");

  return {
    kind: "hold",
    score,
    holdStart: hold[0].t,
    holdEnd: hold[hold.length - 1].t,
    holdDur,
    goodPct,
    segments,
    lines,
    statsText: `플랭크 ${holdDur.toFixed(1)}초 유지 · 정렬 유지율 ${goodPct}%`,
    metrics,
  };
}

// ---- 불가리안 스플릿 스쿼트 분석 ----
function analyzeBulgarian(frames, vw, vh) {
  // 앞다리 찾기: 뒷발은 벤치 위(화면상 높음) → 발목이 더 낮은 쪽이 앞다리
  let ly = 0, ry = 0;
  frames.forEach((f) => {
    ly += f.lm[LM.L_ANKLE].y;
    ry += f.lm[LM.R_ANKLE].y;
  });
  const frontLeft = ly >= ry;
  const S = frontLeft
    ? { sh: LM.L_SHOULDER, hip: LM.L_HIP, knee: LM.L_KNEE, ankle: LM.L_ANKLE, foot: LM.L_FOOT }
    : { sh: LM.R_SHOULDER, hip: LM.R_HIP, knee: LM.R_KNEE, ankle: LM.R_ANKLE, foot: LM.R_FOOT };

  let noseX = 0, hipX = 0;
  frames.forEach((f) => {
    noseX += f.lm[LM.NOSE].x;
    hipX += f.lm[S.hip].x;
  });
  const dir = noseX > hipX ? 1 : -1;

  const metrics = frames.map((f) => {
    const sh = px(f.lm, S.sh, vw, vh);
    const hip = px(f.lm, S.hip, vw, vh);
    const knee = px(f.lm, S.knee, vw, vh);
    const ankle = px(f.lm, S.ankle, vw, vh);
    const foot = px(f.lm, S.foot, vw, vh);
    const kneeAngle = angleAt(hip, knee, ankle);
    const dx = sh.x - hip.x, dy = hip.y - sh.y;
    const torsoLean = (Math.atan2(Math.abs(dx), Math.max(dy, 1)) * 180) / Math.PI;
    const shin = Math.hypot(knee.x - ankle.x, knee.y - ankle.y) || 1;
    const kneeForward = ((knee.x - foot.x) * dir) / shin;
    return { t: f.t, kneeAngle, torsoLean, kneeForward };
  });

  const kneeSeries = median3(metrics.map((m) => m.kneeAngle));
  const reps = detectReps(kneeSeries, { startBelow: 150, endAbove: 160, validBelow: 130 }).map((r) =>
    buildBulgarianRep(metrics, r.startIdx, r.minIdx, r.endIdx)
  );

  const avgScore = reps.length
    ? Math.round(reps.reduce((s, r) => s + r.score, 0) / reps.length)
    : 0;
  const avgDur = reps.length ? reps.reduce((s, r) => s + r.duration, 0) / reps.length : 0;

  return {
    kind: "reps",
    reps,
    avgScore,
    frontView: false,
    statsText: `불가리안 ${reps.length}회 감지 (앞다리: ${frontLeft ? "왼쪽" : "오른쪽"}) · 평균 ${avgDur.toFixed(1)}초/회`,
    kneeSeries,
    metrics,
  };
}

function buildBulgarianRep(metrics, startIdx, minIdx, endIdx) {
  const bottom = metrics[minIdx];
  const duration = metrics[endIdx].t - metrics[startIdx].t;
  const issues = [];
  let score = 100;

  let depth;
  if (bottom.kneeAngle <= 90) {
    depth = { label: "충분한 깊이", cls: "good" };
  } else if (bottom.kneeAngle <= 105) {
    depth = { label: "적정 깊이", cls: "ok" };
  } else if (bottom.kneeAngle <= 120) {
    depth = { label: "약간 얕음", cls: "warn" };
    issues.push("🔽 조금 더 깊이 앉아보세요. 앞허벅지가 바닥과 평행해질 때까지가 목표예요.");
    score -= 12;
  } else {
    depth = { label: "깊이 부족", cls: "bad" };
    issues.push("🔽 깊이가 많이 부족해요. 뒷무릎이 바닥에 가까워지도록 수직으로 앉아보세요.");
    score -= 25;
  }

  if (bottom.torsoLean > 55) {
    issues.push("💡 상체가 많이 숙여졌어요. 엉덩이 자극 목적이면 괜찮지만, 허벅지 위주라면 상체를 좀 더 세워보세요.");
    score -= 8;
  }

  if (bottom.kneeForward > 0.6) {
    issues.push("💡 앞무릎이 발끝을 많이 넘었어요. 앞정강이가 아프다면 앞발을 반 발짝 앞으로 옮겨보세요.");
    score -= 10;
  }

  if (duration < 1.2) {
    issues.push("⏱️ 템포가 빨라요. 균형이 흔들리기 쉬운 운동이라 천천히 통제하는 게 중요해요.");
    score -= 8;
  }

  score = Math.max(40, score);
  return {
    startT: metrics[startIdx].t,
    bottomT: bottom.t,
    endT: metrics[endIdx].t,
    duration,
    depth,
    issues,
    score,
    metaText: `최저점 앞무릎 각도 ${Math.round(bottom.kneeAngle)}° · 상체 기울기 ${Math.round(bottom.torsoLean)}° · ${duration.toFixed(1)}초`,
  };
}

// ---- 루마니안 데드리프트 분석 ----
function analyzeRDL(frames, vw, vh) {
  const useLeft = pickSide(frames, [
    [LM.L_HIP, LM.R_HIP],
    [LM.L_KNEE, LM.R_KNEE],
    [LM.L_WRIST, LM.R_WRIST],
  ]);
  const S = useLeft
    ? { sh: LM.L_SHOULDER, hip: LM.L_HIP, knee: LM.L_KNEE, ankle: LM.L_ANKLE, wr: LM.L_WRIST }
    : { sh: LM.R_SHOULDER, hip: LM.R_HIP, knee: LM.R_KNEE, ankle: LM.R_ANKLE, wr: LM.R_WRIST };

  let noseX = 0, hipX = 0;
  frames.forEach((f) => {
    noseX += f.lm[LM.NOSE].x;
    hipX += f.lm[S.hip].x;
  });
  const dir = noseX > hipX ? 1 : -1;

  const metrics = frames.map((f) => {
    const sh = px(f.lm, S.sh, vw, vh);
    const hip = px(f.lm, S.hip, vw, vh);
    const knee = px(f.lm, S.knee, vw, vh);
    const ankle = px(f.lm, S.ankle, vw, vh);
    const wr = px(f.lm, S.wr, vw, vh);
    const hipAngle = angleAt(sh, hip, knee);
    const kneeAngle = angleAt(hip, knee, ankle);
    const shin = Math.hypot(knee.x - ankle.x, knee.y - ankle.y) || 1;
    // 바(손목)가 발목 기준으로 몸 앞으로 얼마나 떨어졌나
    const barDrift = ((wr.x - ankle.x) * dir) / shin;
    return { t: f.t, hipAngle, kneeAngle, barDrift };
  });

  const hipSeries = median3(metrics.map((m) => m.hipAngle));
  const reps = detectReps(hipSeries, { startBelow: 155, endAbove: 165, validBelow: 140 }).map((r) =>
    buildRdlRep(metrics, r.startIdx, r.minIdx, r.endIdx)
  );

  const avgScore = reps.length
    ? Math.round(reps.reduce((s, r) => s + r.score, 0) / reps.length)
    : 0;
  const avgDur = reps.length ? reps.reduce((s, r) => s + r.duration, 0) / reps.length : 0;

  return {
    kind: "reps",
    reps,
    avgScore,
    frontView: false,
    statsText: `루마니안 데드 ${reps.length}회 감지 · 평균 ${avgDur.toFixed(1)}초/회`,
    hipSeries,
    metrics,
  };
}

function buildRdlRep(metrics, startIdx, minIdx, endIdx) {
  const bottom = metrics[minIdx];
  const duration = metrics[endIdx].t - metrics[startIdx].t;
  const issues = [];
  let score = 100;

  let depth;
  if (bottom.hipAngle <= 100) {
    depth = { label: "충분한 가동범위", cls: "good" };
  } else if (bottom.hipAngle <= 120) {
    depth = { label: "적정 가동범위", cls: "ok" };
  } else {
    depth = { label: "가동범위 짧음", cls: "warn" };
    issues.push("🔽 힌지가 얕아요. 햄스트링(허벅지 뒤)이 당기는 느낌이 들 때까지 엉덩이를 더 뒤로 밀어보세요.");
    score -= 12;
  }

  // 구간 내 최소 무릎 각도 / 최대 바 이탈
  let minKnee = 180, maxDrift = -Infinity;
  for (let i = startIdx; i <= endIdx; i++) {
    if (metrics[i].kneeAngle < minKnee) minKnee = metrics[i].kneeAngle;
    if (metrics[i].barDrift > maxDrift) maxDrift = metrics[i].barDrift;
  }
  if (minKnee < 120) {
    issues.push("🦵 무릎이 많이 굽었어요. 루마니안 데드는 무릎 각도를 거의 고정하고 엉덩이만 뒤로 미는 운동이에요.");
    score -= 12;
  }
  if (maxDrift > 0.45) {
    issues.push("📏 바(손)가 다리에서 멀어졌어요. 바를 허벅지에 스치듯 내려야 허리가 안전해요.");
    score -= 15;
  }

  if (duration < 1.5) {
    issues.push("⏱️ 템포가 빨라요. 내려갈 때 3초 정도로 천천히, 햄스트링의 저항을 느끼며 통제해 보세요.");
    score -= 8;
  }

  score = Math.max(40, score);
  return {
    startT: metrics[startIdx].t,
    bottomT: bottom.t,
    endT: metrics[endIdx].t,
    duration,
    depth,
    issues,
    score,
    metaText: `최저점 힙 각도 ${Math.round(bottom.hipAngle)}° · 무릎 ${Math.round(minKnee)}° · ${duration.toFixed(1)}초`,
  };
}

// ---- 랫풀다운 분석 ----
function analyzeLatpull(frames, vw, vh) {
  const useLeft = pickSide(frames, [
    [LM.L_SHOULDER, LM.R_SHOULDER],
    [LM.L_ELBOW, LM.R_ELBOW],
    [LM.L_WRIST, LM.R_WRIST],
  ]);
  const S = useLeft
    ? { sh: LM.L_SHOULDER, el: LM.L_ELBOW, wr: LM.L_WRIST, hip: LM.L_HIP }
    : { sh: LM.R_SHOULDER, el: LM.R_ELBOW, wr: LM.R_WRIST, hip: LM.R_HIP };

  const metrics = frames.map((f) => {
    const sh = px(f.lm, S.sh, vw, vh);
    const el = px(f.lm, S.el, vw, vh);
    const wr = px(f.lm, S.wr, vw, vh);
    const hip = px(f.lm, S.hip, vw, vh);
    const torso = Math.hypot(sh.x - hip.x, sh.y - hip.y) || 1;
    // 손목이 어깨보다 얼마나 위에 있나 (몸통 길이 배수) — 팔 뻗으면 ~1.2+, 당기면 0 근처
    const wristRel = (sh.y - wr.y) / torso;
    const dx = sh.x - hip.x, dy = hip.y - sh.y;
    const torsoLean = (Math.atan2(Math.abs(dx), Math.max(dy, 1)) * 180) / Math.PI;
    const elbowAngle = angleAt(sh, el, wr);
    return { t: f.t, wristRel, torsoLean, elbowAngle };
  });

  const relSeries = median3(metrics.map((m) => m.wristRel));

  // 기준 상체 기울기: 팔을 뻗고 있는(당기기 전) 프레임들의 중앙값
  const topLeans = metrics.filter((m) => m.wristRel > 0.8).map((m) => m.torsoLean).sort((a, b) => a - b);
  const baseLean = topLeans.length ? topLeans[Math.floor(topLeans.length / 2)] : 10;

  const reps = detectReps(relSeries, { startBelow: 0.7, endAbove: 0.8, validBelow: 0.45 })
    // 진짜 당김은 최저점에서 팔꿈치가 굽어 있어야 함 — 바 잡으러 뻗는 준비 동작 걸러내기
    .filter((r) => metrics[r.minIdx].elbowAngle < 120)
    .map((r) => buildLatpullRep(metrics, r.startIdx, r.minIdx, r.endIdx, baseLean));

  const avgScore = reps.length
    ? Math.round(reps.reduce((s, r) => s + r.score, 0) / reps.length)
    : 0;
  const avgDur = reps.length ? reps.reduce((s, r) => s + r.duration, 0) / reps.length : 0;

  return {
    kind: "reps",
    reps,
    avgScore,
    frontView: false,
    statsText: `랫풀다운 ${reps.length}회 감지 · 평균 ${avgDur.toFixed(1)}초/회`,
    relSeries,
    metrics,
  };
}

function buildLatpullRep(metrics, startIdx, minIdx, endIdx, baseLean) {
  const bottom = metrics[minIdx];
  const duration = metrics[endIdx].t - metrics[startIdx].t;
  const issues = [];
  let score = 100;

  let depth;
  if (bottom.wristRel <= 0.1) {
    depth = { label: "충분히 당김", cls: "good" };
  } else if (bottom.wristRel <= 0.3) {
    depth = { label: "적정 범위", cls: "ok" };
  } else {
    depth = { label: "당김 짧음", cls: "warn" };
    issues.push("🔽 조금 더 당겨보세요. 바가 쇄골 근처까지 내려오는 게 목표예요.");
    score -= 10;
  }

  // 구간 내 최대 상체 기울기 (반동 판정)
  let maxLean = 0;
  for (let i = startIdx; i <= endIdx; i++) {
    if (metrics[i].torsoLean > maxLean) maxLean = metrics[i].torsoLean;
  }
  if (maxLean - baseLean > 18) {
    issues.push("🪑 반동으로 상체가 뒤로 많이 넘어갔어요. 가슴만 살짝 열고, 광배(등) 힘으로만 당겨보세요.");
    score -= 15;
  }

  if (duration < 1.0) {
    issues.push("⏱️ 템포가 빨라요. 특히 바를 올릴 때 천천히 버티면 등 자극이 커져요.");
    score -= 8;
  }

  score = Math.max(40, score);
  return {
    startT: metrics[startIdx].t,
    bottomT: bottom.t,
    endT: metrics[endIdx].t,
    duration,
    depth,
    issues,
    score,
    metaText: `최저점 팔꿈치 각도 ${Math.round(bottom.elbowAngle)}° · 상체 반동 +${Math.max(0, Math.round(maxLean - baseLean))}° · ${duration.toFixed(1)}초`,
  };
}

// ---- 결과 렌더링 ----
function gradeOf(score) {
  return score >= 90 ? "훌륭해요! 💪" :
    score >= 75 ? "좋아요 👍" :
    score >= 60 ? "괜찮아요, 조금만 다듬으면 돼요" :
    "개선 포인트가 보여요";
}

function seekPlayer(t) {
  player.currentTime = t;
  player.pause();
}

function renderResult(duration) {
  showScreen("result");
  $("view-warning").hidden = !(exercise === "squat" && analysis.frontView);
  const note = EX_INFO[exercise].note;
  $("ex-note").textContent = note || "";
  $("ex-note").hidden = !note;

  player.src = currentUrl;
  player.load();

  const timeline = $("timeline");
  timeline.innerHTML = "";

  if (analysis.kind === "reps") {
    const { reps, avgScore, statsText } = analysis;

    const issueCount = {};
    reps.forEach((r) => r.issues.forEach((i) => (issueCount[i] = (issueCount[i] || 0) + 1)));
    const topIssues = Object.entries(issueCount).sort((a, b) => b[1] - a[1]).slice(0, 3);
    const lines = topIssues.length
      ? topIssues
          .map(([msg, n]) => `<p>${msg} <span style="color:var(--muted)">(${n}/${reps.length}회)</span></p>`)
          .join("")
      : "<p>✅ 모든 회차에서 특별한 문제를 찾지 못했어요. 지금 폼을 유지하세요!</p>";

    renderSummary(avgScore, statsText, lines);
    $("timeline-label").textContent = "회차별 최저 지점 — 눌러서 이동";

    reps.forEach((r, i) => {
      timeline.appendChild(makeMarker(i + 1, r.bottomT / duration, r.issues.length > 0, () => seekPlayer(r.bottomT)));
    });

    $("rep-cards").innerHTML = reps
      .map(
        (r, i) => `
      <div class="rep-card" data-t="${r.bottomT}">
        <div class="rep-head">
          <span class="rep-num">${i + 1}회차</span>
          <span class="badge ${r.depth.cls}">${r.depth.label}</span>
          <span class="rep-score">${r.score}점</span>
        </div>
        <div class="rep-meta">${r.metaText}</div>
        ${
          r.issues.length
            ? `<div class="rep-issues">${r.issues.map((x) => `<div>${x}</div>`).join("")}</div>`
            : `<div class="rep-clean">✅ 좋은 자세예요</div>`
        }
      </div>`
      )
      .join("");
  } else {
    // 플랭크(유지형)
    const { score, statsText, lines, segments, holdStart } = analysis;
    renderSummary(score, statsText, lines.map((l) => `<p>${l}</p>`).join(""));
    $("timeline-label").textContent = "자세가 무너진 구간 — 눌러서 이동";

    if (segments.length === 0) {
      timeline.appendChild(makeMarker("✓", holdStart / duration, false, () => seekPlayer(holdStart)));
    }
    segments.forEach((s, i) => {
      timeline.appendChild(makeMarker(i + 1, s.startT / duration, true, () => seekPlayer(s.startT)));
    });

    const fmt = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
    $("rep-cards").innerHTML = segments.length
      ? segments
          .map(
            (s, i) => `
        <div class="rep-card" data-t="${s.startT}">
          <div class="rep-head">
            <span class="rep-num">구간 ${i + 1}</span>
            <span class="badge ${s.type === "sag" ? "bad" : "warn"}">${s.type === "sag" ? "엉덩이 처짐" : "엉덩이 솟음"}</span>
            <span class="rep-score">${fmt(s.startT)}~${fmt(s.endT)}</span>
          </div>
          <div class="rep-meta">${
            s.type === "sag"
              ? "허리에 부담이 가는 자세예요. 배에 힘을 주고 골반을 살짝 말아 올려 보세요."
              : "코어 자극이 줄어드는 자세예요. 엉덩이를 낮춰 일직선을 만들어 보세요."
          }</div>
        </div>`
          )
          .join("")
      : `<div class="rep-card"><div class="rep-clean">✅ 유지 시간 내내 무너진 구간이 없어요. 훌륭한 플랭크예요!</div></div>`;
  }

  document.querySelectorAll(".rep-card[data-t]").forEach((card) => {
    card.onclick = () => seekPlayer(parseFloat(card.dataset.t));
  });

  setupOverlay();
}

function renderSummary(score, statsText, linesHtml) {
  $("summary-card").innerHTML = `
    <div class="summary-top">
      <div class="score-ring" style="--pct:${score}">
        <div class="num">${score}</div>
        <div class="cap">${analysis.kind === "hold" ? "자세 점수" : "평균 점수"}</div>
      </div>
      <div>
        <div class="summary-title">${gradeOf(score)}</div>
        <div class="summary-stats">${statsText}</div>
      </div>
    </div>
    <div class="summary-lines">${linesHtml}</div>
  `;
}

function makeMarker(label, pos, hasIssue, onClick) {
  const btn = document.createElement("button");
  btn.className = "rep-marker" + (hasIssue ? " has-issue" : "");
  btn.style.left = `${Math.min(0.97, Math.max(0.03, pos)) * 100}%`;
  btn.textContent = label;
  btn.onclick = onClick;
  return btn;
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
