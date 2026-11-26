// Persistent User Identity (survives session endings, restarts, and reloads)
let currentUserId = localStorage.getItem("alex_user_id");
if (!currentUserId) {
  currentUserId = "user_" + crypto.randomUUID().substring(0, 12);
  localStorage.setItem("alex_user_id", currentUserId);
}

let currentSessionId = crypto.randomUUID();
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let isProcessingTurn = false;
let currentAudio = null;
let completedTurns = 0;
let pollInterval = null;
let pendingTurnResolve = null;
let lastTurnData = null;
let playbackGenerationId = 0;
let playbackTimer = null;

// DOM Elements — Chat UI Architecture
const ph = document.getElementById("ph");
const chat = document.getElementById("chat");
const mb = document.getElementById("mb");
const pt = document.getElementById("pt");
const keyNotice = document.getElementById("keyNotice");
const keyNoticeText = document.getElementById("keyNoticeText");

// Keep 100dvh synced with visualViewport on mobile keyboards
if (window.visualViewport) {
  const syncViewportHeight = () => {
    if (window.innerWidth < 500) {
      ph.style.height = window.visualViewport.height + "px";
    } else {
      ph.style.height = "";
    }
  };
  window.visualViewport.addEventListener("resize", syncViewportHeight);
  syncViewportHeight();
}

// ---------------------------------------------------------------------------
// Health check on load
// ---------------------------------------------------------------------------
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (!data.groq_configured && !data.gemini_configured) {
      keyNotice.classList.remove("hidden");
      keyNoticeText.innerHTML = `Missing in <strong>.env</strong>: <strong>GROQ_API_KEY</strong>. Please add your key and refresh!`;
    } else {
      keyNotice.classList.add("hidden");
    }
  } catch (e) {
    console.warn("Health check error:", e);
  }
}
checkHealth();

// ---------------------------------------------------------------------------
// UI State Machine
// ---------------------------------------------------------------------------
const STATE_CONFIG = {
  ready: {
    pill: "Ready when you are",
    btn: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round"><rect x="9" y="3" width="6" height="12" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>Tap to speak',
    state: "idle"
  },
  idle: {
    pill: "Ready when you are",
    btn: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round"><rect x="9" y="3" width="6" height="12" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>Tap to speak',
    state: "idle"
  },
  listening: {
    pill: "Listening…",
    btn: '<span class="bars"><i></i><i></i><i></i><i></i></span>Tap when you’re done',
    state: "listening"
  },
  thinking: {
    pill: "Thinking…",
    btn: '<span class="spin"></span>Preparing a reply',
    state: "thinking"
  },
  speaking: {
    pill: "Alex is speaking…",
    btn: '<span class="sq"></span>Tap to interrupt',
    state: "speaking"
  }
};

function setUIState(state) {
  const cfg = STATE_CONFIG[state];
  if (!cfg) return;
  pt.textContent = cfg.pill;
  mb.innerHTML = cfg.btn;
  ph.dataset.state = cfg.state;
  if (window.setGlobeState) window.setGlobeState(cfg.state);
}

setUIState("ready");

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------
async function startRecording() {
  stopCurrentPlayback();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    audioChunks = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = sendAudioToServer;
    mediaRecorder.start();

    // Connect live mic level to globe visualizer
    try {
      const _ac = new (window.AudioContext || window.webkitAudioContext)();
      const _src = _ac.createMediaStreamSource(stream);
      const _an = _ac.createAnalyser();
      _an.fftSize = 256;
      _src.connect(_an);
      if (window.connectGlobeMic) window.connectGlobeMic(_an);
      window._alexMicAudioContext = _ac;
    } catch (_e) { /* non-critical */ }

    isRecording = true;
    setUIState("listening");
  } catch (err) {
    console.error("Microphone access error:", err);
    alert("Please allow microphone access in your browser to speak with Alex.");
    setUIState("ready");
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    if (window.disconnectGlobeMic) window.disconnectGlobeMic();
    if (window._alexMicAudioContext) {
      window._alexMicAudioContext.close().catch(() => {});
      window._alexMicAudioContext = null;
    }
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    isRecording = false;
    setUIState("thinking");
  }
}

// ---------------------------------------------------------------------------
// Chat Bubble & Correction Deduplication Helpers
// ---------------------------------------------------------------------------
function scrollDown() {
  setTimeout(() => { chat.scrollTop = chat.scrollHeight; }, 60);
  setTimeout(() => { chat.scrollTop = chat.scrollHeight; }, 750);
}

/**
 * Deduplicates expression/style cards that overlap with grammar corrections
 * and eliminates identical rendered items while preserving original indices.
 */
function deduplicateCorrections(corrections) {
  if (!corrections || !Array.isArray(corrections) || corrections.length === 0) {
    return [];
  }

  const withIdx = corrections
    .map((c, idx) => ({ item: c, origIdx: idx }))
    .filter(entry => entry.item && typeof entry.item === "object");

  const grammarCards = withIdx.filter(entry => {
    const t = String(entry.item.type || "").trim().toLowerCase();
    return t !== "expression" && t !== "natural_phrasing" && t !== "style" && t !== "word_choice";
  });

  const filtered = [];
  const seenKeys = new Set();

  for (const entry of withIdx) {
    const c = entry.item;
    const t = String(c.type || "").trim().toLowerCase();
    const isExpr = (t === "expression" || t === "natural_phrasing" || t === "style" || t === "word_choice");

    if (isExpr && grammarCards.length > 0) {
      const origLow = String(c.original || "").toLowerCase().trim();
      const corrLow = String(c.corrected || "").toLowerCase().trim();
      const foLow = String(c.full_original_sentence || "").toLowerCase().trim();
      const isDup = grammarCards.some(({ item: g }) => {
        const gOrig = String(g.original || "").toLowerCase().trim();
        const gCorr = String(g.corrected || "").toLowerCase().trim();
        const gFo = String(g.full_original_sentence || "").toLowerCase().trim();
        return (origLow && gOrig && (origLow.includes(gOrig) || gOrig.includes(origLow))) ||
               (corrLow && gCorr && (corrLow.includes(gCorr) || gCorr.includes(corrLow))) ||
               (foLow && gFo && foLow === gFo);
      });
      if (isDup) continue;
    }

    const rawKey = `${(c.original || "").trim()}:::${(c.corrected || "").trim()}`.toLowerCase();
    if (rawKey !== ":::" && seenKeys.has(rawKey)) continue;
    seenKeys.add(rawKey);

    filtered.push(entry);
  }

  return filtered;
}

/**
 * Safely highlights wrong phrases inside the raw transcript with <mark data-k="i">
 * without corrupting HTML entities or overlapping previously inserted tags.
 */
function buildMarkedTranscriptHtml(transcript, entries) {
  if (!transcript) return "";
  if (!entries || entries.length === 0) return escapeHtml(transcript);

  const normalize = (s) => s.replace(/[\u2018\u2019]/g, "'").toLowerCase();
  const normTranscript = normalize(transcript);
  const ranges = [];

  entries.forEach(({ item }, k) => {
    const orig = (item.original || "").trim();
    if (!orig) return;

    let start = transcript.indexOf(orig);
    let len = orig.length;

    if (start === -1) {
      const normOrig = normalize(orig);
      start = normTranscript.indexOf(normOrig);
      len = normOrig.length;
    }

    if (start !== -1) {
      const end = start + len;
      const overlaps = ranges.some(r => start < r.end && end > r.start);
      if (!overlaps) {
        ranges.push({ start, end, k });
      }
    }
  });

  if (ranges.length === 0) return escapeHtml(transcript);

  ranges.sort((a, b) => a.start - b.start);
  let out = "";
  let cursor = 0;

  for (const r of ranges) {
    if (r.start > cursor) {
      out += escapeHtml(transcript.slice(cursor, r.start));
    }
    out += `<mark data-k="${r.k}">${escapeHtml(transcript.slice(r.start, r.end))}</mark>`;
    cursor = r.end;
  }
  if (cursor < transcript.length) {
    out += escapeHtml(transcript.slice(cursor));
  }

  return out;
}

function createUserBubble(transcript, rawCorrections, turnData) {
  const emp = document.getElementById("emp");
  if (emp) emp.remove();

  // Collapse previously opened correction panels when new user turn arrives
  document.querySelectorAll(".grp.open").forEach(g => g.classList.remove("open"));
  ph.dataset.mode = "docked";

  const entries = deduplicateCorrections(rawCorrections);
  const markedHtml = buildMarkedTranscriptHtml(transcript, entries);
  const corrCount = entries.length;

  const grp = document.createElement("div");
  grp.className = "grp";

  const bu = document.createElement("div");
  bu.className = "bu";
  bu.innerHTML = markedHtml;
  grp.appendChild(bu);

  const tipElsMap = {};

  if (corrCount > 0) {
    const fixBtn = document.createElement("button");
    fixBtn.type = "button";
    fixBtn.className = "fix";
    fixBtn.innerHTML = `✎ ${corrCount} correction${corrCount !== 1 ? "s" : ""} <b>▾</b>`;

    const cp = document.createElement("div");
    cp.className = "cp";

    const cd = document.createElement("div");
    cd.className = "cd";

    const th = document.createElement("div");
    th.className = "th";
    th.innerHTML = `<h3>ALEX’S ENGLISH TIPS</h3><span class="chip">${corrCount} targeted improvement${corrCount !== 1 ? "s" : ""}</span>`;
    cd.appendChild(th);

    entries.forEach(({ item: c, origIdx }) => {
      const t = String(c.type || "").trim().toLowerCase();
      const isExpr = (t === "expression" || t === "natural_phrasing" || t === "style" || t === "word_choice");

      const origText = (c.original || c.full_original_sentence || "").trim();
      const corrText = (c.corrected || c.full_corrected_sentence || "").trim();

      const tip = document.createElement("div");
      tip.className = "tip correction-item";

      const tr = document.createElement("div");
      tr.className = "tr";

      const gc = document.createElement("span");
      gc.className = isExpr ? "gc expr" : "gc";
      gc.textContent = isExpr ? "💬 More natural phrasing" : "✎ Grammar correction";

      const playBtn = document.createElement("button");
      playBtn.type = "button";
      playBtn.className = "pl";
      playBtn.textContent = "▶ Tip audio";
      playBtn.onclick = () => replayCorrectionForTurn(turnData, origIdx, c, tip);

      tr.appendChild(gc);
      tr.appendChild(playBtn);
      tip.appendChild(tr);

      const lnOrig = document.createElement("div");
      lnOrig.className = "ln";
      lnOrig.innerHTML = `Instead of:<s>${escapeHtml(origText)}</s>`;

      const lnCorr = document.createElement("div");
      lnCorr.className = "ln";
      lnCorr.innerHTML = `Say:<u>${escapeHtml(corrText)}</u>`;

      tip.appendChild(lnOrig);
      tip.appendChild(lnCorr);

      if (c.explanation) {
        const why = document.createElement("div");
        why.className = "why";
        why.innerHTML = `<b>Why:</b> ${escapeHtml(c.explanation)}`;
        tip.appendChild(why);
      }

      tipElsMap[origIdx] = tip;
      cd.appendChild(tip);
    });

    cp.appendChild(cd);
    grp.appendChild(fixBtn);
    grp.appendChild(cp);

    const toggleOpen = () => {
      grp.classList.toggle("open");
      if (grp.classList.contains("open")) {
        setTimeout(() => grp.scrollIntoView({ block: "nearest", behavior: "smooth" }), 520);
      }
    };

    fixBtn.onclick = toggleOpen;
    bu.querySelectorAll("mark").forEach(m => {
      m.onclick = () => {
        if (!grp.classList.contains("open")) toggleOpen();
      };
    });
  }

  return { grp, tipElsMap, corrCount };
}

function addAlexBubble(text, isTyping) {
  const emp = document.getElementById("emp");
  if (emp) emp.remove();
  ph.dataset.mode = "docked";

  const msg = document.createElement("div");
  msg.className = "msg";
  msg.innerHTML = '<div class="av"></div><div class="ba">' +
    (isTyping ? '<div class="dots"><i></i><i></i><i></i></div>' : escapeHtml(text)) + '</div>';
  chat.appendChild(msg);
  scrollDown();
  return msg;
}

// ---------------------------------------------------------------------------
// Core Turn Processing
// ---------------------------------------------------------------------------
async function sendAudioToServer() {
  if (audioChunks.length === 0) {
    setUIState("ready");
    return;
  }

  isProcessingTurn = true;
  const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
  const formData = new FormData();
  formData.append("audio", audioBlob, "user_turn.webm");
  formData.append("session_id", currentSessionId);
  formData.append("user_id", currentUserId);

  // Immediately show Alex typing bubble while preparing reply
  const hadEmptyBefore = Boolean(document.getElementById("emp"));
  const typingBubble = addAlexBubble("", true);

  try {
    const res = await fetch("/api/turn", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Server error occurred");
    }

    let tipElsMap = {};

    if (data.user_transcript && data.user_transcript.trim()) {
      completedTurns += 1;

      const created = createUserBubble(data.user_transcript, data.corrections || [], data);
      tipElsMap = created.tipElsMap;

      // Place user bubble immediately before Alex's typing bubble
      chat.insertBefore(created.grp, typingBubble);

      // Auto-open newest corrections card after 500ms
      if (created.corrCount > 0) {
        setTimeout(() => {
          created.grp.classList.add("open");
          scrollDown();
        }, 500);
      }
    }

    // Replace typing dots with Alex's reply text
    const aiBubbleEl = typingBubble.querySelector(".ba");
    aiBubbleEl.textContent = data.response_text || "";
    aiBubbleEl.classList.add("playable");
    aiBubbleEl.title = "Tap to replay Alex’s voice";
    aiBubbleEl.onclick = () => playTurnSegments(data, tipElsMap, aiBubbleEl);
    scrollDown();

    logMetric(data);

    lastTurnData = data;
    lastTurnData._tipElsMap = tipElsMap;
    lastTurnData._aiBubbleEl = aiBubbleEl;

    playTurnSegments(data, tipElsMap, aiBubbleEl);

  } catch (err) {
    console.error("Turn processing error:", err);
    if (typingBubble && typingBubble.parentNode) {
      typingBubble.remove();
    }
    if (hadEmptyBefore && completedTurns === 0 && chat.children.length === 0) {
      chat.innerHTML = '<div class="empty" id="emp">Tap the mic and speak. I’ll reply and gently fix your English.</div>';
      ph.dataset.mode = "hero";
    }
    alert("⚠️ " + err.message);
    setUIState("ready");
    checkHealth();
  } finally {
    isProcessingTurn = false;
    if (pendingTurnResolve) {
      pendingTurnResolve();
      pendingTurnResolve = null;
    }
  }
}

// ---------------------------------------------------------------------------
// Grammar Correction Card Rendering & Minimum-Difference Highlighting
// (Kept intact for LCS engine tests & backward compatibility)
// ---------------------------------------------------------------------------
function renderCorrectionPortion(c, cardIndex = 0) {
  if (!c || typeof c !== "object") {
    return { htmlOrig: "", htmlCorr: "" };
  }

  let orig = (c.original || "").trim();
  let corr = (c.corrected || "").trim();

  if (!orig && c.full_original_sentence) orig = c.full_original_sentence.trim();
  if (!corr && c.full_corrected_sentence) corr = c.full_corrected_sentence.trim();

  function escapeHtmlInner(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  const redSpan = (text) => `<span class="text-red-400 font-semibold" style="color: #f87171; font-weight: 600;">${escapeHtmlInner(text)}</span>`;
  const greenSpan = (text) => `<span class="text-green-400 font-semibold" style="color: #4ade80; font-weight: 600;">${escapeHtmlInner(text)}</span>`;
  const neutralSpan = (text) => `<span class="text-slate-300">${escapeHtmlInner(text)}</span>`;

  if (!orig && !corr) return { htmlOrig: "", htmlCorr: "" };
  if (!orig) return { htmlOrig: "", htmlCorr: greenSpan(corr) };
  if (!corr) return { htmlOrig: redSpan(orig), htmlCorr: "" };

  function cleanWord(w) {
    return (w || "").toLowerCase().replace(/^[^\w]+|[^\w]+$/g, "");
  }

  if (orig.startsWith('"') && orig.endsWith('"') && orig.length > 2) orig = orig.slice(1, -1).trim();
  if (corr.startsWith('"') && corr.endsWith('"') && corr.length > 2) corr = corr.slice(1, -1).trim();

  const wordsOrig = orig.split(/\s+/).filter(Boolean);
  const wordsCorr = corr.split(/\s+/).filter(Boolean);

  if (wordsOrig.length === 1 && wordsCorr.length === 1) {
    if (cleanWord(wordsOrig[0]) !== cleanWord(wordsCorr[0]) || wordsOrig[0] !== wordsCorr[0]) {
      return { htmlOrig: redSpan(orig), htmlCorr: greenSpan(corr) };
    }
    return { htmlOrig: neutralSpan(orig), htmlCorr: neutralSpan(corr) };
  }

  const N = wordsOrig.length;
  const M = wordsCorr.length;
  const dp = Array.from({ length: N + 1 }, () => new Array(M + 1).fill(0));

  for (let i = 0; i < N; i++) {
    for (let j = 0; j < M; j++) {
      const w1 = cleanWord(wordsOrig[i]);
      const w2 = cleanWord(wordsCorr[j]);
      if (wordsOrig[i] === wordsCorr[j]) {
        dp[i + 1][j + 1] = dp[i][j] + 2;
      } else if (w1 && w1 === w2) {
        dp[i + 1][j + 1] = dp[i][j] + 1.5;
      } else {
        dp[i + 1][j + 1] = Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
  }

  let i = N, j = M;
  const diff = [];
  while (i > 0 || j > 0) {
    const matchExact = i > 0 && j > 0 && wordsOrig[i - 1] === wordsCorr[j - 1];
    const matchClean = i > 0 && j > 0 && cleanWord(wordsOrig[i - 1]) && cleanWord(wordsOrig[i - 1]) === cleanWord(wordsCorr[j - 1]);

    if (matchExact || matchClean) {
      diff.unshift({ type: "equal", orig: wordsOrig[i - 1], corr: wordsCorr[j - 1] });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      diff.unshift({ type: "insert", orig: null, corr: wordsCorr[j - 1] });
      j--;
    } else if (i > 0) {
      diff.unshift({ type: "delete", orig: wordsOrig[i - 1], corr: null });
      i--;
    }
  }

  const isParticipleToFinite = wordsOrig.length === 2 && wordsCorr.length === 2 &&
    wordsOrig[1].toLowerCase().endsWith("ing") && !wordsCorr[1].toLowerCase().endsWith("ing");

  const equalCount = diff.filter(d => d.type === "equal").length;
  const maxWords = Math.max(N, M);
  const isCompleteReconstruction = isParticipleToFinite || (equalCount === 0) || (N <= 8 && M <= 8 && equalCount <= 2 && (equalCount / maxWords) < 0.30);

  if (isCompleteReconstruction) {
    return { htmlOrig: redSpan(orig), htmlCorr: greenSpan(corr) };
  }

  for (let k = 0; k < diff.length - 2; k++) {
    if (diff[k].type === "delete" && diff[k + 1].type === "equal" && diff[k + 2].type === "insert") {
      if (cleanWord(diff[k].orig) === cleanWord(diff[k + 2].corr)) {
        diff.splice(k, 3, { type: "swap", origWords: [diff[k].orig, diff[k + 1].orig], corrWords: [diff[k + 1].corr, diff[k + 2].corr] });
      }
    } else if (diff[k].type === "insert" && diff[k + 1].type === "equal" && diff[k + 2].type === "delete") {
      if (cleanWord(diff[k].corr) === cleanWord(diff[k + 2].orig)) {
        diff.splice(k, 3, { type: "swap", origWords: [diff[k + 1].orig, diff[k + 2].orig], corrWords: [diff[k].corr, diff[k + 1].corr] });
      }
    }
  }

  let windowDiff = diff;
  const changedIndices = [];
  diff.forEach((item, idx) => {
    if (item.type !== "equal") changedIndices.push(idx);
  });

  if (changedIndices.length > 0 && diff.length > 5) {
    const clusters = [];
    let curCluster = [changedIndices[0]];
    for (let cIdx = 1; cIdx < changedIndices.length; cIdx++) {
      if (changedIndices[cIdx] - changedIndices[cIdx - 1] <= 2) {
        curCluster.push(changedIndices[cIdx]);
      } else {
        clusters.push(curCluster);
        curCluster = [changedIndices[cIdx]];
      }
    }
    clusters.push(curCluster);

    const targetCluster = (cardIndex !== undefined && cardIndex < clusters.length) ? clusters[cardIndex] : clusters[0];
    const startIdx = Math.max(0, targetCluster[0] - 1);
    const endIdx = Math.min(diff.length, targetCluster[targetCluster.length - 1] + 2);
    windowDiff = diff.slice(startIdx, endIdx);
  }

  const origSegments = [];
  let currOrigChunk = null;
  for (const item of windowDiff) {
    if (item.type === "equal") {
      if (!currOrigChunk || currOrigChunk.type !== "equal") {
        currOrigChunk = { type: "equal", words: [] };
        origSegments.push(currOrigChunk);
      }
      currOrigChunk.words.push(item.orig);
    } else if (item.type === "delete") {
      if (!currOrigChunk || currOrigChunk.type !== "changed") {
        currOrigChunk = { type: "changed", words: [] };
        origSegments.push(currOrigChunk);
      }
      currOrigChunk.words.push(item.orig);
    } else if (item.type === "swap") {
      currOrigChunk = { type: "changed", words: item.origWords };
      origSegments.push(currOrigChunk);
    }
  }

  const htmlOrig = origSegments.map(seg => {
    const text = seg.words.join(" ");
    return seg.type === "changed" ? redSpan(text) : neutralSpan(text);
  }).join(" ");

  const corrSegments = [];
  let currCorrChunk = null;
  for (const item of windowDiff) {
    if (item.type === "equal") {
      if (!currCorrChunk || currCorrChunk.type !== "equal") {
        currCorrChunk = { type: "equal", words: [] };
        corrSegments.push(currCorrChunk);
      }
      currCorrChunk.words.push(item.corr);
    } else if (item.type === "insert") {
      if (!currCorrChunk || currCorrChunk.type !== "changed") {
        currCorrChunk = { type: "changed", words: [] };
        corrSegments.push(currCorrChunk);
      }
      currCorrChunk.words.push(item.corr);
    } else if (item.type === "swap") {
      currCorrChunk = { type: "changed", words: item.corrWords };
      corrSegments.push(currCorrChunk);
    }
  }

  const htmlCorr = corrSegments.map(seg => {
    const text = seg.words.join(" ");
    return seg.type === "changed" ? greenSpan(text) : neutralSpan(text);
  }).join(" ");

  return { htmlOrig, htmlCorr };
}

function renderCorrectionCard(corrections) {
  console.log("[renderCorrectionCard] corrections:", corrections);
}

// ---------------------------------------------------------------------------
// Card Highlighting & Visual Synchronization
// ---------------------------------------------------------------------------
function highlightCard(cardIndex, tipElsMap, aiBubbleEl) {
  clearHighlights();
  if (cardIndex !== null && cardIndex !== undefined) {
    const el = tipElsMap && tipElsMap[cardIndex];
    if (el) el.classList.add("highlighted");
  } else if (aiBubbleEl) {
    aiBubbleEl.classList.add("highlighted");
  }
}

function clearHighlights() {
  document.querySelectorAll(".tip.highlighted, .ba.highlighted").forEach(el => {
    el.classList.remove("highlighted");
  });
}

// ---------------------------------------------------------------------------
// Audio Playback & Adaptive Speech Timing
// ---------------------------------------------------------------------------
function playTurnSegments(turnData, tipElsMap, aiBubbleEl) {
  stopCurrentPlayback();
  if (!turnData) return;

  const currentGen = playbackGenerationId;
  const segments = turnData.speech_segments || [];
  const tips = tipElsMap || turnData._tipElsMap || {};
  const bubble = aiBubbleEl || turnData._aiBubbleEl || null;

  if (!segments || segments.length === 0) {
    setUIState("speaking");
    if (turnData.audio_base64 && turnData.tts_engine === "edge-tts") {
      playBase64Audio(turnData.audio_base64);
    } else {
      playBrowserSpeechFallback(turnData.spoken_text || turnData.response_text || "");
    }
    return;
  }

  setUIState("speaking");

  let index = 0;

  function playNext() {
    if (playbackGenerationId !== currentGen) return;

    if (index >= segments.length) {
      clearHighlights();
      setUIState("ready");
      return;
    }

    const seg = segments[index];
    index++;

    if (!seg.audio_base64) {
      playNext();
      return;
    }

    if (seg.type && seg.type.startsWith("corr")) {
      highlightCard(seg.card_index, tips, bubble);
    } else {
      highlightCard(null, tips, bubble);
    }

    currentAudio = new Audio(`data:audio/mp3;base64,${seg.audio_base64}`);

    currentAudio.onended = () => {
      if (playbackGenerationId !== currentGen) return;
      const pauseMs = seg.pause_after_ms || 0;
      if (pauseMs > 0 && index < segments.length) {
        playbackTimer = setTimeout(() => {
          if (playbackGenerationId !== currentGen) return;
          playNext();
        }, pauseMs);
      } else {
        playNext();
      }
    };

    currentAudio.onerror = (err) => {
      console.warn("Audio segment error:", err);
      if (playbackGenerationId !== currentGen) return;
      playNext();
    };

    currentAudio.play().catch(e => {
      console.warn("Autoplay blocked on segment, falling back to next:", e);
      if (playbackGenerationId !== currentGen) return;
      playNext();
    });
  }

  playNext();
}

function replayCurrentTurn() {
  if (!lastTurnData) return;
  playTurnSegments(lastTurnData, lastTurnData._tipElsMap, lastTurnData._aiBubbleEl);
}

function replayAllCorrections() {
  if (!lastTurnData || !lastTurnData.speech_segments) return;
  const matched = lastTurnData.speech_segments.filter(s => s.type && s.type.startsWith("corr"));
  if (matched.length > 0) {
    playTurnSegments({ speech_segments: matched }, lastTurnData._tipElsMap, null);
  }
}

function replayCorrection(cardIndex) {
  if (!lastTurnData) return;
  const c = (lastTurnData.corrections || [])[cardIndex];
  const tipEl = lastTurnData._tipElsMap ? lastTurnData._tipElsMap[cardIndex] : null;
  replayCorrectionForTurn(lastTurnData, cardIndex, c, tipEl);
}

function replayCorrectionForTurn(turnData, cardIndex, correctionObj, tipEl) {
  const segs = (turnData && turnData.speech_segments) ? turnData.speech_segments : [];
  const matched = segs.filter(s => s.type && s.type.startsWith("corr") && s.card_index === cardIndex && s.audio_base64);

  if (matched.length > 0) {
    const map = tipEl ? { [cardIndex]: tipEl } : {};
    playTurnSegments({ speech_segments: matched }, map, null);
    return;
  }

  // Fallback for tips beyond synthesized top-2 segments
  if (correctionObj && (correctionObj.corrected || correctionObj.original)) {
    stopCurrentPlayback();
    setUIState("speaking");
    if (tipEl) tipEl.classList.add("highlighted");
    const orig = (correctionObj.original || "").trim();
    const corr = (correctionObj.corrected || "").trim();
    const spoken = (orig && corr) ? `Say "${corr}" instead of "${orig}".` : (corr || orig);
    playBrowserSpeechFallback(spoken);
  }
}

function playBase64Audio(base64Data) {
  stopCurrentPlayback();
  currentAudio = new Audio(`data:audio/mp3;base64,${base64Data}`);
  currentAudio.onended = () => {
    clearHighlights();
    setUIState("ready");
  };
  currentAudio.play().catch(e => {
    console.warn("Autoplay blocked, falling back to browser speech:", e);
    playBrowserSpeechFallback(lastTurnData ? lastTurnData.response_text || "" : "");
  });
}

function playBrowserSpeechFallback(text) {
  stopCurrentPlayback();
  if ("speechSynthesis" in window && text) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.onend = () => {
      clearHighlights();
      setUIState("ready");
    };
    window.speechSynthesis.speak(utterance);
  } else {
    clearHighlights();
    setUIState("ready");
  }
}

function stopCurrentPlayback() {
  playbackGenerationId++;
  if (playbackTimer) {
    clearTimeout(playbackTimer);
    playbackTimer = null;
  }
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.currentTime = 0;
    currentAudio = null;
  }
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  clearHighlights();
}

// ---------------------------------------------------------------------------
// Metrics Logging
// ---------------------------------------------------------------------------
function logMetric(data) {
  const m = data.latency_ms;
  if (!m) return;
  const corrCount = (data.corrections || []).length;
  const corrNote = corrCount > 0 ? ` | Corrections: ${corrCount}` : "";
  console.log(`[Turn ${completedTurns}] Total: ${m.total}ms | STT: ${m.stt}ms | LLM: ${m.llm}ms | TTS: ${m.tts}ms (${data.tts_engine})${corrNote}`);
}

// ---------------------------------------------------------------------------
// New Conversation / Topic Reset
// ---------------------------------------------------------------------------
function handleStartNewConversation() {
  stopCurrentPlayback();
  currentSessionId = crypto.randomUUID();
  completedTurns = 0;
  isProcessingTurn = false;
  pendingTurnResolve = null;
  chat.innerHTML = '<div class="empty" id="emp">Tap the mic and speak. I’ll reply and gently fix your English.</div>';
  ph.dataset.mode = "hero";
  document.getElementById("tst").classList.remove("show");
  setUIState("ready");
  console.log(`--- Started New Topic (${currentSessionId.substring(0, 8)}) ---`);
}

// ---------------------------------------------------------------------------
// Security: HTML injection prevention
// ---------------------------------------------------------------------------
function escapeHtml(str) {
  if (!str) return "";
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Event Listeners
// ---------------------------------------------------------------------------
mb.onclick = function() {
  if (isProcessingTurn) return;
  const currentState = ph.dataset.state;
  if (currentState === "idle") {
    startRecording();
  } else if (currentState === "listening") {
    stopRecording();
  } else if (currentState === "speaking") {
    stopCurrentPlayback();
    setUIState("ready");
  }
};

// ✦ button — toggles test-state chips + new topic reset
document.getElementById("ts").onclick = function() {
  document.getElementById("tst").classList.toggle("show");
};

// Populate test-state chips + reset chip
(function() {
  const tst = document.getElementById("tst");
  [["ready", "Idle"], ["listening", "User speaking"], ["speaking", "Alex speaking"], ["thinking", "Thinking"]].forEach(function(a) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = a[1];
    b.onclick = function() { setUIState(a[0]); };
    tst.appendChild(b);
  });
  const resetChip = document.createElement("button");
  resetChip.type = "button";
  resetChip.textContent = "↻ New topic";
  resetChip.onclick = handleStartNewConversation;
  tst.appendChild(resetChip);
})();

// Keyboard button handler
const kbBtn = document.getElementById("kbBtn");
if (kbBtn) {
  kbBtn.onclick = function() {
    document.getElementById("tst").classList.toggle("show");
  };
}

// ── SAPPHIRE PARTICLE GLOBE ─────────────────────────────────────────────────
(function() {
  var TP = 6.2832, globeSt = "idle", globeLv = 0, globeSp = 1, globePt = 0, globeT = 0;
  var globeMicAn = null, globeMicBuf = null;
  var G = [];
  for (var i = 0, N = 950; i < N; i++) {
    var y = 1 - 2 * (i + .5) / N, r = Math.sqrt(1 - y * y), a = i * 2.39996;
    G.push([Math.cos(a) * r, y, Math.sin(a) * r]);
  }
  var ce = document.querySelector(".gw canvas");
  if (!ce) return;
  var W = 280, D = Math.min(window.devicePixelRatio || 1, 2);
  ce.width = ce.height = W * D;
  var gx = ce.getContext("2d");

  window.setGlobeState = function(s) {
    var m = { ready: "idle", idle: "idle", listening: "listening", thinking: "thinking", speaking: "speaking" };
    globeSt = m[s] || "idle";
  };
  window.connectGlobeMic = function(an) {
    globeMicAn = an;
    globeMicBuf = new Uint8Array(an.frequencyBinCount);
  };
  window.disconnectGlobeMic = function() {
    globeMicAn = null;
    globeMicBuf = null;
  };

  function drawGlobe() {
    var c0 = W / 2, R = W * .26, ca = Math.cos(globePt * .5), sa = Math.sin(globePt * .5), cb = Math.cos(.42), sb = Math.sin(.42);
    gx.setTransform(D, 0, 0, D, 0, 0);
    gx.clearRect(0, 0, W, W);
    gx.globalCompositeOperation = "source-over";
    gx.strokeStyle = "rgba(154,185,255,.16)";
    gx.lineWidth = 1.2;
    gx.beginPath();
    gx.arc(c0, c0, R * 1.5, 0, TP);
    gx.stroke();

    if (globeSt === "thinking") {
      var an2 = globePt * 2.2;
      gx.fillStyle = "#ffe3a3";
      gx.shadowColor = "#ffd36e";
      gx.shadowBlur = 16;
      gx.beginPath();
      gx.arc(c0 + Math.cos(an2) * R * 1.5, c0 + Math.sin(an2) * R * 1.5, 3.6, 0, TP);
      gx.fill();
      gx.shadowBlur = 0;
    }

    gx.globalCompositeOperation = "lighter";
    var g = gx.createRadialGradient(c0, c0, 0, c0, c0, R * 1.4);
    g.addColorStop(0, "rgba(70,120,255," + (.18 + globeLv * .3) + ")");
    g.addColorStop(1, "rgba(0,0,0,0)");
    gx.fillStyle = g;
    gx.beginPath();
    gx.arc(c0, c0, R * 1.4, 0, TP);
    gx.fill();

    for (var i = 0; i < G.length; i++) {
      var p = G[i];
      var n = Math.sin(p[0] * 3 + globeT * 1.5) * Math.sin(p[1] * 4 - globeT * 1.2) * Math.sin(p[2] * 3 + globeT * .9);
      var rr = R * (1 + globeLv * .2 * n + globeLv * .04);
      var X = p[0] * rr, Y = p[1] * rr, Z = p[2] * rr;
      var x1 = X * ca + Z * sa, z1 = -X * sa + Z * ca;
      var y1 = Y * cb - z1 * sb, z2 = Y * sb + z1 * cb;
      var f = 300 / (300 - z2), d = (z2 + R) / (2 * R);
      gx.fillStyle = "hsla(" + (205 + (p[1] + 1) * 25 + n * 10) + ",92%," + (58 + d * 26) + "%," + (.12 + .74 * d) + ")";
      gx.beginPath();
      gx.arc(c0 + x1 * f, c0 + y1 * f, .55 + d * 1.4, 0, TP);
      gx.fill();
    }
  }

  function globeFrame() {
    globeT += .016;
    var e = Math.abs(Math.sin(globeT * 5.3) * Math.sin(globeT * 1.9 + 1) + .5 * Math.sin(globeT * 11.7));
    var tgLv = .05, tgSp = .35;
    if (globeSt === "listening") {
      tgLv = .15 + .3 * e;
      tgSp = .8;
      if (globeMicAn && globeMicBuf) {
        globeMicAn.getByteFrequencyData(globeMicBuf);
        var s = 0;
        for (var i = 0; i < globeMicBuf.length; i++) s += globeMicBuf[i];
        var micLevel = Math.min(1, s / globeMicBuf.length / 55);
        tgLv = Math.max(micLevel, .10 + .12 * e);
      }
    } else if (globeSt === "speaking") {
      tgLv = .3 + .65 * e;
      tgSp = 1.3;
    } else if (globeSt === "thinking") {
      tgLv = .12;
      tgSp = 2.6;
    }

    globeLv += (tgLv - globeLv) * .14;
    globeSp += (tgSp - globeSp) * .05;
    globePt += .016 * globeSp;

    drawGlobe();
    requestAnimationFrame(globeFrame);
  }

  globeFrame();
})();
