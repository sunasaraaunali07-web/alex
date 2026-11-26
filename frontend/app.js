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

// DOM Elements — Voice View
const voiceView = document.getElementById("voiceView");
const controlFooter = document.getElementById("controlFooter");
const talkBtn = document.getElementById("talkBtn");
const talkBtnText = document.getElementById("talkBtnText");
const stopAudioBtn = document.getElementById("stopAudioBtn");
const newChatBtn = document.getElementById("newChatBtn");
const statusLabel = document.getElementById("statusLabel");
const voiceOrb = document.getElementById("voiceOrb");
const stateEmoji = document.getElementById("stateEmoji");
const pulseRing = document.getElementById("pulseRing");
const metricsLog = document.getElementById("metricsLog");
const keyNotice = document.getElementById("keyNotice");
const keyNoticeText = document.getElementById("keyNoticeText");

// DOM Elements — Subtitle Cards
const userCard = document.getElementById("userCard");
const userTranscriptText = document.getElementById("userTranscriptText");
const correctionCard = document.getElementById("correctionCard");
const correctionCardContent = document.getElementById("correctionCardContent");
const aiCard = document.getElementById("aiCard");
const aiResponseText = document.getElementById("aiResponseText");

// ---------------------------------------------------------------------------
// Health check on load
// ---------------------------------------------------------------------------
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (!data.groq_configured && !data.gemini_configured) {
      keyNotice.classList.remove("hidden");
      keyNoticeText.innerHTML = `Missing in <code class="font-semibold text-amber-100">.env</code>: <strong>GROQ_API_KEY</strong>. Please add your key and refresh!`;
    } else {
      keyNotice.classList.add("hidden");
    }
  } catch (e) {
    console.warn("Health check error:", e);
  }
}
checkHealth();
setUIState("ready");

// ---------------------------------------------------------------------------
// UI State Machine
// ---------------------------------------------------------------------------
function setUIState(state) {
  if (window.setGlobeState) window.setGlobeState(state);

  talkBtn.classList.remove("state-listening", "state-speaking");
  if (statusLabel && statusLabel.parentElement) {
    statusLabel.parentElement.classList.remove("state-listening");
  }

  if (state === "ready") {
    statusLabel.textContent = "Ready when you are";
    if (stateEmoji) stateEmoji.textContent = "👋";
    talkBtnText.textContent = "Ready when you are";
    talkBtn.disabled = false;
    talkBtn.style.opacity = "";
    if (pulseRing) pulseRing.classList.add("opacity-0");
  } else if (state === "listening") {
    statusLabel.textContent = "User speaking…";
    if (statusLabel && statusLabel.parentElement) {
      statusLabel.parentElement.classList.add("state-listening");
    }
    talkBtn.classList.add("state-listening");
    if (stateEmoji) stateEmoji.textContent = "👂";
    talkBtnText.textContent = "Tap when done";
    talkBtn.disabled = false;
    talkBtn.style.opacity = "";
    if (pulseRing) pulseRing.classList.remove("opacity-0");
  } else if (state === "thinking") {
    statusLabel.textContent = "Thinking…";
    if (stateEmoji) stateEmoji.textContent = "🤔";
    talkBtnText.textContent = "Thinking…";
    talkBtn.disabled = true;
    talkBtn.style.opacity = "0.45";
    if (pulseRing) pulseRing.classList.add("opacity-0");
  } else if (state === "speaking") {
    statusLabel.textContent = "Alex speaking…";
    talkBtn.classList.add("state-speaking");
    if (stateEmoji) stateEmoji.textContent = "🗣️";
    talkBtnText.textContent = "Alex speaking…";
    talkBtn.disabled = false;
    talkBtn.style.opacity = "";
    stopAudioBtn.classList.remove("hidden");
  }
}

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
      const _an = _ac.createAnalyser(); _an.fftSize = 256;
      _src.connect(_an);
      if (window.connectGlobeMic) window.connectGlobeMic(_an);
      window._alexMicAudioContext = _ac;
    } catch(_e) { /* non-critical */ }

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
    if (window._alexMicAudioContext) { window._alexMicAudioContext.close().catch(()=>{}); window._alexMicAudioContext = null; }
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    isRecording = false;
    setUIState("thinking");
  }
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

  try {
    const res = await fetch("/api/turn", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Server error occurred");
    }

    if (data.user_transcript && data.user_transcript.trim()) {
      completedTurns += 1;
      if (completedTurns >= 1 && newChatBtn) {
        newChatBtn.classList.remove("hidden");
      }
      userCard.classList.remove("hidden");
      userTranscriptText.textContent = data.user_transcript;
    } else {
      userCard.classList.add("hidden");
    }

    // 2. Show correction card (only when corrections exist)
    renderCorrectionCard(data.corrections || []);

    // 3. Show Alex's conversational response
    aiCard.classList.remove("hidden");
    aiResponseText.textContent = data.response_text;

    // 4. Log metrics
    logMetric(data);

    const cardsStream = document.getElementById("cardsStream");
    if (cardsStream) {
      cardsStream.scrollTo({ top: 0, behavior: "smooth" });
    }

    // 5. Play speech segments with adaptive timing & card highlighting
    lastTurnData = data;
    playTurnSegments(data);

  } catch (err) {
    console.error("Turn processing error:", err);
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
// ---------------------------------------------------------------------------

/**
 * Extracts and renders minimal difference highlighting for a single correction item.
 * Strictly adheres to visual design:
 * - Incorrect text: text-red-400 font-semibold (style="color: #f87171; font-weight: 600;")
 * - Corrected text: text-green-400 font-semibold (style="color: #4ade80; font-weight: 600;")
 * - Unchanged text: normal neutral styling (text-slate-300)
 * - Added words: shown only in green on corrected side, no invented word in original
 * - Removed words: shown only in red on original side, no invented word in corrected
 * - Complete sentence/phrase reconstruction: entire relevant portion highlighted
 */
function renderCorrectionPortion(c, cardIndex = 0) {
  if (!c || typeof c !== "object") {
    return { htmlOrig: "", htmlCorr: "" };
  }

  // 1. Determine relevant original and corrected strings
  let orig = (c.original || "").trim();
  let corr = (c.corrected || "").trim();

  // If either is missing, fall back to full sentence fields
  if (!orig && c.full_original_sentence) orig = c.full_original_sentence.trim();
  if (!corr && c.full_corrected_sentence) corr = c.full_corrected_sentence.trim();

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  const redSpan = (text) => `<span class="text-red-400 font-semibold" style="color: #f87171; font-weight: 600;">${escapeHtml(text)}</span>`;
  const greenSpan = (text) => `<span class="text-green-400 font-semibold" style="color: #4ade80; font-weight: 600;">${escapeHtml(text)}</span>`;
  const neutralSpan = (text) => `<span class="text-slate-300">${escapeHtml(text)}</span>`;

  if (!orig && !corr) {
    return { htmlOrig: "", htmlCorr: "" };
  }
  if (!orig) {
    return { htmlOrig: "", htmlCorr: greenSpan(corr) };
  }
  if (!corr) {
    return { htmlOrig: redSpan(orig), htmlCorr: "" };
  }

  function cleanWord(w) {
    return (w || "").toLowerCase().replace(/^[^\w]+|[^\w]+$/g, "");
  }

  // Strip boundary quotes if present
  if (orig.startsWith('"') && orig.endsWith('"') && orig.length > 2) orig = orig.slice(1, -1).trim();
  if (corr.startsWith('"') && corr.endsWith('"') && corr.length > 2) corr = corr.slice(1, -1).trim();

  const wordsOrig = orig.split(/\s+/).filter(Boolean);
  const wordsCorr = corr.split(/\s+/).filter(Boolean);

  // 2. Single-word mistake: direct minimal highlight
  if (wordsOrig.length === 1 && wordsCorr.length === 1) {
    if (cleanWord(wordsOrig[0]) !== cleanWord(wordsCorr[0]) || wordsOrig[0] !== wordsCorr[0]) {
      return {
        htmlOrig: redSpan(orig),
        htmlCorr: greenSpan(corr)
      };
    }
    return {
      htmlOrig: neutralSpan(orig),
      htmlCorr: neutralSpan(corr)
    };
  }

  // 3. Word-level alignment via LCS
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

  // Example B: Complete grammatical construction change (e.g. "she going" -> "she went")
  const isParticipleToFinite = wordsOrig.length === 2 && wordsCorr.length === 2 &&
    wordsOrig[1].toLowerCase().endsWith("ing") && !wordsCorr[1].toLowerCase().endsWith("ing");

  // Check for complete phrase reconstruction:
  // ONLY if participle clause reconstructed into finite tense (e.g. "she going" -> "she went"),
  // OR if zero words match (equalCount === 0),
  // OR if it is a short phrase (<= 8 words) where word order/construction is scrambled (ratio < 0.30)
  const equalCount = diff.filter(d => d.type === "equal").length;
  const maxWords = Math.max(N, M);
  const isCompleteReconstruction = isParticipleToFinite || (equalCount === 0) || (N <= 8 && M <= 8 && equalCount <= 2 && (equalCount / maxWords) < 0.30);

  if (isCompleteReconstruction) {
    return {
      htmlOrig: redSpan(orig),
      htmlCorr: greenSpan(corr)
    };
  }

  // Handle word transpositions (e.g. "you are" -> "are you")
  for (let k = 0; k < diff.length - 2; k++) {
    if (diff[k].type === "delete" && diff[k + 1].type === "equal" && diff[k + 2].type === "insert") {
      if (cleanWord(diff[k].orig) === cleanWord(diff[k + 2].corr)) {
        const swappedOrig = [diff[k].orig, diff[k + 1].orig];
        const swappedCorr = [diff[k + 1].corr, diff[k + 2].corr];
        diff.splice(k, 3, {
          type: "swap",
          origWords: swappedOrig,
          corrWords: swappedCorr
        });
      }
    } else if (diff[k].type === "insert" && diff[k + 1].type === "equal" && diff[k + 2].type === "delete") {
      if (cleanWord(diff[k].corr) === cleanWord(diff[k + 2].orig)) {
        const swappedOrig = [diff[k + 1].orig, diff[k + 2].orig];
        const swappedCorr = [diff[k].corr, diff[k + 1].corr];
        diff.splice(k, 3, {
          type: "swap",
          origWords: swappedOrig,
          corrWords: swappedCorr
        });
      }
    }
  }

  // 4. Compact Window Slicing:
  // If the diff is long (> 5 words) with unchanged equal words, we NEVER want the entire sentence.
  // We locate the changed region and slice the window to at most 1 context word before/after!
  let windowDiff = diff;
  const changedIndices = [];
  diff.forEach((item, idx) => {
    if (item.type !== "equal") changedIndices.push(idx);
  });

  if (changedIndices.length > 0 && diff.length > 5) {
    // Group changed indices into clusters separated by >= 2 equal words
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

    // Pick target cluster for this card
    const targetCluster = (cardIndex !== undefined && cardIndex < clusters.length) ? clusters[cardIndex] : clusters[0];
    const minDiffIdx = targetCluster[0];
    const maxDiffIdx = targetCluster[targetCluster.length - 1];

    // Take at most 1 equal item before and 1 equal item after
    const startIdx = Math.max(0, minDiffIdx - 1);
    const endIdx = Math.min(diff.length, maxDiffIdx + 2);
    windowDiff = diff.slice(startIdx, endIdx);
  }

  // Build rendered HTML for original side
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

  // Build rendered HTML for corrected side
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

/**
 * Renders distinct grammar mistakes into separate cards.
 * Displays only the specific relevant portion on each card.
 */
function renderCorrectionCard(corrections) {
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  correctionCardContent.innerHTML = "";

  if (!corrections || !Array.isArray(corrections) || corrections.length === 0) {
    correctionCard.classList.add("hidden");
    return;
  }

  correctionCard.classList.remove("hidden");

  // Deduplicate: If any "expression" (More Natural Phrasing) card covers the same sentence or words as a "grammar" card, remove it
  const grammarCards = corrections.filter(c => {
    const t = String(c.type || "").trim().toLowerCase();
    return t !== "expression" && t !== "natural_phrasing" && t !== "style" && t !== "word_choice";
  });

  const cardsToRender = [];
  corrections.forEach(c => {
    const t = String(c.type || "").trim().toLowerCase();
    const isExpr = (t === "expression" || t === "natural_phrasing" || t === "style" || t === "word_choice");
    if (isExpr && grammarCards.length > 0) {
      const origLow = String(c.original || "").toLowerCase().trim();
      const corrLow = String(c.corrected || "").toLowerCase().trim();
      const foLow = String(c.full_original_sentence || "").toLowerCase().trim();
      const isDup = grammarCards.some(g => {
        const gOrig = String(g.original || "").toLowerCase().trim();
        const gCorr = String(g.corrected || "").toLowerCase().trim();
        const gFo = String(g.full_original_sentence || "").toLowerCase().trim();
        return (origLow && gOrig && (origLow.includes(gOrig) || gOrig.includes(origLow))) ||
               (corrLow && gCorr && (corrLow.includes(gCorr) || gCorr.includes(corrLow))) ||
               (foLow && gFo && foLow === gFo);
      });
      if (isDup) return;
    }
    cardsToRender.push(c);
  });

  if (cardsToRender.length === 0) {
    correctionCard.classList.add("hidden");
    return;
  }

  // Deduplicate rendered cards so no identical cards appear
  const renderedPortions = new Set();

  const _cntChip = document.getElementById("correctionCountChip");
  if (_cntChip) _cntChip.textContent = cardsToRender.length + " targeted improvement" + (cardsToRender.length !== 1 ? "s" : "");

  // Render a SEPARATE card for each distinct mistake
  cardsToRender.forEach((c, idx) => {
    if (!c || typeof c !== "object") return;

    const t = String(c.type || "").trim().toLowerCase();
    const isGrammar = (t !== "expression" && t !== "natural_phrasing" && t !== "style" && t !== "word_choice");

    const { htmlOrig, htmlCorr } = renderCorrectionPortion(c, idx);

    // Filter duplicates by rendered text
    const cleanKey = `${htmlOrig}:::${htmlCorr}`.replace(/<[^>]+>/g, "").toLowerCase().trim();
    if (cleanKey && renderedPortions.has(cleanKey)) {
      return;
    }
    renderedPortions.add(cleanKey);

    const card = document.createElement("div");
    card.id = `corr-card-${idx}`;
    card.className = "tip correction-item";

    const pillHtml = isGrammar
      ? `<span class="gc-pill">✎ Grammar correction</span>`
      : `<span class="expr-pill">💬 More natural phrasing</span>`;

    const headerHtml = `
      <div class="tip-row">
        ${pillHtml}
        <button type="button" onclick="replayCorrection(${idx})" class="play-btn">▶ Tip audio</button>
      </div>`;

    const bodyHtml = `
      <div class="ln">Instead of: <span>${htmlOrig}</span></div>
      <div class="ln">Say: <span>${htmlCorr}</span></div>
      ${c.explanation ? `<div class="why"><b>Why:</b> ${escapeHtml(c.explanation)}</div>` : ''}`;

    card.innerHTML = headerHtml + bodyHtml;
    correctionCardContent.appendChild(card);
  });
}

// ---------------------------------------------------------------------------
// Card Highlighting & Visual Synchronization
// ---------------------------------------------------------------------------

function highlightCard(cardIndex) {
  clearHighlights();
  if (cardIndex !== null && cardIndex !== undefined) {
    const el = document.getElementById(`corr-card-${cardIndex}`) ||
               document.getElementById(`corr-item-${cardIndex}`) ||
               document.getElementById("consolidated-correction-card");
    if (el) {
      el.classList.add("highlighted");
      try { el.scrollIntoView({ behavior: "smooth", block: "nearest" }); } catch(e) {}
    }
  } else {
    if (aiCard) {
      aiCard.classList.add("highlighted");
      try { aiCard.scrollIntoView({ behavior: "smooth", block: "nearest" }); } catch(e) {}
    }
  }
}

function clearHighlights() {
  document.querySelectorAll(".correction-item").forEach(el => {
    el.classList.remove("highlighted");
  });
  if (aiCard) aiCard.classList.remove("highlighted");
}

// ---------------------------------------------------------------------------
// Audio Playback & Adaptive Speech Timing
// ---------------------------------------------------------------------------

function playTurnSegments(turnData) {
  stopCurrentPlayback();
  if (!turnData) return;

  const currentGen = playbackGenerationId;
  const segments = turnData.speech_segments || [];

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
  stopAudioBtn.classList.remove("hidden");

  let index = 0;

  function playNext() {
    if (playbackGenerationId !== currentGen) return;

    if (index >= segments.length) {
      clearHighlights();
      stopAudioBtn.classList.add("hidden");
      setUIState("ready");
      return;
    }

    const seg = segments[index];
    index++;

    if (!seg.audio_base64) {
      playNext();
      return;
    }

    // Highlight card corresponding to this segment
    if (seg.type && seg.type.startsWith("corr")) {
      highlightCard(seg.card_index);
    } else {
      highlightCard(null); // conversational reply
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
  playTurnSegments(lastTurnData);
}

function replayAllCorrections() {
  if (!lastTurnData || !lastTurnData.speech_segments) return;
  const matched = lastTurnData.speech_segments.filter(s => s.type && s.type.startsWith("corr"));
  if (matched.length > 0) {
    playTurnSegments({ speech_segments: matched });
  }
}

function replayCorrection(cardIndex) {
  if (!lastTurnData || !lastTurnData.speech_segments) return;
  let matched = [];
  if (cardIndex !== undefined && cardIndex !== null) {
    matched = lastTurnData.speech_segments.filter(s =>
      s.type && s.type.startsWith("corr") && s.card_index === cardIndex
    );
  }
  if (!matched || matched.length === 0) {
    matched = lastTurnData.speech_segments.filter(s => s.type && s.type.startsWith("corr"));
  }
  if (matched.length > 0) {
    playTurnSegments({ speech_segments: matched });
  }
}

function playBase64Audio(base64Data) {
  stopCurrentPlayback();
  currentAudio = new Audio(`data:audio/mp3;base64,${base64Data}`);
  currentAudio.onended = () => {
    stopAudioBtn.classList.add("hidden");
    setUIState("ready");
  };
  currentAudio.play().catch(e => {
    console.warn("Autoplay blocked, falling back to browser speech:", e);
    playBrowserSpeechFallback(aiResponseText.textContent);
  });
}

function playBrowserSpeechFallback(text) {
  stopCurrentPlayback();
  if ("speechSynthesis" in window) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.onend = () => {
      stopAudioBtn.classList.add("hidden");
      setUIState("ready");
    };
    window.speechSynthesis.speak(utterance);
  } else {
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
  stopAudioBtn.classList.add("hidden");
}

// ---------------------------------------------------------------------------
// Metrics Logging
// ---------------------------------------------------------------------------
function logMetric(data) {
  const m = data.latency_ms;
  const corrCount = (data.corrections || []).length;
  const corrNote = corrCount > 0 ? ` | Corrections: ${corrCount}` : "";
  const line = document.createElement("div");
  line.textContent = `[Turn ${completedTurns}] Total: ${m.total}ms | STT: ${m.stt}ms | LLM: ${m.llm}ms | TTS: ${m.tts}ms (${data.tts_engine})${corrNote}`;
  metricsLog.prepend(line);
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

  // Reset subtitle cards
  userCard.classList.add("hidden");
  correctionCard.classList.add("hidden");
  correctionCardContent.innerHTML = "";
  aiCard.classList.add("hidden");
  if (newChatBtn) newChatBtn.classList.add("hidden");

  setUIState("ready");
  const logLine = document.createElement("div");
  logLine.textContent = `--- Started New Topic (${currentSessionId.substring(0, 8)}) ---`;
  metricsLog.prepend(logLine);
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
talkBtn.addEventListener("click", () => {
  if (isProcessingTurn) return;
  if (!isRecording) startRecording();
  else stopRecording();
});

stopAudioBtn.addEventListener("click", () => {
  stopCurrentPlayback();
  setUIState("ready");
});

if (newChatBtn) {
  newChatBtn.addEventListener("click", handleStartNewConversation);
}


// ── SAPPHIRE PARTICLE GLOBE ─────────────────────────────────────────────────
(function() {
  var TP = 6.2832, globeSt = "idle", globeLv = 0, globeSp = 1, globePt = 0, globeT = 0;
  var globeMicAn = null, globeMicBuf = null;
  var G = [];
  for (var i = 0, N = 950; i < N; i++) {
    var y = 1 - 2*(i + .5)/N, r = Math.sqrt(1 - y*y), a = i*2.39996;
    G.push([Math.cos(a)*r, y, Math.sin(a)*r]);
  }
  var ce = document.getElementById("globeCanvas");
  if (!ce) return;
  var W = 280, D = Math.min(window.devicePixelRatio || 1, 2);
  ce.width = ce.height = W * D;
  ce.style.width = W + "px";
  ce.style.height = W + "px";
  var gx = ce.getContext("2d");

  window.setGlobeState = function(s) {
    var m = { ready: "idle", listening: "listening", thinking: "thinking", speaking: "speaking" };
    globeSt = m[s] || "idle";
  };
  window.connectGlobeMic = function(an) {
    globeMicAn = an;
    globeMicBuf = new Uint8Array(an.frequencyBinCount);
    globeSt = "mic";
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
    gx.lineWidth = 1;
    gx.beginPath();
    gx.arc(c0, c0, R * 1.5, 0, TP);
    gx.stroke();

    if (globeSt === "thinking") {
      var an2 = globePt * 2.2;
      gx.fillStyle = "#cfe0ff";
      gx.shadowColor = "#7fa4ff";
      gx.shadowBlur = 14;
      gx.beginPath();
      gx.arc(c0 + Math.cos(an2) * R * 1.5, c0 + Math.sin(an2) * R * 1.5, 3.2, 0, TP);
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
      gx.arc(c0 + x1 * f, c0 + y1 * f, .5 + d * 1.3, 0, TP);
      gx.fill();
    }
  }

  function globeFrame() {
    globeT += .016;
    var e = Math.abs(Math.sin(globeT * 5.3) * Math.sin(globeT * 1.9 + 1) + .5 * Math.sin(globeT * 11.7));
    var tgLv = .05, tgSp = .35;
    if (globeSt === "listening")     { tgLv = .15 + .3 * e;  tgSp = .8;  }
    else if (globeSt === "speaking") { tgLv = .3 + .65 * e;  tgSp = 1.3; }
    else if (globeSt === "thinking") { tgLv = .12;           tgSp = 2.6; }
    else if (globeSt === "mic")      { tgLv = .08;           tgSp = 1;   }

    if (globeSt === "mic" && globeMicAn) {
      globeMicAn.getByteFrequencyData(globeMicBuf);
      var s = 0;
      for (var i = 0; i < globeMicBuf.length; i++) s += globeMicBuf[i];
      tgLv = Math.min(1, s / globeMicBuf.length / 70);
    }

    globeLv += (tgLv - globeLv) * .14;
    globeSp += (tgSp - globeSp) * .05;
    globePt += .016 * globeSp;

    drawGlobe();
    requestAnimationFrame(globeFrame);
  }

  globeFrame();
})();
