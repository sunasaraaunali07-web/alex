// Node.js test script for Alex Phase 2 Step 4: Correction Card Presentation
const fs = require("fs");
const path = require("path");
const assert = require("assert");

// Read app.js and extract renderCorrectionCard & escapeHtml
const appJsPath = path.join(__dirname, "frontend", "app.js");
const appJsContent = fs.readFileSync(appJsPath, "utf-8");

// Mock DOM elements
function createMockElement(id, classes = []) {
  return {
    id,
    classList: {
      _classes: new Set(classes),
      add(cls) { this._classes.add(cls); },
      remove(cls) { this._classes.delete(cls); },
      contains(cls) { return this._classes.has(cls); }
    },
    innerHTML: "",
    children: [],
    appendChild(child) {
      this.children.push(child);
      this.innerHTML += child.innerHTML;
    }
  };
}

const mockCorrectionCard = createMockElement("correctionCard", ["hidden"]);
const mockCorrectionCardContent = createMockElement("correctionCardContent");

// Mock document
global.document = {
  createElement(tag) {
    return {
      tagName: tag,
      className: "",
      innerHTML: "",
      textContent: ""
    };
  }
};
global.escapeHtml = function(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
};
global.correctionCard = mockCorrectionCard;
global.correctionCardContent = mockCorrectionCardContent;

// Eval renderCorrectionCard from app.js in this sandbox context
const fnMatch = appJsContent.match(/function renderCorrectionCard\(corrections\)\s*\{[\s\S]*?\n\}/);
if (!fnMatch) {
  console.error("Could not find renderCorrectionCard in app.js");
  process.exit(1);
}
eval(fnMatch[0]);

console.log("=== Testing Alex Phase 2 - Step 4: Correction Card Presentation ===\n");

// Test 1: Genuine Grammar Correction
console.log("[Test 1] Genuine Grammar Correction");
mockCorrectionCard.classList.add("hidden");
mockCorrectionCardContent.children = [];
mockCorrectionCardContent.innerHTML = "";

renderCorrectionCard([
  {
    original: "Yesterday I go to market",
    corrected: "Yesterday I went to the market",
    explanation: "Use past tense 'went' for actions that happened yesterday.",
    type: "grammar"
  }
]);

assert.strictEqual(mockCorrectionCard.classList.contains("hidden"), false, "Card must be visible");
assert.strictEqual(mockCorrectionCardContent.children.length, 1, "Should render 1 item");
const item1 = mockCorrectionCardContent.children[0];
assert(item1.innerHTML.includes("Grammar Correction"), "Must have 'Grammar Correction' label");
assert(item1.className.includes("rose"), "Must have rose/red styling");
assert(item1.innerHTML.includes("Instead of:"), "Must have 'Instead of:' label");
assert(item1.innerHTML.includes("Say:"), "Must have 'Say:' label");
assert(item1.innerHTML.includes("Yesterday I went to the market"), "Must contain corrected text");
assert(item1.innerHTML.includes("Use past tense &#039;went&#039;"), "Must contain explanation and be escaped");
console.log("  PASS: Rendered rose Grammar Correction card with 'Instead of' / 'Say' and explanation.\n");

// Test 2: Optional Natural Expression
console.log("[Test 2] Optional Natural-Expression Suggestion");
mockCorrectionCard.classList.add("hidden");
mockCorrectionCardContent.children = [];
mockCorrectionCardContent.innerHTML = "";

renderCorrectionCard([
  {
    original: "Everything is going out of my hand",
    corrected: "Things are getting out of my control",
    explanation: "The natural English idiom is 'getting out of my control'.",
    type: "expression"
  }
]);

assert.strictEqual(mockCorrectionCard.classList.contains("hidden"), false, "Card must be visible");
assert.strictEqual(mockCorrectionCardContent.children.length, 1, "Should render 1 item");
const item2 = mockCorrectionCardContent.children[0];
assert(item2.innerHTML.includes("More Natural Phrasing"), "Must have 'More Natural Phrasing' label");
assert(item2.className.includes("sky"), "Must have sky/blue styling");
assert(item2.innerHTML.includes("Optional Alternative"), "Must state it is optional");
assert(item2.innerHTML.includes("grammatically fine"), "Must explain sentence is grammatically fine");
assert(item2.innerHTML.includes("Natural Alternative:"), "Must have 'Natural Alternative:' label");
console.log("  PASS: Rendered blue/amber More Natural Phrasing card clarifying the original is grammatically fine.\n");

// Test 3: Multiple Corrections of Both Types in One Turn
console.log("[Test 3] Multiple Corrections of Both Types in One Message");
mockCorrectionCard.classList.add("hidden");
mockCorrectionCardContent.children = [];
mockCorrectionCardContent.innerHTML = "";

renderCorrectionCard([
  {
    original: "Yesterday I go",
    corrected: "Yesterday I went",
    explanation: "Past tense required.",
    type: "grammar"
  },
  {
    original: "My head is not working",
    corrected: "I can't think clearly",
    explanation: "Natural English expression.",
    type: "expression"
  }
]);

assert.strictEqual(mockCorrectionCard.classList.contains("hidden"), false, "Card must be visible");
assert.strictEqual(mockCorrectionCardContent.children.length, 2, "Should render 2 distinct cards");
assert(mockCorrectionCardContent.children[0].innerHTML.includes("Grammar Correction"), "First card is grammar");
assert(mockCorrectionCardContent.children[0].className.includes("rose"), "First card is rose");
assert(mockCorrectionCardContent.children[1].innerHTML.includes("More Natural Phrasing"), "Second card is expression");
assert(mockCorrectionCardContent.children[1].className.includes("sky"), "Second card is sky/blue");
console.log("  PASS: Both distinct cards rendered side-by-side with appropriate distinct styles.\n");

// Test 4: Message with Zero Corrections
console.log("[Test 4] Message with Zero Corrections");
mockCorrectionCard.classList.remove("hidden"); // start unhidden
mockCorrectionCardContent.children = [];
mockCorrectionCardContent.innerHTML = "";

renderCorrectionCard([]);
assert.strictEqual(mockCorrectionCard.classList.contains("hidden"), true, "Card must be hidden for empty list");

renderCorrectionCard(null);
assert.strictEqual(mockCorrectionCard.classList.contains("hidden"), true, "Card must be hidden for null");
console.log("  PASS: Card safely hidden when corrections array is empty or null.\n");

// Test 5: Missing or Unexpected Correction Types
console.log("[Test 5] Missing or Unexpected Correction Types (Fallback Safety)");
mockCorrectionCard.classList.add("hidden");
mockCorrectionCardContent.children = [];
mockCorrectionCardContent.innerHTML = "";

renderCorrectionCard([
  {
    original: "He don't know",
    corrected: "He doesn't know",
    explanation: "Subject-verb agreement."
    // type is completely undefined
  },
  {
    original: "We was there",
    corrected: "We were there",
    explanation: "Plural subject.",
    type: "unknown_custom_type"
  },
  {
    original: "I am having doubt",
    corrected: "I have a question",
    explanation: "Natural expression.",
    type: "word_choice" // Should classify under natural phrasing
  }
]);

assert.strictEqual(mockCorrectionCardContent.children.length, 3, "All 3 items rendered safely");
assert(mockCorrectionCardContent.children[0].innerHTML.includes("Grammar Correction"), "Undefined type defaults to Grammar Correction");
assert(mockCorrectionCardContent.children[0].className.includes("rose"), "Undefined type gets rose styling");
assert(mockCorrectionCardContent.children[1].innerHTML.includes("Grammar Correction"), "Unknown type defaults to Grammar Correction");
assert(mockCorrectionCardContent.children[1].className.includes("rose"), "Unknown type gets rose styling");
assert(mockCorrectionCardContent.children[2].innerHTML.includes("More Natural Phrasing"), "word_choice maps to natural phrasing");
console.log("  PASS: Missing and unexpected types safely fall back to Grammar Correction with zero errors.\n");

// Test 6: XSS Sanitization Check
console.log("[Test 6] XSS Sanitization Check");
mockCorrectionCardContent.children = [];
mockCorrectionCardContent.innerHTML = "";

renderCorrectionCard([
  {
    original: "<script>alert('xss')</script>",
    corrected: "<b>safe</b>",
    explanation: "<img src=x onerror=alert(1)>",
    type: "grammar"
  }
]);

const rawHtml = mockCorrectionCardContent.innerHTML;
assert(!rawHtml.includes("<script>"), "Must NOT contain unescaped script tags");
assert(!rawHtml.includes("<img src=x"), "Must NOT contain unescaped img tags");
assert(rawHtml.includes("&lt;script&gt;"), "Must contain escaped script tags");
console.log("  PASS: All dynamic text successfully HTML entity encoded.\n");

console.log("ALL STEP 4 FRONTEND TESTS PASSED SUCCESSFULLY!");
