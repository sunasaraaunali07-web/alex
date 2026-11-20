const fs = require("fs");
const path = require("path");

const appJs = fs.readFileSync(path.join(__dirname, "..", "frontend", "app.js"), "utf-8");

// Mock document
global.document = {
  createElement(tag) {
    return {
      tagName: tag,
      _text: "",
      set textContent(v) { this._text = v; this.innerHTML = v; },
      get textContent() { return this._text; },
      innerHTML: ""
    };
  }
};

const fnMatch = appJs.match(/function renderCorrectionPortion\(c, cardIndex = 0\)\s*\{[\s\S]*?\n\}/);
if (!fnMatch) {
  console.error("renderCorrectionPortion not found");
  process.exit(1);
}
eval(fnMatch[0]);

const testCases = [
  { original: "I go", corrected: "I went", explanation: "Past tense" },
  { original: "because I have to", corrected: "because I had to", explanation: "Past tense obligation" },
  { original: "he don't understand", corrected: "he doesn't understand", explanation: "Subject-verb agreement" },
  { original: "start arguing with", corrected: "starts arguing", explanation: "Singular verb and preposition" }
];

console.log("=== CARD RENDERING VERIFICATION ===");
testCases.forEach((tc, idx) => {
  const rendered = renderCorrectionPortion(tc, idx);
  console.log(`\nCard ${idx + 1}:`);
  console.log("  Original HTML: ", rendered.htmlOrig);
  console.log("  Corrected HTML:", rendered.htmlCorr);
});
