// UI contract checks for app/wejha.html and app/planner.html (Claude Code teammate 2).
//
//   node app/check_ui.js            # schema + page-ranking checks
//   node app/check_ui.js --cli      # also compare top-3 with `python src/basket_recommender.py recommend`
//
// 1. Both pages' inline scripts parse.
// 2. app/recommendation_data.js and app/wejha_data.js follow WEJHA_BUILD_PLAN.md (schema v1).
// 3. The ranking function used by wejha.html (extracted from the page itself, not re-implemented)
//    never returns basket/unavailable items, returns at most 3 rows, is sorted by score then name,
//    and falls back as the plan specifies.
// 4. With --cli, the page's top-3 equals the Python CLI's top-3 for a set of baskets.
// Exit code 1 on any failure.

const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

const APP = __dirname;
const ROOT = path.dirname(APP);
let failures = 0;
const ok = (cond, msg) => { if (cond) console.log("  ok   " + msg); else { failures++; console.log("  FAIL " + msg); } };

function inlineScript(file) {
  const html = fs.readFileSync(path.join(APP, file), "utf8");
  return html.split("<script>").pop().split("</script>")[0];
}
function loadWindowScript(file) {
  const p = path.join(APP, file);
  if (!fs.existsSync(p)) return undefined;
  const window = {};
  new Function("window", fs.readFileSync(p, "utf8"))(window);
  return window;
}

console.log("1. Page scripts");
for (const f of ["wejha.html", "planner.html"]) {
  try { new Function(inlineScript(f)); ok(true, `${f} script parses`); } catch (e) { ok(false, `${f}: ${e.message}`); }
}

console.log("2. Data contracts");
const R = (loadWindowScript("recommendation_data.js") || {}).STORIES_RECOMMENDATIONS;
const W = (loadWindowScript("wejha_data.js") || {}).STORIES_WEJHA;
ok(!!R, "recommendation_data.js defines window.STORIES_RECOMMENDATIONS");
ok(!!W, "wejha_data.js defines window.STORIES_WEJHA");
const ROW = ["product", "category", "score", "conditional_probability", "baseline_probability", "lift",
  "pair_orders", "anchor_orders", "unit_margin_usd", "reason", "evidence_status"];
if (R) {
  ok(R.schema_version === 1, `recommendation schema_version is 1 (got ${R.schema_version})`);
  for (const k of ["status", "trained_through", "selected_method", "branches", "dayparts", "cohorts", "products", "contexts", "evaluation"])
    ok(k in R, `recommendations have '${k}'`);
  const products = new Set((R.products || []).map((p) => p.product));
  ok(products.size > 0, `${products.size} products listed`);
  for (const b of R.branches || []) ok(!!R.contexts[`${b}|all|all`], `fallback context ${b}|all|all exists`);
  let rows = 0, bad = 0, unknown = 0, ctxBad = 0;
  for (const [key, ctx] of Object.entries(R.contexts || {})) {
    if (!Array.isArray(ctx.baseline) || typeof ctx.by_anchor !== "object") ctxBad++;
    const lists = [ctx.baseline || [], ...Object.values(ctx.by_anchor || {})];
    for (const list of lists) for (const r of list) {
      rows++;
      if (ROW.some((f) => !(f in r)) || typeof r.score !== "number") bad++;
      if (!products.has(r.product)) unknown++;
    }
  }
  ok(ctxBad === 0, `every context has baseline[] and by_anchor{} (${Object.keys(R.contexts).length} contexts)`);
  ok(bad === 0, `all ${rows} recommendation rows have the 11 contract fields (${bad} missing)`);
  ok(unknown === 0, `every recommended product is in products[] (${unknown} unknown)`);
}
if (W) {
  ok(W.schema_version === 1, `wejha schema_version is 1 (got ${W.schema_version})`);
  for (const k of ["status", "products", "pairs", "layouts", "assumptions", "pilot"]) ok(k in W, `wejha has '${k}'`);
  const names = (W.layouts || []).map((l) => l.name);
  ok((W.layouts || []).length >= 3, `at least 3 layouts (${names.join(", ")})`);
  for (const L of W.layouts || []) {
    const s = L.slots || [];
    ok(new Set(s).size === s.length, `${L.name}: no product repeated (${s.length} slots)`);
    ok(typeof L.weighted_pair_distance === "number", `${L.name}: numeric weighted_pair_distance`);
  }
  const sets = (W.layouts || []).map((l) => JSON.stringify([...(l.slots || [])].sort()));
  ok(new Set(sets).size <= 1, "all layouts use the same products");
  ok((W.pairs || []).every((p) => ["item_a", "item_b", "pair_orders", "lift", "weight", "evidence_status"].every((f) => f in p)), "pair rows have the contract fields");
}

console.log("3. Page ranking rule (function taken from wejha.html)");
// \r?\n: Windows checkouts (core.autocrlf) store the page with CRLF line endings
const src = inlineScript("wejha.html").match(/function recommend\([\s\S]*?\r?\n}\r?\n/);
ok(!!src, "recommend() found in wejha.html");
const recommend = src ? new Function(src[0] + ";return recommend;")() : null;
const queries = [];
if (R && recommend) {
  const prods = (R.products || []).map((p) => p.product);
  const pick = (i) => prods[(i * 7) % prods.length];
  for (const b of R.branches) for (const d of R.dayparts) for (const c of R.cohorts) {
    queries.push({ b, d, c, items: [], un: [] });
    queries.push({ b, d, c, items: [pick(queries.length)], un: [] });
    queries.push({ b, d, c, items: [pick(queries.length), pick(queries.length + 3)], un: [pick(queries.length + 1)] });
  }
  let violations = 0;
  for (const q of queries) {
    const res = recommend(R, q.b, q.d, q.c, q.items, new Set(q.un));
    const rows = res.rows;
    const excl = new Set([...q.items, ...q.un]);
    if (rows.length > 3 || rows.some((r) => excl.has(r.product))) violations++;
    for (let i = 1; i < rows.length; i++) {
      const a = rows[i - 1], z = rows[i];
      if (a.score < z.score || (a.score === z.score && String(a.product).localeCompare(String(z.product)) > 0)) violations++;
    }
    if (new Set(rows.map((r) => r.product)).size !== rows.length) violations++;
    q.top = rows.map((r) => r.product);
  }
  ok(violations === 0, `${queries.length} queries: ≤3 rows, no basket/unavailable items, no duplicates, sorted by score then name`);
  const missing = recommend(R, R.branches[0], "no-such-daypart", "all", [], new Set());
  ok(missing.key === `${R.branches[0]}|all|all` && missing.notes.length > 0, "absent context falls back to branch|all|all and says so");
  const unk = recommend(R, R.branches[0], "all", "all", ["__not_a_product__"], new Set());
  ok(unk.notes.some((n) => n.includes("baseline")), "unknown anchor falls back to baseline and says so");
}

if (process.argv.includes("--cli") && R && recommend) {
  console.log("4. Page top-3 vs Python CLI");
  const py = process.env.PYTHON || "python";
  let agree = 0, tried = 0;
  for (const q of queries.filter((_, i) => i % 5 === 0).slice(0, 24)) {
    const args = ["-X", "utf8", path.join(ROOT, "src", "basket_recommender.py"), "recommend", "--branch", q.b, "--daypart", q.d, "--cohort", q.c];
    q.items.forEach((it) => args.push("--item", it));
    q.un.forEach((it) => args.push("--unavailable", it));
    let out;
    try { out = execFileSync(py, args, { cwd: ROOT, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }); }
    catch (e) { ok(false, `CLI failed for ${JSON.stringify(q)}: ${(e.stderr || e.message).slice(0, 200)}`); continue; }
    let cliTop = null;
    try {
      const j = JSON.parse(out);
      const list = j.recommendations || j.rows || j;
      cliTop = (Array.isArray(list) ? list : []).map((r) => r.product);
    } catch { cliTop = null; }
    tried++;
    if (cliTop && JSON.stringify(cliTop) === JSON.stringify(q.top)) agree++;
    else console.log(`       differs: ${q.b}|${q.d}|${q.c} [${q.items}] -x[${q.un}] page=${JSON.stringify(q.top)} cli=${cliTop ? JSON.stringify(cliTop) : out.slice(0, 160)}`);
  }
  ok(tried > 0 && agree === tried, `page and CLI agree on ${agree}/${tried} baskets`);
}

console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exit(failures ? 1 : 0);
