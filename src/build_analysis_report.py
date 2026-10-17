"""Build the research and implementation report from reproducible result files."""
from pathlib import Path
import json
from html import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
DEST = ROOT / "output/pdf/deep_analysis_report.pdf"
INK = colors.HexColor("#172b3a")
ACCENT = colors.HexColor("#176b70")
MUTED = colors.HexColor("#52616c")
LIGHT = colors.HexColor("#edf4f4")
WIDTH = A4[0] - 84
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleX",fontName="Helvetica-Bold",fontSize=25,leading=29,textColor=INK,spaceAfter=15))
styles.add(ParagraphStyle(name="H1X",fontName="Helvetica-Bold",fontSize=18,leading=22,textColor=INK,spaceAfter=12))
styles.add(ParagraphStyle(name="H2X",fontName="Helvetica-Bold",fontSize=11.5,leading=15,textColor=ACCENT,spaceBefore=10,spaceAfter=6))
styles.add(ParagraphStyle(name="BodyX",fontName="Helvetica",fontSize=10,leading=14,textColor=INK,spaceAfter=8))
styles.add(ParagraphStyle(name="SmallX",fontName="Helvetica",fontSize=8.4,leading=11.4,textColor=MUTED,spaceAfter=6))
styles.add(ParagraphStyle(name="CellX",fontName="Helvetica",fontSize=8.8,leading=11.7,textColor=INK))
styles.add(ParagraphStyle(name="CellHeadX",fontName="Helvetica-Bold",fontSize=8.8,leading=11.7,textColor=colors.white))
story = []


def p(text, style="BodyX"):
    story.append(Paragraph(text, styles[style]))


def title(text, number):
    if story:
        story.append(PageBreak())
    p(f"{number:02d} / STORIES COFFEE", "SmallX")
    p(text, "H1X")


def table(headers, rows, widths):
    data = [[Paragraph(escape(str(x)),styles["CellHeadX"]) for x in headers]]
    data += [[Paragraph(str(x),styles["CellX"]) for x in row] for row in rows]
    t = Table(data, colWidths=[WIDTH*w for w in widths], repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),INK),("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT]),
        ("LINEBELOW",(0,-1),(-1,-1),.5,colors.HexColor("#cbd8da")),
    ]))
    story.extend([t,Spacer(1,10)])


def dollars(v):
    return f"${v:,.0f}"


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor("#cbd8da"))
    canvas.line(42,39,A4[0]-42,39)
    canvas.setFont("Helvetica",8)
    canvas.setFillColor(MUTED)
    canvas.drawString(42,26,"Jan-Jun 2026 data | Research completed 17 Oct 2026 | USD")
    canvas.drawRightString(A4[0]-42,26,str(doc.page))


def main():
    ext = json.loads((R/"external_analysis_results.json").read_text())
    model = json.loads((R/"model_results.json").read_text())
    holiday = json.loads((R/"holiday_robustness.json").read_text())
    metrics = pd.DataFrame(model["overall_metrics"])
    finance = pd.DataFrame(model["financial_results_all_perishables"])
    fm = finance[(finance.period=="test_June") & (finance.salvage_fraction_of_cost==0) & (finance.policy=="cost_aware_poisson_quantile")].set_index("method")
    gain = fm.loc["internal","simulated_contribution_usd"] - fm.loc["four_weekdays","simulated_contribution_usd"]
    direct_gain = fm.loc["internal","simulated_contribution_usd"] - fm.loc["direct_sku_four_weekdays","simulated_contribution_usd"]
    metric = lambda level,method,period="test_June": metrics[(metrics.level==level)&(metrics.method==method)&(metrics.period==period)].iloc[0].wape_pct
    examples = pd.read_csv(R/"preparation_recommendations.csv")
    sources = {s["id"]:s for s in json.loads((R/"external_evidence.json").read_text(encoding="utf-8"))["sources"]}

    p("STORIES COFFEE / HACKATHON RESEARCH", "SmallX")
    p("Predict demand.<br/>Choose preparation.<br/>Measure profit.", "TitleX")
    p("Recommendation: build a daily preparation planner with external-context tests and a profit-based ordering rule. Start with the internal demand model that won validation. Keep external adjustments as tested candidates, rather than automatically reacting to every news event.")
    table(["What this work establishes", "Result"], [
        ["Forecast benefit",f"June branch-category error: <b>{metric('branch_category_day','internal'):.2f}%</b> versus {metric('branch_category_day','four_weekdays'):.2f}% for a four-weekday baseline."],
        ["Useful external lead","Sandwich sales were about 4 units/day lower across the three branches on defined Muslim holiday closure dates. The pattern persists across sensitivity checks."],
        ["External model comparison","Calendar and weather additions did not improve validation or June overall forecast error in the tested models."],
        ["Financial scenario",f"June preparation simulation: <b>+{dollars(gain)}</b> versus a strong category-based baseline, or +{dollars(direct_gain)} versus a direct item baseline. These are assumptions-based comparisons, not measured company savings."],
    ], [.35,.65])
    p("The business problem", "H2X")
    p("A manager must decide how much to prepare before demand is known. Too little loses contribution margin; too much ties up money and may be discarded. A sales summary does not make that decision. This prototype forecasts uncertainty, applies each branch's product economics, and returns a quantity that can be tested operationally.")
    p("The honest commercial decision", "H2X")
    p(f"Build a low-cost pilot, not an expensive rollout justified by a large revenue promise. The selected model's simulated advantage over the stronger baseline is only {dollars(gain)} across three branches in June. Waste, stockouts, lead times, shelf lives and labor are absent, so actual incremental profit is not identified. The company gains a way to test and improve daily decisions, with a clear stop rule if it does not pay.")
    p("Delivered: reproducible analysis, source/date registry, chronological model comparison, item preparation replay, financial scenarios, corrected methodology, and this implementation plan.","SmallX")

    title("Start by explaining the right surprise",2)
    busy = ext["busiest_tuesday"][0]
    unusual = ext["most_unusual_tuesday"][0]
    p(f"<b>June 30 was the busiest Tuesday: {busy['orders']} orders.</b> A forecast trained only on earlier dates expected {busy['expected_prior_data']:.0f}; the surprise was just {busy['residual_pct']:.1f}%. Beirut proxy weather was {busy['temp_max_c_proxy']:.1f}C with no rain. A World Cup window also overlaps this date, but coincidence does not explain the sales.")
    p(f"<b>March 3 was more unusual:</b> {unusual['orders']} orders versus {unusual['expected_prior_data']:.0f} expected, or +{unusual['residual_pct']:.1f}%. It falls in Ramadan and immediately after renewed escalation on March 2 [1]. Plausible mechanisms include access changes, displacement, opening hours or an internal campaign. There is no verified branch location or operating log that lets us choose among those causes.")
    story.append(Image(str(R/"figures/context_forecast_audit.png"),width=WIDTH,height=WIDTH*7/11))
    p("The order forecast above uses weekday and a time trend, refitted each day after 42 days of history. It is separate from the category/SKU preparation model. A large raw total is not necessarily an anomaly; inspect actual minus expected before searching for a cause.","SmallX")
    p("Data facts that change interpretation", "H2X")
    p("The brief describes <b>real transactions with replacement IDs</b>; SYN prefixes do not prove synthetic data. All 51,177 raw lines are retained: 33,887 orders, 54,087 units, $305,925.14 revenue and $179,430.49 recorded gross profit. The Downloads and repository CSVs have identical parsed records despite different byte hashes. Removing 132 repeated lines would remove $741.43 revenue, although the schema cannot distinguish import duplicates from repeated items in one basket.")
    p("Harbour, Garden and Uptown are aliases without a verified address map. Every branch has recorded sales on all 181 dates, which does not establish full opening hours or absence of outages. Each branch-product list price is constant, so price elasticity is not identified.","SmallX")

    title("Which outside relationships survive scrutiny?",3)
    p("The exploratory audit uses weekday controls, linear and quadratic time trends, joint weather/calendar/conflict covariates, relevant promotion controls, and seven-day HAC uncertainty. It examines 40 network/branch/category series plus weather effects for 45 products. Sparse civic and April 8 coefficients are descriptive only.")
    p(f"Across {ext['test_count']} eligible comparisons, {ext['bh_discoveries']} pass Benjamini-Hochberg adjustment; only {ext['by_discoveries']} passes the more conservative Benjamini-Yekutieli adjustment. These corrections do not solve confounding, small event counts, or the fact that hypotheses are explored on the same period.","SmallX")
    table(["Relationship", "Adjusted association", "Decision status"], [
        ["Muslim closure dates / Sandwiches","-4.06 units/day; about -18.8% of the overall category daily mean. BY q=0.0026.","Best lead for a future preparation test. Four event episodes only."],
        ["Month-edge window / all products","First/last 3 days: +13.37 units/day (+4.5%). BH passes; BY does not.","Hypothesis, not proof of salary timing. Verify payroll/campaign mechanism first."],
        ["May 4-17 exams / Harbour Cold Drinks","+3.01 units/day (+20.7%); BY does not pass.","Catchment and actual campus attendance unknown. Do not advertise an exam effect."],
        ["June 11 onward / Cold Drinks","+5.12 units/day (+16.9%); BY does not pass.","Late-June change could be weather, growth or screenings. Confirm branch screens and kickoff times."],
        ["Temperature / Garden Matcha &amp; Tea","-1.84 units/day per +5C; BY does not pass.","Category includes both hot and iced items. Single Beirut grid is an unverified proxy."],
        ["Christian dates / Uptown Meals","+1.25 units/day on a mean of only 1.65/day.","Large percentage comes from a tiny category. BY does not pass; do not prioritize on percentage alone."],
    ], [.25,.39,.36])
    p("Weather, war and Ramadan", "H2X")
    p("At network level, +5C is associated with +2.72 daily units, with an interval from -6.24 to +11.67. Rain and the conflict regime also have intervals crossing zero. This is insufficient evidence for automatic network-wide adjustments, not proof that these factors never matter. Time and exposure are strongly confounded: temperature correlates 0.81 with the quadratic time term, and the post-truce indicator 0.87.")
    p("Ramadan's 08:00-16:59 order share is 42.7%, versus 42.0% otherwise. A separate four-series analysis estimates a +1.54 percentage-point adjusted network difference. This is a wall-clock proxy, not an iftar analysis, and does not support a daytime-collapse story. Future work needs sunset-relative hours and actual store hours.","SmallX")

    title("A concrete holiday insight, with its limits",4)
    p("Sandwich demand was lower across four Muslim holiday closure episodes. Dates come from the official calendar and final government closure memos [4-8]. Christian dates include both Western and Eastern Easter calendars. Calendar labels are not classifications of customers' religion.")
    events = [r for r in holiday["off_all_muslim_events"] if r["specification"]=="full_context_quadratic"]
    table(["Event window", "Actual units", "Expected units", "Difference"], [
        [r["event"],f"{r['actual_units']:.0f}",f"{r['expected_units']:.1f}",f"{r['deviation_pct_of_expected']:.1f}%"] for r in events
    ], [.37,.19,.22,.22])
    p("Expected units here come from a model fitted outside all nine closure dates. It still uses surrounding and later dates, so this is retrospective sensitivity analysis, not a new chronological test. Fitr has four days, Adha three, and the other events one each.","SmallX")
    p("Why this deserves a test", "H2X")
    p("The estimate stays near -4 sandwiches/day with linear or quadratic growth controls. Removing an entire event block and refitting leaves -3.63 to -4.44/day. Nearby same-weekday comparisons give -3.57/day. Restricting the flags to the first religious day gives -3.04/day. The result is not solely caused by one holiday or one closure-window choice.")
    p("The formal Muslim-minus-Christian sandwich contrast is -5.20 units/day (model interval -7.38 to -3.01). The analogous all-product contrast is inconclusive. This suggests testing a product-specific adjustment, rather than reducing preparation across the whole menu.")
    table(["Sandwich item", "Estimated change / day, all branches"], [
        ["Turkey &amp; Cheese Sandwich","-2.47 (about 61% of the category reduction)"],
        ["Chicken Pesto Sandwich","-1.29"],["Halloumi Wrap","-0.87"],["Tuna Sandwich","+0.57"],
    ], [.48,.52])
    p("These item estimates are post-selection diagnostics. Branch effects are all negative, but every branch interval crosses zero. Do not automatically reduce every sandwich item at every branch.","SmallX")
    p("What money could it save?", "H2X")
    p("About 36.6 fewer units across nine days, at an average recorded cost of $5.06, represents roughly <b>$185 of ingredient exposure</b>. It becomes avoided waste only if those units would otherwise be prepared and discarded, with no salvage or lost sales. This is a small supporting opportunity, not a large business case by itself.")
    p("Four events remain weak replication: a simple two-sided event-sign test gives p=0.125. The mechanism could be lower demand, shorter hours, unavailable stock or changed traffic. Log preparation, waste and stockouts before turning the association into a permanent rule.","SmallX")

    title("The exact forecasting model to build",5)
    p("<b>Model: a regularized Poisson demand regression, followed by item allocation and a cost-based preparation rule.</b> It produces counts, shares evidence across related series, and is small enough to inspect. A more complex model is a challenger only if it improves the same held-out decision metric [14].")
    table(["Component", "Implemented specification"], [
        ["Target","Units per branch x category x day, including zero-sales combinations: 27 daily series."],
        ["Internal inputs","Branch/category identity; category-weekday and branch-weekday effects; category trends; yesterday/last-week sales; prior 7/28-day means; four previous matching weekdays."],
        ["External challengers","Known holiday/Ramadan calendar; prior-day weather. Actual target-day weather and retrospective conflict masks are separately labelled hindsight diagnostics."],
        ["Item forecast","Category mean x preceding 28-day item share, with a small +1 smoothing count. Direct SKU seasonal baselines are also tested."],
        ["Selection and evaluation","January-March initial training; April-May choose family and regularization; June rolling final evaluation. Refit every 7 days using only earlier observations, updating sales lags daily."],
    ], [.28,.72])
    labels = {"last_weekday":"Last matching weekday", "four_weekdays":"Mean of four matching weekdays", "internal":"Internal Poisson model (selected)", "calendar":"Internal + calendar", "calendar_lag_weather":"Internal + calendar + prior weather", "oracle_actual_weather_conflict":"Hindsight weather/conflict diagnostic"}
    table(["Branch-category model", "Apr-May WAPE", "June WAPE"], [
        [label,f"{metric('branch_category_day',name,'validation_Apr_May'):.2f}%",f"{metric('branch_category_day',name):.2f}%"] for name,label in labels.items()
    ], [.60,.20,.20])
    p("WAPE is total absolute unit error divided by total observed units; lower is better. The internal model improves June WAPE by 1.59 percentage points (6.4% relative) versus the four-weekday baseline. It is a one-day-ahead test, not a month-ahead June forecast. Earlier June observations may enter later weekly refits, but June never chooses the model.","SmallX")
    p(f"Item-level June WAPE remains {metric('branch_sku_day','internal'):.2f}%, versus {metric('branch_sku_day','four_weekdays'):.2f}% for category baseline plus shares and {metric('branch_sku_day','direct_sku_four_weekdays'):.2f}% for direct item averages. Sparse daily items are noisy: do not promise exact SKU counts or confuse category accuracy with item accuracy.")
    p("Keep the external data pipeline, but exclude unsupported adjustments from the first operating model. Prior-day weather is not a forecast issued for tomorrow; the cached archive is also not a timestamped real-time feed. Production weather evaluation needs forecast runs available before preparation [9].", "SmallX")

    title("Convert predictions into a financial decision",6)
    p("Choose the number of units that balances the margin lost when demand exceeds stock against the cost of leftovers. Under a one-period inventory assumption, use the demand quantile at <b>(price - cost) / (price - salvage)</b>. Price and cost are branch-specific and estimated from January-March only [15].")
    p("Scenario contribution = price x units sold + salvage x unsold units - unit cost x units prepared. Every forecast below uses the same cost-aware decision rule, so the comparison does not credit a better rule to only one model.","SmallX")
    table(["June preparation scenario", "Contribution", "Unsold", "Unmet"], [
        [label,dollars(fm.loc[name,"simulated_contribution_usd"]),f"{fm.loc[name,'unsold_units']:.0f}",f"{fm.loc[name,'unmet_units']:.0f}"]
        for name,label in [("direct_sku_four_weekdays","Direct item four-weekday baseline"),("four_weekdays","Category four-weekday + item shares"),("internal","Selected internal model + item shares")]
    ], [.49,.19,.16,.16])
    p(f"The selected model adds <b>{dollars(gain)} (+{100*gain/fm.loc['four_weekdays','simulated_contribution_usd']:.2f}%)</b> versus the stronger baseline. It sells 105 more units while leaving 89 more unsold. The mechanism in this replay is more fulfilled demand, not lower waste. The +{dollars(direct_gain)} comparison against direct item averages is also valid, but should not replace the stronger comparison.")
    p("Seven-day date-block resampling gives an exploratory June interval of approximately +$45 to +$295 versus the stronger baseline, conditional on these frozen model and inventory assumptions. It does not measure the company's current stock policy, causal uplift or annual profit.")
    p("Assumptions that could reverse this conclusion", "H2X")
    p("Recorded sales are treated as uncensored demand. Bakery, Sandwiches, Breakfast and Desserts are assumed prepared once before demand, with zero initial stock, no replenishment, no capacity limits, no substitution and zero salvage in this table. Actual shelf lives, stockouts, waste, staffing and commissions are unknown. A second scenario returns 50% of unit cost for leftovers; inspect both before choosing a policy.")
    p("Poisson uncertainty is a starting assumption. The selected model's nominal central 80% discrete intervals cover 88.0% of observed perishable SKU-days. This aggregate result can hide poor coverage by item or branch and does not establish calibrated service levels.","SmallX")
    p("Make the pilot earn its keep", "H2X")
    p("If an illustrative $600 setup cost is spread over six months, it costs $100/month. With a further $100/month for manager time and operation, the hurdle is $200/month - above this replay's $179. A larger rollout therefore needs measured savings, more coverage or lower implementation cost. These cost inputs are examples, not quotations or known company costs.")
    p("The saved preparation replay contains 1,710 branch-item-day recommendations for June, with predicted means, quantities, price, cost, actual units and contribution. It demonstrates the decision end-to-end without pretending the company followed it.","SmallX")

    title("Season, location and the next useful data",7)
    season = pd.read_csv(R/"season_category_summary.csv")
    season = season.groupby(["season_window","Category"],as_index=False).units_per_day.sum()
    sp = season.pivot(index="Category",columns="season_window",values="units_per_day")
    table(["Units/day, all branches", "Jan-Feb", "Mar-May", "June"], [
        [escape(cat),f"{sp.loc[cat,'winter_Jan-Feb']:.1f}",f"{sp.loc[cat,'spring_Mar-May']:.1f}",f"{sp.loc[cat,'early_summer_Jun']:.1f}"]
        for cat in ["Hot Coffee","Cold Coffee","Cold Drinks","Sandwiches","Bakery"] if cat in sp.index
    ], [.43,.19,.19,.19])
    p("These are observed season windows, not estimated annual seasonality. Overall sales grew strongly, so higher summer units do not establish a temperature effect. Explicitly iced/cold-brew/frappe-named products account for 21.2% of January units and 21.0% of June units. That definition is narrower than all cold drinks. No product-level temperature or rain association survives the conservative BY screen. Autumn and another annual cycle are absent.","SmallX")
    table(["Outside mechanism", "Evidence / missing link", "Action if validated"], [
        ["Conflict and displacement","Escalation Mar 2; temporary truce Apr 17; disruption persisted [1-3]. No alias-to-address mapping.","Adjust preparation and deliveries to actual opening/access conditions. Track local incidents and receiving-area demand; do not forecast future attacks."],
        ["University attendance","LAU/AUB exams overlap May 4-17, but some LAU classes were online [10-12]. Branch catchments unknown.","Prepare study snacks and staff the right hours where on-campus activity actually changes footfall."],
        ["Football and local events","World Cup began June 11 [13]. No record of screens, reservations or affected branches.","Match-time stock and seating decisions; test contribution per seat-hour and incremental basket margin."],
        ["Rain, heat, outdoor seating","Single Beirut proxy grid. Outdoor capacity, delivery availability and forecast vintage missing [9].", "Test branch/daypart effects with known coordinates and forecasts issued before ordering."],
        ["Pay cycle and internal campaigns","Month-edge association is exploratory. Payroll dates, offers and app ranking changes unobserved.","Verify internal campaigns and availability before attributing a spike to outside news."],
    ], [.24,.40,.36])
    p("Regional and holiday comparisons must stay aggregate. Do not infer a customer's religion, displacement status or neighborhood from purchases. Official Stories listings do not map the aliases to particular districts.","SmallX")

    title("Exactly what the team should finish",8)
    table(["Workstream", "Concrete deliverable and acceptance check"], [
        ["1. Data and context","Preserve source lines. Complete branch x date x item grids. Add verified branch coordinates, opening hours, preparations, waste, stockout times, shelf life, batch size, lead time and event known_at timestamps. Do not treat a closed store as zero demand."],
        ["2. Forecast service","Use the validated Poisson model and retain the category-four-weekday baseline. Predict tomorrow after closing. Export mean, interval, model version, training cutoff and known input status. Future dates need an updated calendar; current code is a historical replay."],
        ["3. Decision service","Apply branch-item price, incremental preparation cost and salvage. Return integer quantities. Add batch/capacity constraints and an explicit manager override before live use. A profitable SKU can justify more safety stock than a low-margin SKU."],
        ["4. Manager interface","Three views: tomorrow's preparation list; evidence explaining selected adjustments and uncertainty; profit/waste/availability comparison after the day ends. Include forecast vs actual and override reason. Avoid an automatic news-to-cause narrative."],
        ["5. Pilot and evaluation","Start with selected perishables. Record current decisions first. Then randomize or balance planner-vs-current-policy branch-days by branch and weekday, with carryover-safe periods. Compare actual contribution after waste and incremental labor; monitor availability and service."],
    ], [.23,.77])
    p("Practical pilot", "H2X")
    p("Use an initial two-week logging/shadow period, then a planned four-week comparison if the measured daily variance supports useful precision. With only three branches, do not randomize one whole branch per arm and call the result definitive. Choose switchback periods that exceed shelf-life/carryover, balance calendar exposure, analyze by assigned unit and report date-block uncertainty. Extend the pilot if the profit interval is too wide.")
    p("Adopt only if incremental contribution exceeds setup/maintenance/manager-time costs and availability remains acceptable under a pre-agreed service floor. Log closures and supply failures; suspend normal recommendations during operational disruption. Holiday adjustments require future events or additional historical years; the present sensitivity study is not that test.")
    p("Hackathon demo sequence", "H2X")
    p("Replay June 30 without using its sales in the forecast. Show a branch-item quantity, the price/cost tradeoff, and the result after revealing actual sales. Compare it with both strong and simple baselines. Then show the sandwich holiday candidate, why it is not yet an automatic rule, and the experiment that could establish its profit value.")
    p("Pitch: 'We convert expected demand and item economics into daily preparation choices. We tested external context, rejected additions that did not improve forecasting, and identified a holiday-specific sandwich pattern for a controlled pilot. Our replay quantifies both upside and waste; live profit is the next test.'", "SmallX")

    title("Evidence, reproducibility and sources",9)
    p("Run from the repository root", "H2X")
    p("<font face='Courier'>python -m pip install -r requirements.txt<br/>python src/external_analysis.py<br/>python src/demand_model.py<br/>python src/holiday_robustness.py<br/>python src/build_analysis_report.py</font>","SmallX")
    p("The initial repository pull advanced main to 33ed971. Original notebooks and figures are preserved; README and reports/methodology_audit.txt identify claims superseded by this analysis. Raw rows are unchanged. Outputs under reports include association/sensitivity tables, daily anomaly audit, category/item economics, model metrics/predictions, scenario finances and preparation replay. The source registry records event dates, publication dates and limitations.","SmallX")
    p("Source documents: the supplied one-page EECE490 Stories Coffee Hackathon brief and stories_transactions.csv. Brief requirements are treated as assignment context; they are not permission to publish or contact anyone. The current work remains local.","SmallX")
    refs = [
        ("war_onset_un","UN Lebanon, March 17: escalation began March 2"),
        ("ceasefire_unhcr","UNHCR, April 17 report: temporary ceasefire"),
        ("continued_displacement","UNHCR, May 5: displacement continued"),
        ("official_holidays_bdl","Banque du Liban: 2026 holiday calendar"),
        ("eid_fitr_public_closure","Government memo March 19: Fitr closure through March 23"),
        ("eid_adha_public_closure","Government memo May 18: Adha closure May 27-29"),
        ("hijri_new_year","Government memo June 9: Hijri New Year June 16"),
        ("ashoura","Government memo June 22: Ashoura June 26"),
        ("weather_forecast_asof","Open-Meteo: previous model runs and forecast lead times"),
        ("lau_calendar","LAU academic calendar 2025-26"),
        ("aub_calendar","AUB academic calendar 2025-26 (retrieved revision June 25)"),
        ("lau_online_march","LAU: primarily online lectures March 23-27"),
        ("world_cup_schedule","FIFA: 2026 World Cup schedule"),
    ]
    for i,(key,label) in enumerate(refs,1):
        p(f"[{i}] <link href='{escape(sources[key]['url'],quote=True)}' color='#176b70'>{escape(label)}</link>","SmallX")
    p("[14] <link href='https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.PoissonRegressor.html' color='#176b70'>scikit-learn: PoissonRegressor (Poisson GLM with log link)</link>","SmallX")
    p("[15] <link href='https://ocw.mit.edu/courses/15-772j-d-lab-supply-chains-fall-2014/6952be57b43aa185119c6f114908bcc5_MIT15_772JF14_Newsboy.pdf' color='#176b70'>MIT OpenCourseWare: newsvendor inventory decision</link>","SmallX")
    p("Additional sources, including Ramadan declarations, April 8 strikes, school closures, Stories locations and historical weather documentation, are in reports/external_evidence.json. Public-sector holiday closure does not establish cafe closure. The historical weather model/version was not recorded in the team's original cached file.","SmallX")
    DEST.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(DEST),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=40,bottomMargin=52,title="Stories Coffee: external context, demand and profit",author="EECE490 team research").build(story,onFirstPage=footer,onLaterPages=footer)
    print(DEST)


if __name__ == "__main__":
    main()
