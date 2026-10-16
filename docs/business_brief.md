# Business Risk Assessment: Card-Not-Present Fraud & Chargeback Exposure

**Company:** NovaCart (fictional online retailer, used as the case study for this project)
**Prepared by:** Risk & Analytics
**Date:** October 2026
**Classification:** Portfolio project. NovaCart is fictional; its transactions are the real IEEE-CIS dataset, and industry figures are sourced below.

> **Executive summary.** NovaCart's fraud ratio is 3.5%, more than double Visa's 1.5% limit, and doing nothing costs about **$556k–$619k a month**. A machine-learning risk score plus a cost-optimised decision policy cuts the total cost of fraud by **60% (about $334k a month)** on an unseen test month, brings the fraud ratio to **1.27%** (under Visa's limit, no fines), and still approves **91% of genuine customers instantly**. The 0.5% target is reachable only at the price of extra checks for half of all genuine customers, so we recommend adopting the policy now and reaching 0.5% through better data. See **Section 8**.

---

## 1. Situation

NovaCart is a fast-growing online retailer that accepts card payments through a third-party payment processor. Over the last six months (590,540 transactions, about **3,050 a day**), **3.5% of transactions and 3.9% of revenue were fraudulent**. Almost all of this is card-not-present fraud: stolen card details used at checkout, which the real cardholder later disputes as a **chargeback**.

Today NovaCart has **no automated fraud screening**. Every order that passes the card issuer's authorization is shipped.

### Measured baseline: the cost of doing nothing
*Measured on the IEEE-CIS transaction data (182 days); see `notebooks/02_eda_cost_of_doing_nothing.ipynb`.*

| Metric | Value |
|---|---|
| Fraud ratio by count (what card networks monitor) | **3.50%**, 2.3× Visa's 1.5% limit, in every month |
| Fraud as % of revenue | **3.87%** |
| Lost orders + chargeback fees ($25) + network fines ($8) | **$3.77M** over 182 days (about **$619k a month**) |
| Total cost as % of revenue | **4.7%** |
| Frauds per day (median) | **113** |

## 2. Why this is a business risk, not just a cost

| Risk | What it means for NovaCart |
|---|---|
| **Direct loss** | The goods are shipped and the money is pulled back. NovaCart loses both. |
| **Chargeback fees** | Processors charge a fee on every dispute, typically **$15–$100** each (Stripe and Shopify Payments charge $15). |
| **Hidden costs** | Industry research estimates US e-commerce merchants lose about **$4.61 in total cost for every $1 of fraud**, once fees, shipping, labour and operations are counted. |
| **Card-network monitoring** | Visa's **VAMP** program flags merchants whose combined fraud-and-dispute ratio exceeds **1.5%** (US, Canada, EU, Asia Pacific, since April 2026), with fines of about **$8 per disputed or fraudulent transaction** while over the threshold. Mastercard's excessive-chargeback program starts at **1.5% and 100+ chargebacks a month**. |
| **Losing the processor** | Acquirers are judged on their *whole portfolio* (Visa flags them from **0.5%**), so they pressure individual merchants to stay well below the network limit, and can offboard them. **Without a processor, NovaCart cannot take card payments at all.** |

At 3.5%, NovaCart is more than double the network threshold, every month. **This is the most serious risk: continued operation, not just margin.**

## 3. The trade-off we must manage

Blocking fraud is easy if we block a lot of customers. That is also expensive:

- **False declines** (rejecting a genuine customer) cost US merchants more than fraud itself, according to industry research, and roughly **4 in 10 shoppers say they never return** after being wrongly declined.
- **Step-up verification** (SMS one-time code / 3-D Secure) stops most fraud, but some genuine customers abandon checkout at that step.

So the goal is not "catch every fraudster". It is **minimise the total cost of fraud plus the cost of friction**.

## 4. Objectives

| # | Objective | Target |
|---|---|---|
| 1 | Fraud chargeback ratio (by count) | From **3.5% to below 0.5%** (stays clear of acquirer pressure, well under the 1.5% network limit) |
| 2 | Fraud losses as % of revenue | From **3.9% to below 0.5%** |
| 3 | Merchant friction | **At least 90%** of genuine customers approved instantly, with no extra step |
| 4 | Manual review workload | **≤ 100 orders/day** (2 analysts × 50 reviews each, about 3% of daily volume) |
| 5 | Total cost of fraud | Show a net saving after all model and friction costs |

## 5. Proposed policy: three-tier risk triage

Every transaction receives a **fraud risk score** from a machine-learning model, then follows one of three paths:

| Tier | Action | Purpose |
|---|---|---|
| 🟢 **Low risk** | Approve instantly | Zero friction for the large majority of genuine customers |
| 🟡 **Medium risk** | Step-up verification (SMS code / 3-D Secure) | Stops most fraud while letting genuine customers through |
| 🔴 **High risk** | Decline and send to the manual review queue | Blocks the clearest fraud; analysts recover false declines |

**The score cut-offs will not be fixed by guesswork.** They will be chosen by finding the thresholds that **minimise total financial cost**, subject to the review-capacity limit (Objective 4). The capacity limit is why the model must be precise at the top of the score range: a model that sends 1,000 orders a day to two analysts is unusable, however many frauds it catches. With about 113 frauds a day against 100 review slots, analysts could not keep up even if the model sent them only fraud, so **most risk must be handled automatically** (decline or step-up), with analysts reserved for the most uncertain cases.

## 6. Cost assumptions

Each decision has a dollar cost, and costs scale with the order value rather than being flat. Industry-sourced values are marked; the rest are stated assumptions, tested in the sensitivity analysis (Section 8).

| Outcome | Cost to NovaCart | Values used |
|---|---|---|
| Fraud approved (missed) | Order amount + chargeback fee (+ Visa fine in months above 1.5%) | $25 fee (industry range $15–$100); $8 VAMP fine (Visa) |
| Genuine customer declined | Lost margin on the order + chance the customer never returns × their future margin | 30% margin; 30% churn (surveys say ~40% "never return"; discounted for stated intent); $80 future margin |
| Customer sent to step-up | SMS cost + share of genuine customers who abandon (lost margin) | $0.05 per check; 10% abandon |
| Fraud sent to step-up | Share of fraudsters who pass verification | 10% get through |
| Manual review | Analyst time + analyst errors | $5 per case; analysts miss 10% of fraud and wrongly decline 5% of genuine orders |

We deliberately **do not** add the LexisNexis $4.61-per-$1 multiplier on top: it is an industry average of *all* hidden costs and would double-count the items above.

## 7. Approach

1. **Score:** a gradient-boosted model (LightGBM, 42 interpretable features) trained on real, anonymised e-commerce transactions (IEEE-CIS Fraud Detection dataset). Data is split by **time**: train (102 days) → validation (20 days) → a 30-day gap mimicking chargeback delay → an untouched **test month**. The model does not see card numbers. It learns from checkout signals: order amount, product type, masked card attributes, billing region, email domains, and count and timing patterns.
2. **Calibrate:** scores are turned into honest probabilities, because the cost maths needs them.
3. **Decide:** each order gets the action with the **lowest expected dollar cost**, using fraud probability **and** order value, within two operating limits: ≤ 100 manual reviews a day (enforced day by day) and ≥ 90% instant approval for genuine customers. Every setting is chosen on the validation month; the test month is used once, to report results.
4. **Explain:** SHAP values give every flagged order its top reasons, so analysts and customer service can see *why*.

## 8. Results and recommendation

*Untouched test month: 85,430 orders, 30 days. Notebooks 03–05; all figures in `reports/`.*

### Policies compared

| Policy | Cost per 30 days | Saving | Fraud ratio | Genuine customers approved instantly | Reviews/day (max) |
|---|---|---|---|---|---|
| 0. Do nothing | $556k | — | 3.50% | 100% | 0 |
| A. Textbook thresholds (0.30 / 0.75) | $405k | 27% | 2.17% ✗ over Visa limit | 98.8% | 64 |
| B. Cost-optimised thresholds | $225k | 60% | 1.08% | 84.6% ✗ | 20 |
| C. Expected-cost (no friction limit) | $210k | 62% | 0.85% | 77.7% ✗ | 70 |
| **D. Expected-cost within limits (recommended)** | **$222k** | **60%** | **1.27%** | **91.3%** ✓ | **58** ✓ |
| E. Expected-cost forced to 0.5% | $257k | 54% | 0.45% | 50.6% ✗ | 100 |

### Scorecard against the objectives (Policy D)

| # | Objective | Result | |
|---|---|---|---|
| 1 | Fraud ratio below 0.5% | **1.27%** (from 3.50%) | ✗ under Visa's 1.5% limit, but above target |
| 2 | Fraud losses below 0.5% of revenue | **1.2%** of revenue (from 3.9%); 69% of fraud dollars stopped | ✗ |
| 3 | ≥ 90% of genuine customers approved instantly | **91.3%** | ✓ |
| 4 | ≤ 100 manual reviews a day | average **39**, maximum **58** | ✓ |
| 5 | Net saving | **$334k per 30 days (60%)**, roughly $4M a year if the test month is typical | ✓ |

### What we learned

- **Textbook thresholds fail.** Intuitive cut-offs save only 27% and leave NovaCart above Visa's limit, still paying fines.
- **Order value matters.** The model catches cheap fraud first (its most confident catches are small orders), so deciding on *probability × amount* rather than probability alone delivers the same savings as optimised thresholds (B) with **about 7 points less customer friction** (91.3% vs 84.6% instant approval).
- **Protecting customer experience is cheap.** Enforcing the 90% instant-approval limit costs only about $11k a month versus the unconstrained optimum (C), which would add checks for 22% of genuine customers.
- **The 0.5% target is the expensive part.** Reaching it with this model means extra checks for half of all genuine customers, 5× more lost genuine orders than Policy D, a review queue at full capacity, and $35k a month more cost than D.
- **The decisions are explainable and sensible.** Declines are driven mainly by extreme count patterns typical of card testing (e.g. an $8 order on a card linked to 203 addresses); reviews mainly by order value, so expensive, uncertain orders go to people.
- **The result is robust.** Under seven alternative assumptions (step-up abandonment 5–20%, $50 chargeback fee, 15% margin, 45% churn, 50 reviews a day), savings stay between **58% and 64%**.

### Recommendation

1. **Adopt Policy D now.** It removes Visa fines, cuts the cost of fraud by about 60%, and keeps checkout friction within the agreed limit.
2. **Monitor monthly** the fraud ratio (D's 1.27% sits close to the 1.5% limit), instant-approval rate, review volume and score calibration; retrain when they drift.
3. **Pursue 0.5% through better data, not stricter rules.** The fraud Policy D misses looks like an ordinary customer in transaction data (e.g. a $2,260 credit-card order from a gmail.com address). Adding **device and identity data** (the IEEE-CIS identity file) is the next step; stricter thresholds would mostly add friction.

## 9. Key risks and limitations

- **Concept drift:** fraud patterns change. The model needs regular monitoring and retraining.
- **Delayed labels:** chargebacks arrive weeks after the purchase, so in production the most recent transactions have no confirmed label yet. The 30-day gap between validation and test mimics this.
- **Assumed costs:** margin, churn, abandonment and analyst figures are assumptions, not NovaCart data. The sensitivity analysis shows the conclusion holds across reasonable ranges, but they should be replaced with real figures before deployment.
- **One test month:** results come from a single 30-day period; seasonal peaks may behave differently.
- **Masked features:** most IEEE-CIS columns are anonymised, so some explanations ("count feature C1") cannot be mapped to an exact business meaning.
- **Step-up liability shift:** successful 3-D Secure authentication usually shifts fraud liability to the card issuer. We conservatively did not count this benefit.

---

### Sources
- [Visa VAMP thresholds and fines (Chargeflow)](https://www.chargeflow.io/blog/vamp-visa-acquirer-monitoring-program)
- [Visa VAMP 2026 thresholds (Redo)](https://redo.com/resources/articles/chargebacks/vamp-2026-shopify)
- [Mastercard chargeback monitoring thresholds (Chargeback.io)](https://www.chargeback.io/blog/mastercard-chargeback-guide)
- [Chargeback fees by processor (Chargebacks911)](https://chargebacks911.com/chargeback-management/chargeback-fees/how-much-is-a-chargeback-fee/)
- [LexisNexis True Cost of Fraud Study 2025](https://risk.lexisnexis.com/about-us/press-room/press-release/20250402-tcof-ecommerce-and-retail)
- [False declines and customer churn (ClearSale)](https://www.clear.sale/blog/everything-you-need-to-know-about-false-declines)
- [Checkout.com / Oxford Economics false-declines research](https://www.checkout.com/ja-jp/newsroom/checkout-com-research-finds-us-businesses-lag-behind-in-payment-performance-losing-2-1-of-global-revenues-annually)

*Card-network rules change often. Figures were checked in October 2026.*
