# Business Risk Assessment: Card-Not-Present Fraud & Chargeback Exposure

**Company:** NovaCart (fictional online retailer, used as the case study for this project)
**Prepared by:** Risk & Analytics
**Date:** October 2026
**Classification:** Portfolio project. Figures for NovaCart are illustrative; industry figures are sourced below.

---

## 1. Situation

NovaCart is a fast-growing online retailer that accepts card payments through a third-party payment processor. Over the last two quarters, **fraudulent transactions have grown to roughly 2.5% of revenue**. Almost all of this is card-not-present fraud: stolen card details used at checkout, which the real cardholder later disputes as a **chargeback**.

Today NovaCart has **no automated fraud screening**. Every order that passes the card issuer's authorization is shipped.

## 2. Why this is a business risk, not just a cost

| Risk | What it means for NovaCart |
|---|---|
| **Direct loss** | The goods are shipped and the money is pulled back. NovaCart loses both. |
| **Chargeback fees** | Processors charge a fee on every dispute, typically **$15–$100** each (Stripe and Shopify Payments charge $15). |
| **Hidden costs** | Industry research estimates US e-commerce merchants lose about **$4.61 in total cost for every $1 of fraud**, once fees, shipping, labour and operations are counted. |
| **Card-network monitoring** | Visa's **VAMP** program flags merchants whose combined fraud-and-dispute ratio exceeds **1.5%** (US, Canada, EU, Asia Pacific, since April 2026), with fines of about **$8 per disputed or fraudulent transaction** while over the threshold. Mastercard's excessive-chargeback program starts at **1.5% and 100+ chargebacks a month**. |
| **Losing the processor** | Acquirers are judged on their *whole portfolio* (Visa flags them from **0.5%**), so they pressure individual merchants to stay well below the network limit, and can offboard them. **Without a processor, NovaCart cannot take card payments at all.** |

At 2.5%, NovaCart is already above the network thresholds. **This is the most serious risk: continued operation, not just margin.**

## 3. The trade-off we must manage

Blocking fraud is easy if we block a lot of customers. That is also expensive:

- **False declines** (rejecting a genuine customer) cost US merchants more than fraud itself, according to industry research, and roughly **4 in 10 shoppers say they never return** after being wrongly declined.
- **Step-up verification** (SMS one-time code / 3-D Secure) stops most fraud, but some genuine customers abandon checkout at that step.

So the goal is not "catch every fraudster". It is **minimise the total cost of fraud plus the cost of friction**.

## 4. Objectives

| # | Objective | Target |
|---|---|---|
| 1 | Fraud chargeback ratio (by count) | **Below 0.5%** (stays clear of acquirer pressure, well under the 1.5% network limit) |
| 2 | Fraud losses as % of revenue | From **2.5% to below 0.5%** |
| 3 | Merchant friction | **At least 90%** of genuine customers approved instantly, with no extra step |
| 4 | Manual review workload | **≤ 100 orders/day** (2 analysts × 50 reviews each; to be checked against daily volume in Step 2) |
| 5 | Total cost of fraud | Show a net saving after all model and friction costs |

## 5. Proposed policy: three-tier risk triage

Every transaction receives a **fraud risk score** from a machine-learning model, then follows one of three paths:

| Tier | Action | Purpose |
|---|---|---|
| 🟢 **Low risk** | Approve instantly | Zero friction for the large majority of genuine customers |
| 🟡 **Medium risk** | Step-up verification (SMS code / 3-D Secure) | Stops most fraud while letting genuine customers through |
| 🔴 **High risk** | Decline and send to the manual review queue | Blocks the clearest fraud; analysts recover false declines |

**The score cut-offs will not be fixed by guesswork.** They will be chosen by finding the thresholds that **minimise total financial cost**, subject to the review-capacity limit (Objective 4). The capacity limit is why the model must be precise at the top of the score range: a model that sends 1,000 orders a day to two analysts is unusable, however many frauds it catches.

## 6. Cost assumptions (to be finalised in Step 4)

Each decision has a dollar cost. Costs scale with the order value rather than being flat.

| Outcome | Cost to NovaCart (starting assumptions) |
|---|---|
| Fraud approved (missed) | Order amount + chargeback fee ($25) |
| Genuine customer declined | Lost margin on the order + share of customers who never return |
| Customer sent to step-up | SMS cost + share of genuine customers who abandon |
| Fraud sent to step-up | Small residual loss (most fraudsters fail verification) |
| Manual review | Analyst time per case |

These assumptions are documented openly and can be changed in the final dashboard, so a stakeholder can test "what if chargeback fees double?"

## 7. Approach

1. **Score:** a gradient-boosted model (LightGBM) trained on real, anonymised e-commerce transactions (IEEE-CIS Fraud Detection dataset), evaluated on a later time period it has never seen. The model does not see card numbers. It learns from checkout signals: order amount, product type, masked card attributes (network, debit/credit, issuer), billing region, purchaser and recipient email domains, and timing and frequency patterns.
2. **Decide:** turn scores into the three-tier policy, with thresholds chosen to minimise cost.
3. **Prove:** a financial impact dashboard comparing "no screening" with "model + policy" on fraud ratio, total cost, and customer friction.

## 8. Key risks of the solution itself

- **Concept drift:** fraud patterns change. The model needs regular monitoring and retraining.
- **Delayed labels:** chargebacks arrive weeks after the purchase, so in production the most recent transactions have no confirmed label yet. To evaluate the model honestly, we leave a **gap of about 30 days between the training and test periods**, mimicking a model that is retrained without the latest, still-unlabelled weeks.
- **Over-blocking:** an aggressive policy can cost more in lost customers than it saves in fraud. That is why friction is measured as a first-class metric.

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
