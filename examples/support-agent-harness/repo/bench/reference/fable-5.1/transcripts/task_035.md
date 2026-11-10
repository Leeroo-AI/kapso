# task_035 · trial 0 · resolved · ended: user_stop

**assistant:** Hi! How can I help you today?

**user:** Hi, I need help urgently. I just got an alert that my credit score dropped 127 points overnight. Credit Karma says Rho-Bank reported a 60-day delinquency on my credit card—but I’ve never missed a payment. I have autopay. I checked my Rho-Bank account and all my payments are current.

I’m supposed to apply for a mortgage next week and this could ruin everything. What is going on, and how do you fix this today?

**assistant:** I understand how stressful this is, especially with a mortgage application coming up. Let me look into how we handle credit reporting disputes so I can give you accurate guidance.

**assistant → KB_search_dense:** {"query": "credit bureau reporting error delinquency dispute incorrect late payment reported"}

**assistant → KB_search_bm25:** {"query": "credit report delinquency inaccurate reporting dispute"}

**tool:** 1. Internal: Filing a Debit Card Transaction Dispute
   ID: doc_bank_accounts_bank_accounts_(general)_031
   Score: 0.6736
   Content: When a customer needs to file a dispute for a debit card transaction (unauthorized charges, ATM errors, merchant issues, or incorrect amounts), the agent must gather comprehensive information and follow Regulation E requirements. Before proceeding, inform the customer of their liability exposure based on when they noticed the unauthorized activity:
- Reported within 2 business days of statement: Maximum liability $50
- Reported within 60 days of statement: Maximum liability $500
- Reported after 60 days: Unlimited liability - customer may not recover funds

Dispute the earliest (first) transaction when multiple duplicates exist.

**Pre-Filing Requirements:**
1. Customer must be verified
2. Transaction must be at least $1.00
3. Transaction must be within 60 days old
4. Customer cannot exceed the maximum open disputes for their checking account tier: Entry Tier max 2, Mid Tier max 3, Premium Tier max 4, Elite Tier max 5. Dispute limits are per account, not per customer.
5. The debit card must be linked to an OPEN checking account
6. For ATM disputes, determine if it was a Rho-Bank ATM or third-party ATM (different processes apply)

**Tool: file_debit_card_transaction_dispute_6281**

**Tool Arguments:**

1. **transaction_id** (string) - ID of the transaction being disputed. Use get_bank_account_transactions_9173 to find it.

2. **account_id** (string) - The checking account ID linked to the debit card.

3. **card_id** (string) - The debit card ID. 

4. **user_id** (string) - The customer's Rho-Bank user ID.

5. **dispute_category** (string) - Must be exactly one of:
   - 'unauthorized_transaction': Transaction customer did not make or authorize (use only when fraud is NOT suspected)
   - 'atm_cash_discrepancy': ATM dispensed wrong amount or no cash
   - 'atm_deposit_not_credited': ATM deposit not reflected in account
   - 'duplicate_charge': Same transaction charged multiple times
   - 'incorrect_amount': Charged different amount than expected
   - 'goods_services_not_received': Paid but never received item/service
   - 'recurring_charge_after_cancellation': Subscription cancelled but still charging
   - 'card_present_fraud': Physical card used fraudulently (not by customer) - USE THIS when fraud suspected and card was physically present
   - 'card_not_present_fraud': Online/phone transaction customer didn't make - USE THIS when fraud suspected for online/phone transactions

 When a transaction is unauthorized, determine if fraud is suspected. If YES, use 'card_present_fraud' (for in-store/physical transactions) or 'card_not_present_fraud' (for online/phone transactions). Only use 'unauthorized_transaction' when fraud is NOT suspected (e.g., family member used card without permission, customer forgot about a transaction, etc.).

6. **transaction_date** (string, MM/DD/YYYY) - Date the disputed transaction occurred.

7. **discovery_date** (string, MM/DD/YYYY) - Date customer first noticed the issue.

8. **disputed_amount** (float) - The dollar amount being disputed.

9. **transaction_type** (string) - Determine from user circumstances. Must be exactly one of:
   - 'pin_purchase': In-store purchase with PIN
   - 'signature_purchase': In-store purchase with signature
   - 'online_purchase': Online or card-not-present transaction
   - 'atm_withdrawal': ATM cash withdrawal
   - 'atm_deposit': ATM deposit
   - 'recurring_payment': Subscription or automatic payment
   - 'person_to_person': P2P transfer (EveryonePay, etc.)

10. **card_in_possession** (boolean) - Ask: "Do you still have your physical debit card in your possession?" This affects fraud classification.

11. **pin_compromised** (string) - Ask: "Do you believe your PIN may have been compromised?" Must be exactly one of:
    - 'yes_shared': Customer shared PIN with someone
    - 'yes_observed': Customer believes PIN was observed/skimmed
    - 'no': PIN not compromised
    - 'unknown': Customer unsure

12. **contacted_merchant** (boolean) - Ask: "Have you attempted to resolve this directly with the merchant?" Required for non-fraud disputes.

13. **police_report_filed** (boolean) - For fraud disputes over $500, ask if customer has filed a police report. If not, recommend they do so.

14. **written_statement_provided** (boolean) - Whether the customer has provided a written statement describing what happened. Required for Reg E provisional credit eligibility. Ask the customer: "Are you willing to provide a written statement describing what happened? We can use this conversation as your written statement if you agree." Set to true if the customer agrees.

15. **provisional_credit_eligible** (boolean) - Agent must determine based on Debit Card Provisional Credit Guidelines.

16. **card_action** (string) - Determine based on dispute category using the mapping below. Must be exactly one of:
    - 'keep_active': Keep card active
    - 'freeze_pending_investigation': Temporarily freeze card during investigation
    - 'close_and_reissue': Close card and issue replacement

**Note:** This parameter records metadata only; the agent must separately perform the indicated card action after filing the dispute.

**Card Action Mapping by Dispute Category:**
- 'card_present_fraud' → 'close_and_reissue'
- 'card_not_present_fraud' → 'close_and_reissue'
- 'unauthorized_transaction' → 'freeze_pending_investigation'
- 'atm_cash_discrepancy' → 'keep_active'
- 'atm_deposit_not_credited' → 'keep_active'
- 'duplicate_charge' → 'keep_active'
- 'incorrect_amount' → 'keep_active'
- 'goods_services_not_received' → 'keep_active'
- 'recurring_charge_after_cancellation' → 'keep_active'

**Multiple Disputes on Same Card:** When filing multiple disputes for the same card, record each dispute's card_action based on its own category mapping (do NOT change individual dispute parameters). However, when performing the actual card action after filing all disputes, use the MOST SEVERE action across all disputes. Severity order (highest to lowest): 'close_and_reissue' > 'freeze_pending_investigation' > 'keep_active'. For example, if Dispute A maps to 'keep_active' and Dispute B maps to 'freeze_pending_investigation', record 'keep_active' for Dispute A and 'freeze_pending_investigation' for Dispute B, but then call freeze_debit_card_3892 (the most severe action) once for the card.

2. Filing a Credit Card Transaction Dispute (Internal)
   ID: doc_credit_cards_credit_cards_(general)_014
   Score: 0.6468
   Content: ## Process Summary
When a customer needs to file a formal dispute for a credit card transaction (such as unauthorized charges, merchant issues, or billing errors), the agent must gather comprehensive information and call the file_credit_card_transaction_dispute_4829 tool. First unlock the tool, then call it using call_discoverable_agent_tool with the tool name and a JSON string containing all the required arguments.

## Tool Arguments - Each numbered item below corresponds to a required argument:
1. transaction_id (string) - ID of the transaction being disputed. 

2. card_action (string) - Determine the appropriate card action based on the user's situation. Must be exactly one of these values:
   - 'keep_active': Keep the card active, just dispute this charge (use when user wants to continue using their current card)
   - 'cancel_and_reissue': The card is being cancelled and a replacement issued (use when user wants card replaced - whether you've already ordered a replacement card via order_replacement_credit_card_7291 or the cancellation is happening as part of this dispute) 

3. card_last_4_digits (string) - Last 4 digits of the credit card under which the disputed transaction took place. ". 

4. full_name (string) - The full name of the user. 

5. user_id (string) - The Rho-Bank user ID of the user. 

6. phone (string) - The registered phone number of the user. 

7. email (string) - The registered email address of the user. 

8. address (string) - The registered home address of the user. 

9. contacted_merchant (boolean) - Ask user: Did you try to resolve this with the merchant first? Pass true or false

10. purchase_date (string, format MM/DD/YYYY) - The date in which the disputed transaction occurred. 

11. issue_noticed_date (string, format MM/DD/YYYY) - Ask user when they noticed the issue. 

12. dispute_reason (string) - Ask user to select one. Must be exactly one of these values:
   - 'unauthorized_fraudulent_charge': Charge was not authorized or is fraudulent
   - 'duplicate_charge': Same charge appeared multiple times
   - 'incorrect_amount': Amount charged differs from expected
   - 'goods_services_not_received': Never received what was paid for
   - 'goods_services_not_as_described': Received item/service differs from description
   - 'canceled_subscription_still_charging': Subscription was cancelled but charges continue
   - 'refund_never_processed': Merchant promised refund but it was never applied

13. resolution_requested (string) - Ask user what resolution they want. Must be exactly one of these values:
    - 'full_refund': Complete refund of the transaction amount
    - 'partial_refund': Partial amount (must also provide partial_refund_amount)
    - 'reversal_of_charge': Charge reversal/chargeback

14. partial_refund_amount (number, optional) - Only required if resolution_requested is 'partial_refund'. The dollar amount for the partial refund.

15. eligible_for_provisional_credit (boolean) - Agent must determine this based on the Provisional Credit Eligibility Guidelines article in this knowledge base. Pass true or false.

3. Internal: Checking Debit Card Dispute Status
   ID: doc_bank_accounts_bank_accounts_(general)_035
   Score: 0.6255
   Content: To retrieve a customer's debit card dispute history and check the status of open disputes, use the get_debit_dispute_status_7483 tool.

**Tool: get_debit_dispute_status_7483(user_id: str)**

Returns a list of all debit card disputes filed by the customer, including:
- dispute_id: Unique identifier for the dispute
- transaction_id: The disputed transaction
- account_id: The checking account involved
- dispute_category: Type of dispute
- disputed_amount: Amount in dispute
- filing_date: When the dispute was filed
- status: Current status (see below)
- provisional_credit_issued: Boolean
- provisional_credit_amount: Amount of provisional credit if issued
- provisional_credit_date: Date provisional credit was applied
- expected_resolution_date: Estimated completion date
- resolution: Final outcome (if resolved)
- resolution_date: Date of resolution (if resolved)

**Dispute Statuses:**
- OPEN: Dispute filed, investigation in progress
- PENDING_DOCUMENTATION: Waiting for customer to provide additional documentation (affidavit, police report, etc.)
- UNDER_REVIEW: Investigation complete, under final review
- PROVISIONAL_CREDIT_ISSUED: Provisional credit applied, investigation ongoing
- RESOLVED_CUSTOMER_FAVOR: Dispute resolved in customer's favor, credit is permanent
- RESOLVED_BANK_FAVOR: Investigation found transaction was valid, no credit issued
- RESOLVED_PARTIAL: Partial credit issued
- PROVISIONAL_REVERSED: Provisional credit was reversed after investigation
- CLOSED_NO_RESPONSE: Closed due to customer not providing required documentation

**Timeline Monitoring:**
When checking dispute status, verify that regulatory timelines are being met:
- Provisional credit should be issued within 10 days (20 for new accounts)
- Investigation should complete within 45 days (90 for international/POS)

If a dispute appears to be exceeding timelines, escalate to a supervisor.

4. Internal: Submitting Interest Discrepancy Reports
   ID: doc_bank_accounts_bank_accounts_(general)_044
   Score: 0.5741
   Content: When a customer's savings account interest calculation is incorrect, agents must submit an interest discrepancy report to the backend team for investigation. Use the submit_interest_discrepancy_report_7294 tool to create this report.

When to Submit a Report:

1. Missing APY Boost: Customer has a qualifying checking-savings account pairing but the linked checking APY boost was not applied to their interest calculation.

2. Incorrect Tier Rate: Customer's balance qualifies for a higher APY tier but the lower tier rate was applied.

3. Missing Relationship Bonus: Customer maintains multiple Rho-Bank products but the relationship bonus was not applied.

4. System Calculation Error: Any other discrepancy between expected and actual interest credited.

Tool: submit_interest_discrepancy_report_7294
Parameters:
- account_id (string): The savings account ID with the discrepancy
- user_id (string): The customer's unique identifier
- expected_apy (number): The APY percentage the customer should have received (e.g., 2.775 for 2.775%)
- actual_apy (number): The APY percentage that was actually applied (e.g., 2.5 for 2.5%)
- amount_difference (number): The dollar amount difference between expected and actual interest credited

Procedure:
1) Verify the customer's identity and account ownership
2) Look up the customer's accounts using get_all_user_accounts_by_user_id_3847
3) Check transaction history using get_bank_account_transactions_9173 to find the interest credit
4) Review documentation for the savings account type to determine all applicable APY components (base rate, tier rate, linked checking boost, credit card bonuses, relationship bonus)
5) Calculate the expected APY by adding all applicable components
6) Calculate the discrepancy between expected and actual interest
7) If a discrepancy exists, first apply a credit using apply_savings_account_credit_6831 to correct the customer's account
8) Then submit the discrepancy report using submit_interest_discrepancy_report_7294

Important: Always apply the credit to the customer's account BEFORE submitting the report. The report is for backend investigation to fix the underlying system issue, while the credit immediately resolves the customer's concern.

5. Submitting a Cash Back Dispute (Internal)
   ID: doc_credit_cards_credit_cards_(general)_003
   Score: 0.5637
   Content: ## When to use this
Use this process when a customer believes there is a discrepancy between the cash back they received for a particular transaction and the cash back they should have received. This process applies to all credit card transactions.

## How to submit the dispute
Instead of a step-by-step workflow, instruct the customer to use the tool directly:

- Tool to provide to the user: `submit_cash_back_dispute_0589(user_id: str, transaction_id: str)`
- Tell the customer to run the tool with their own user_id and the specific transaction_id for the purchase in question.

## Agent notes
- Confirm the customer has the correct transaction_id before they submit.
- Do not collect sensitive card details; the tool uses the identifiers provided by the user.
- Advise the customer that supporting context (e.g., category or promotion expectations) may be requested later during review, but the submission itself is initiated with the tool above.

6. (Internal) Applying a Credit Card Statement Credit
   ID: doc_credit_cards_credit_cards_(general)_017
   Score: 0.5350
   Content: ## Purpose

When an agent needs to apply a statement credit to a customer's credit card account (for goodwill adjustments, promotional credits, fee reversals, or other account credits), use the apply_statement_credit_8472 tool. First unlock the tool using unlock_discoverable_agent_tool, then call it using call_discoverable_agent_tool with the tool name and a JSON string containing all required arguments.

## Steps to Apply a Statement Credit

1. Unlock the tool:
   - Call unlock_discoverable_agent_tool with the tool name apply_statement_credit_8472.

2. Prepare the arguments JSON:
   - Include all required fields exactly as specified in Tool Arguments below.

3. Call the tool:
   - Use call_discoverable_agent_tool with:
     - Tool name: apply_statement_credit_8472
     - Arguments: JSON string containing user_id, credit_card_account_id, amount, and reason

4. Confirm the result:
   - Verify the credit appears as a negative transaction in the customer’s credit card transaction history and reduces the statement balance.

## Tool Arguments

1. user_id (string, required) - The customer's unique user identifier in the system.

2. credit_card_account_id (string, required) - The credit card account ID to apply the credit to. This can be found by calling get_credit_card_accounts_by_user.

3. amount (number, required) - The credit amount in dollars. Must be a positive number (e.g., 25.00 for a $25 credit).

4. reason (string, required) - The reason for the statement credit. Must be exactly one of:
   - 'goodwill_adjustment': One-time courtesy credit for customer satisfaction
   - 'promotional_credit': Credit from a promotional offer or campaign
   - 'annual_fee_reversal': Reversal of an annual fee charge
   - 'late_fee_reversal': Reversal of a late payment fee
   - 'interest_charge_reversal': Reversal of interest charges
   - 'dispute_resolution': Credit issued as part of dispute resolution
   - 'price_match': Credit for a price match guarantee
   - 'retention_offer': Credit offered to retain a customer
   - 'error_correction': Credit to correct a billing error
   - 'other': Other reasons not covered above

7. Internal: Applying Credits and Rebates to Checking Accounts
   ID: doc_bank_accounts_bank_accounts_(general)_017
   Score: 0.5279
   Content: ## Overview

This document describes the policy and procedure for agents to apply credits to customer checking accounts. Credits may only be applied to checking accounts (not savings accounts or other account types). The apply_checking_account_credit_5829 tool is used to add a credit transaction to a checking account.

## Eligible Circumstances for Applying Credits

Agents are authorized to apply credits to checking accounts ONLY in the following circumstances:

### 1. Missing Rebates

If a rebate that should have been applied to the customer's account has not yet been applied, and the customer meets all eligibility requirements for that rebate, the agent may apply the credit. Before applying:
- Verify the customer is eligible for the rebate based on documented rebate policies
- Confirm the rebate has not already been applied by checking the account's transaction history
- Ensure the rebate amount matches what is specified in the applicable rebate policy

### 2. Fee mischarges

If any fees on the customer's account do not match the correct amounts per the documented fee schedule, the agent must identify all fee discrepancies and apply a credit for the net correction. Before applying:
- Review the account terms and documented fee structures to identify all fees that differ from what should have been charged
- Check the transaction history to confirm each fee discrepancy
- Ensure the credit amount reflects the net correction across all identified fee discrepancies

## Credit Types

When applying a credit, use the appropriate credit_type:
- `rebate_credit`: For rebates that should have been applied but were missing
- `fee_refund`: For refunding fees that were incorrectly charged

## Procedure

1. Verify the customer's identity and account ownership
2. Confirm the account is a checking account (credits cannot be applied to savings or other account types)
3. Verify the customer meets one of the eligible circumstances listed above
4. Calculate the correct credit amount based on the rebate policy or fee that was overcharged
5. Use the apply_checking_account_credit_5829 tool to apply the credit
6. Inform the customer of the applied credit and new account balance

## Agent Tool Usage (Internal Only)

- Tool: apply_checking_account_credit_5829(account_id, amount, credit_type)
  - When to call: After confirming the customer meets one of the eligible circumstances above
  - How to set parameters:
    - account_id: The checking account ID to credit
    - amount: The positive dollar amount to credit (must be greater than 0)
    - credit_type: Must be one of 'rebate_credit' or 'fee_refund'
  - Expected outcome: Adds a credit transaction to the checking account and updates the account balance

The apply_checking_account_credit_5829 tool may only be called ONCE per checking account per customer interaction. After a credit is applied to a checking account, the system enforces a 14-day cooldown period before another credit can be applied to that same account. Because only one credit call is allowed per account, if multiple corrections are needed for the same account (e.g., both fee refunds and missing rebates), combine them into a single credit with the total amount and use the credit_type that applies to the majority of the corrections.

## Important Restrictions

- Credits may ONLY be applied to checking accounts. Savings accounts and other account types are not eligible.
- Credits may only be applied for the two circumstances listed above. Do not apply credits for any other reason without supervisor approval.
- Always document the reason for the credit by selecting the appropriate credit_type. 
- The credit amount must match the exact rebate or fee amount - do not round or estimate.

8. Internal: Backend Incident 11/14 - Erroneous Credit Bureau Reporting
   ID: doc_credit_cards_credit_cards_(general)_012
   Score: 0.5200
   Content: This protocol is active from 11/14/2025 through 11/18/2025, 11:59 PM EST. On 11/14/2025, a backend batch processing error caused erroneous delinquency reports to be transmitted to credit bureaus for certain credit card accounts that are in good standing. Affected customers may report: receiving alerts from credit monitoring services about sudden score drops, being notified of a delinquency they did not incur, or credit score decreasing significantly (often 80-150 points). This is Rho-Bank's fault and requires IMMEDIATE escalation. Identity verification is NOT required for this incident protocol. IMMEDIATELY use emergency_credit_bureau_incident_transfer_1114() then transfer_to_human_agents. Use empathetic, urgent language acknowledging the severity.

9. BNPL Diamond: Concierge Dispute Support, Response Times, and What to Expect
   ID: doc_buy_now_pay_later_bnpl_diamond_007
   Score: 0.5154
   Content: ## Availability
- Dedicated concierge dispute support enabled: Yes.

## Response times
- When concierge support is enabled, you can expect an initial response within 36 hours from first contact.

## What to expect during a dispute
- Triage and acknowledgement: You receive confirmation of receipt within the target window and a case reference.
- Documentation request: You are guided on the exact evidence required and how to submit it securely.
- Case handling: A specialist manages outreach to counterparties and provides periodic status updates.
- Resolution: You receive a written outcome and next steps, including any credits or adjustments when applicable.

## If concierge support is not enabled
- Your dispute is routed through the standard support workflow, and the 36 target does not apply.

10. Checking User Dispute History (Internal)
   ID: doc_credit_cards_credit_cards_(general)_016
   Score: 0.5134
   Content: ## Summary

To retrieve a user's credit card dispute history, use the get_user_dispute_history_7291 tool. Call it with the user's user_id to get a list of all disputes filed by that user, including dispute dates, statuses, and transaction details.

## When to Use

- You need a consolidated list of all credit card disputes filed by a specific user.
- You are reviewing the current status or historical progression of a user’s disputes.
- You need transaction-level context for each dispute.

## Required Input

- user_id (required): The user’s canonical internal identifier.

Tip: Ensure you are using the correct and current user_id before making the call.

## Procedure

1. Obtain the user_id for the user whose dispute history you need to review.
2. Invoke the tool with the user_id parameter.
3. Review the returned list of disputes and associated transaction details.

Example invocation (pseudocode):
```
result = get_user_dispute_history_7291(user_id="<user_id>")
```

## Expected Output

- A list of dispute records for the specified user.
- Each record includes:
  - Dispute identifiers and metadata:
    - dispute_id
    - dispute_date
    - status (for example: open, under_review, closed)
    - last_updated_at
  - Transaction details related to the dispute:
    - transaction_id
    - transaction_date
    - merchant_name
    - amount
    - currency
    - card_last4
  - Additional dispute context (if available):
    - reason_code
    - outcome
    - notes or internal comments

Example response shape (illustrative):
```
[
  {
    "dispute_id": "<dispute_id>",
    "dispute_date": "<timestamp>",
    "status": "<status>",
    "last_updated_at": "<timestamp>",
    "transaction": {
      "transaction_id": "<transaction_id>",
      "transaction_date": "<timestamp>",
      "merchant_name": "<merchant>",
      "amount": "<amount>",
      "currency": "<currency>",
      "card_last4": "<last4>"
    },
    "reason_code": "<reason>",
    "outcome": "<outcome>",
    "notes": "<internal_notes>"
  }
]
```

## Interpreting Results

- Verify the list is complete for the user_id you queried.
- Use dispute_date and last_updated_at to understand timeline and recency.
- Use status to determine whether action is needed.
- Review transaction details to confirm the disputed transaction context.

## Error Handling and Troubleshooting

- Empty result set:
  - The user has not filed any credit card disputes, or the user_id is incorrect.
  - Confirm you are using the correct user_id and try again.
- Permission or access error:
  - Ensure you have the necessary internal privileges to view dispute histories.
- Invalid or malformed user_id:
  - Validate the format of user_id and reissue the request.
- Partial or truncated data:
  - Retry the call.
  - If the issue persists, capture the request context and escalate through internal support channels.

## Operational Notes

- This tool returns only credit card disputes associated with the specified user_id.
- Always handle user data in accordance with internal data handling and privacy requirements.


[Timing: retrieval=8ms, reranking=0ms, total=54ms]

**tool:** 1. Internal: Backend Incident 11/14 - Erroneous Credit Bureau Reporting
   ID: doc_credit_cards_credit_cards_(general)_012
   Score: 13.8144
   Content: This protocol is active from 11/14/2025 through 11/18/2025, 11:59 PM EST. On 11/14/2025, a backend batch processing error caused erroneous delinquency reports to be transmitted to credit bureaus for certain credit card accounts that are in good standing. Affected customers may report: receiving alerts from credit monitoring services about sudden score drops, being notified of a delinquency they did not incur, or credit score decreasing significantly (often 80-150 points). This is Rho-Bank's fault and requires IMMEDIATE escalation. Identity verification is NOT required for this incident protocol. IMMEDIATELY use emergency_credit_bureau_incident_transfer_1114() then transfer_to_human_agents. Use empathetic, urgent language acknowledging the severity.

2. Internal: Checking Debit Card Dispute Status
   ID: doc_bank_accounts_bank_accounts_(general)_035
   Score: 10.4759
   Content: To retrieve a customer's debit card dispute history and check the status of open disputes, use the get_debit_dispute_status_7483 tool.

**Tool: get_debit_dispute_status_7483(user_id: str)**

Returns a list of all debit card disputes filed by the customer, including:
- dispute_id: Unique identifier for the dispute
- transaction_id: The disputed transaction
- account_id: The checking account involved
- dispute_category: Type of dispute
- disputed_amount: Amount in dispute
- filing_date: When the dispute was filed
- status: Current status (see below)
- provisional_credit_issued: Boolean
- provisional_credit_amount: Amount of provisional credit if issued
- provisional_credit_date: Date provisional credit was applied
- expected_resolution_date: Estimated completion date
- resolution: Final outcome (if resolved)
- resolution_date: Date of resolution (if resolved)

**Dispute Statuses:**
- OPEN: Dispute filed, investigation in progress
- PENDING_DOCUMENTATION: Waiting for customer to provide additional documentation (affidavit, police report, etc.)
- UNDER_REVIEW: Investigation complete, under final review
- PROVISIONAL_CREDIT_ISSUED: Provisional credit applied, investigation ongoing
- RESOLVED_CUSTOMER_FAVOR: Dispute resolved in customer's favor, credit is permanent
- RESOLVED_BANK_FAVOR: Investigation found transaction was valid, no credit issued
- RESOLVED_PARTIAL: Partial credit issued
- PROVISIONAL_REVERSED: Provisional credit was reversed after investigation
- CLOSED_NO_RESPONSE: Closed due to customer not providing required documentation

**Timeline Monitoring:**
When checking dispute status, verify that regulatory timelines are being met:
- Provisional credit should be issued within 10 days (20 for new accounts)
- Investigation should complete within 45 days (90 for international/POS)

If a dispute appears to be exceeding timelines, escalate to a supervisor.

3. Emerald Saver vs Green Account (savings): What is the difference?
   ID: doc_business_savings_accounts_emerald_saver_003
   Score: 10.0523
   Content: ## Key differences to consider

### Statement delivery
- Emerald Saver requires paperless statements: Yes.
- If you need mailed statements, review the Green Account (savings) terms to confirm its delivery options before choosing.

### Environmental impact reporting cadence
- Emerald Saver provides environmental impact reporting on this schedule: monthly.
- If you prefer a different reporting cadence, compare the Green Account (savings) specification to see how it aligns with your needs.

## Choosing between the two
- Select Emerald Saver when you want mandatory e‑statements and an impact update cadence of monthly.
- If printed statements or a different report timing is essential for your operations, verify the Green Account (savings) details prior to opening.

4. Green Rewards Card: Understanding Your Carbon Footprint Report
   ID: doc_business_credit_cards_green_rewards_card_003
   Score: 9.7934
   Content: ## Report Structure
- Summary page: High-level view of total categorized spend, estimated emissions, and total offsets applied.
- Category analysis: Emissions estimates by spending category with variance highlights period over period.
- Offset details: A breakdown of offsets accrued using the per-dollar rate of $0.06.
- Net impact: Your emissions minus applied offsets for the reporting window.

## Reporting Cadence
- Reports are delivered on a monthly basis. Each report covers activity that posted in that period.

## How Offsets Are Calculated
- Formula: Eligible spend multiplied by $0.06 yields pounds of CO₂ offset.
- Returns and credits: These reduce eligible spend and therefore reduce calculated offsets in the period they post.

## Reconciling to Your Transactions
- Match the total eligible spend reported to your card ledger for the same monthly period.
- Review category mappings for large vendors to ensure the appropriate emissions factors were applied.

## Interpreting Trends
- Use the category analysis to identify high-impact spend areas.
- Compare net impact across consecutive monthly reports to assess progress.

## Exporting and Sharing
- Export the report for internal review, sustainability disclosures, or stakeholder updates.
- Keep the exported file with your accounting records so that your emissions narrative and financials remain aligned.

5. What is Provisional Credit?
   ID: doc_credit_cards_credit_cards_(general)_022
   Score: 9.3729
   Content: When you file a dispute for a credit card transaction, you may hear the term "provisional credit" mentioned. We want to help you understand what this means and how it might apply to your situation.

## Understanding Provisional Credit

Provisional credit is a temporary credit that may be applied to your account while we investigate a disputed transaction. Think of it as Rho-Bank giving you the benefit of the doubt—we temporarily restore the disputed amount to your account so you're not out of pocket while we work to resolve the issue. This allows you to continue using your available credit without being impacted by charges that may ultimately be reversed.

## How It Works

If you're eligible for provisional credit and you file a dispute, the disputed amount will be temporarily credited back to your account, usually within a few business days of the dispute being filed. This credit remains on your account while our team investigates the dispute, which can take anywhere from a few days to several weeks depending on the complexity of the case.

## After the Investigation

Once the investigation is complete, one of two things will happen:

- **If the dispute is resolved in your favor**: The provisional credit becomes permanent, and the matter is closed. You won't owe anything for that transaction.
- **If the dispute is not resolved in your favor**: The provisional credit will be reversed, and the original charge will reappear on your account. You'll be responsible for paying the amount as part of your regular statement.

## Eligibility

Not all disputes automatically qualify for provisional credit. Eligibility depends on several factors, including your account history, the type of dispute, the amount in question, and other criteria. When you file a dispute with our customer service team, they will let you know whether provisional credit applies to your situation.

## Questions?

If you have questions about provisional credit or want to check on the status of a dispute you've filed, please contact our customer service team at 1-800-RHO-BANK or through the Rho-Bank mobile app. We're here to help guide you through the process and keep you informed every step of the way.

6. Disputing a payment
   ID: doc_everyone_pay_everyone_pay_008
   Score: 8.9581
   Content: ## When to file a dispute
- You believe the payment was unauthorized.
- The amount or recipient does not match what you intended.
- You encounter suspected fraud or a scam.

## How to start a dispute
- Open the transaction from your activity and select Report a problem.
- Choose the reason that best describes the issue.
- Provide a clear description and attach any supporting evidence such as messages or receipts.
- Submit the dispute and monitor status from the transaction thread.

## What to include
- Why the payment is incorrect or unauthorized.
- Any communications with the recipient.
- Screenshots or documents that support your claim.

## What happens next
- We review your submission and may contact you for additional details.
- If the payment is eligible, we will pursue recovery with the counterparty.
- You will receive updates in-app and by email as the case progresses.

## Tips
- Act quickly once you notice an issue.
- Keep all relevant correspondence until the case is resolved.

7. Tracking your environmental impact
   ID: doc_checking_accounts_evergreen_account_004
   Score: 8.9262
   Content: ## How often your impact is reported
- You receive an environmental impact report on a monthly basis.
- Each report summarizes impact for transactions that posted during that reporting period.

## What your report includes
Your environmental impact report focuses on three key measures:
- Total posted Evergreen Account spend in the period
- Estimated CO2 offset, calculated using $1.25 grams per dollar
- Estimated number of trees planted based on your spend (Rho plants a tree for every $1,000 spent)

## Calculating your CO2 offset
Use this formula to verify or estimate your offset at any time:
- CO2 offset (grams) = Total posted Evergreen spend (USD) × $1.25

Examples:
- $750 spend → 750 × $1.25 grams
- $1,000 spend → 1,000 × $1.25 grams
- $2,500 spend → 2,500 × $1.25 grams

Tip: Convert to larger units if needed
- Kilograms: grams ÷ 1,000
- Metric tons: grams ÷ 1,000,000

## Tracking trees planted
Rho plants one tree for every $1,000 spent.
- To estimate trees: Divide your posted Evergreen spend by 1,000 and take the whole-number result.
  - $750 spend → 0 trees
  - $1,000 spend → 1 tree
  - $2,500 spend → 2 trees
- Your report lists the estimated tree count for the period based on posted transactions.

## Reconciling your report step-by-step
1. Confirm the period: Identify the exact monthly window shown on the report.
2. Sum posted spend: Add all Evergreen Account transactions that posted within the period.
3. Compute offset: Multiply the sum by $1.25 to get total grams of CO2 offset.
4. Count trees: Divide the same spend total by 1,000 and take the whole-number result.
5. Compare: The calculated figures should match your report, subject to the timing considerations below.

## Timing and data considerations
- Pending vs. posted: Only posted transactions are included; pending authorizations are excluded until they post.
- Posting dates: Transactions post on the date they settle, which may differ from the purchase date near period boundaries.
- Refunds/chargebacks: These appear as negative adjustments and reduce both spend and calculated impact within the period in which they post.
- Reversals or corrections: Any corrections posted after the period will be reflected in the subsequent monthly report.

## Keeping your own running totals
- Maintain a simple spreadsheet with two columns: posted spend date and amount.
- At any point, sum the amounts for your target date range and apply:
  - Offset (grams) = Sum × $1.25
  - Trees = floor(Sum ÷ 1,000)
- Save each monthly report alongside your reconciliation for audit and sustainability reporting needs.

8. Green Rewards Card: How the Sustainability Tracker Works
   ID: doc_business_credit_cards_green_rewards_card_002
   Score: 8.8332
   Content: ## Overview of Data Flow
- Transaction categorization: Card purchases are automatically classified by merchant category and spend type.
- Impact mapping: Eligible categories map to emissions factors to estimate your footprint, then apply offsets.

## Offset Mechanics
- For every dollar you spend, the tracker applies an automatic offset of $0.06 pounds of CO₂.
- Offsets are tallied continuously and displayed in your environmental dashboard.

## Reporting Cadence
- Sustainability and carbon reports are compiled on a monthly basis.
- Each report aggregates categorized spend, estimated emissions, and total offsets applied.

## What You Will See in the Tracker
- Real-time spend segmentation by category and by cardholder
- Estimated emissions generated versus offsets applied using $0.06
- Net impact summaries aligned to your monthly reporting period

## Data Quality and Adjustments
- Returns and refunds are netted against the reporting period in which they post.
- If a merchant reclassifies, category updates flow through to your next monthly report.

## Tips for Accurate Tracking
- Assign clear categories to recurring vendors so future transactions auto-map.
- Encourage consistent receipt tagging to improve activity-level reporting.

9. What To Do If You Notice an Unauthorized Debit Card Transaction
   ID: doc_bank_accounts_bank_accounts_(general)_036
   Score: 8.7533
   Content: Discovering an unfamiliar charge on your debit card can be alarming, but don't worry—Rho-Bank is here to help you every step of the way. Because debit card transactions draw directly from your checking account, we take these matters very seriously and will work quickly to investigate and resolve the issue.

## Time is Important

With debit card disputes, how quickly you report the issue can affect your liability. Federal law provides strong protections, but they depend on timely reporting:

- **Report within 2 business days**: Your maximum liability is just $50
- **Report within 60 days**: Your maximum liability is $500
- **Report after 60 days**: You may not be able to recover the funds

If you notice something suspicious on your account, please contact us right away. Don't wait to see if additional charges appear.

## Types of Issues We Can Help With

You should contact us if you experience any of the following:

- **Unauthorized Transactions**: Charges you didn't make or approve
- **ATM Problems**: Cash not dispensed, wrong amount dispensed, or deposits not credited
- **Duplicate Charges**: The same transaction appearing multiple times
- **Incorrect Amounts**: Being charged more than you expected
- **Subscription Issues**: Being charged for a subscription you already cancelled
- **Missing Purchases**: You paid for something but never received it

## What Happens When You File a Dispute

When you report an issue to us, here's what you can expect:

1. **We'll gather information**: Our representative will ask you questions about the transaction and what happened
2. **Provisional credit**: For many types of disputes, we'll temporarily credit your account while we investigate, so you're not out of pocket. Provisional credit is typically issued within 10 business days.
3. **Investigation**: We'll research the transaction, which may involve contacting the merchant or ATM network
4. **Resolution**: We'll notify you of our findings, typically within 10 to 45 business days depending on the complexity

## What You'll Need

To help us process your dispute quickly, please have the following ready:

- Your account information
- Details about the transaction (date, amount, merchant name if known)
- When you first noticed the issue
- Whether you still have your physical debit card
- Any communication you've had with the merchant

## How to Reach Us

To report an unauthorized transaction or file a dispute, contact us immediately:

- **Phone**: 1-800-RHO-BANK (available 24/7 for fraud reports)
- **Mobile App**: Use the "Report Issue" feature on any transaction
- **Online**: Visit rhobank.com/disputes

For suspected fraud, we recommend also filing a police report, especially for amounts over $500. This can help with the investigation and may be required for certain claims.

At Rho-Bank, protecting your money is our priority. Don't hesitate to reach out if something doesn't look right—we're here to help.

10. Internal: Debit Card Provisional Credit Guidelines
   ID: doc_bank_accounts_bank_accounts_(general)_032
   Score: 8.6535
   Content: Under Regulation E, Rho-Bank is REQUIRED to provide provisional credit for debit card disputes under certain conditions. 

**Timeline for Provisional Credit:**
- Standard accounts: Must issue within 10 business days of dispute filing
- New accounts (open less than 30 days): Must issue within 20 business days

**Provisional Credit is REQUIRED when ALL of these conditions are met:**

1. **Timely Reporting**: Customer reported the unauthorized transaction within 60 days of the statement date showing the transaction

2. **Dispute Category**: The dispute is for one of these categories:
   - 'unauthorized_transaction'
   - 'card_present_fraud'
   - 'card_not_present_fraud'
   - 'atm_cash_discrepancy'
   - 'duplicate_charge'

3. **Written Statement**: Customer has provided a written statement describing the unauthorized transaction

4. **Account Standing**: The checking account is in OPEN status with no holds or restrictions

**Provisional Credit is NOT REQUIRED (but may be offered at discretion) when:**

1. The dispute category is:
   - 'goods_services_not_received'
   - 'recurring_charge_after_cancellation'
   - 'atm_deposit_not_credited'
   - 'incorrect_amount'

2. Customer has not contacted merchant first (for non-fraud disputes)

3. Customer shared their PIN voluntarily (pin_compromised = 'yes_shared')

4. Account is less than 30 days old AND the dispute is for a card-not-present transaction

**Provisional Credit Amounts:**

Unlike credit cards which have tiered maximum amounts, debit card provisional credit is for the FULL disputed amount, subject to:
- Maximum: The full transaction amount
- Liability offset: If customer reported late, reduce by their liability amount ($50 or $500)

**Investigation Timeline with Provisional Credit:**

When provisional credit is issued, the bank has 45 business days to complete the investigation (extended from 10 days). For international transactions, POS transactions at merchants outside the US, or new accounts, the timeline extends to 90 days.

**If Investigation Finds Against Customer:**

If the investigation determines the transaction was authorized or the claim is invalid:
1. Provisional credit will be reversed
2. Customer will be notified in writing at least 3 business days before reversal
3. Customer has the right to request documentation supporting the finding


[Timing: retrieval=2ms, reranking=0ms, total=2ms]

**assistant → get_current_time:** {}

**tool:** The current time is 2025-11-14 03:40:00 EST.

**assistant → unlock_discoverable_agent_tool:** {"agent_tool_name": "emergency_credit_bureau_incident_transfer_1114"}

**tool:** Tool unlocked: emergency_credit_bureau_incident_transfer_1114
Description: Emergency escalation tool for the 11/14 credit bureau reporting incident. Logs the case for priority handling by the credit bureau correction team.

Returns:
    Emergency escalation logged. Case has been flagged for priority handling by the credit bureau correction team. Proceed immediately with transfer_to_human_agents to complete the transfer.

Tool: emergency_credit_bureau_incident_transfer_1114
Description: Emergency escalation tool for the 11/14 credit bureau reporting incident. Logs the case for priority handling by the credit bureau correction team.

Returns:
    Emergency escalation logged. Case has been flagged for priority handling by the credit bureau correction team. Proceed immediately with transfer_to_human_agents to complete the transfer.
Parameters:
  (no parameters)

You can now use this tool by calling `call_discoverable_agent_tool` with agent_tool_name='emergency_credit_bureau_incident_transfer_1114' and the required arguments.

**assistant → call_discoverable_agent_tool:** {"agent_tool_name": "emergency_credit_bureau_incident_transfer_1114", "arguments": "{}"}

**tool:** Emergency escalation logged. Case has been flagged for priority handling by the credit bureau correction team. Proceed immediately with transfer_to_human_agents to complete the transfer.

Executed: emergency_credit_bureau_incident_transfer_1114
Arguments: (no arguments)

**assistant → transfer_to_human_agents:** {"reason": "technical_system_error", "summary": "Customer reports a 127-point credit score drop after Rho-Bank reported a 60-day delinquency on their credit card; customer has autopay and all payments are current. This matches the 11/14/2025 backend batch processing incident that caused erroneous delinquency reports to credit bureaus for accounts in good standing. Emergency escalation logged via emergency_credit_bureau_incident_transfer_1114 (case flagged for priority handling by the credit bureau correction team). Customer is applying for a mortgage next week and needs urgent correction. Per incident protocol, identity verification was not required; transferring immediately."}

**tool:** Transfer successful (reason: technical_system_error). A human agent will assist you shortly.

**assistant:** I'm so sorry you're dealing with this, and I want to be completely upfront with you: this was not your fault. Rho-Bank experienced a backend processing error today that caused incorrect delinquency reports to be sent to the credit bureaus for certain credit card accounts that are actually in good standing. Everything you described — the sudden alert, the large score drop, a delinquency you never incurred while your payments are current — matches this incident exactly.

Here's what I've done for you right now:

1. **Flagged your case for emergency priority handling** with our credit bureau correction team, the group responsible for getting these erroneous reports corrected with the bureaus.
2. **Transferred you to a human agent** who will take it from here. I've included in the handoff that you have a mortgage application next week so they understand the urgency.

A human agent will be with you shortly to confirm your details and walk you through the correction process and timeline. I know how much is riding on this, and I'm truly sorry for the stress this has caused.

**user:** ###TRANSFER###

