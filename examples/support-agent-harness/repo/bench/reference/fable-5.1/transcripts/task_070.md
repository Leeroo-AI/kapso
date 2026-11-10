# task_070 · trial 0 · resolved · ended: user_stop

**assistant:** Hi! How can I help you today?

**user:** Yeah, hi — so here’s the deal. I run a small digital marketing agency (Brightwave Digital) and I’m looking to open a business checking account that’s easiest to use and gives me the most perks. I also need stuff like no overdraft fees and a few other requirements, and I’m hoping you can tell me which account is the right fit.

I really don’t want to compare a bunch of options — just point me to the one that works best for my situation.

**assistant:** I'd be happy to help you find the right business checking account. Let me look into our options.

**assistant → KB_search_bm25:** {"query": "business checking account overdraft fees perks"}

**assistant → KB_search_dense:** {"query": "business checking account options for small business comparison"}

**tool:** 1. Business Platinum Rewards Card: Understanding Your Annual Fee and Premium Perks
   ID: doc_business_credit_cards_business_platinum_rewards_card_008
   Score: 11.6576
   Content: ## Annual Fee Structure
- Standard annual fee: $450.00
- First-year promotional annual fee: 0

## Premium Perks Included
- Airport lounge visits unlimited: Yes
- Annual travel credits: $400
- Travel insurance maximum per trip: $62,500
- Purchase protection maximum per claim: $17,500
- Foreign transaction fee on international purchases: 0%
- Dedicated account manager assigned: Yes

## How Fees Are Assessed
- The annual fee is billed to your account and renews automatically unless the account is closed before renewal.
- If the promotional first-year fee applies, your first statement will reflect 0 for that period.

## Getting the Most from Your Perks
- Enroll eligible travel services early to use $400.
- Add traveler and loyalty details to streamline lounge access and insurance eligibility.
- Use your card for covered purchases to activate benefits tied to $17,500 and $62,500.

2. Why does my account have a lower balance than I expected or have a negative balance?
   ID: doc_checking_accounts_checking_accounts_(general)_006
   Score: 11.3661
   Content: ## What can cause a lower or negative balance
- Overdraft protections may temporarily cover transactions, which can display a negative balance until incoming funds post or adjustments complete.
- Authorization holds reduce your available balance while a merchant finalizes a purchase or security deposit.
- Pending card transactions and checks can decrease available funds before they appear as posted.
- Scheduled payments or transfers may be earmarked, lowering your available balance in advance.

## How to review and resolve
- Check both pending and posted activity to understand current holds and recent payments.
- Add funds or transfer money from another account to restore a positive available balance.
- If a hold seems higher than expected, contact the merchant to request a release or adjustment.
- Review and adjust overdraft settings to align with your preferences for coverage and fees.
- If something looks incorrect, contact support with transaction details and timestamps.

## Decline Codes Related to Balance Issues

If your debit card was declined due to balance-related reasons, you may have encountered one of these codes:

### CODE 51 - Insufficient Funds

The account doesn't have enough funds for the transaction. This is the most common decline but requires careful diagnosis.

1. **Check Current Balance**: Look up the customer's accounts to get the checking account balance.

2. **If balance APPEARS sufficient** for the transaction amount:
   
   a. **Check Authorization Holds**: Ask the customer if they have any recent authorization holds that might be reducing their available balance.
      - Authorization holds reduce available balance but don't show as posted transactions.
      - Common sources: gas stations (often $75-$150 pre-auth), hotels, car rentals, restaurants (tip buffer).
      - If holds exist, explain: 'Your posted balance is $[balance], but you may have pending authorization holds reducing your available balance. These holds typically release in 1-3 business days.'

   b. **Check Pending Transactions**: Review recent transaction history and look for transactions with status 'pending'.
      - Pending debits reduce available balance.
      - Explain: 'You have pending transactions totaling $[amount] that haven't posted yet.'

   c. **Check Overdraft Settings**: The account's overdraft_pos_enabled field from the account lookup shows if overdraft is enabled for POS.
      - Regulation E requires customer opt-in for POS/ATM overdraft coverage.
      - If overdraft_pos_enabled is FALSE: 'Your account isn't opted into overdraft coverage for debit card purchases. Would you like me to explain your options?' Then reference the overdraft features documentation.

3. **If balance is genuinely insufficient**:
   - Inform customer of their balance.
   - Offer options: 'Would you like to transfer funds from another account, or would you prefer to make a smaller transaction?'
   - If they want to transfer, help them do so.

### CODE 52 - No Checking Account

The PIN was accepted but the underlying checking account has issues.

1. Look up the customer's accounts to check the account status.

2. If account status is CLOSED: 'The checking account linked to this debit card has been closed. This card can no longer be used for transactions.'
   - If customer wants to continue banking with Rho-Bank, discuss opening a new checking account.

3. If account exists and is OPEN but still getting this code: This may be a system synchronization issue. Advise waiting 10-15 minutes and retrying.

3. Internal: Ordering a Debit Card for a Bank Account
   ID: doc_bank_accounts_bank_accounts_(general)_023
   Score: 10.9907
   Content: Procedure for if the customer inquires about ordering a debit card linked to a specific checking account. Debit cards can only be ordered for checking accounts (personal or business) - savings accounts are not eligible for debit cards.

Eligibility requirements:
1) Customer must be verified
2) The account must be a checking account (account_type must be 'checking')
3) Account status must be OPEN
4) Account must have been open for at least 3 business days (excluding weekends)
5) Customer cannot have more than 1 active debit cards per checking account
6) Account must have a minimum balance of $25 (to cover potential fees)
7) Customer must be at least 18 years old (verify using date_of_birth)
8) Customer cannot have a pending debit card order for the same account (check debit_cards table for PENDING status)
9) Customer's address on file must be a valid US domestic address (international shipping is not available)
 

Delivery Options:
- STANDARD: Free shipping, arrives in 7-10 business days
- EXPEDITED: $15 fee, arrives in 3-5 business days
- RUSH: $35 fee, arrives in 1-2 business days, fees may vary based on account tier. 

Card Design Options:
- CLASSIC: Standard Rho-Bank blue design (default, no fee)
- PREMIUM: Metallic silver finish ($10 one-time fee)
- CUSTOM: Customer-uploaded image ($25 one-time fee, subject to approval), fees may vary ased on account tier. 

Steps:
1) Verify customer identity
2) Confirm which checking account the debit card should be linked to. 
3) Check eligibility requirements for the specified account
4) Ask customer for preferred delivery option (STANDARD, EXPEDITED, or RUSH) and explain fees. 
5) Ask customer for preferred card design (CLASSIC, PREMIUM, or CUSTOM) and explain fees. 
6) Confirm the address that the customer would like to mail the card to. 
7) Use order_debit_card_5739 to order the card. 
8) Inform customer of expected delivery timeframe and any applicable fees

Important Notes:
- Expedited and rush delivery fees are automatically deducted from the linked checking account
- If the account has insufficient funds for delivery or design fees, the order will fail
- Customers can track their card shipment status using the Rho-Bank mobile app
- New cards are automatically activated upon first use with PIN entry
- The customer's existing debit card (if any) will remain active until the new card is activated.

4. Claiming AWS Credits and Stripe Discounts
   ID: doc_business_checking_accounts_sky_blue_003
   Score: 10.7205
   Content: ## AWS Credits
- Value: $2,750.

### How to Claim
- Open your perks portal and locate the AWS benefit.
- Follow the redemption flow to associate your AWS account.
- Confirm credit application status in your perks portal after submission.

## Stripe Processing Discount
- Discount on payment processing: 12.0%.

### How to Activate
- Access the Stripe perk link from your perks portal.
- Complete the partner verification steps and connect your Stripe account.
- Verify the applied discount in your Stripe dashboard after activation.

## Tips for a Smooth Redemption
- Use the same business identity details across partner portals.
- Keep confirmation emails or reference IDs for your records.

5. Gold Plus: Exclusive perks and features
   ID: doc_savings_accounts_gold_plus_account_004
   Score: 10.6562
   Content: ## Brokerage commission discount

- Enjoy a 25% percent discount on Rho-Bank brokerage trading commissions.

## Digital asset access

- Cryptocurrency trading access: Yes

## Family benefits

- Link up to 4 family member accounts for shared benefits within your household.

### How to activate these perks

- Enable brokerage perks from your investment dashboard after account verification.
- Confirm crypto access in your trading preferences if Yes is available to you.
- Add family members from your profile by submitting their account details, up to 4 linked accounts.


6. Blue Account features and fees
   ID: doc_checking_accounts_blue_account_003
   Score: 10.4324
   Content: ## Key Features
- No overdraft fees on the account
- Waivable monthly maintenance fee: $20.00; waived when your minimum daily balance is at least $625

## Optional Add-On
### Overdraft Protection Transfers
- If you opt in to overdraft protection, each transfer from a linked account to cover a negative balance incurs a $12.50 fee
- No overdraft fees are charged by the account itself

## Fees
| Item | Amount | When it applies |
|---|---:|---|
| Monthly maintenance fee | $20.00 | Charged monthly; waived with a minimum daily balance of $625 |
| Paper statement fee | $2.50 | Per month if you choose mailed paper statements |
| Returned deposit | $15.00 | Per deposited item returned unpaid |
| Incoming domestic wire transfer | $12.50 | Per incoming domestic wire |
| Stop payment on a check | $30 | Per stop payment request |

## Linked Savings APY Boosts
- When linked to a Silver Plus Savings Account: +0.35% APY boost
- When linked to a Platinum Savings Account: +0.8% APY boost

## Ways to Reduce or Avoid Fees
- Maintain a minimum daily balance of $625 to waive the $20.00 monthly maintenance fee
- Choose e-statements to avoid the $2.50 paper statement fee
- Use in-network ATMs to avoid out-of-network ATM fees
- Only enable overdraft protection if you need it, as each transfer incurs a $12.50 fee

7. Getting Started with Hunter Green Business Checking
   ID: doc_business_checking_accounts_hunter_green_001
   Score: 10.4002
   Content: ## Quick start checklist
- Confirm your delivery preference: paperless statements are required (Yes). You will manage statements digitally from day one.
- Review pricing: the monthly maintenance fee is $25.00. You can avoid this charge by maintaining at least $5,000 in your account.
- Understand earnings: your account earns 1.0% APY on your balance.
- Know the overdraft policy: the overdraft fee is $0.00.

## Set up essentials
- Activate e-statements in your profile settings to ensure uninterrupted delivery, since Yes applies.
- Add your funding source and make an initial deposit according to your business needs.
- Confirm your balance strategy so that you consistently meet the $5,000 waiver threshold if you want to avoid the $25.00.

## Interest and statements
- Track your accrued interest at the 1.0% APY in your account activity.
- Download statements from your online documents center; statements will reflect any $25.00 assessed and interest credited.

## Tips for a smooth start
- Set internal reminders to check that your balance remains at or above $5,000 before statement cycle close to avoid $25.00.
- If your operations are sensitive to overdrafts, note that the overdraft fee is $0.00.

## Referral program

Once you're settled in, consider referring other eco-conscious businesses:

- **Referrer reward**: $175 for each successful referral
- **New member bonus**: $125 for the business you refer
- **Annual limit**: Up to 10 referral bonuses per year
- **Requirement**: Referred business must deposit $10,000 within 90 days
- **Eligibility**: A minimum relationship duration of 60 days as a checking account holder is required. 

Share the Hunter Green commitment to sustainability while growing your professional network.

8. Navy Blue Account: Overdraft Protection and Fees
   ID: doc_business_checking_accounts_navy_blue_002
   Score: 10.0124
   Content: ## Overdraft options
You can choose whether transactions are covered when your available balance is not sufficient.

### If an overdraft occurs
- Overdraft fee per occurrence: $0.00
- The fee is assessed when a covered transaction settles into a negative available balance.

### Overdraft protection via linked account
- When enabled, funds can be transferred automatically from a linked eligible account to cover a shortfall.
- Transfer fee for overdraft protection: $0.00

### How to manage your settings
- In your online or mobile banking, navigate to account settings and choose your overdraft preference.
- If you enable protection, confirm the linked funding source and authorize automatic transfers.
- You can change your preference at any time before a transaction settles.

### When fees do not apply
- If a transaction is declined due to insufficient funds, no overdraft occurs and $0.00 does not apply.
- If overdraft protection fully covers the shortfall, only the protection transfer fee $0.00 may apply.

### Disputes and adjustments
- If you believe an overdraft was charged in error, contact support promptly with the transaction details.
- Reversals are evaluated case by case based on posting order and available balance at the time of settlement.

9. Green Account (checking) at a glance
   ID: doc_checking_accounts_green_account_(checking)_001
   Score: 9.9677
   Content: ## Highlights
- No overdraft fees and no overdraft coverage; transactions that exceed your available balance are declined
- Earn 0.11% APY on your checking balance
- Get direct deposits up to 1 day early
- Mobile check deposits up to $3,000 per day
- Boost a linked savings account's APY: Gold +0.75% or Silver +0.25%

## Fees and limits
| Item | Amount |
|---|---|
| Monthly maintenance fee | $22.50 |
| Minimum daily balance to waive fee | $1,350 |
| Paper statement fee (monthly) | $2.50 |
| Out-of-network ATM withdrawal fee | $3.00 |
| Returned deposit fee | $17.50 |
| Incoming domestic wire (receive) | $15.00 |
| APY on balance | 0.11% |
| Daily mobile check deposit limit | $3,000 |
| Early direct deposit | Up to 1 day early |

## Referral program
Refer friends and family to open a Green Account and earn rewards:
- You earn: $20 per successful referral
- They receive: $30 welcome bonus
- Maximum referrals per year: 5
- Qualifying deposit required: $500 within 60 days
- Account tenure requirement: 30 days since becoming a Rho-Bank checking customer.

## Important details
- To avoid the monthly maintenance fee, maintain a minimum daily balance of $1,350 each statement cycle
- If you opt for paper statements, a $2.50 monthly fee applies
- Using a non-network ATM incurs a $3.00 fee; the ATM owner may charge additional fees
- Deposited checks returned unpaid result in a $17.50 fee
- Receiving a domestic wire transfer costs $15.00
- Early direct deposit availability depends on when your payer submits the deposit; funds can post up to 1 day early
- Savings APY boosts apply when you link an eligible Gold or Silver Savings Account: +0.75% or +0.25% respectively

10. Dark Green Account at a glance
   ID: doc_checking_accounts_dark_green_account_001
   Score: 9.8895
   Content: ## Snapshot

Quick reference to core details you will use most often.

### Eligibility
- Minimum primary holder age: 17
- Maximum primary holder age: 26

### Fees and rates
- Monthly maintenance fee: $10.00
- Overdraft protection transfer fee: 0
- Out-of-network ATM withdrawal fee: 1% (min $1.50)
- Annual percentage yield (APY): 1.5%

### Rewards and perks
- Student loan payment cashback: 1.25%
- Annual academic standing bonus: $62.50
- Partner textbook retailer discount: 10.0%
- Graduation transition bonus: $100.00

### Limits
- Mobile check deposit daily limit: $1,500

### Direct deposit
- Early direct deposit: 1 day(s) before payday

### Quick tables

#### Eligibility and costs
| Item | Value |
| --- | --- |
| Minimum age | 17 |
| Maximum age | 26 |
| Monthly maintenance fee | $10.00 |
| Overdraft protection transfer fee | 0 |
| Out-of-network ATM withdrawal fee | 1% (min $1.50) |

#### Rewards and limits
| Item | Value |
| --- | --- |
| APY | 1.5% |
| Student loan payment cashback | 1.25% |
| Annual academic standing bonus | $62.50 |
| Partner textbook discount | 10.0% |
| Graduation transition bonus | $100.00 |
| Mobile deposit daily limit | $1,500 |
| Early direct deposit | 1 day(s) before payday |

### Notes
- Rewards and limits apply when eligibility conditions are met and verification is provided where required.
- Fees and rates may change with required notice.
- Get paid 1 day(s) early with direct deposit.


[Timing: retrieval=3ms, reranking=0ms, total=3ms]

**tool:** 1. Internal: Opening Business Savings Accounts
   ID: doc_bank_accounts_bank_accounts_(general)_004
   Score: 0.7006
   Content: ## Description

Procedure for opening business savings accounts. Eligibility requirements: 1) Customer must be verified, 2) Customer must already have at least one business checking account with status OPEN, 3) Customer cannot have more than 4 business savings accounts, 4) Customer must not have any accounts with negative balances, 5) Existing business checking account must have been open for at least 30 days, 6) Existing business checking account must have a balance of at least $2,500. Steps: 1) Verify customer identity, 2) Check eligibility requirements, 3) Confirm account selection with customer (business savings account_class options include Bronze Saver Account, Silver Saver Account, etc.), 4) Use open_bank_account_4821 to open the account, 5) Ask the customer if they would like you to transfer the opening deposit from their business checking account now. If yes, use transfer_funds_between_bank_accounts_7291 to transfer the required amount. If no, inform them they have 30 days to fund the account (via internal transfer or external deposit) or the account will be closed.

## Eligibility Requirements

Confirm all of the following before proceeding:
- Customer identity is verified.
- Customer has at least one business checking account with status OPEN.
- Customer has fewer than 4 existing business savings accounts.
- Customer has no accounts with negative balances.
- At least one existing business checking account has been open for at least 30 days.
- That business checking account has a current balance of at least $2,500.

Notes:
- Use the qualifying OPEN business checking account that meets both the tenure and balance thresholds as the source for the optional opening deposit transfer.

## Step-by-Step Procedure

1) Verify customer identity.
2) Check eligibility requirements (see list above).
3) Confirm account selection with the customer:
   - Ask for the desired business savings account_class (e.g., Bronze Saver Account, Silver Saver Account, etc.).
   - Ensure you capture the exact official account_class name ending with “Account.”
4) Open the new business savings account using the agent tool (see Tool Instructions below).
5) Funding the opening deposit:
   - Ask the customer if they want you to transfer the opening deposit now from their eligible business checking account.
   - If yes: initiate the internal transfer using the agent tool (see Tool Instructions below).
   - If no: inform the customer they have 30 days to fund the account via internal transfer or external deposit; otherwise, the account will be closed.

## Agent Tool Instructions

The AGENT calls these tools directly to perform actions on behalf of the customer. Do not expose tool details to the customer.

### Tool: open_bank_account_4821

- Signature:
  - open_bank_account_4821(user_id, account_type, account_class)
- When to call:
  - After eligibility is confirmed and the customer has selected the desired business savings account_class.
- How to call:
  - Set account_type to 'savings'.
  - Set account_class to the exact official name provided by the customer (e.g., 'Bronze Saver Account').
- Expected outcome:
  - Returns a new savings account record (capture the new account_id for subsequent actions).

### Tool: transfer_funds_between_bank_accounts_7291

- Signature:
  - transfer_funds_between_bank_accounts_7291(source_account_id, destination_account_id, amount)
- When to call:
  - Only if the customer authorizes transferring the opening deposit now.
- How to call:
  - source_account_id: the qualifying OPEN business checking account that meets the 30-day tenure and $2,500 balance requirements.
  - destination_account_id: the newly opened business savings account_id.
  - amount: the required opening deposit amount confirmed with the customer.
- If the transfer fails (e.g., insufficient funds):
  - Inform the customer and remind them they have 30 days to fund the account via internal transfer or external deposit, or the account will be closed.

## Checklist Before Opening

- Customer identity verified.
- OPEN business checking account identified and qualified (≥ 30 days open and ≥ $2,500 balance).
- Savings account count confirmed is < 4.
- No negative balances across any accounts.
- Customer-confirmed account_class captured exactly.

## Post-Opening Actions

- If the customer funds now: complete the transfer and confirm success.
- If the customer defers funding: clearly communicate the 30-day deadline and acceptable methods (internal transfer or external deposit), and note that the account will be closed if not funded within that timeframe.

2. Internal: Opening Business Checking Accounts
   ID: doc_bank_accounts_bank_accounts_(general)_003
   Score: 0.6064
   Content: ## Eligibility Requirements

To open a business checking account, ensure all of the following are true:
1. The customer is verified.
2. The customer has at least one existing personal checking account with status OPEN.
3. The customer does not exceed 6 business checking accounts.
4. The customer has no accounts with status CLOSED.
5. The customer's existing checking account has a balance of at least $500.

## Opening Procedure

1. Verify customer identity.
2. Check eligibility requirements listed above.
3. Confirm the customer's desired account_class selection.
   - Business checking account_class options include Navy Blue, Cobalt Blue, True Blue, etc.
4. Use open_bank_account_4821 to open the account.

3. Internal: Retrieving Debit Card Information
   ID: doc_bank_accounts_bank_accounts_(general)_028
   Score: 0.5695
   Content: Tool for retrieving debit card information for a customer's checking account. Use get_debit_cards_by_account_id_7823 to look up all debit cards associated with a specific checking account.

## Tool Usage

get_debit_cards_by_account_id_7823(account_id) - account_id is the checking account ID to retrieve debit cards for.

## Debit Card Fields Returned

- card_id: Unique identifier for the debit card
- account_id: The checking account ID the card is linked to
- user_id: The user ID of the cardholder
- card_number_last_4: Last 4 digits of the card number
- status: Current status of the card (ACTIVE, PENDING, FROZEN, CLOSED)
- issue_reason: Why the card was issued (new_account, first_card, lost, stolen, fraud, expired, damaged, upgrade, bank_reissue)
- expiration_date: Card expiration date (MM/YY format)
- date_issued: Date the card was issued
- card_design: Design type (CLASSIC, PREMIUM, CUSTOM)
- daily_purchase_limit: Maximum daily purchase amount
- daily_atm_limit: Maximum daily ATM withdrawal amount

## Common Use Cases

1. Before ordering a new debit card: Check if customer already has an active or pending card for the account
2. Before activating a card: Look up the issue_reason to determine which activation tool to use
3. Before freezing/unfreezing: Verify the card exists and check its current status
4. Before closing a card: Confirm the card_id and current status
5. Customer inquiries: Look up card details when customer asks about their debit card

## Important Notes

- This tool only returns debit cards for checking accounts (savings accounts do not have debit cards)
- Multiple cards may be returned if the account has card history (e.g., old closed cards plus current active card)
- For privacy, full card numbers are never returned - only the last 4 digits
- If no cards exist for the account, an empty list is returned

4. Light Blue Account at a glance
   ID: doc_checking_accounts_light_blue_account_002
   Score: 0.5673
   Content: ## Quick facts
- APY on your balance: 0%
- Daily mobile check deposit limit: $2,000
- Early direct deposit: 0 day(s) before payday

## Notes
- If you plan to deposit multiple checks in one day, track your total to stay within the $2,000 daily limit.
- This account offers 0 day(s) early access to direct deposits.

5. Bronze Saver Account specifications and requirements
   ID: doc_business_savings_accounts_bronze_saver_account_002
   Score: 0.5645
   Content: ## Specifications
| Item | Details |
| --- | --- |
| Minimum opening deposit | $0 |
| Interest compounding frequency | daily |
| Linked Rho-Bank business checking required | Yes |

## Requirements guidance
- Ensure a linked Rho-Bank business checking account remains active if Yes is required for your setup.
- Confirm your compounding preference is set to daily in your account details.
- Verify the initial funding meets or exceeds $0.

6. Internal: Opening Personal Savings Accounts
   ID: doc_bank_accounts_bank_accounts_(general)_002
   Score: 0.5546
   Content: ## Scope and Focus

Procedure for opening personal savings accounts. Eligibility requirements: 1) Customer must be verified, 2) Customer must already have at least one active Rho-Bank checking account, 3) Cannot have more than 5 personal savings accounts, 4) Must not have any accounts in collections or with negative balances, 5) Must have held their checking account for at least 14 days. Steps: 1) Verify customer identity, 2) Check eligibility requirements, 3) Confirm account selection with customer, 4) Use open_bank_account_4821 to open the account (note: account_class must use the full official name ending with 'Account', e.g., 'Silver Plus Account', 'Gold Account'), 5) Ask the customer if they would like you to transfer the opening deposit from their checking account now. If yes, use transfer_funds_between_bank_accounts_7291 to transfer the required amount. If no, inform them they have 30 days to fund the account (via internal transfer or external deposit) or the account will be closed.

## Eligibility Requirements (Internal Checklist)

Confirm all of the following before proceeding:
- Customer identity is verified in our systems.
- Customer has at least one active Rho-Bank checking account.
- Customer currently holds fewer than 5 personal savings accounts.
- Customer has no accounts in collections and no negative balances.
- The customer’s checking account tenure is at least 14 days.

Do not proceed if any item above is not met.

## Step-by-Step Procedure

1) Verify identity
- Authenticate the customer and confirm identity verification status on file.

2) Check eligibility
- Confirm an active checking account exists and meets the 14-day tenure requirement.
- Count existing personal savings accounts; ensure the customer is below the 5 limit.
- Review account status; there must be no collections activity and no negative balances.

3) Confirm account selection
- Discuss available personal savings account options with the customer.
- Capture the exact account_class string. It must be the full official name ending with “Account” (for example, “Silver Plus Account”, “Gold Account”).

4) Open the savings account (agent action)
- Use the open_bank_account_4821 tool with account_type set to 'savings' and the confirmed account_class.

5) Arrange opening deposit
- Ask the customer if they want you to transfer the opening deposit from their checking account now.
  - If yes: use transfer_funds_between_bank_accounts_7291 to transfer the required amount from the customer’s checking account to the newly opened savings account.
  - If no: inform the customer they have 30 days to fund the account (via internal transfer or external deposit) or the account will be closed.

6) Confirm completion
- Provide the new account details and confirm the funding status or the funding deadline.

## Agent Tool Usage (Internal Only)

The AGENT calls these tools directly to perform actions on behalf of the customer. Do not ask the customer to call tools or provide tool parameters.

- Tool: open_bank_account_4821(user_id, account_type, account_class)
  - When to call: After steps 1–3, once eligibility is confirmed and the customer has selected an account_class.
  - How to set parameters:
    - user_id: the authenticated customer’s user identifier.
    - account_type: 'savings' for personal savings accounts.
    - account_class: the full official account name ending with 'Account' exactly as confirmed with the customer.
  - Expected outcome: Creates a new personal savings account for the customer.

- Tool: transfer_funds_between_bank_accounts_7291(source_account_id, destination_account_id, amount)
  - When to call: In step 5, only if the customer authorizes an immediate transfer for the opening deposit.
  - How to set parameters:
    - source_account_id: the customer's Rho-Bank checking account to be debited.
    - destination_account_id: the newly opened personal savings account to be credited.
    - amount: the required opening deposit amount confirmed with the customer.
  - Expected outcome: Moves funds from checking to savings to complete the opening deposit.

## Decision Points and Handling

- Exceeds savings account limit: If the customer already has 5 personal savings accounts, do not open a new one. Inform the customer they have reached the maximum.
- Insufficient checking tenure: If the checking account has been open fewer than 14 days, advise the customer when they will become eligible.
- Collections or negative balances: Resolve these issues first; do not proceed until all accounts are in good standing.
- Customer defers funding: Clearly communicate the 30-day funding window and the consequence of closure if unfunded. Document the acknowledgment in the interaction notes.

7. Business Bronze Rewards Card: Exceptions and Exclusions
   ID: doc_business_credit_cards_business_bronze_rewards_card_003
   Score: 0.5526
   Content: ## Merchants Not Eligible for Cash Back
The following specific merchants earn **0% cash back** on your Business Bronze Rewards Card:

**Coworking Space Merchants:**
- WeWork
- Regus
- Industrious

**Payroll Processor Merchants:**
- Gusto
- ADP
- Paychex
- Rippling

**SaaS Subscriptions (after first 12 months only):**
The following SaaS merchants earn the 1.0% rate for the first 12 months of your subscription only. After the initial 12-month period, auto-renewals and continued subscription payments to these merchants earn 0% cash back:
- Slack
- Zoom
- HubSpot
- Salesforce

8. Paying Off Credit Card Balance from Checking Account (Internal)
   ID: doc_credit_cards_credit_card_account_logistics_009
   Score: 0.5467
   Content: ## Purpose

When a customer needs to pay off their credit card balance using funds from their Rho-Bank checking account, you can process the payment directly.

## Prerequisites

1. Verify the customer's identity using standard procedures.
2. Confirm the customer has a Rho-Bank checking account with sufficient funds to cover the payment amount.
3. Look up both the checking account and credit card account to obtain account IDs and verify balances.

## Process

1. Verify the customer's identity.
2. Look up the customer's checking account to confirm sufficient funds.
3. Look up the customer's credit card account to confirm the outstanding balance.
4. Confirm the payment amount with the customer and obtain authorization to transfer.
5. Unlock the pay_credit_card_from_checking_9182 tool using unlock_discoverable_agent_tool.
6. Call the tool using call_discoverable_agent_tool with the required arguments listed below.

## Tool Arguments for pay_credit_card_from_checking_9182

- user_id (string, required): The customer's unique identifier in the system.
- checking_account_id (string, required): The ID of the Rho-Bank checking account to debit.
- credit_card_account_id (string, required): The ID of the credit card account to credit.
- amount (float, required): The payment amount in dollars. Must be positive and cannot exceed the checking account balance or the credit card outstanding balance.

The tool will debit the specified amount from the checking account and reduce the credit card balance by the same amount. A confirmation will include the new balances for both accounts.

## Instructions for Passing the Tool

- Use unlock_discoverable_agent_tool to enable pay_credit_card_from_checking_9182 for the session.
- Then pass the arguments via call_discoverable_agent_tool exactly as listed above, ensuring the account IDs and user_id match the authenticated customer and the amount is authorized.

9. Internal: Transferring Funds Between Bank Accounts
   ID: doc_bank_accounts_bank_accounts_(general)_010
   Score: 0.5367
   Content: ## Description

Tool for transferring funds between a customer's bank accounts. Use transfer_funds_between_bank_accounts_7291 to move money from one checking or savings account to another. This tool is essential for: 1) Funding new savings accounts with the required opening deposit from an existing checking account, 2) Moving funds between accounts at customer request, 3) Consolidating balances before account closure. Requirements: Both the source and destination accounts must be in ACTIVE or OPEN status. The source account must have sufficient funds to cover the transfer amount. Parameters: source_account_id (the account ID to transfer from), destination_account_id (the account ID to transfer to), amount (the amount in USD to transfer). The tool will return an error if the source account has insufficient funds.

## Agent Discoverable Tool

- Tool signature (call directly; do not surface to the customer):
  - transfer_funds_between_bank_accounts_7291(source_account_id, destination_account_id, amount)

- Parameters
  - source_account_id: the account ID to transfer from
  - destination_account_id: the account ID to transfer to
  - amount: the amount in USD to transfer

- The AGENT calls these tools directly to perform actions on behalf of the customer.

## When the Agent Should Call This Tool

- Funding a new savings account’s opening deposit from an existing checking account
- Moving funds between a customer’s checking and savings accounts at their request
- Consolidating balances prior to closing one of the customer’s accounts

## Preconditions and Validation

Before calling transfer_funds_between_bank_accounts_7291, the agent should:

- Confirm both the source and destination accounts are in ACTIVE or OPEN status
- Verify the source account has sufficient available funds to cover the transfer amount
- Ensure the two accounts belong to the same customer and the customer has authorized the transfer
- Validate the amount is a positive USD value
- Confirm the source and destination account IDs are distinct and valid

## How to Execute (Agent Only)

1. Gather required inputs:
   - source_account_id
   - destination_account_id
   - amount (USD)
2. Validate account statuses and available funds per the Preconditions and Validation section.
3. Call the tool:
   - transfer_funds_between_bank_accounts_7291(source_account_id, destination_account_id, amount)
4. On success, confirm completion to the customer and, if requested, share updated balances.
5. Document the action per internal procedures.

## Post-Call Checks

- Verify the transfer posted as expected between the specified accounts
- Confirm no duplicate transfer was initiated
- If funding a new savings account, confirm the required opening deposit is reflected

## Error Handling

- Insufficient funds
  - The tool will return an error if the source account has insufficient funds
  - Offer to adjust the amount or select a different source account after re-validating available funds
- Invalid account status
  - If either account is not ACTIVE or OPEN, do not proceed; resolve the account status first
- Invalid or identical account IDs
  - If IDs are invalid or identical, correct the inputs before retrying
- Amount validation failures
  - Ensure the amount is present, positive, and in USD before re-attempting the transfer

10. Internal: Opening Personal Checking Accounts
   ID: doc_bank_accounts_bank_accounts_(general)_001
   Score: 0.5357
   Content: ## Eligibility Requirements

To open a personal checking account, ensure all of the following are true:
1. The customer is verified.
2. The customer is at least 18 years old.
3. The customer does not exceed 4 personal checking accounts.
4. The customer has no checking accounts closed for cause in the past 6 months.

## Opening Procedure

1. Verify customer identity.
2. Check eligibility requirements listed above.
3. Confirm the customer's desired account_class selection.
   - Personal checking account_class options must use the full official name ending with 'Account' (e.g., 'Blue Account', 'Green Account (checking)').
4. Use open_bank_account_4821 to open the account.


[Timing: retrieval=8ms, reranking=0ms, total=53ms]

**assistant → shell:** {"command": "ls | grep -i business_checking; echo ---; grep -i \"business checking\" INDEX.md | head -50"}

**tool:** doc_business_checking_accounts_beige_001.md
doc_business_checking_accounts_beige_002.md
doc_business_checking_accounts_beige_003.md
doc_business_checking_accounts_beige_004.md
doc_business_checking_accounts_beige_005.md
doc_business_checking_accounts_beige_006.md
doc_business_checking_accounts_beige_007.md
doc_business_checking_accounts_beige_008.md
doc_business_checking_accounts_beige_009.md
doc_business_checking_accounts_beige_010.md
doc_business_checking_accounts_beige_011.md
doc_business_checking_accounts_beige_012.md
doc_business_checking_accounts_cobalt_blue_001.md
doc_business_checking_accounts_cobalt_blue_002.md
doc_business_checking_accounts_cobalt_blue_003.md
doc_business_checking_accounts_cobalt_blue_004.md
doc_business_checking_accounts_cobalt_blue_005.md
doc_business_checking_accounts_cobalt_blue_006.md
doc_business_checking_accounts_cobalt_blue_007.md
doc_business_checking_accounts_cobalt_blue_008.md
doc_business_checking_accounts_cobalt_blue_009.md
doc_business_checking_accounts_cobalt_blue_010.md
doc_business_checking_accounts_hunter_green_001.md
doc_business_checking_accounts_hunter_green_002.md
doc_business_checking_accounts_hunter_green_003.md
doc_business_checking_accounts_hunter_green_004.md
doc_business_checking_accounts_hunter_green_005.md
doc_business_checking_accounts_hunter_green_006.md
doc_business_checking_accounts_hunter_green_007.md
doc_business_checking_accounts_hunter_green_008.md
doc_business_checking_accounts_hunter_green_009.md
doc_business_checking_accounts_hunter_green_010.md
doc_business_checking_accounts_lime_green_001.md
doc_business_checking_accounts_lime_green_002.md
doc_business_checking_accounts_lime_green_003.md
doc_business_checking_accounts_lime_green_004.md
doc_business_checking_accounts_lime_green_005.md
doc_business_checking_accounts_lime_green_006.md
doc_business_checking_accounts_lime_green_007.md
doc_business_checking_accounts_lime_green_008.md
doc_business_checking_accounts_navy_blue_001.md
doc_business_checking_accounts_navy_blue_002.md
doc_business_checking_accounts_navy_blue_003.md
doc_business_checking_accounts_navy_blue_004.md
doc_business_checking_accounts_navy_blue_005.md
doc_business_checking_accounts_navy_blue_006.md
doc_business_checking_accounts_navy_blue_007.md
doc_business_checking_accounts_navy_blue_008.md
doc_business_checking_accounts_navy_blue_009.md
doc_business_checking_accounts_sky_blue_001.md
doc_business_checking_accounts_sky_blue_002.md
doc_business_checking_accounts_sky_blue_003.md
doc_business_checking_accounts_sky_blue_004.md
doc_business_checking_accounts_sky_blue_005.md
doc_business_checking_accounts_sky_blue_006.md
doc_business_checking_accounts_sky_blue_007.md
doc_business_checking_accounts_sky_blue_008.md
doc_business_checking_accounts_sky_blue_009.md
doc_business_checking_accounts_sky_blue_010.md
doc_business_checking_accounts_true_blue_001.md
doc_business_checking_accounts_true_blue_002.md
doc_business_checking_accounts_true_blue_003.md
doc_business_checking_accounts_true_blue_004.md
doc_business_checking_accounts_true_blue_005.md
doc_business_checking_accounts_true_blue_006.md
doc_business_checking_accounts_true_blue_007.md
doc_business_checking_accounts_true_blue_008.md
doc_business_checking_accounts_true_blue_009.md
doc_business_checking_accounts_true_blue_010.md
doc_business_checking_accounts_true_blue_011.md
doc_business_checking_accounts_world_blue_001.md
doc_business_checking_accounts_world_blue_002.md
doc_business_checking_accounts_world_blue_003.md
doc_business_checking_accounts_world_blue_004.md
doc_business_checking_accounts_world_blue_005.md
doc_business_checking_accounts_world_blue_006.md
doc_business_checking_accounts_world_blue_007.md
doc_business_checking_accounts_world_blue_008.md
doc_business_checking_accounts_world_blue_009.md
---
- **doc_bank_accounts_bank_accounts__general__003.md** - Internal: Opening Business Checking Accounts
- **doc_bank_accounts_bank_accounts__general__007.md** - Internal: Closing Business Checking Accounts
- **doc_bank_accounts_bank_accounts__general__013.md** - Internal: Business Checking Account Promotion - November 2025
- **doc_bank_accounts_bank_accounts__general__014.md** - Internal: Business Checking Account Promotion - October 2025
- **doc_business_checking_accounts_cobalt_blue_001.md** - Getting Started with Cobalt Blue Business Checking
- **doc_business_checking_accounts_hunter_green_001.md** - Getting Started with Hunter Green Business Checking
- **doc_business_checking_accounts_lime_green_001.md** - Lime Green Business Checking: Complete Guide
- **doc_business_checking_accounts_navy_blue_001.md** - Navy Blue Business Checking: Complete Guide
- **doc_business_checking_accounts_true_blue_001.md** - True Blue Business Checking: Complete Guide
- **doc_business_savings_accounts_gold_saver_account_002.md** - Gold Saver Account: How Automated Sweeps Work with Business Checking
- **doc_business_savings_accounts_gold_saver_account_004.md** - Gold Saver Account: Integration with Business Checking

**assistant → shell:** {"command": "grep -il \"recommend\\|which account\\|best fit\\|digital marketing\\|agency\" doc_business_checking_accounts_*.md doc_bank_accounts_*.md; echo ---; grep -i \"business_checking\" INDEX.md | sed 's/^- //' "}

**tool:** doc_business_checking_accounts_beige_003.md
doc_business_checking_accounts_beige_004.md
doc_business_checking_accounts_beige_006.md
doc_business_checking_accounts_beige_008.md
doc_business_checking_accounts_navy_blue_007.md
doc_bank_accounts_bank_accounts__general__013.md
doc_bank_accounts_bank_accounts__general__014.md
doc_bank_accounts_bank_accounts__general__015.md
doc_bank_accounts_bank_accounts__general__016.md
doc_bank_accounts_bank_accounts__general__024.md
doc_bank_accounts_bank_accounts__general__025.md
doc_bank_accounts_bank_accounts__general__026.md
doc_bank_accounts_bank_accounts__general__031.md
doc_bank_accounts_bank_accounts__general__036.md
doc_bank_accounts_bank_accounts__general__041.md
doc_bank_accounts_bank_accounts__general__047.md
---
**doc_business_checking_accounts_beige_001.md** - Implementing Beige: Enterprise Treasury Onboarding
**doc_business_checking_accounts_beige_002.md** - Connecting Your ERP for Beige Account: API Integration Guide
**doc_business_checking_accounts_beige_003.md** - Beige: Balance Aggregation and Sub-Account Management
**doc_business_checking_accounts_beige_004.md** - Beige: Configuring Auto-Fund Sweeps
**doc_business_checking_accounts_beige_005.md** - Security: Dual Authorization and Transaction Controls
**doc_business_checking_accounts_beige_006.md** - Beige: Your Dedicated Account Management Team
**doc_business_checking_accounts_beige_007.md** - Earning Interest on Enterprise Balances
**doc_business_checking_accounts_beige_008.md** - API Rate Limits and Technical Specifications
**doc_business_checking_accounts_beige_009.md** - Wire Transfers and Payment Processing on Beige
**doc_business_checking_accounts_beige_010.md** - Troubleshooting: Beige Fee and Balance Issues
**doc_business_checking_accounts_beige_011.md** - Beige Account: Enterprise ATM Benefits
**doc_business_checking_accounts_beige_012.md** - Beige Account Referral Program
**doc_business_checking_accounts_cobalt_blue_001.md** - Getting Started with Cobalt Blue Business Checking
**doc_business_checking_accounts_cobalt_blue_002.md** - Cobalt Blue: No Overdraft Fees — How It Works
**doc_business_checking_accounts_cobalt_blue_003.md** - Managing Transactions and Staying Within Limits
**doc_business_checking_accounts_cobalt_blue_004.md** - Earning Rewards with Your Cobalt Blue Debit Card
**doc_business_checking_accounts_cobalt_blue_005.md** - Cobalt Blue: Payments and Transfers Explained
**doc_business_checking_accounts_cobalt_blue_006.md** - Mobile Deposits and ATM Access on Cobalt Blue
**doc_business_checking_accounts_cobalt_blue_007.md** - Using Bill Pay and Invoice Tools
**doc_business_checking_accounts_cobalt_blue_008.md** - How Interest Accrues on Your Cobalt Blue Balance
**doc_business_checking_accounts_cobalt_blue_009.md** - Troubleshooting: Unexpected Charges on Cobalt Blue
**doc_business_checking_accounts_cobalt_blue_010.md** - Cobalt Blue Account: ATM Fees and Out-of-Network Usage
**doc_business_checking_accounts_hunter_green_001.md** - Getting Started with Hunter Green Business Checking
**doc_business_checking_accounts_hunter_green_002.md** - Hunter Green: Your Carbon Offset Contribution
**doc_business_checking_accounts_hunter_green_003.md** - Earning Cashback from Eco-Friendly Vendors
**doc_business_checking_accounts_hunter_green_004.md** - Reading Your Sustainability Impact Report
**doc_business_checking_accounts_hunter_green_005.md** - Spending Alerts and Budget Controls on Hunter Green
**doc_business_checking_accounts_hunter_green_006.md** - Hunter Green: Interest and APY Explained
**doc_business_checking_accounts_hunter_green_007.md** - Mobile Deposits and Wire Transfers on Hunter Green
**doc_business_checking_accounts_hunter_green_008.md** - Paperless Banking: Why Hunter Green Requires E-Statements
**doc_business_checking_accounts_hunter_green_009.md** - Troubleshooting: Hunter Green Fees and Charges
**doc_business_checking_accounts_hunter_green_010.md** - Hunter Green Account: ATM Fees and Out-of-Network Usage
**doc_business_checking_accounts_lime_green_001.md** - Lime Green Business Checking: Complete Guide
**doc_business_checking_accounts_lime_green_002.md** - Lime Green Account: Fee Waivers and Balance Requirements
**doc_business_checking_accounts_lime_green_003.md** - Lime Green Account: Your Dedicated Account Manager
**doc_business_checking_accounts_lime_green_004.md** - Lime Green Account: Wire Transfer Pricing
**doc_business_checking_accounts_lime_green_005.md** - Lime Green Account: Interest and APY Explained
**doc_business_checking_accounts_lime_green_006.md** - Lime Green Account: Mobile Deposits and Daily Limits
**doc_business_checking_accounts_lime_green_007.md** - Lime Green Account: ATM Access and Fee Rebates
**doc_business_checking_accounts_lime_green_008.md** - Why am I being charged a fee on my Lime Green Account?
**doc_business_checking_accounts_navy_blue_001.md** - Navy Blue Business Checking: Complete Guide
**doc_business_checking_accounts_navy_blue_002.md** - Navy Blue Account: Overdraft Protection and Fees
**doc_business_checking_accounts_navy_blue_003.md** - Navy Blue Account: Wire Transfer Pricing
**doc_business_checking_accounts_navy_blue_004.md** - Navy Blue Account: Interest Explained
**doc_business_checking_accounts_navy_blue_005.md** - Navy Blue Account: Mobile Deposits and Daily Limits
**doc_business_checking_accounts_navy_blue_006.md** - Navy Blue Account: ACH Transfers and Processing Times
**doc_business_checking_accounts_navy_blue_007.md** - Navy Blue Account: Paper vs E-Statements
**doc_business_checking_accounts_navy_blue_008.md** - Why am I being charged a fee on my Navy Blue Account?
**doc_business_checking_accounts_navy_blue_009.md** - Navy Blue Account: ATM Fees and Out-of-Network Usage
**doc_business_checking_accounts_sky_blue_001.md** - Getting Started with Sky Blue: Startup Account Setup
**doc_business_checking_accounts_sky_blue_002.md** - Sky Blue: Understanding Your Free Period
**doc_business_checking_accounts_sky_blue_003.md** - Claiming AWS Credits and Stripe Discounts
**doc_business_checking_accounts_sky_blue_004.md** - Sky Blue: Burn-Rate Analytics and Runway Projections
**doc_business_checking_accounts_sky_blue_005.md** - Founder Cards: How to Request and Manage
**doc_business_checking_accounts_sky_blue_006.md** - Sky Blue: Preparing Investor Reports
**doc_business_checking_accounts_sky_blue_007.md** - Mobile Banking and Deposits on Sky Blue
**doc_business_checking_accounts_sky_blue_008.md** - After the Free Period: What Changes and What Stays
**doc_business_checking_accounts_sky_blue_009.md** - Troubleshooting: Sky Blue Eligibility Issues
**doc_business_checking_accounts_sky_blue_010.md** - Sky Blue Account: ATM Fees and Out-of-Network Usage
**doc_business_checking_accounts_true_blue_001.md** - True Blue Business Checking: Complete Guide
**doc_business_checking_accounts_true_blue_002.md** - True Blue Account: Fee Waivers and Balance Requirements
**doc_business_checking_accounts_true_blue_003.md** - True Blue Account: Cash Management Dashboard
**doc_business_checking_accounts_true_blue_004.md** - True Blue Account: Treasury Services Integration
**doc_business_checking_accounts_true_blue_005.md** - True Blue Account: Payroll Processing Discounts
**doc_business_checking_accounts_true_blue_006.md** - True Blue Account: Wire Transfers and Free Wires
**doc_business_checking_accounts_true_blue_007.md** - True Blue Account: Same-Day ACH and Mobile Deposits
**doc_business_checking_accounts_true_blue_008.md** - True Blue Account: Interest and APY Explained
**doc_business_checking_accounts_true_blue_009.md** - True Blue Account: Dedicated Support and Response Times
**doc_business_checking_accounts_true_blue_010.md** - Why am I being charged a fee on my True Blue Account?
**doc_business_checking_accounts_true_blue_011.md** - True Blue Account: ATM Fees and Out-of-Network Usage
**doc_business_checking_accounts_world_blue_001.md** - Getting Started with World Blue International Checking
**doc_business_checking_accounts_world_blue_002.md** - World Blue: How to Avoid Monthly Fees
**doc_business_checking_accounts_world_blue_003.md** - Navigating Currency Conversions on World Blue
**doc_business_checking_accounts_world_blue_004.md** - Sending International Payments: SWIFT and SEPA Options
**doc_business_checking_accounts_world_blue_005.md** - World Blue: Traveling Abroad with Your Business Account
**doc_business_checking_accounts_world_blue_006.md** - Configuring Team Permissions on World Blue
**doc_business_checking_accounts_world_blue_007.md** - Earning Interest on Your World Blue Balance
**doc_business_checking_accounts_world_blue_008.md** - Troubleshooting: My World Blue fee wasn't waived
**doc_business_checking_accounts_world_blue_009.md** - World Blue: Free International ATM Access

**assistant → shell:** {"command": "for f in doc_business_checking_accounts_cobalt_blue_001.md doc_business_checking_accounts_navy_blue_001.md doc_business_checking_accounts_true_blue_001.md doc_business_checking_accounts_lime_green_001.md doc_business_checking_accounts_sky_blue_001.md doc_business_checking_accounts_sky_blue_002.md doc_business_checking_accounts_world_blue_001.md doc_business_checking_accounts_beige_001.md; do echo \"=== $f ===\"; cat $f; echo; done"}

**tool:** === doc_business_checking_accounts_cobalt_blue_001.md ===
# Getting Started with Cobalt Blue Business Checking

## Set up your account
Follow these steps to start using your account immediately:

- Fund your account to begin transacting and to work toward waiving the monthly fee
- Enroll in online and mobile banking to monitor balances, interest, and transaction activity
- Set account alerts for balance thresholds and transaction activity so you can stay on top of usage
- Add authorized users and establish permissions as needed for your business operations

## Know your core costs and how to avoid them
- Monthly maintenance fee: $20.00
- How to waive it: maintain a daily balance at or above $2,500

Tip: Use balance alerts to help you remain above $2,500 throughout the statement cycle.

## Start earning interest
- Your balance earns an APY of 0.5%
- Interest begins accruing once your initial deposit is received and available

## Manage your monthly activity
- You get $175 free transactions per month (deposits, withdrawals, and transfers)
- After you use your included allotment, additional transactions may incur a fee

## Grow your network with referrals
Once you're established, refer other businesses to Cobalt Blue:
- You earn: $150 per successful referral
- They receive: $100 as a welcome bonus
- Maximum referrals: 10 per calendar year

## First-week checklist
- Make your initial deposit and confirm it has posted
- Set a balance alert at or slightly above $2,500
- Review your current month's usage against the $175 limit
- Verify interest is accruing at 0.5% on your available balance

## Where to find these details
- Account summary: shows current balance against the $2,500 waiver requirement
- Activity tracker: displays monthly usage versus the $175 limit
- Interest section: displays the current APY of 0.5% and accrued interest to date
=== doc_business_checking_accounts_navy_blue_001.md ===
# Navy Blue Business Checking: Complete Guide

## Key Account Costs and Limits

- Monthly maintenance fee: $0.00
- Annual Percentage Yield (APY) on balances: 0.5%
- Daily digital transfer limit: $25,000

### What you can expect day to day
- No minimum balance requirement, so you can use the account flexibly.
- Unlimited digital transfers within the daily limit of $25,000.
- Optional overdraft settings are available if you want coverage on eligible transactions.
- E-statements are included at no additional cost.

### Practical tips for managing your account
- Use e-statements to keep records organized and accessible.
- Plan larger transfers with the daily limit of $25,000 in mind; schedule across days if needed.
- Keep an eye on your balance to maximize earnings at 0.5%.

### Quick reference
| Item | Amount |
| --- | --- |
| Monthly maintenance fee | $0.00 |
| APY | 0.5% |
| Daily digital transfer limit | $25,000 |

## Business Referral Program

Grow your network and earn rewards by referring other businesses to Navy Blue:

| Referral Detail | Value |
| --- | --- |
| Your referral bonus | $100 |
| New business welcome bonus | $75 |
| Maximum referrals per year | 10 |
| Required qualifying deposit | $5,000 |
| Deposit must be made within | 90 days |
| Your account age requirement | 60 days |

To qualify: The referred business must open a Navy Blue account and deposit at least $5,000 within 90 days of account opening. Referrer eligibility requires having been a Rho-Bank checking customer for 60 days or more.
=== doc_business_checking_accounts_true_blue_001.md ===
# True Blue Business Checking: Complete Guide

## What you pay
- Monthly maintenance fee: $75.00

## Balance requirements
- Maintain at least $25,000 to keep the account in good standing and access premium features.

## Transactions included
- Transactions included per month: -1 (unlimited)

## Interest on balances
- Annual Percentage Yield (APY): 2.0%
- Interest accrues on your collected balance and is credited according to the account terms.

## Practical guidance
- Keep your end-of-day balance at or above $25,000 to avoid interruptions to premium capabilities.
- Use the account for high-volume activity without worrying about transaction limits, as the included transactions per month are -1.
=== doc_business_checking_accounts_lime_green_001.md ===
# Lime Green Business Checking: Complete Guide

## Key account values

| Item | Value |
| --- | --- |
| Monthly maintenance fee | $25.00 |
| Annual Percentage Yield (APY) | 1.5% |
| Minimum balance requirement | $5,000 |
| Daily transaction limit | $100,000 |

## Monthly maintenance fee
- You are charged $25.00 for premium account maintenance and dedicated service.
- The fee is drawn directly from the account. If your available balance is insufficient at assessment time, the charge posts when funds are available.

## Minimum balance requirement
- Maintain at least $5,000 to preserve account benefits and avoid balance-related charges that may apply when you are below the requirement.
- Practical tips:
  - Schedule large outgoing payments after confirming that the remaining balance will stay at or above $5,000.
  - Set balance alerts so you can move funds in advance if needed.

## Interest earnings
- Your account earns interest at an APY of 1.5%.
- Interest accrual is based on your balance; keeping funds in the account consistently helps maximize earnings.

## Daily transaction limit
- You can transact up to $100,000 per day across eligible outgoing activity.
- Plan ahead for high-payment days:
  - Confirm the remaining capacity before initiating additional payments.
  - If a single payment could exceed $100,000, split payments across days or queue them to post on different business days.

=== doc_business_checking_accounts_sky_blue_001.md ===
# Getting Started with Sky Blue: Startup Account Setup

## Verify Eligibility
- Confirm your company is within 4 years of formation.
- Check whether you meet the funding requirement: 0.

## Understand Key Cost Terms Before You Apply
- Your startup receives a free period of 6 months.
- After the free period, the monthly maintenance fee is $25.00.
- If your balance goes negative, the overdraft fee is $0.00.

## Prepare Your Application
- Gather company formation documents and ownership details.
- Be ready to provide basic business information and authorized signers.
- Plan initial account settings, such as user permissions and card controls, so you can activate quickly after approval.

## Submit and Configure
- Complete the online application and upload requested documents.
- After approval, log in to set user roles, spending controls, and alerts.
- Link external accounts to enable seamless transfers.

## Startup Referral Program
Help other startups get started with Sky Blue:
- Your bonus: $150 for each successful referral
- Their bonus: $250 welcome bonus
- Annual limit: 8 referrals per year

## First-Funding and Go-Live Checklist
- Make an initial deposit from a linked account.
- Confirm your free period of 6 months is visible in your billing section.
- Review your fee settings to ensure you understand the transition to $25.00 and the overdraft policy of $0.00.
=== doc_business_checking_accounts_sky_blue_002.md ===
# Sky Blue: Understanding Your Free Period

## What Your Free Period Covers
- Monthly maintenance fees are waived for 6 months.
- Your account features continue to function normally during this time.

## When It Begins and Ends
- The free period starts when your account is opened and runs for 6 months.
- You can track remaining time in your billing or plan details screen.

## What Happens After
- When the free period ends, the monthly maintenance fee becomes $25.00.
- No action is required to continue service; billing updates automatically.

## Managing Your Timeline
- Review your billing settings to confirm the free period end date.
- Set internal reminders to prepare for the transition to $25.00.

## Referral Program During Free Period
Even during your free period, you can participate in the referral program once eligible:
- Qualifying deposit: Referred startups must deposit at least $10,000 to trigger the bonus
- Time window: The deposit must be made within 90 days of account opening
- Your eligibility: Referrers must meet the 45-day tenure threshold based on their banking history with Rho-Bank.
=== doc_business_checking_accounts_world_blue_001.md ===
# Getting Started with World Blue International Checking

## Quick checklist before you begin
- Confirm you can maintain at least $10,000 across currencies to keep the account in good standing
- Be aware of the monthly maintenance fee of $50.00
- Decide which of the up to 35 supported currencies you plan to hold and transact in
- Verify that your intended counterparties and destinations are within the $140 supported regions for sending and receiving payments

## Initial setup steps
- Fund your account so you can meet ongoing balance needs
- Add the currency wallets you will use for operations and settlements
- Set a primary operating currency for reporting and reconciliation
- Configure payee profiles for your frequent counterparties within the supported regions
- Review user access and prepare to assign permissions based on roles and responsibilities

## Key thresholds and limits at a glance

| Item | Value |
|---|---|
| Monthly maintenance fee | $50.00 |
| Minimum balance requirement | $10,000 |
| Supported currencies | 35 |
| Supported regions for payments | $140 |

## First-week best practices
- Keep an operating buffer above $10,000 to avoid service interruptions
- Start with a small set of currency wallets and expand as your payment flows stabilize
- Validate a test payment to a known counterparty in one of the supported regions and confirm posting and reconciliation
- Set alerts and reports aligned to your treasury cadence to monitor balances and activity

## Referring other businesses to World Blue

As a Rho-Bank checking account holder, you can refer other businesses to World Blue. To be eligible to refer:
- You must have maintained checking account status with Rho-Bank for at least 90 days

For a referral to qualify:
- The referred business must open a World Blue account
- They must deposit at least $25,000 within 90 days of account opening

=== doc_business_checking_accounts_beige_001.md ===
# Implementing Beige: Enterprise Treasury Onboarding

## Onboarding timeline and responsibilities
- Confirm executive sponsorship and authorized signers before initiating onboarding.
- Prepare onboarding documentation, treasury policy, and internal approval matrix.
- Identify primary funding sources for initial deposits and recurring cash flows.

## Funding and balance setup
- Maintain at least $250,000 to keep your enterprise-level treasury account in good standing.
- The monthly maintenance fee is $200.00. This fee is waived when your average balance meets or exceeds $500,000.
- If you anticipate balance variability during early implementation, align funding schedules to preserve eligibility for the fee waiver threshold.

## Dedicated support structure
- You are assigned 2 dedicated account managers who coordinate onboarding milestones, entitlements, and operational readiness.
- Establish a recurring operating cadence with your account team for progress tracking and rapid issue resolution.

## Overdraft settings during go-live
- Overdrafts incur a fee of $0.00.
- If you plan to stagger incoming funds across multiple institutions, configure conservative payment windows and internal approvals to avoid negative balances.

## Operational readiness checklist
- Set signer roles and entitlements consistent with your treasury policy.
- Add internal approval rules for payments and administrative changes in line with your governance model.
- Validate statement delivery preferences, reporting formats, and reconciliation identifiers prior to first close.
- Confirm that funding sources are scheduled to achieve at least $250,000 and, if desired, the waiver threshold of $500,000.

## Reference values for onboarding
| Item | Value |
| --- | --- |
| Monthly maintenance fee | $200.00 |
| Minimum balance requirement | $250,000 |
| Fee waiver balance threshold | $500,000 |
| Dedicated account managers | 2 |
| Overdraft fee | $0.00 |

**assistant → shell:** {"command": "for f in doc_bank_accounts_bank_accounts__general__013.md doc_bank_accounts_bank_accounts__general__014.md; do echo \"=== $f ===\"; cat $f; echo; done; grep -il \"overdraft\" doc_business_checking_accounts_*.md"}

**tool:** === doc_bank_accounts_bank_accounts__general__013.md ===
# Internal: Business Checking Account Promotion - November 2025

PROMOTION NOTICE - ACTIVE FROM 11/01/2025 TO 11/30/2025: When recommending business checking accounts to customers, if multiple accounts meet all of the customer's stated requirements, agents should prioritize recommending accounts in the following order: (1) Sky Blue account should be recommended first, as it is currently under promotional priority; (2) Lime Green account should be recommended second, as it is also under promotional priority; (3) All other qualifying accounts may be recommended only if neither Sky Blue nor Lime Green meet the customer's requirements. This promotion is designed to increase adoption of our startup and premium business offerings. Always ensure the recommended account still meets ALL of the customer's stated requirements - do not recommend a promotional account if it does not satisfy the customer's needs.
=== doc_bank_accounts_bank_accounts__general__014.md ===
# Internal: Business Checking Account Promotion - October 2025

PROMOTION NOTICE - ACTIVE FROM 10/12/2025 TO 11/12/2025: When recommending business checking accounts to customers, if multiple accounts meet all of the customer's stated requirements, agents should prioritize recommending accounts in the following order: (1) Lime Green account should be recommended first, as it is currently under promotional priority; (2) Hunter Green account should be recommended second, as it is also under promotional priority; (3) All other qualifying accounts may be recommended only if neither Lime Green nor Hunter Green meet the customer's requirements. This promotion is designed to increase adoption of our premium and eco-conscious business offerings. Always ensure the recommended account still meets ALL of the customer's stated requirements - do not recommend a promotional account if it does not satisfy the customer's needs.
doc_business_checking_accounts_beige_001.md
doc_business_checking_accounts_beige_004.md
doc_business_checking_accounts_cobalt_blue_002.md
doc_business_checking_accounts_cobalt_blue_009.md
doc_business_checking_accounts_hunter_green_001.md
doc_business_checking_accounts_lime_green_002.md
doc_business_checking_accounts_navy_blue_001.md
doc_business_checking_accounts_navy_blue_002.md
doc_business_checking_accounts_sky_blue_001.md
doc_business_checking_accounts_true_blue_002.md
doc_business_checking_accounts_world_blue_002.md

**assistant → shell:** {"command": "for f in doc_business_checking_accounts_lime_green_002.md doc_business_checking_accounts_cobalt_blue_002.md doc_business_checking_accounts_true_blue_002.md doc_business_checking_accounts_world_blue_002.md; do echo \"=== $f ===\"; cat $f; echo; done"}

**tool:** === doc_business_checking_accounts_lime_green_002.md ===
# Lime Green Account: Fee Waivers and Balance Requirements

## Monthly maintenance fee and how to waive it
- The monthly maintenance fee is $25.00.
- The fee is waived when your balance meets or exceeds $15,000 as defined for this account.
- If your balance falls below $15,000 during the evaluation period, the $25.00 may apply.

## Minimum balance requirement
- Keep your end-of-day balance at or above $5,000 to maintain account benefits and avoid balance-related charges that can trigger when you are below the requirement.
- Practical guidance:
  - Before sending payments, ensure your projected end-of-day balance remains at or above $5,000.
  - Use internal transfers to top up if you anticipate a shortfall.

## Overdrafts
- The overdraft fee is $0.00.
- To avoid negative balances:
  - Track pending debits before initiating new transfers.
  - Hold off on non-urgent payments until deposits have cleared.

## Quick reference

| Topic | Amount / Threshold |
| --- | --- |
| Monthly maintenance fee | $25.00 |
| Fee waiver balance threshold | $15,000 |
| Minimum balance requirement | $5,000 |
| Overdraft fee | $0.00 |

## How to check your fee waiver status
- Review your current balance and recent activity before the fee assessment date to ensure you are at or above $15,000.
- If you expect large outgoing payments, pre-fund the account so your balance remains comfortably above $5,000.

=== doc_business_checking_accounts_cobalt_blue_002.md ===
# Cobalt Blue: No Overdraft Fees — How It Works

## What you are charged when your balance goes negative
- Overdraft fee: $0.00

## How transactions are handled
- If a transaction would take your available balance below zero, it may be declined or returned
- You will not be charged an overdraft fee of any kind: the fee is $0.00

## Practical tips to avoid declines
- Enable low-balance alerts so you can transfer funds before authorizing new payments
- Review pending card authorizations that may temporarily reduce your available balance
- Schedule outgoing payments for dates when incoming funds are expected to arrive first

## If a payment is returned or declined
- Check your available balance and pending items
- Resubmit the payment after funds are available
- Consider splitting a large outgoing payment into multiple payments scheduled around expected deposits (when feasible for your payee)

## Monitoring tools to stay in control
- Use real-time balance and transaction alerts
- Review daily activity to ensure sufficient funds are available before initiating new payments
=== doc_business_checking_accounts_true_blue_002.md ===
# True Blue Account: Fee Waivers and Balance Requirements

## Waiving your monthly maintenance fee
- You will be charged $75.00 each month unless you meet the waiver threshold.
- The monthly maintenance fee is waived when your balance is at least $50,000 during the statement period.

## Minimum balance requirement vs. fee waiver threshold
- Minimum balance requirement: $25,000 to maintain the account and access premium features.
- Fee waiver threshold: $50,000 to waive the $75.00 monthly maintenance fee.
- Falling below $25,000 may limit access to premium features until your balance is restored.

## Overdrafts and shortfalls
- If your balance goes negative, a per-occurrence overdraft fee of $25.00 may apply.

## How to avoid fees
- Maintain a balance of at least $50,000 to waive the monthly maintenance fee.
- Set internal processes to monitor balances against both $25,000 and $50,000.
- If a shortfall is anticipated, transfer funds promptly to avoid the $25.00 and preserve eligibility for the fee waiver.

## True Blue Referral Program

Refer other high-volume businesses to True Blue and earn significant rewards:

| Benefit | Amount |
| --- | --- |
| Referrer bonus | $350 per referral |
| New member bonus | $250 |
| Maximum referrals per year | 15 |
| Qualifying deposit | $50,000 |
| Deposit window | 120 days |
| Account age requirement | 90 days |

### Referral requirements
- The referred business must deposit $50,000 within 120 days of opening their True Blue account.
- Eligibility is based on how long you've been a Rho-Bank checking customer. The account age threshold is 90 days, calculated from when you opened your oldest checking account.
- Referral bonuses are paid after qualification is confirmed and are subject to the annual limit of 15.
=== doc_business_checking_accounts_world_blue_002.md ===
# World Blue: How to Avoid Monthly Fees

## Know your thresholds
- The monthly maintenance fee is $50.00
- You can avoid the fee by maintaining an eligible balance at or above $25,000
- Always keep your total balance at or above $10,000 to maintain account status

## Practical tactics to stay above the waiver threshold
- Maintain a prudent buffer over $25,000 to accommodate FX movements and timing of payments
- Consolidate excess funds into the account ahead of major disbursements so you remain above $25,000 throughout the cycle
- Schedule recurring funding to top up balances before expected outflows
- Monitor balances in real time and set internal alerts that nudge you before you drop below $25,000

## Avoid avoidable charges
- Prevent overdrafts; an overdraft triggers a fee of $30.00 when the account goes negative
- Reconcile pending items and expected settlements so you do not rely on funds that are not yet available toward $25,000

## Verify and follow up
- At the end of each statement period, confirm whether you met $25,000
- If you believe the fee was assessed in error, document your balance history and contact support from within your dashboard

**assistant → get_current_time:** {}

**tool:** The current time is 2025-11-14 03:40:00 EST.

**assistant → shell:** {"command": "cat doc_business_checking_accounts_sky_blue_009.md; echo ====; cat doc_business_checking_accounts_sky_blue_008.md"}

**tool:** # Troubleshooting: Sky Blue Eligibility Issues

## Quick Checks
- Verify your company age is within 4. Confirm using formation documents that match your application details.
- Ensure you meet the funding requirement shown as 0. If documentation was unclear, provide updated evidence.

## If Your Free Period Is Missing or Incorrect
- Confirm your enrollment met the criteria for a free period of 6 months.
- Review your onboarding confirmation to see whether the free period was applied at account opening.

## How to Resolve Documentation Mismatches
- Re-upload formation documents if dates or legal names differ from what you submitted.
- Provide capitalization records or investor confirmations if funding details were questioned.
- Add a brief explanation letter to clarify name changes, conversions, or reorganizations.

## When to Contact Support
- If your records confirm compliance with 4 and 0 but your status has not updated, share copies of your formation certificate and funding proof.
- Include a screenshot of your plan details showing the expected 6 so support can reconcile your account configuration.====
# After the Free Period: What Changes and What Stays

## What Changes
- The monthly maintenance fee becomes $25.00.
- Billing updates automatically; you will see the fee on your next statement after the transition.

## What Stays the Same
- Interest continues to accrue at 1.25% according to your account terms.
- Your existing users, cards, controls, and analytics remain active without interruption.

## About Eligibility Post-Onboarding
- Eligibility criteria tied to company age are evaluated when you open the account; the threshold is 4.
- Your account status after the free period does not require re-checking this onboarding criterion.

**assistant:** Thanks for the details — I can definitely point you to a single best fit rather than making you compare everything. To make sure I recommend the right one, I just need a couple of quick things from you:

1. You mentioned "no overdraft fees and a few other requirements" — what are the other requirements? (For example: a specific monthly fee ceiling, a minimum balance you're comfortable maintaining, interest earned on your balance, transaction volume, wire needs, etc.)
2. How long ago was Brightwave Digital formed? Some of our business accounts have eligibility tied to company age, so this helps me rule things in or out.

Once I have those, I'll give you one clear recommendation.

**user:** Other requirements, yeah:

- I travel to meet clients a lot and I use ATMs all over the place. I need at least **$15/month in ATM fee rebates** — those fees add up fast.
- I **absolutely cannot have overdraft fees**. Cash flow swings with client payments and I’ve been burned before. **Zero overdraft fees is non‑negotiable.**
- I’m a small agency, not a huge company. I **can’t keep $10,000+** sitting in there all the time. If there’s a minimum balance, it needs to be **under $10k**.
- I’d also like to earn **decent interest** on the balance — **at least 1% APY** if possible.

On Brightwave Digital’s age: I don’t have the exact formation date in front of me. It’s not brand new, but I’d need to check the paperwork for the exact month/year. Does it have to be, like, “must be 2+ years” or something?

**assistant → shell:** {"command": "for f in doc_business_checking_accounts_sky_blue_010.md doc_business_checking_accounts_lime_green_007.md doc_business_checking_accounts_hunter_green_010.md doc_business_checking_accounts_cobalt_blue_010.md doc_business_checking_accounts_navy_blue_009.md doc_business_checking_accounts_true_blue_011.md; do echo \"=== $f ===\"; cat $f; echo; done"}

**tool:** === doc_business_checking_accounts_sky_blue_010.md ===
# Sky Blue Account: ATM Fees and Out-of-Network Usage

## Domestic out-of-network ATM usage (U.S.)

- When you withdraw cash at a non‑Rho (out‑of‑network) ATM in the U.S., you are charged an out‑of‑network ATM fee of $1.50 per transaction.
- ATM operators may impose their own surcharges at the terminal. These are separate from Rho’s fees and are disclosed on the ATM screen before you complete the withdrawal.

### Ways to minimize domestic fees
- Use in‑network ATMs when possible to avoid the out‑of‑network fee of $1.50.
- If you anticipate multiple withdrawals, consider taking out cash less frequently in larger amounts to reduce the number of per‑transaction fees.

## International ATM withdrawals

- For cash withdrawals at foreign (international) ATMs, you are charged 2% of the withdrawal amount, with a minimum fee of $3.00 per withdrawal.
- Local ATM operators may assess additional surcharges. These are set by the operator and are separate from Rho’s fees.

### Practical tips for international use
- Because a minimum fee of $3.00 applies, fewer, larger withdrawals may reduce the effective fee rate on small cash needs.
- Review the ATM’s on‑screen disclosures before confirming a withdrawal to see any operator surcharges.

## Monthly ATM fee rebates for out-of-network usage

- You are eligible for up to 15 in ATM fee rebates per month for out‑of‑network ATM usage.
- The rebate cap is monthly. If total eligible fees for out‑of‑network usage exceed 15 in a given month, any amount above the cap is your responsibility.

## Common scenarios

- Two domestic out‑of‑network withdrawals in one month:
  - You are charged $1.50 per withdrawal. If the total eligible out‑of‑network fees for the month remain within 15, they are rebated up to that monthly cap.

- International withdrawal of a small amount:
  - The fee is 2% of the withdrawal, subject to the minimum of $3.00 if the calculated percentage is lower.

## If something looks off

- Keep your ATM receipt and check your statement. If a fee or rebate does not appear as expected, contact support and include the date, location, and amount of the withdrawal so we can review it.
=== doc_business_checking_accounts_lime_green_007.md ===
# Lime Green Account: ATM Access and Fee Rebates

## ATM Access

- You can use out-of-network ATMs for cash withdrawals. Each out-of-network ATM transaction is assessed a fee of $1.00.
- International (foreign) ATM withdrawals incur a fee equal to 1.5% of the withdrawal amount, with a minimum of $2.50 per withdrawal.

### Quick reference

| Item | Amount |
|---|---|
| Out-of-network ATM transaction fee (domestic) | $1.00 per transaction |
| Foreign ATM withdrawal fee | 1.5% of amount; minimum $2.50 |
| Monthly ATM fee rebate cap (out-of-network usage) | Up to $25 per month |

## ATM Fee Rebates

- Your account provides monthly rebates of eligible fees from out-of-network ATM usage, up to $25 per month.
- Rebates are applied until the monthly cap is reached; any eligible fees above the cap are your responsibility.
- You do not need to take additional action for eligible rebates—fees and corresponding credits will appear as separate line items on your account activity once processed.

### What counts toward the monthly cap

- Fees assessed for out-of-network ATM usage that are charged to your account are counted toward the $25 monthly rebate limit.

## How fees are calculated

- Domestic out-of-network usage: Each withdrawal is charged $1.00.
- Foreign ATM withdrawals: The fee is 1.5% of the withdrawal amount, with a minimum of $2.50. If the percentage-based amount is less than $2.50, the $2.50 minimum applies.

## Examples

- Domestic, multiple withdrawals:
  - If you make several out-of-network withdrawals in a month, each is charged $1.00. Eligible out-of-network ATM fees will be rebated up to a total of $25 for that month.

- Foreign withdrawal:
  - For a single international withdrawal, your fee is calculated at 1.5% of the withdrawn amount, subject to a $2.50 minimum. The applicable amount is charged at the time the transaction is processed.

## Tips to reduce fees

- Consolidate cash needs into fewer withdrawals to limit per-transaction charges of $1.00.
- Monitor your monthly total of out-of-network ATM fees to understand how close you are to the $25 rebate cap.
=== doc_business_checking_accounts_hunter_green_010.md ===
# Hunter Green Account: ATM Fees and Out-of-Network Usage

## Fee Summary

- Out-of-network (non‑Rho) ATM transactions (domestic): $2.00 per transaction
- International ATM withdrawals: 2.5% of the withdrawal amount, with a minimum of $4.00 per withdrawal
- Monthly ATM fee rebates for out-of-network usage: Up to $20 per month

Note: ATM owner/operator surcharges are set by the ATM and are separate from Rho fees.

## How Fees Apply

### Domestic out-of-network usage
- Each cash withdrawal at a non‑Rho ATM incurs a $2.00 fee.
- Any surcharge displayed by the ATM owner/operator will be charged in addition to the Rho fee.
- Eligible out-of-network ATM fees are automatically rebated until you reach the monthly cap of $20.

### International ATM withdrawals
- Each withdrawal outside the U.S. incurs a fee of 2.5% of the withdrawal amount, with a minimum charge of $4.00 per withdrawal.
- If the international withdrawal is made at a non‑Rho ATM, the $2.00 out-of-network fee also applies.
- ATM owner/operator surcharges at foreign ATMs may also be charged.
- Eligible out-of-network ATM fees are rebated up to the monthly maximum of $20.

## Calculating Your Total Cost

For each ATM withdrawal:
1. Add any ATM owner/operator surcharge shown at the terminal.
2. If the ATM is non‑Rho, add $2.00.
3. If the withdrawal is international, add 2.5% of the withdrawal amount (minimum $4.00).
4. Subtract any applicable rebates, up to a total of $20 per month.

## Examples

- Domestic, non‑Rho ATM, $100 withdrawal:
  - Rho out-of-network fee: $2.00
  - Plus any ATM owner surcharge
  - Rebated up to your remaining monthly $20 cap

- International ATM, $100 withdrawal:
  - International fee: 2.5% of $100 = $2.50, but minimum applies → $4.00
  - Out-of-network fee (if the ATM is non‑Rho): $2.00
  - Plus any ATM owner surcharge
  - Rebated up to your remaining monthly $20 cap

- International ATM, $400 withdrawal:
  - International fee: 2.5% of $400 = $10.00
  - Out-of-network fee (if the ATM is non‑Rho): $2.00
  - Plus any ATM owner surcharge
  - Rebated up to your remaining monthly $20 cap

## Posting and Visibility

- Fees and any related rebates appear as separate line items on your account activity and statements.
- You can track how much of the $20 monthly rebate you have used by reviewing your statement or recent transactions.
=== doc_business_checking_accounts_cobalt_blue_010.md ===
# Cobalt Blue Account: ATM Fees and Out-of-Network Usage

## Summary of ATM Fees

- Domestic out-of-network ATM withdrawal: $2.00 per transaction
- International ATM withdrawal: 2.5% of the withdrawal amount (minimum $4.00 per transaction)
- Monthly ATM fee rebates: Up to $15 in out-of-network ATM fee rebates per month

Note: ATM owners/operators may charge additional surcharges that you will see at the terminal before you complete the transaction.

## Domestic Out-of-Network ATM Usage

When you use an ATM outside the in-network system within the United States:
- You are charged $2.00 per withdrawal.
- Some ATM owners add their own surcharge. You can choose to cancel the transaction if you do not want to accept an operator surcharge.
- You can receive up to $15 in ATM fee rebates per month for out-of-network usage. These rebates can offset qualifying out-of-network ATM fees you incur, up to the monthly cap.

Tips to minimize domestic fees:
- Combine cash needs into fewer withdrawals to reduce per-transaction fees.
- Check the ATM’s on-screen disclosure for any operator surcharges before accepting.

## International ATM Withdrawals

For withdrawals at foreign (international) ATMs:
- You are charged 2.5% of the withdrawal amount, with a minimum of $4.00 per transaction.
- Local ATM owners may add their own surcharges at the terminal.
- Exchange rates and any third-party conversion choices at the ATM can affect your total cost.

Examples:
- Withdraw $200: 2.5% of $200 = $5.00; the fee is $5.00 (exceeds the $4.00 minimum).
- Withdraw $100: 2.5% of $100 = $2.50; the $4.00 minimum applies, so the fee is $4.00.

## Monthly ATM Fee Rebates

- You can receive up to $15 in ATM fee rebates per month for out-of-network ATM usage.
- The rebate limit applies on a monthly basis; once you reach $15, additional out-of-network ATM fees will not be rebated that month.

Example:
- If you make two domestic out-of-network withdrawals, your bank-assessed fees would be 2 × $2.00. Up to $15 of qualifying out-of-network ATM fees may be rebated that month.
=== doc_business_checking_accounts_navy_blue_009.md ===
# Navy Blue Account: ATM Fees and Out-of-Network Usage

## Where ATM fees apply for Navy Blue
- Out-of-network usage: Any transaction at a non‑Rho ATM is assessed a per‑transaction fee of $2.50.
- International usage: ATM cash withdrawals at foreign (international) ATMs are assessed a fee of 3% of the withdrawal amount, with a minimum of $5.00.
- Operator surcharges: The ATM owner may charge an additional surcharge at the terminal. This surcharge is separate from Rho fees.

## Fee schedule
| Scenario | Fee | Notes |
|---|---|---|
| Domestic out‑of‑network ATM transaction | $2.50 per transaction | Applies at non‑Rho ATMs in the U.S. |
| International ATM cash withdrawal | 3% of withdrawal amount (min. $5.00) | Applies at foreign ATMs outside the U.S. |
| Monthly rebates for out‑of‑network usage | Up to $10 per calendar month | Rebates are capped at the monthly limit. |

## How international ATM withdrawal fees are calculated
- Calculation: The fee equals the greater of:
  - 3% of the cash withdrawal amount, or
  - $5.00 (minimum)
- Threshold for the minimum to apply: The minimum applies when the withdrawal amount is less than or equal to $5.00 divided by 3%.

## When fees can combine
- If you use an international ATM that is also out‑of‑network, both the $2.50 out‑of‑network fee and the international withdrawal fee of 3% (minimum $5.00) can apply.
- ATM operator surcharges, if any, are in addition to the above.

## Monthly rebate details
- Rho rebates up to $10 in out‑of‑network ATM fees per calendar month.
- The monthly cap is the maximum total rebated; fees above this amount are your responsibility after the cap is reached.

## Tips to minimize ATM costs
- Use Rho ATMs whenever possible to avoid the out‑of‑network fee of $2.50.
- For international cash needs, consider making larger, less frequent withdrawals to reduce the impact of the $5.00 minimum relative to the withdrawal amount.
- Always review any on‑screen ATM operator surcharge disclosures before completing a transaction.
=== doc_business_checking_accounts_true_blue_011.md ===
# True Blue Account: ATM Fees and Out-of-Network Usage

## Fees at a glance

| Transaction type | Where it applies | Fee |
|---|---|---|
| Out-of-network ATM withdrawal | Domestic (U.S.) non-Rho ATMs | $1.00 per transaction |
| Foreign ATM cash withdrawal | International ATMs | 2% of the withdrawal amount; minimum $3.00 per withdrawal |
| Monthly ATM fee rebates (cap) | Out-of-network ATM usage | Up to $30 per month |

Note: ATM owners may assess their own surcharges on out-of-network transactions. Those surcharges are separate from Rho’s fees.

## Domestic out-of-network ATM usage (U.S.)

- When you use a non-Rho ATM domestically, you are charged $1.00 per transaction.
- The ATM owner may also charge a separate surcharge at the terminal; you can accept or cancel the transaction after the surcharge is disclosed.

Example:
- If you complete 5 out-of-network withdrawals in a month, Rho’s fees would total 5 × $1.00. Any ATM owner surcharges would be in addition to this amount.

## International ATM withdrawals

- For cash withdrawals at foreign (international) ATMs, the fee is 2% of the withdrawal amount, with a minimum of $3.00 per withdrawal.
- The percentage is applied to each withdrawal; if the calculated amount is below $3.00, the minimum applies.

Examples:
- $100 withdrawal: 2% = $2.00, which is below the $3.00 minimum, so the fee is $3.00.
- $200 withdrawal: 2% = $4.00, so the fee is $4.00.

## Monthly ATM fee rebates

- You can receive up to $30 in ATM fee rebates per month for out-of-network ATM usage.
- The cap is applied on a monthly basis; once total eligible ATM fees reach $30 for the month, no additional ATM fee rebates are provided for that month.

Examples:
- If your monthly out-of-network ATM fees total $12, up to $12 may be rebated (within the $30 cap).
- If your monthly out-of-network ATM fees total $36, the maximum rebate is $30.

## Tips to manage costs

- Prefer Rho ATMs to avoid the $1.00 out-of-network fee.
- Abroad, consider fewer, larger withdrawals to reduce the impact of the $3.00 per-withdrawal minimum.
- Track your monthly ATM fees so you know when you are approaching the $30 cap.

**assistant → shell:** {"command": "grep -i \"minimum\\|balance requirement\\|APY\" doc_business_checking_accounts_sky_blue_*.md doc_business_checking_accounts_lime_green_005.md | head -30"}

**tool:** doc_business_checking_accounts_sky_blue_007.md:- Your account earns an APY of 1.25%.
doc_business_checking_accounts_sky_blue_010.md:- For cash withdrawals at foreign (international) ATMs, you are charged 2% of the withdrawal amount, with a minimum fee of $3.00 per withdrawal.
doc_business_checking_accounts_sky_blue_010.md:- Because a minimum fee of $3.00 applies, fewer, larger withdrawals may reduce the effective fee rate on small cash needs.
doc_business_checking_accounts_sky_blue_010.md:  - The fee is 2% of the withdrawal, subject to the minimum of $3.00 if the calculated percentage is lower.
doc_business_checking_accounts_lime_green_005.md:# Lime Green Account: Interest and APY Explained
doc_business_checking_accounts_lime_green_005.md:## Your APY
doc_business_checking_accounts_lime_green_005.md:- Monitor your interest accrual within account activity and statements. Interest amounts reflect the APY of 1.5% with daily compounding.

**assistant:** Good news — your requirements narrow this down cleanly, and the only thing left is the company-age question. To answer that directly: it's not a "must be 2+ years" rule — it's the opposite. The account that best fits you is for businesses **within 4 years of formation**.

**My recommendation: Sky Blue Business Checking**

Here's how it stacks up against each of your requirements:

- **Zero overdraft fees** — Overdraft fee is $0.00. ✅
- **ATM rebates** — Up to $15/month in out-of-network ATM fee rebates (domestic out-of-network fee is $1.50/transaction, so that cap covers roughly 10 withdrawals a month). ✅
- **No big balance requirement** — Funding requirement is $0; there's no minimum balance you need to keep parked. ✅
- **Interest** — 1.25% APY on your balance. ✅

Extra perks you'd get since you run a digital business:
- **6-month free period** with no monthly maintenance fee; after that it's $25.00/month.
- **$2,750 in AWS credits** and a **12% Stripe payment-processing discount** via the perks portal.
- Burn-rate/runway analytics, founder cards, and spending controls.
- Referral program: $150 for you per successful referral, $250 for the business you refer.

**The one thing to confirm:** Brightwave Digital must have been formed within the last 4 years. If you check your paperwork and it turns out the agency is older than that, let me know — there's one other account (Lime Green) that also hits every one of your requirements (zero overdraft fees, $25/month ATM rebates, $5,000 minimum balance, 1.5% APY), though it carries a $25 monthly fee unless you keep $15,000 in it.

If you'd like to move forward with opening a Sky Blue account, I can help with that — I'd just need to verify your identity first. Would you like to proceed, or check the formation date first?

**user:** Okay — I’m good with your recommendation. Let’s move forward with **Sky Blue Business Checking**.

And yeah, I’m pretty sure Brightwave Digital is within that **4-year** window, I just don’t have the exact formation date in front of me right this second.

Here’s what you need for identity verification:

- **Name:** Yuki Nakamura  
- **Phone:** 503-555-0842  
- **Email:** yuki.nakamura@simba.com  
- **DOB:** 05/12/1991  
- **Address:** 3421 Sakura Avenue, Portland, OR 97205  

I also already have a personal **Green Account** with you (balance is around **$2,850**), if that matters for linking/funding.

**assistant → get_user_information_by_name:** {"customer_name": "Yuki Nakamura"}

**tool:** Found 1 record(s) in 'users':

1. Record ID: 224959b99e
   name: Yuki Nakamura
   user_id: 224959b99e
   address: 3421 Sakura Avenue, Portland, OR 97205
   email: yuki.nakamura@simba.com
   phone_number: 503-555-0842
   date_of_birth: 05/12/1991

**assistant → log_verification:** {"name": "Yuki Nakamura", "user_id": "224959b99e", "address": "3421 Sakura Avenue, Portland, OR 97205", "email": "yuki.nakamura@simba.com", "phone_number": "503-555-0842", "date_of_birth": "05/12/1991", "time_verified": "2025-11-14 03:40:00 EST"}

**tool:** Verification logged successfully.
  - User: Yuki Nakamura (ID: 224959b99e)
  - Verified at: 2025-11-14 03:40:00 EST

**assistant → shell:** {"command": "cat doc_bank_accounts_bank_accounts__general__003.md; echo ====; grep -l \"get_bank_accounts\\|bank accounts for a user\\|retrieve.*bank account\" doc_bank_accounts_*.md"}

**tool:** # Internal: Opening Business Checking Accounts

## Eligibility Requirements

To open a business checking account, ensure all of the following are true:
1. The customer is verified.
2. The customer has at least one existing personal checking account with status OPEN.
3. The customer does not exceed 6 business checking accounts.
4. The customer has no accounts with status CLOSED.
5. The customer's existing checking account has a balance of at least $500.

## Opening Procedure

1. Verify customer identity.
2. Check eligibility requirements listed above.
3. Confirm the customer's desired account_class selection.
   - Business checking account_class options include Navy Blue, Cobalt Blue, True Blue, etc.
4. Use open_bank_account_4821 to open the account.====
doc_bank_accounts_bank_accounts__general__009.md
doc_bank_accounts_bank_accounts__general__018.md

**assistant → shell:** {"command": "cat doc_bank_accounts_bank_accounts__general__009.md; echo ====; head -30 doc_bank_accounts_bank_accounts__general__018.md"}

**tool:** # Internal: Retrieving Customer Account Information

## Overview

Use get_all_user_accounts_by_user_id_3847 to retrieve the bank accounts (checkings, savings) for a customer.

## When to Use

This tool is essential for:
1. Checking eligibility requirements for opening new accounts (existing account status, balances, account tenure, number of accounts)
2. Verifying pre-closure requirements when closing accounts
3. Looking up account details for customer service inquiries

## Parameters

The tool requires the customer's user_id and returns all account information including:
- account_id
- account_type
- account_class
- status
- balance
- date_opened====
# Internal: Retrieving Bank Account Transaction History

## Overview

Agents can retrieve the transaction history for a customer's bank account (checking or savings) using the get_bank_account_transactions_9173 tool. This is useful when reviewing account activity, verifying fees, checking for applied rebates, or investigating customer inquiries.

## Agent Tool Usage

- Tool: get_bank_account_transactions_9173(account_id)
  - account_id: The bank account ID to retrieve transactions for
  - Returns: A list of all transactions for the account. Transactions are returned in reverse chronological order (most recent first).

## Transaction Fields

Each transaction record contains the following fields:

- **transaction_id**: Unique identifier for the transaction
- **account_id**: The bank account ID this transaction belongs to
- **date**: Date of the transaction (MM/DD/YYYY format)
- **description**: Description of the transaction (e.g., 'ATM WITHDRAWAL - CHASE BANK #2847 CHICAGO IL')
- **amount**: Transaction amount in USD. Positive values are credits (money in), negative values are debits (money out)
- **type**: Transaction type, one of:
  - direct_deposit
  - debit_card_purchase
  - atm_withdrawal
  - atm_balance_inquiry
  - atm_fee
  - ach_transfer_in
  - ach_transfer_out
  - wire_transfer_in

**assistant → unlock_discoverable_agent_tool:** {"agent_tool_name": "get_all_user_accounts_by_user_id_3847"}

**tool:** Tool unlocked: get_all_user_accounts_by_user_id_3847
Description: Retrieve all accounts (checking, savings, credit cards) for a customer.

Tool: get_all_user_accounts_by_user_id_3847
Description: Retrieve all accounts (checking, savings, credit cards) for a customer.
Parameters:
  - user_id: string (required) - The customer's unique identifier in the system

You can now use this tool by calling `call_discoverable_agent_tool` with agent_tool_name='get_all_user_accounts_by_user_id_3847' and the required arguments.

**assistant → call_discoverable_agent_tool:** {"agent_tool_name": "get_all_user_accounts_by_user_id_3847", "arguments": "{\"user_id\": \"224959b99e\"}"}

**tool:** User accounts retrieved successfully.

Executed: get_all_user_accounts_by_user_id_3847
Accounts for user 224959b99e:

Bank Accounts:
Found 1 record(s) in 'accounts':

1. Record ID: 05
   account_id: 05
   user_id: 224959b99e
   class: checking
   level: Green Account
   date_opened: 11/10/2022
   status: OPEN
   current_holdings: 2850.00


Credit Card Accounts:
Found 2 record(s) in 'credit_card_accounts':

1. Record ID: cc_224959b99e_plat
   account_id: cc_224959b99e_plat
   user_id: 224959b99e
   card_type: Platinum Rewards Card
   date_of_account_open: 11/10/2022
   current_balance: $75.00
   reward_points: 0 points

2. Record ID: cc_224959b99e_bsilver
   account_id: cc_224959b99e_bsilver
   user_id: 224959b99e
   card_type: Business Silver Rewards Card
   date_of_account_open: 03/15/2023
   current_balance: $125.00
   reward_points: 4500 points

**assistant:** Thank you, Yuki — you're verified, and I've checked your eligibility to open a business checking account. Everything checks out:

- Your personal Green Account is open with a $2,850 balance (above the $500 minimum required)
- You have no closed accounts and no existing business checking accounts

Before I open the Sky Blue account, one last thing: Sky Blue eligibility is tied to the business being **within 4 years of formation**, and that's evaluated at account opening. You mentioned you're "pretty sure" Brightwave Digital qualifies. Since this is a firm eligibility criterion, can you confirm the agency was formed **on or after November 2021**? If you'd rather double-check your paperwork first, that's completely fine — I can open the account as soon as you confirm.

Once you confirm, I'll open the Sky Blue Business Checking account for you right away.

**user:** Yep — I can confirm Brightwave Digital was formed **on or after November 2021**. Go ahead and open the **Sky Blue Business Checking** account.

**assistant → unlock_discoverable_agent_tool:** {"agent_tool_name": "open_bank_account_4821"}

**tool:** Tool unlocked: open_bank_account_4821
Description: Open a new bank account for a customer.

Tool: open_bank_account_4821
Description: Open a new bank account for a customer.
Parameters:
  - user_id: string (required) - The customer's unique identifier in the system
  - account_type: string (required) - Type of account to open. Must be one of: 'checking' (personal checking), 'savings' (personal savings), 'business_checking', 'business_savings'
  - account_class: string (required) - The full official account class name

You can now use this tool by calling `call_discoverable_agent_tool` with agent_tool_name='open_bank_account_4821' and the required arguments.

**assistant → call_discoverable_agent_tool:** {"agent_tool_name": "open_bank_account_4821", "arguments": "{\"user_id\": \"224959b99e\", \"account_type\": \"business_checking\", \"account_class\": \"Sky Blue\"}"}

**tool:** Bank account opened successfully!
  - Account ID: 3befd297b57bc9c1
  - User ID: 224959b99e
  - Account Type: business_checking
  - Account Class: Sky Blue
  - Status: OPEN
  - Initial Balance: $0.00
  - Date Opened: 11/14/2025

**assistant:** Your new account is open, Yuki. Here are the details:

**Sky Blue Business Checking**
- **Account ID:** 3befd297b57bc9c1
- **Status:** OPEN
- **Date opened:** 11/14/2025
- **Current balance:** $0.00

**What you're getting:**
- $0.00 overdraft fees
- Up to $15/month in out-of-network ATM fee rebates
- 1.25% APY on your balance
- No minimum balance requirement
- 6-month free period on the monthly maintenance fee (it becomes $25.00/month after that — you can track the end date in your billing/plan details screen)
- $2,750 in AWS credits and a 12% Stripe processing discount, redeemable through your perks portal

**Suggested next steps:**
- Make an initial deposit from your Green Account or an external account to start earning interest
- Log in to set user roles, spending controls, and alerts
- Enroll in e-statements and set a low-balance alert, since transactions that would take you negative may be declined rather than overdrafted

Is there anything else I can help you with — for example, transferring an opening deposit from your Green Account into the new account?

**user:** Perfect, thanks for making that easy. I think I’m all set for now! ###STOP###

