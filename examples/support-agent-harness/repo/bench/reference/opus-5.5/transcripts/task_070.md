# task_070 · trial 0 · not resolved · ended: user_stop

**assistant:** Hi! How can I help you today?

**user:** Yeah, hi — so here’s the problem. I run a small digital marketing agency and I’m looking to open a business checking account that’s the easiest to use and gives me the most perks. I also want stuff like no overdraft fees, plus a few other requirements, and I’m hoping you can tell me which account is the right fit.

I really don’t want to compare a bunch of options — just point me to the one that works best for my situation.

→ assistant calls `KB_search_bm25` with `{"query": "business checking account overdraft fees"}`

**tool:** 1. Why does my account have a lower balance than I expected or have a negative balance?
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

2. Internal: Ordering a Debit Card for a Bank Account
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

3. Blue Account features and fees
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

4. Getting Started with Hunter Green Business Checking
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

5. Navy Blue Account: Overdraft Protection and Fees
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

6. Green Account (checking) at a glance
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

7. Dark Green Account specifications and requirements
   ID: doc_checking_accounts_dark_green_account_002
   Score: 9.7021
   Content: ## Eligibility requirements
- The primary account holder must be at least 17.
- You can maintain the account through 26.

## Fee schedule
The following fees apply to this account.

| Fee | Amount |
| --- | --- |
| Monthly maintenance | $10.00 |
| Overdraft protection transfer | 0 |
| Foreign ATM withdrawal | 2.5% of amount (max $6.00) |
| Returned deposit | $10.00 |
| Incoming domestic wire | $7.50 |

## Foreign ATM withdrawals
- When you withdraw cash from an ATM abroad, a fee of 2.5% of the withdrawal amount is charged.
- The fee is capped at a maximum of $6.00 per transaction, protecting you from excessive fees on larger withdrawals.
- Third-party ATM operator fees may still apply.

## Transaction limits
- Debit card daily purchase limit: $1,500
- Daily ATM withdrawal limit: $300

## Overdraft protection transfers
- When enabled and available, transfers from a linked funding source to cover an overdraft incur a transfer fee of 0.

8. Light Green Account specifications and requirements
   ID: doc_checking_accounts_light_green_account_002
   Score: 9.6021
   Content: ## Eligibility requirements
- You must be the primary account holder and be between 13 and 24 years old to open the account.
- You must remain within the 13–24 age range to maintain the account.

### Age verification and ongoing eligibility
- Your eligibility is determined using your date of birth at account opening and on an ongoing basis.
- Once you are no longer within the 13–24 range, you are no longer eligible to maintain the Light Green Account.

## Key specifications
- Overdraft fees: None (no overdraft fee is charged on this account)

## Rates
- Interest earned: 0.05% APY on the account balance

## Fees and limits
| Item | Amount/Limit |
|---|---|
| Monthly maintenance fee | $0.00 |
| Returned deposit fee (per returned item) | $12.50 |
| Incoming domestic wire transfer fee (per incoming wire) | $10.00 |
| Debit card daily purchase limit | $250 per day |

## Linked savings APY boosts
- If you link a Platinum Savings Account, you receive a 0.65% APY boost on that savings account.
- If you link a Diamond Elite Savings Account, you receive a 0.2% APY boost on that savings account.

## Referral program requirements
To participate in the Light Green Account referral program:
- The person you refer must deposit at least $100 within 90 days of opening their account.
- Referrer eligibility is based on account tenure: you must have opened your first Rho-Bank checking account at least 14 days ago.

9. Internal: Opening Business Savings Accounts
   ID: doc_bank_accounts_bank_accounts_(general)_004
   Score: 9.5070
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

10. Why am I being charged an overdraft fee on my Blue Account?
   ID: doc_checking_accounts_blue_account_008
   Score: 9.4349
   Content: ## Check the fee you see
- The overdraft fee on this account is $0.00. If a charge appears labeled as an overdraft fee, review the transaction details to confirm the fee type.

## Daily limits, if a fee ever posts in error
- Even if a non-zero overdraft fee were posted in error, the number of overdraft fees that can be charged in a single day is capped at 3.

## What to do next
- Open the transaction in your activity view and review the description to verify the fee type.
- Check your monthly statement for the same entry to confirm how it is categorized.
- If the fee appears misclassified as an overdraft fee, contact us so we can investigate and correct it.


[Timing: retrieval=3ms, reranking=0ms, total=3ms]

→ assistant calls `shell` with `{"command": "ls | grep business_checking; grep -i \"business\" INDEX.md | head -50"}`

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
- **doc_bank_accounts_bank_accounts__general__003.md** - Internal: Opening Business Checking Accounts
- **doc_bank_accounts_bank_accounts__general__004.md** - Internal: Opening Business Savings Accounts
- **doc_bank_accounts_bank_accounts__general__007.md** - Internal: Closing Business Checking Accounts
- **doc_bank_accounts_bank_accounts__general__008.md** - Internal: Closing Business Savings Accounts
- **doc_bank_accounts_bank_accounts__general__013.md** - Internal: Business Checking Account Promotion - November 2025
- **doc_bank_accounts_bank_accounts__general__014.md** - Internal: Business Checking Account Promotion - October 2025
- **doc_bank_accounts_bank_accounts__general__015.md** - Internal: Business Savings Account Promotion - November 2025
- **doc_bank_accounts_bank_accounts__general__016.md** - Internal: Business Savings Account Promotion - October 2025
- **doc_business_checking_accounts_beige_001.md** - Implementing Beige: Enterprise Treasury Onboarding
- **doc_business_checking_accounts_beige_002.md** - Connecting Your ERP for Beige Account: API Integration Guide
- **doc_business_checking_accounts_beige_003.md** - Beige: Balance Aggregation and Sub-Account Management
- **doc_business_checking_accounts_beige_004.md** - Beige: Configuring Auto-Fund Sweeps
- **doc_business_checking_accounts_beige_005.md** - Security: Dual Authorization and Transaction Controls
- **doc_business_checking_accounts_beige_006.md** - Beige: Your Dedicated Account Management Team
- **doc_business_checking_accounts_beige_007.md** - Earning Interest on Enterprise Balances
- **doc_business_checking_accounts_beige_008.md** - API Rate Limits and Technical Specifications
- **doc_business_checking_accounts_beige_009.md** - Wire Transfers and Payment Processing on Beige
- **doc_business_checking_accounts_beige_010.md** - Troubleshooting: Beige Fee and Balance Issues
- **doc_business_checking_accounts_beige_011.md** - Beige Account: Enterprise ATM Benefits
- **doc_business_checking_accounts_beige_012.md** - Beige Account Referral Program
- **doc_business_checking_accounts_cobalt_blue_001.md** - Getting Started with Cobalt Blue Business Checking
- **doc_business_checking_accounts_cobalt_blue_002.md** - Cobalt Blue: No Overdraft Fees — How It Works
- **doc_business_checking_accounts_cobalt_blue_003.md** - Managing Transactions and Staying Within Limits
- **doc_business_checking_accounts_cobalt_blue_004.md** - Earning Rewards with Your Cobalt Blue Debit Card
- **doc_business_checking_accounts_cobalt_blue_005.md** - Cobalt Blue: Payments and Transfers Explained
- **doc_business_checking_accounts_cobalt_blue_006.md** - Mobile Deposits and ATM Access on Cobalt Blue
- **doc_business_checking_accounts_cobalt_blue_007.md** - Using Bill Pay and Invoice Tools
- **doc_business_checking_accounts_cobalt_blue_008.md** - How Interest Accrues on Your Cobalt Blue Balance
- **doc_business_checking_accounts_cobalt_blue_009.md** - Troubleshooting: Unexpected Charges on Cobalt Blue
- **doc_business_checking_accounts_cobalt_blue_010.md** - Cobalt Blue Account: ATM Fees and Out-of-Network Usage
- **doc_business_checking_accounts_hunter_green_001.md** - Getting Started with Hunter Green Business Checking
- **doc_business_checking_accounts_hunter_green_002.md** - Hunter Green: Your Carbon Offset Contribution
- **doc_business_checking_accounts_hunter_green_003.md** - Earning Cashback from Eco-Friendly Vendors
- **doc_business_checking_accounts_hunter_green_004.md** - Reading Your Sustainability Impact Report
- **doc_business_checking_accounts_hunter_green_005.md** - Spending Alerts and Budget Controls on Hunter Green
- **doc_business_checking_accounts_hunter_green_006.md** - Hunter Green: Interest and APY Explained
- **doc_business_checking_accounts_hunter_green_007.md** - Mobile Deposits and Wire Transfers on Hunter Green
- **doc_business_checking_accounts_hunter_green_008.md** - Paperless Banking: Why Hunter Green Requires E-Statements
- **doc_business_checking_accounts_hunter_green_009.md** - Troubleshooting: Hunter Green Fees and Charges
- **doc_business_checking_accounts_hunter_green_010.md** - Hunter Green Account: ATM Fees and Out-of-Network Usage
- **doc_business_checking_accounts_lime_green_001.md** - Lime Green Business Checking: Complete Guide
- **doc_business_checking_accounts_lime_green_002.md** - Lime Green Account: Fee Waivers and Balance Requirements
- **doc_business_checking_accounts_lime_green_003.md** - Lime Green Account: Your Dedicated Account Manager
- **doc_business_checking_accounts_lime_green_004.md** - Lime Green Account: Wire Transfer Pricing
- **doc_business_checking_accounts_lime_green_005.md** - Lime Green Account: Interest and APY Explained
- **doc_business_checking_accounts_lime_green_006.md** - Lime Green Account: Mobile Deposits and Daily Limits
- **doc_business_checking_accounts_lime_green_007.md** - Lime Green Account: ATM Access and Fee Rebates
- **doc_business_checking_accounts_lime_green_008.md** - Why am I being charged a fee on my Lime Green Account?
- **doc_business_checking_accounts_navy_blue_001.md** - Navy Blue Business Checking: Complete Guide
- **doc_business_checking_accounts_navy_blue_002.md** - Navy Blue Account: Overdraft Protection and Fees


→ assistant calls `shell` with `{"command": "grep -ril \"recommend\" . | head; for f in doc_business_checking_accounts_*_001.md; do echo \"== $f\"; head -40 $f; done"}`

**tool:** ./doc_checking_accounts_purple_account_007.md
./doc_bank_accounts_bank_accounts__general__036.md
./doc_bank_accounts_bank_accounts__general__047.md
./doc_business_checking_accounts_navy_blue_007.md
./doc_credit_cards_credit_card_replacements_002.md
./doc_bank_accounts_bank_accounts__general__013.md
./doc_checking_accounts_green_account__checking__010.md
./doc_checking_accounts_checking_accounts__general__003.md
./doc_everyone_pay_scheduled_payments_004.md
./doc_business_savings_accounts_gold_plus_saver_001.md
== doc_business_checking_accounts_beige_001.md
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
== doc_business_checking_accounts_cobalt_blue_001.md
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
- Interest section: displays the current APY of 0.5% and accrued interest to date== doc_business_checking_accounts_hunter_green_001.md
# Getting Started with Hunter Green Business Checking

## Quick start checklist
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

Share the Hunter Green commitment to sustainability while growing your professional network.== doc_business_checking_accounts_lime_green_001.md
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
== doc_business_checking_accounts_navy_blue_001.md
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

To qualify: The referred business must open a Navy Blue account and deposit at least $5,000 within 90 days of account opening. Referrer eligibility requires having been a Rho-Bank checking customer for 60 days or more.== doc_business_checking_accounts_sky_blue_001.md
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
- Review your fee settings to ensure you understand the transition to $25.00 and the overdraft policy of $0.00.== doc_business_checking_accounts_true_blue_001.md
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
- Use the account for high-volume activity without worrying about transaction limits, as the included transactions per month are -1.== doc_business_checking_accounts_world_blue_001.md
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


**assistant:** I can help you find the best fit. Good news first: several of our business checking accounts charge **$0 overdraft fees**, so that requirement alone doesn't narrow it down. The right choice depends on your other needs.

So I can point you to one account instead of a list, could you tell me:

1. **Your "other requirements."** What else matters to you? For example: no monthly fee, earning interest, debit card rewards, international payments or multiple currencies, or a dedicated account manager.
2. **Typical balance.** About how much do you expect to keep in the account? Some accounts waive their monthly fee or require a minimum balance at certain levels.
3. **Business age.** When was your agency formed? Some accounts are only for newer businesses.
4. **Transaction volume.** Roughly how many deposits, withdrawals and transfers do you make each month? And do you send large digital transfers, such as over $25,000 in a single day?

Once I have these, I'll recommend the single account that suits your agency best.

**user:** Other requirements, yeah — a few key ones:

1) **ATM access:** I travel to meet clients pretty regularly and use ATMs all over. I need **at least $15/month in ATM fee rebates**.

2) **Overdraft protection:** I **absolutely cannot have overdraft fees**. Zero overdraft fees is non‑negotiable.

3) **Typical balance / minimums:** I’m not a huge company. I **can’t commit to keeping $10,000+** in there at all times. If there’s a minimum balance requirement, it needs to be **under ten grand**.

4) **Interest:** I’d like to earn **at least 1% APY** on whatever I keep in there.

For your other questions:
- **Typical balance:** it’ll fluctuate, but think **a few thousand**, not $10k+.
- **Business age:** I’ve been running Brightwave for a while (it’s not brand-new), but I don’t have the exact formation date in front of me.
- **Transaction volume:** pretty normal/low volume for a small agency — I can pull exact counts if you truly need them, but I’m not doing huge daily wires or anything like that.

→ assistant calls `shell` with `{"command": "cat doc_business_checking_accounts_lime_green_007.md doc_business_checking_accounts_hunter_green_010.md; grep -il \"rebate\" doc_business_checking_accounts_*; grep -i \"overdraft\" doc_business_checking_accounts_lime_green_*"}`

**tool:** # Lime Green Account: ATM Access and Fee Rebates

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
- Monitor your monthly total of out-of-network ATM fees to understand how close you are to the $25 rebate cap.# Hunter Green Account: ATM Fees and Out-of-Network Usage

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
- You can track how much of the $20 monthly rebate you have used by reviewing your statement or recent transactions.doc_business_checking_accounts_beige_009.md
doc_business_checking_accounts_beige_011.md
doc_business_checking_accounts_cobalt_blue_006.md
doc_business_checking_accounts_cobalt_blue_010.md
doc_business_checking_accounts_hunter_green_007.md
doc_business_checking_accounts_hunter_green_010.md
doc_business_checking_accounts_lime_green_007.md
doc_business_checking_accounts_navy_blue_005.md
doc_business_checking_accounts_navy_blue_009.md
doc_business_checking_accounts_sky_blue_007.md
doc_business_checking_accounts_sky_blue_010.md
doc_business_checking_accounts_true_blue_007.md
doc_business_checking_accounts_true_blue_011.md
doc_business_checking_accounts_world_blue_005.md
doc_business_checking_accounts_world_blue_009.md
doc_business_checking_accounts_lime_green_002.md:## Overdrafts
doc_business_checking_accounts_lime_green_002.md:- The overdraft fee is $0.00.
doc_business_checking_accounts_lime_green_002.md:| Overdraft fee | $0.00 |


→ assistant calls `shell` with `{"command": "grep -i \"apy\\|rebate\\|minimum\" doc_business_checking_accounts_sky_blue_* doc_business_checking_accounts_hunter_green_* doc_business_checking_accounts_lime_green_002.md | head -60"}`

**tool:** doc_business_checking_accounts_sky_blue_007.md:- Your account earns an APY of 1.25%.
doc_business_checking_accounts_sky_blue_007.md:## ATM Access and Rebates
doc_business_checking_accounts_sky_blue_007.md:- Out-of-network ATM fees are rebated up to 15 per month.
doc_business_checking_accounts_sky_blue_007.md:- Keep ATM receipts and review posted rebates in your statements.
doc_business_checking_accounts_sky_blue_010.md:- For cash withdrawals at foreign (international) ATMs, you are charged 2% of the withdrawal amount, with a minimum fee of $3.00 per withdrawal.
doc_business_checking_accounts_sky_blue_010.md:- Because a minimum fee of $3.00 applies, fewer, larger withdrawals may reduce the effective fee rate on small cash needs.
doc_business_checking_accounts_sky_blue_010.md:## Monthly ATM fee rebates for out-of-network usage
doc_business_checking_accounts_sky_blue_010.md:- You are eligible for up to 15 in ATM fee rebates per month for out‑of‑network ATM usage.
doc_business_checking_accounts_sky_blue_010.md:- The rebate cap is monthly. If total eligible fees for out‑of‑network usage exceed 15 in a given month, any amount above the cap is your responsibility.
doc_business_checking_accounts_sky_blue_010.md:  - You are charged $1.50 per withdrawal. If the total eligible out‑of‑network fees for the month remain within 15, they are rebated up to that monthly cap.
doc_business_checking_accounts_sky_blue_010.md:  - The fee is 2% of the withdrawal, subject to the minimum of $3.00 if the calculated percentage is lower.
doc_business_checking_accounts_sky_blue_010.md:- Keep your ATM receipt and check your statement. If a fee or rebate does not appear as expected, contact support and include the date, location, and amount of the withdrawal so we can review it.
doc_business_checking_accounts_hunter_green_001.md:- Understand earnings: your account earns 1.0% APY on your balance.
doc_business_checking_accounts_hunter_green_001.md:- Track your accrued interest at the 1.0% APY in your account activity.
doc_business_checking_accounts_hunter_green_001.md:- **Eligibility**: A minimum relationship duration of 60 days as a checking account holder is required. 
doc_business_checking_accounts_hunter_green_006.md:# Hunter Green: Interest and APY Explained
doc_business_checking_accounts_hunter_green_006.md:## Your APY
doc_business_checking_accounts_hunter_green_006.md:- Your account earns 1.0% APY on eligible balances. This is the rate used to calculate your interest earnings over a year.
doc_business_checking_accounts_hunter_green_007.md:## ATM fee rebates
doc_business_checking_accounts_hunter_green_007.md:- We rebate out-of-network ATM fees up to $20 per month. Submit receipts if prompted to validate the surcharge amount.
doc_business_checking_accounts_hunter_green_009.md:- Fee: 2.5% of the withdrawal amount, with a minimum of $4.00.
doc_business_checking_accounts_hunter_green_009.md:- Withdrawing 80.00 in local currency equivalent: 80.00 × 2.5% = 2.00; minimum $4.00 applies, so the fee is $4.00.
doc_business_checking_accounts_hunter_green_009.md:| Foreign ATM withdrawal              | 2.5% (minimum $4.00) |
doc_business_checking_accounts_hunter_green_010.md:- International ATM withdrawals: 2.5% of the withdrawal amount, with a minimum of $4.00 per withdrawal
doc_business_checking_accounts_hunter_green_010.md:- Monthly ATM fee rebates for out-of-network usage: Up to $20 per month
doc_business_checking_accounts_hunter_green_010.md:- Eligible out-of-network ATM fees are automatically rebated until you reach the monthly cap of $20.
doc_business_checking_accounts_hunter_green_010.md:- Each withdrawal outside the U.S. incurs a fee of 2.5% of the withdrawal amount, with a minimum charge of $4.00 per withdrawal.
doc_business_checking_accounts_hunter_green_010.md:- Eligible out-of-network ATM fees are rebated up to the monthly maximum of $20.
doc_business_checking_accounts_hunter_green_010.md:3. If the withdrawal is international, add 2.5% of the withdrawal amount (minimum $4.00).
doc_business_checking_accounts_hunter_green_010.md:4. Subtract any applicable rebates, up to a total of $20 per month.
doc_business_checking_accounts_hunter_green_010.md:  - Rebated up to your remaining monthly $20 cap
doc_business_checking_accounts_hunter_green_010.md:  - International fee: 2.5% of $100 = $2.50, but minimum applies → $4.00
doc_business_checking_accounts_hunter_green_010.md:  - Rebated up to your remaining monthly $20 cap
doc_business_checking_accounts_hunter_green_010.md:  - Rebated up to your remaining monthly $20 cap
doc_business_checking_accounts_hunter_green_010.md:- Fees and any related rebates appear as separate line items on your account activity and statements.
doc_business_checking_accounts_hunter_green_010.md:- You can track how much of the $20 monthly rebate you have used by reviewing your statement or recent transactions.
doc_business_checking_accounts_lime_green_002.md:## Minimum balance requirement
doc_business_checking_accounts_lime_green_002.md:| Minimum balance requirement | $5,000 |


→ assistant calls `shell` with `{"command": "cat doc_business_checking_accounts_lime_green_002.md; grep -i \"minimum balance\\|requirement\" doc_business_checking_accounts_hunter_green_*.md doc_business_checking_accounts_sky_blue_*.md | head -30"}`

**tool:** # Lime Green Account: Fee Waivers and Balance Requirements

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
doc_business_checking_accounts_hunter_green_001.md:- **Requirement**: Referred business must deposit $10,000 within 90 days
doc_business_checking_accounts_hunter_green_008.md:## Requirement and intent
doc_business_checking_accounts_sky_blue_001.md:- Check whether you meet the funding requirement: 0.
doc_business_checking_accounts_sky_blue_009.md:- Ensure you meet the funding requirement shown as 0. If documentation was unclear, provide updated evidence.


→ assistant calls `shell` with `{"command": "cat doc_business_checking_accounts_hunter_green_009.md doc_business_checking_accounts_hunter_green_007.md | head -80; grep -i \"rebate\\|apy\" doc_business_checking_accounts_navy_blue_00[59].md doc_business_checking_accounts_cobalt_blue_0*.md | head"}`

**tool:** # Troubleshooting: Hunter Green Fees and Charges

## Identify and Resolve Unexpected Charges

Use the sections below to diagnose why a fee posted, how it’s calculated, and what to do if you believe it’s incorrect.

---

## Monthly Maintenance Fee

You were charged $25.00 if the account did not meet the waiver criteria for the statement period.

Troubleshooting steps:
- Confirm your statement period and check the “Fees & Charges” section for the $25.00 line.
- Verify whether your balance met the waiver threshold of $5,000 for the period.
- If you recently moved funds near the end of the cycle, the timing of deposits/withdrawals may have affected the period-end balance used to determine the waiver.
- If you believe your balance qualified for the waiver but the fee posted, request a review and include your statement period and balance details for the period.

Tips to avoid future charges:
- Keep your balance at or above $5,000 throughout the statement period.
- Set internal alerts to notify you before the cycle ends if your balance is trending below $5,000.

---

## Domestic Wire Transfer Fee

An outgoing domestic wire triggers a fee of $15.00.

Troubleshooting steps:
- Review your transaction history for outgoing domestic wires on the date the fee posted.
- Check whether a team member initiated a wire or if a scheduled/recurring wire executed.
- If you see multiple fees for a single wire, compare timestamps and reference numbers; submit both entries for review if they appear duplicative.

How to avoid:
- Where possible, consider alternative transfer methods for domestic payments that do not incur the $15.00.

---

## ATM Fees

### Domestic out-of-network ATM usage
- Fee: $2.00 per transaction when using non‑Rho ATMs within the U.S.
- You may also see a separate surcharge from the ATM operator; this is set by the ATM owner.

Troubleshooting steps:
- Match the ATM withdrawal date/time and location to the fee posted.
- If you were using an in-network ATM but still see $2.00, provide the ATM address and receipt for review.

### Foreign (international) ATM withdrawals
- Fee: 2.5% of the withdrawal amount, with a minimum of $4.00.
- In addition, the ATM operator may add its own surcharge.

How the fee is calculated:
- The fee is the greater of:
  - Withdrawal amount × 2.5%
  - $4.00

Examples:
- Withdrawing 80.00 in local currency equivalent: 80.00 × 2.5% = 2.00; minimum $4.00 applies, so the fee is $4.00.
- Withdrawing 400.00 in local currency equivalent: 400.00 × 2.5% = 10.00; since this exceeds $4.00, the fee is 10.00.

If a foreign ATM fee looks incorrect:
- Confirm the final settled withdrawal amount (in USD equivalent) used for the percentage calculation.
- Check for any separate ATM-operator surcharge on your receipt.
- If you see both a foreign ATM fee and a domestic out-of-network fee for the same international withdrawal, share the transaction details for review.

---

## Sustainability Allocations Within Fees

Some fees on Hunter Green include a carbon-offset allocation. If you see a note or line item indicating a contribution tied to a fee:
- This does not increase the total you pay beyond the published fee amounts (e.g., $25.00, $15.00, $2.00).
- If you believe a separate charge was added in error rather than noted as an allocation within an existing fee, share the statement line items for review.

---

## Quick Reference: Fees Discussed Here

| Charge type                          | Amount/Calculation |
|-------------------------------------|--------------------|
doc_business_checking_accounts_navy_blue_005.md:## ATM fee rebates
doc_business_checking_accounts_navy_blue_005.md:- Monthly cap on ATM fee rebates for out-of-network usage: 10
doc_business_checking_accounts_navy_blue_005.md:- Rebates are applied to qualifying ATM surcharge amounts up to the monthly cap.
doc_business_checking_accounts_navy_blue_009.md:| Monthly rebates for out‑of‑network usage | Up to $10 per calendar month | Rebates are capped at the monthly limit. |
doc_business_checking_accounts_navy_blue_009.md:## Monthly rebate details
doc_business_checking_accounts_navy_blue_009.md:- Rho rebates up to $10 in out‑of‑network ATM fees per calendar month.
doc_business_checking_accounts_navy_blue_009.md:- The monthly cap is the maximum total rebated; fees above this amount are your responsibility after the cap is reached.
doc_business_checking_accounts_cobalt_blue_001.md:- Your balance earns an APY of 0.5%
doc_business_checking_accounts_cobalt_blue_001.md:- Interest section: displays the current APY of 0.5% and accrued interest to date
doc_business_checking_accounts_cobalt_blue_006.md:## ATM access and rebates


**assistant:** Based on what you've told me, **Lime Green Business Checking** is the account I'd recommend. It meets all four of your requirements and has the most perks of the accounts that do:

| Your requirement | Lime Green |
|---|---|
| At least $15/month in ATM fee rebates | **Up to $25/month** rebated on out-of-network ATM fees (our out-of-network fee is $1.00 per withdrawal) |
| Zero overdraft fees | **$0.00** overdraft fee |
| Minimum balance under $10,000 | **$5,000** minimum balance |
| At least 1% APY | **1.5% APY** |

You'd also get a dedicated account manager and a daily transaction limit of $100,000.

**Two things to check before you commit:**
- **The $5,000 minimum.** You said your balance will usually be "a few thousand." If it often drops below $5,000, you could face balance-related charges and lose account benefits.
- **The monthly fee.** There's a $25 monthly maintenance fee, and it's only waived if your balance is $15,000 or more. With your balances, you should expect to pay it.

**If your balance often dips below $5,000:** Hunter Green Business Checking is the safer choice. It has no minimum balance requirement, $0 overdraft fees, up to $20/month in ATM fee rebates and 1.0% APY. It also has a $25 monthly fee, but that's waived if you keep at least $5,000 in the account. It requires paperless statements.

If you can confirm that Brightwave was formed less than 4 years ago, you'd also qualify for Sky Blue. It has a $0 overdraft fee, 1.25% APY, up to $15/month in ATM fee rebates and no monthly fee for the first 6 months. After that, the fee is $25 a month.

Would you like to go ahead with Lime Green? If so, I'll need to verify your identity before opening the account.

**user:** Look, I appreciate the detail, but I literally said I don’t want to compare options.

Given what you just laid out, **Lime Green doesn’t actually fit my situation** because I’m not going to keep **$5,000 minimum** consistently and I’m definitely not keeping **$15,000** around to waive a fee. So recommending that as “the one” doesn’t work.

Based on my requirements and the “few thousand” balance reality, **which ONE account do you recommend** — just pick it and I’ll go with it.

**assistant:** I recommend **Hunter Green Business Checking**. It meets all four of your requirements, even with a balance of a few thousand:

- **ATM rebates:** up to $20/month on out-of-network ATM fees, above the $15 you need
- **Overdraft fees:** $0.00
- **Minimum balance:** none required
- **Interest:** 1.0% APY

**Before you open it, two things to know:**
- **Monthly fee:** $25, and it's only waived when your balance is at least $5,000. At your usual balance, you'll probably pay it most months.
- **Statements:** paperless e-statements are required.

If you'd like to open it, I'll need to verify your identity first. Please share your full name and any two of these: date of birth, email on file, phone number on file, or address on file.

**user:** Alright — **Hunter Green Business Checking** sounds like the right fit. Let’s set it up.

My info:
- **Full name:** Yuki Nakamura  
- **Date of birth:** 05/12/1991  
- **Email:** yuki.nakamura@simba.com

→ assistant calls `get_user_information_by_name` with `{"customer_name": "Yuki Nakamura"}`
→ assistant calls `shell` with `{"command": "cat doc_bank_accounts_bank_accounts__general__003.md"}`

**tool:** Found 1 record(s) in 'users':

1. Record ID: 224959b99e
   name: Yuki Nakamura
   user_id: 224959b99e
   address: 3421 Sakura Avenue, Portland, OR 97205
   email: yuki.nakamura@simba.com
   phone_number: 503-555-0842
   date_of_birth: 05/12/1991


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
4. Use open_bank_account_4821 to open the account.

→ assistant calls `get_current_time` with `{}`

**tool:** The current time is 2025-11-14 03:40:00 EST.

→ assistant calls `log_verification` with `{"name": "Yuki Nakamura", "user_id": "224959b99e", "address": "3421 Sakura Avenue, Portland, OR 97205", "email": "yuki.nakamura@simba.com", "phone_number": "503-555-0842", "date_of_birth": "05/12/1991", "time_verified": "2025-11-14 03:40:00 EST"}`

**tool:** Verification logged successfully.
  - User: Yuki Nakamura (ID: 224959b99e)
  - Verified at: 2025-11-14 03:40:00 EST

→ assistant calls `shell` with `{"command": "grep -rhoi \"[a-z_]*_[0-9]\\{4\\}\" . | sort | uniq -c | sort -rn | head -40"}`

**tool:**      10 transfer_funds_between_bank_accounts_7291
      8 open_bank_account_4821
      5 initial_transfer_to_human_agent_0218
      5 get_pending_replacement_orders_5765
      5 get_bank_account_transactions_9173
      4 submit_interest_discrepancy_report_7294
      4 deposit_check_3847
      4 close_bank_account_7392
      4 apply_savings_account_credit_6831
      4 apply_checking_account_credit_5829
      3 pay_credit_card_from_checking_9182
      3 order_replacement_credit_card_7291
      3 initial_transfer_to_human_agent_1822
      3 get_all_user_accounts_by_user_id_3847
      3 clear_debit_card_fraud_alert_4892
      3 apply_statement_credit_8472
      2 update_transaction_rewards_3847
      2 submit_credit_limit_increase_request_7392
      2 order_debit_card_5739
      2 log_credit_card_closure_reason_4521
      2 get_user_dispute_history_7291
      2 get_payment_history_6183
      2 get_debit_dispute_status_7483
      2 get_debit_cards_by_account_id_7823
      2 get_credit_limit_increase_history_4829
      2 get_closure_reason_history_8293
      2 freeze_debit_card_3892
      2 downgrade_credit_card_3847
      2 deny_credit_limit_increase_5848
      2 dbc_1234
      2 approve_credit_limit_increase_5847
      2 apply_credit_card_account_flag_6147
      2 activate_debit_card_8293
      2 activate_debit_card_8292
      2 activate_debit_card_8291
      1 unfreeze_debit_card_3893
      1 submit_cash_back_dispute_0589
      1 set_debit_card_recurring_block_7382
      1 reset_debit_card_pin_6284
      1 request_temporary_debit_card_limit_increase_8374


→ assistant calls `unlock_discoverable_agent_tool` with `{"agent_tool_name": "get_all_user_accounts_by_user_id_3847"}`

**tool:** Tool unlocked: get_all_user_accounts_by_user_id_3847
Description: Retrieve all accounts (checking, savings, credit cards) for a customer.

Tool: get_all_user_accounts_by_user_id_3847
Description: Retrieve all accounts (checking, savings, credit cards) for a customer.
Parameters:
  - user_id: string (required) - The customer's unique identifier in the system

You can now use this tool by calling `call_discoverable_agent_tool` with agent_tool_name='get_all_user_accounts_by_user_id_3847' and the required arguments.

→ assistant calls `call_discoverable_agent_tool` with `{"agent_tool_name": "get_all_user_accounts_by_user_id_3847", "arguments": "{\"user_id\": \"224959b99e\"}"}`

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


→ assistant calls `shell` with `{"command": "cat doc_bank_accounts_bank_accounts__general__013.md"}`

**tool:** # Internal: Business Checking Account Promotion - November 2025

PROMOTION NOTICE - ACTIVE FROM 11/01/2025 TO 11/30/2025: When recommending business checking accounts to customers, if multiple accounts meet all of the customer's stated requirements, agents should prioritize recommending accounts in the following order: (1) Sky Blue account should be recommended first, as it is currently under promotional priority; (2) Lime Green account should be recommended second, as it is also under promotional priority; (3) All other qualifying accounts may be recommended only if neither Sky Blue nor Lime Green meet the customer's requirements. This promotion is designed to increase adoption of our startup and premium business offerings. Always ensure the recommended account still meets ALL of the customer's stated requirements - do not recommend a promotional account if it does not satisfy the customer's needs.

→ assistant calls `shell` with `{"command": "cat doc_business_checking_accounts_sky_blue_00[2-9].md | grep -iv \"^$\" | head -120"}`

**tool:** # Sky Blue: Understanding Your Free Period
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
- Your eligibility: Referrers must meet the 45-day tenure threshold based on their banking history with Rho-Bank.# Claiming AWS Credits and Stripe Discounts
## AWS Credits
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
- Keep confirmation emails or reference IDs for your records.# Sky Blue: Burn-Rate Analytics and Runway Projections
## Access and Inclusion
- Burn-rate analytics are included at no extra cost: Yes.
## Using the Dashboard
- Open the analytics section to review monthly spend, inflows, and net burn.
- Categorize transactions to improve accuracy and trend visibility.
## Runway Projections
- The runway tool forecasts up to 18 months based on your historical burn and current balance assumptions.
- Adjust categorization and filters to reflect one-time or planned changes and refine the projection.
## Best Practices
- Reconcile transactions frequently so burn and runway reflect current operations.
- Annotate material changes such as new hires or major vendor shifts to inform stakeholders.
- Export visuals for sharing during planning sessions or board updates.# Founder Cards: How to Request and Manage
## Requesting Founder Cards
- You can request up to 3.
- From your cards section, choose Add card, supply recipient details, and set spending controls.
- Select physical or virtual issuance, then confirm delivery or activation preferences.
## Managing Cards
- Activate, freeze or unfreeze, and replace cards from the card management screen.
- Adjust category controls, merchant restrictions, and individual limits for each cardholder.
- Review transactions in real time and reconcile expenses to maintain accurate records.
## Related Transfers That May Affect Card Workflows
- If you reimburse founders or fund settlements via domestic wire, the fee is $15.00.
- Your account includes 5 free incoming domestic wires per month, which can help when receiving reimbursements or capital infusions tied to card activity.
## Operational Tips
- Create a card policy for founders covering allowed merchants and required receipts.
- Use alerts for large or unusual transactions to speed up reviews.# Sky Blue: Preparing Investor Reports
## Included Tools
- Investor-ready reporting tools are included: Yes.
- Burn-rate analytics are included and integrate with reporting: Yes.
## Building Your Investor Packet
- Choose the reporting period in your analytics or reporting section.
- Include key exhibits: cash position, operating inflows and outflows, burn summary, and runway overview.
- Add narrative context for material variances and one-time items.
## Workflow Tips
- Lock your dataset before exporting to prevent late changes.
- Attach notes explaining accounting treatments, vendor changes, or hiring plans.
- Export your packet and store a versioned copy for audit consistency.# Mobile Banking and Deposits on Sky Blue
## Mobile Check Deposits
- Your daily mobile deposit limit is $25,000.
- Endorse the check, capture clear images, and submit through the mobile app.
- Monitor deposit status in your deposit history and retain the check until confirmation.
## Interest on Balances
- Your account earns an APY of 1.25%.
- Interest is compounded daily.
## ATM Access and Rebates
- Out-of-network ATM fees are rebated up to 15 per month.
- Keep ATM receipts and review posted rebates in your statements.
## Best Practices
- Use well-lit photos for deposits and avoid folded or damaged checks.
- Set balance and deposit alerts to track funds availability and interest accrual.# After the Free Period: What Changes and What Stays
## What Changes
- The monthly maintenance fee becomes $25.00.
- Billing updates automatically; you will see the fee on your next statement after the transition.
## What Stays the Same
- Interest continues to accrue at 1.25% according to your account terms.
- Your existing users, cards, controls, and analytics remain active without interruption.
## About Eligibility Post-Onboarding
- Eligibility criteria tied to company age are evaluated when you open the account; the threshold is 4.
- Your account status after the free period does not require re-checking this onboarding criterion.# Troubleshooting: Sky Blue Eligibility Issues
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
- Include a screenshot of your plan details showing the expected 6 so support can reconcile your account configuration.


**assistant:** Thanks, Yuki. Your identity is verified, and you're eligible to open a new business checking account.

Before I open Hunter Green, one quick question: **was Brightwave formed within the last 4 years?**

I'm asking because if it was, Sky Blue Business Checking would be my single recommendation instead. It meets all four of your requirements, and it beats Hunter Green on several points:
- **Overdraft fees:** $0.00
- **Minimum balance:** none required
- **ATM rebates:** up to $15/month, which meets your minimum but is less than Hunter Green's $20
- **Interest:** 1.25% APY, compared with Hunter Green's 1.0%
- **Monthly fee:** none for the first 6 months, then $25. Hunter Green charges $25 a month from the start unless your balance is $5,000 or more.

Sky Blue is only open to businesses formed within the last 4 years. Your formation documents should show the date. Here's what to do:
- **Formed within 4 years:** reply "Sky Blue" and I'll open that account.
- **Formed more than 4 years ago, or you'd rather not check:** reply "Hunter Green" and I'll open that account right away.

**user:** I don’t have the formation date in front of me and I’m not going digging for documents right now.

Open **Hunter Green Business Checking**.

→ assistant calls `unlock_discoverable_agent_tool` with `{"agent_tool_name": "open_bank_account_4821"}`

**tool:** Tool unlocked: open_bank_account_4821
Description: Open a new bank account for a customer.

Tool: open_bank_account_4821
Description: Open a new bank account for a customer.
Parameters:
  - user_id: string (required) - The customer's unique identifier in the system
  - account_type: string (required) - Type of account to open. Must be one of: 'checking' (personal checking), 'savings' (personal savings), 'business_checking', 'business_savings'
  - account_class: string (required) - The full official account class name

You can now use this tool by calling `call_discoverable_agent_tool` with agent_tool_name='open_bank_account_4821' and the required arguments.

→ assistant calls `call_discoverable_agent_tool` with `{"agent_tool_name": "open_bank_account_4821", "arguments": "{\"user_id\": \"224959b99e\", \"account_type\": \"business_checking\", \"account_class\": \"Hunter Green Account\"}"}`

**tool:** Bank account opened successfully!
  - Account ID: 23a286625140d97b
  - User ID: 224959b99e
  - Account Type: business_checking
  - Account Class: Hunter Green Account
  - Status: OPEN
  - Initial Balance: $0.00
  - Date Opened: 11/14/2025

**assistant:** Your **Hunter Green Business Checking** account is open.

- **Account ID:** 23a286625140d97b
- **Status:** Open
- **Opened:** 11/14/2025
- **Starting balance:** $0.00

**Getting started:**
1. **Turn on e-statements** in your profile settings. This account requires paperless statements.
2. **Make your first deposit** from your funding source.
3. **Avoiding the $25 monthly fee:** the fee is waived when your balance is at least $5,000. Otherwise it applies each month.

As a reminder, the account has no overdraft fees and earns 1.0% APY. It rebates up to $20 a month in out-of-network ATM fees. A Rho fee of $2.00 applies to each out-of-network withdrawal in the U.S. International withdrawals cost 2.5% of the amount, with a $4.00 minimum.

Is there anything else I can help you with?

**user:** Perfect — thanks for making that easy. I think I’m all set for now!  

###STOP###
