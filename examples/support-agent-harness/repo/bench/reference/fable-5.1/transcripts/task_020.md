# task_020 · trial 0 · not resolved · ended: user_stop

**assistant:** Hi! How can I help you today?

**user:** Hi there, I’ve been going through my rewards statements and something doesn’t seem right with my cash back. Could you help me figure out what’s going on? I have both a Business Silver Rewards Card and a Silver Rewards Card, and the totals just don’t seem to add up.

**assistant:** I'd be happy to help you look into your cash back rewards. Before I can access your account information, I need to verify your identity.

Could you please provide your full name, along with two of the following: your date of birth, email address, phone number, or home address?

**user:** Sure. My name is Amara Okonkwo.  

For verification:  
- Date of birth: 08/11/1997  
- Phone number: 713-555-0963

**assistant → get_user_information_by_name:** {"customer_name": "Amara Okonkwo"}

**tool:** Found 1 record(s) in 'users':

1. Record ID: 755bcb4d5d
   name: Amara Okonkwo
   user_id: 755bcb4d5d
   address: 305 Magnolia Street, Houston, TX 77002
   email: dancing_star_amara@icloud.com
   phone_number: 713-555-0963
   date_of_birth: 08/11/1997

**assistant → get_current_time:** {}

**tool:** The current time is 2025-11-14 03:40:00 EST.

**assistant → log_verification:** {"name": "Amara Okonkwo", "user_id": "755bcb4d5d", "address": "305 Magnolia Street, Houston, TX 77002", "email": "dancing_star_amara@icloud.com", "phone_number": "713-555-0963", "date_of_birth": "08/11/1997", "time_verified": "2025-11-14 03:40:00 EST"}

**tool:** Verification logged successfully.
  - User: Amara Okonkwo (ID: 755bcb4d5d)
  - Verified at: 2025-11-14 03:40:00 EST

**assistant → get_credit_card_accounts_by_user:** {"user_id": "755bcb4d5d"}

**tool:** Found 2 record(s) in 'credit_card_accounts':

1. Record ID: cc_755bcb4d5d_bsilver
   account_id: cc_755bcb4d5d_bsilver
   user_id: 755bcb4d5d
   card_type: Business Silver Rewards Card
   date_of_account_open: 02/13/2025
   current_balance: $3,105.12
   reward_points: 37848 points

2. Record ID: cc_755bcb4d5d_silver
   account_id: cc_755bcb4d5d_silver
   user_id: 755bcb4d5d
   card_type: Silver Rewards Card
   date_of_account_open: 01/20/2025
   current_balance: $2,465.79
   reward_points: 9875 points

**assistant → get_credit_card_transactions_by_user:** {"user_id": "755bcb4d5d"}

**tool:** Found 26 record(s) in 'credit_card_transaction_history':

1. Record ID: txn_b7e2d4c5f501
   transaction_id: txn_b7e2d4c5f501
   user_id: 755bcb4d5d
   credit_card_type: Silver Rewards Card
   merchant_name: Trader Joe's
   transaction_amount: $87.25
   transaction_date: 02/01/2025
   category: Groceries
   status: COMPLETED
   rewards_earned: 87 points

2. Record ID: txn_b7e2d4c5f502
   transaction_id: txn_b7e2d4c5f502
   user_id: 755bcb4d5d
   credit_card_type: Silver Rewards Card
   merchant_name: Delta Airlines
   transaction_amount: $367.50
   transaction_date: 02/18/2025
   category: Travel
   status: COMPLETED
   rewards_earned: 1470 points

3. Record ID: txn_a8f1c2d3e401
   transaction_id: txn_a8f1c2d3e401
   user_id: 755bcb4d5d
   credit_card_type: Business Silver Rewards Card
   merchant_name: Delta Airlines
   transaction_amount: $450.00
   transaction_date: 02/20/2025
   category: Travel
   status: COMPLETED
   rewards_earned: 9000 points

4. Record ID: txn_a8f1c2d3e402
   transaction_id: txn_a8f1c2d3e402
   user_id: 755bcb4d5d
   credit_card_type: Business Silver Rewards Card
   merchant_name: Adobe Creative Cloud
   transaction_amount: $54.99
   transaction_date: 03/05/2025
   category: Software
   status: COMPLETED
   rewards_earned: 1100 points

5. Record ID: txn_b7e2d4c5f503
   transaction_id: txn_b7e2d4c5f503
   user_id: 755bcb4d5d
   credit_card_type: Silver Rewards Card
   merchant_name: Slack Technologies
   transaction_amount: $44.99
   transaction_date: 03/12/2025
   category: Software
   status: COMPLETED
   rewards_earned: 180 points

6. Record ID: txn_a8f1c2d3e403
   transaction_id: txn_a8f1c2d3e403
   user_id: 755bcb4d5d
   credit_card_type: Business Silver Rewards Card
   merchant_name: JetBlue Airways
   transaction_amount: $315.00
   transaction_date: 03/22/2025
   category: Travel
   status: COMPLETED
   rewards_earned: 3150 points

7. Record ID: txn_b7e2d4c5f504
   transaction_id: txn_b7e2d4c5f504
   user_id: 755bcb4d5d
   credit_card_type: Silver Rewards Card
   merchant_name: Olive Garden
   transaction_amount: $65.00
   transaction_date: 04/05/2025
   category: Dining
   status: COMPLETED
   rewards_earned: 65 points

8. Record ID: txn_a8f1c2d3e404
   transaction_id: txn_a8f1c2d3e404
   user_id: 755bcb4d5d
   credit_card_type: Business Silver Rewards Card
   merchant_name: Microsoft 365
   transaction_amount: $79.99
   transaction_date: 04/10/2025
   category: Software
   status: COMPLETED
   rewards_earned: 160 points

9. Record ID: txn_b7e2d4c5f505
   transaction_id: txn_b7e2d4c5f505
   user_id: 755bcb4d5d
   credit_card_type: Silver Rewards Card
   merchant_name: Expedia
   transaction_amount: $520.00
   transaction_date: 04/25/2025
   category: Travel
   status: COMPLETED
   rewards_earned: 2080 points

10. Record ID: txn_a8f1c2d3e405
   transaction_id: txn_a8f1c2d3e405
   user_id: 755bcb4d5d
   credit_card_type: Business Silver Rewards Card
   merchant_name: Marriott Hotels
   transaction_amount: $425.50
   transaction_date: 05/03/2025
   category: Travel
   status: COMPLETED
   rewards_earned: 8510 points

11. Record ID: txn_b7e2d4c5f506
   transaction_id: txn_b7e2d4c5f506
   user_id: 755bcb4d5d
   credit_card_type: Silver Rewards Card
   merchant_name: GitHub Enterprise
   transaction_amount: $255.00
   transaction_date: 05/18/2025
   category: Software
   status: COMPLETED
   rewards_earned: 2550 points

12. Record ID: txn_a8f1c2d3e406
   transaction_id: txn_a8f1c2d3e406
   user_id: 755bcb4d5d
   credit_card_type: Business Silver Rewards Card
   merchant_name: Whole Foods Market
   transaction_amount: $127.43
   transaction_date: 05/28/2025
   category: Groceries
   status: COMPLETED
   rewards_earned: 255 points

13. Record ID: txn_b7e2d4c5f507
   transaction_id: txn_b7e2d4c5f507
   user_id: 755bcb4d5d
   credit_card_type: Silver Rewards Card
   merchant_name: Netflix
   transaction_amount: $22.99
   transaction_date: 06/08/2025
   category: Entertainment
   status: COMPLETED
   rewards_earned: 23 points

14. Record ID: txn_a8f1c2d3e407
   transaction_id: txn_a8f1c2d3e407
   user_id: 755bcb4d5d
   credit_card_type: Business Silver Rewards Card
   merchant_name: Salesforce
   transaction_amount: $199.00
   transaction_date: 06/15/2025
   category: Software
   status: COMPLETED
   rewards_earned: 3980 points

15. Record ID: txn_b7e2d4c5f508
   transaction_id: txn_b7e2d4c5f508
   user_id: 755bcb4d5d
   credit_card_type: Silver Rewards Card
   merchant_name: American Airlines
   transaction_amount: $298.00
   transaction_date: 07/02/2025
   category: Travel
   status: COMPLETED
   rewards_earned: 1192 points

16. Record ID: txn_a8f1c2d3e408
   transaction_id: txn_a8f1c2d3e408
   user_id: 755bcb4d5d
   credit_card_type: Business Silver Rewards Card
   merchant_name: United Airlines
   transaction_amount: $285.00
   transaction_date: 07/08/2025
   category: Travel
   status: COMPLETED
   rewards_earned: 5700 points

17. Record ID: txn_b7e2d4c5f509
   transaction_id: txn_b7e2d4c5f509
   user_id: 755bcb4d5d
   credit_card_type: Silver Rewards Card
   merchant_name: Costco
   transaction_amount: $143.56
   transaction_date: 07/28/2025
   category: Groceries
   status: COMPLETED
   rewards_earned: 144 points

18. Record ID: txn_a8f1c2d3e409
   transaction_id: txn_a8f1c2d3e409
   user_id: 755bcb4d5d
   credit_card_type: Business Silver Rewards Card
   merchant_name: Amazon
   transaction_amount: $156.78
   transaction_date: 08/01/2025
   category: Shopping
   status: COMPLETED
   rewards_earned: 314 points

19. Record ID: txn_b7e2d4c5f510
   transaction_id: txn_b7e2d4c5f510
   user_id: 755bcb4d5d
   credit_card_type: Silver Rewards Card
   merchant_name: JetBrains
   transaction_amount: $199.00
   transaction_date: 08/20/2025
   category: Software
   status: COMPLETED
   rewards_earned: 796 points

20. Record ID: txn_a8f1c2d3e410
   transaction_id: txn_a8f1c2d3e410
   user_id: 755bcb4d5d
   credit_card_type: Business Silver Rewards Card
   merchant_name: Southwest Airlines
   transaction_amount: $380.00
   transaction_date: 08/25/2025
   category: Travel
   status: COMPLETED
   rewards_earned: 1520 points

21. Record ID: txn_a8f1c2d3e411
   transaction_id: txn_a8f1c2d3e411
   user_id: 755bcb4d5d
   credit_card_type: Business Silver Rewards Card
   merchant_name: Zoom Video
   transaction_amount: $149.99
   transaction_date: 09/15/2025
   category: Software
   status: COMPLETED
   rewards_earned: 600 points

22. Record ID: txn_b7e2d4c5f511
   transaction_id: txn_b7e2d4c5f511
   user_id: 755bcb4d5d
   credit_card_type: Silver Rewards Card
   merchant_name: Hilton Seattle
   transaction_amount: $275.00
   transaction_date: 09/30/2025
   category: Travel
   status: COMPLETED
   rewards_earned: 1100 points

23. Record ID: txn_a8f1c2d3e412
   transaction_id: txn_a8f1c2d3e412
   user_id: 755bcb4d5d
   credit_card_type: Business Silver Rewards Card
   merchant_name: Hilton Hotels
   transaction_amount: $342.00
   transaction_date: 10/10/2025
   category: Travel
   status: COMPLETED
   rewards_earned: 3420 points

24. Record ID: txn_a8f1c2d3e414
   transaction_id: txn_a8f1c2d3e414
   user_id: 755bcb4d5d
   credit_card_type: Business Silver Rewards Card
   merchant_name: Coursera
   transaction_amount: $49.99
   transaction_date: 10/15/2025
   category: Software
   status: COMPLETED
   rewards_earned: 50 points

25. Record ID: txn_b7e2d4c5f512
   transaction_id: txn_b7e2d4c5f512
   user_id: 755bcb4d5d
   credit_card_type: Silver Rewards Card
   merchant_name: Nordstrom
   transaction_amount: $187.50
   transaction_date: 10/28/2025
   category: Shopping
   status: COMPLETED
   rewards_earned: 188 points

26. Record ID: txn_a8f1c2d3e413
   transaction_id: txn_a8f1c2d3e413
   user_id: 755bcb4d5d
   credit_card_type: Business Silver Rewards Card
   merchant_name: Target
   transaction_amount: $89.45
   transaction_date: 11/05/2025
   category: Shopping
   status: COMPLETED
   rewards_earned: 89 points

**assistant → shell:** {"command": "cat INDEX.md; grep -ril \"Silver Rewards\" ."}

**tool:** # Knowledge Base Index

**Total documents:** 698

## Documents

- **doc_bank_accounts_bank_accounts__general__001.md** - Internal: Opening Personal Checking Accounts
- **doc_bank_accounts_bank_accounts__general__002.md** - Internal: Opening Personal Savings Accounts
- **doc_bank_accounts_bank_accounts__general__003.md** - Internal: Opening Business Checking Accounts
- **doc_bank_accounts_bank_accounts__general__004.md** - Internal: Opening Business Savings Accounts
- **doc_bank_accounts_bank_accounts__general__005.md** - Internal: Closing Personal Checking Accounts
- **doc_bank_accounts_bank_accounts__general__006.md** - Internal: Closing Personal Savings Accounts
- **doc_bank_accounts_bank_accounts__general__007.md** - Internal: Closing Business Checking Accounts
- **doc_bank_accounts_bank_accounts__general__008.md** - Internal: Closing Business Savings Accounts
- **doc_bank_accounts_bank_accounts__general__009.md** - Internal: Retrieving Customer Account Information
- **doc_bank_accounts_bank_accounts__general__010.md** - Internal: Transferring Funds Between Bank Accounts
- **doc_bank_accounts_bank_accounts__general__011.md** - Depositing a Check Into Your Account
- **doc_bank_accounts_bank_accounts__general__012.md** - Linked Checking Account APY Boosts for Savings Accounts
- **doc_bank_accounts_bank_accounts__general__013.md** - Internal: Business Checking Account Promotion - November 2025
- **doc_bank_accounts_bank_accounts__general__014.md** - Internal: Business Checking Account Promotion - October 2025
- **doc_bank_accounts_bank_accounts__general__015.md** - Internal: Business Savings Account Promotion - November 2025
- **doc_bank_accounts_bank_accounts__general__016.md** - Internal: Business Savings Account Promotion - October 2025
- **doc_bank_accounts_bank_accounts__general__017.md** - Internal: Applying Credits and Rebates to Checking Accounts
- **doc_bank_accounts_bank_accounts__general__018.md** - Internal: Retrieving Bank Account Transaction History
- **doc_bank_accounts_bank_accounts__general__019.md** - Notice Something Wrong on Your Statement? We Can Help
- **doc_bank_accounts_bank_accounts__general__020.md** - What to Know Before Closing Your Account
- **doc_bank_accounts_bank_accounts__general__021.md** - Need Your Transaction History? Our Team Can Help
- **doc_bank_accounts_bank_accounts__general__022.md** - Moving Money Between Your Rho-Bank Accounts
- **doc_bank_accounts_bank_accounts__general__023.md** - Internal: Ordering a Debit Card for a Bank Account
- **doc_bank_accounts_bank_accounts__general__024.md** - Internal: Activating a Debit Card
- **doc_bank_accounts_bank_accounts__general__025.md** - Internal: Closing/Cancelling a Debit Card
- **doc_bank_accounts_bank_accounts__general__026.md** - Internal: Freezing and Unfreezing a Debit Card
- **doc_bank_accounts_bank_accounts__general__027.md** - Internal: Resetting or Changing a Debit Card PIN
- **doc_bank_accounts_bank_accounts__general__028.md** - Internal: Retrieving Debit Card Information
- **doc_bank_accounts_bank_accounts__general__029.md** - Internal: Debit Card Replacement Options by Account Tier
- **doc_bank_accounts_bank_accounts__general__030.md** - Internal: Lost/Stolen Card - Cross-Product Security Protocol
- **doc_bank_accounts_bank_accounts__general__031.md** - Internal: Filing a Debit Card Transaction Dispute
- **doc_bank_accounts_bank_accounts__general__032.md** - Internal: Debit Card Provisional Credit Guidelines
- **doc_bank_accounts_bank_accounts__general__033.md** - Internal: ATM Dispute Special Procedures
- **doc_bank_accounts_bank_accounts__general__034.md** - Internal: Recurring Payment Disputes vs. Stop Future Payments
- **doc_bank_accounts_bank_accounts__general__035.md** - Internal: Checking Debit Card Dispute Status
- **doc_bank_accounts_bank_accounts__general__036.md** - What To Do If You Notice an Unauthorized Debit Card Transaction
- **doc_bank_accounts_bank_accounts__general__037.md** - Understanding Regulation E: Your Debit Card Consumer Protections
- **doc_bank_accounts_bank_accounts__general__039.md** - Internal: Managing Debit Card Security Alerts and Blocks
- **doc_bank_accounts_bank_accounts__general__040.md** - Internal: Requesting Temporary Debit Card Limit Increase
- **doc_bank_accounts_bank_accounts__general__041.md** - Internal: PIN Lock Investigation Protocol - Fraud Risk Scoring
- **doc_bank_accounts_bank_accounts__general__042.md** - Internal: Human Agent Transfer Reason Codes
- **doc_bank_accounts_bank_accounts__general__043.md** - Internal: Applying Credits to Savings Accounts
- **doc_bank_accounts_bank_accounts__general__044.md** - Internal: Submitting Interest Discrepancy Reports
- **doc_bank_accounts_bank_accounts__general__045.md** - Credit Card APY Bonuses: Stacking Policy
- **doc_bank_accounts_bank_accounts__general__046.md** - Linked Checking Account APY Boost: Selection Policy
- **doc_bank_accounts_bank_accounts__general__047.md** - Internal: Checking Account Referral Program Terms and Restrictions
- **doc_bank_accounts_bank_accounts__general__048.md** - FAQ: Checking Account Referrals
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
- **doc_business_checking_accounts_navy_blue_003.md** - Navy Blue Account: Wire Transfer Pricing
- **doc_business_checking_accounts_navy_blue_004.md** - Navy Blue Account: Interest Explained
- **doc_business_checking_accounts_navy_blue_005.md** - Navy Blue Account: Mobile Deposits and Daily Limits
- **doc_business_checking_accounts_navy_blue_006.md** - Navy Blue Account: ACH Transfers and Processing Times
- **doc_business_checking_accounts_navy_blue_007.md** - Navy Blue Account: Paper vs E-Statements
- **doc_business_checking_accounts_navy_blue_008.md** - Why am I being charged a fee on my Navy Blue Account?
- **doc_business_checking_accounts_navy_blue_009.md** - Navy Blue Account: ATM Fees and Out-of-Network Usage
- **doc_business_checking_accounts_sky_blue_001.md** - Getting Started with Sky Blue: Startup Account Setup
- **doc_business_checking_accounts_sky_blue_002.md** - Sky Blue: Understanding Your Free Period
- **doc_business_checking_accounts_sky_blue_003.md** - Claiming AWS Credits and Stripe Discounts
- **doc_business_checking_accounts_sky_blue_004.md** - Sky Blue: Burn-Rate Analytics and Runway Projections
- **doc_business_checking_accounts_sky_blue_005.md** - Founder Cards: How to Request and Manage
- **doc_business_checking_accounts_sky_blue_006.md** - Sky Blue: Preparing Investor Reports
- **doc_business_checking_accounts_sky_blue_007.md** - Mobile Banking and Deposits on Sky Blue
- **doc_business_checking_accounts_sky_blue_008.md** - After the Free Period: What Changes and What Stays
- **doc_business_checking_accounts_sky_blue_009.md** - Troubleshooting: Sky Blue Eligibility Issues
- **doc_business_checking_accounts_sky_blue_010.md** - Sky Blue Account: ATM Fees and Out-of-Network Usage
- **doc_business_checking_accounts_true_blue_001.md** - True Blue Business Checking: Complete Guide
- **doc_business_checking_accounts_true_blue_002.md** - True Blue Account: Fee Waivers and Balance Requirements
- **doc_business_checking_accounts_true_blue_003.md** - True Blue Account: Cash Management Dashboard
- **doc_business_checking_accounts_true_blue_004.md** - True Blue Account: Treasury Services Integration
- **doc_business_checking_accounts_true_blue_005.md** - True Blue Account: Payroll Processing Discounts
- **doc_business_checking_accounts_true_blue_006.md** - True Blue Account: Wire Transfers and Free Wires
- **doc_business_checking_accounts_true_blue_007.md** - True Blue Account: Same-Day ACH and Mobile Deposits
- **doc_business_checking_accounts_true_blue_008.md** - True Blue Account: Interest and APY Explained
- **doc_business_checking_accounts_true_blue_009.md** - True Blue Account: Dedicated Support and Response Times
- **doc_business_checking_accounts_true_blue_010.md** - Why am I being charged a fee on my True Blue Account?
- **doc_business_checking_accounts_true_blue_011.md** - True Blue Account: ATM Fees and Out-of-Network Usage
- **doc_business_checking_accounts_world_blue_001.md** - Getting Started with World Blue International Checking
- **doc_business_checking_accounts_world_blue_002.md** - World Blue: How to Avoid Monthly Fees
- **doc_business_checking_accounts_world_blue_003.md** - Navigating Currency Conversions on World Blue
- **doc_business_checking_accounts_world_blue_004.md** - Sending International Payments: SWIFT and SEPA Options
- **doc_business_checking_accounts_world_blue_005.md** - World Blue: Traveling Abroad with Your Business Account
- **doc_business_checking_accounts_world_blue_006.md** - Configuring Team Permissions on World Blue
- **doc_business_checking_accounts_world_blue_007.md** - Earning Interest on Your World Blue Balance
- **doc_business_checking_accounts_world_blue_008.md** - Troubleshooting: My World Blue fee wasn't waived
- **doc_business_checking_accounts_world_blue_009.md** - World Blue: Free International ATM Access
- **doc_business_credit_cards_business_bronze_rewards_card_001.md** - Business Bronze Rewards Card: How to Apply and Get Started
- **doc_business_credit_cards_business_bronze_rewards_card_002.md** - Business Bronze Rewards Card: Understanding Your 1% Cash Back Rewards
- **doc_business_credit_cards_business_bronze_rewards_card_003.md** - Business Bronze Rewards Card: Exceptions and Exclusions
- **doc_business_credit_cards_business_bronze_rewards_card_004.md** - Business Bronze Rewards Card: When and How Are Cash Back Rewards Paid?
- **doc_business_credit_cards_business_bronze_rewards_card_005.md** - Business Bronze Rewards Card: Credit Limit and Spending Guidelines
- **doc_business_credit_cards_business_bronze_rewards_card_006.md** - Business Bronze Rewards Card: No Annual Fee - What Does This Mean?
- **doc_business_credit_cards_business_bronze_rewards_card_007.md** - Business Bronze Rewards Card: Making Payments and Managing Your Balance
- **doc_business_credit_cards_business_bronze_rewards_card_008.md** - Business Bronze Rewards Card: Travel Benefits and Purchase Protection
- **doc_business_credit_cards_business_bronze_rewards_card_009.md** - Business Bronze Rewards Card: Adding Employee Cards to Your Account
- **doc_business_credit_cards_business_bronze_rewards_card_010.md** - Business Bronze Rewards Card: New Customer Promo Offer Details
- **doc_business_credit_cards_business_bronze_rewards_card_011.md** - Business Bronze Rewards Card: Referral Program - Earn Rewards for Referring Businesses
- **doc_business_credit_cards_business_gold_rewards_card_001.md** - Business Gold Rewards Card: Premium Business Credit Card Overview
- **doc_business_credit_cards_business_gold_rewards_card_002.md** - Business Gold Rewards Card: Earning 2.5% Cash Back on Operations Spending
- **doc_business_credit_cards_business_gold_rewards_card_003.md** - Business Gold Rewards Card: What Qualifies as Operations Spend?
- **doc_business_credit_cards_business_gold_rewards_card_004.md** - Business Gold Rewards Card: How to Access Airport Lounges Worldwide
- **doc_business_credit_cards_business_gold_rewards_card_005.md** - Business Gold Rewards Card: Lounge Access Network and Participating Airports
- **doc_business_credit_cards_business_gold_rewards_card_006.md** - Business Gold Rewards Card: Adding Authorized Users with Lounge Access
- **doc_business_credit_cards_business_gold_rewards_card_007.md** - Business Gold Rewards Card: Annual Fee and Premium Benefits Guide
- **doc_business_credit_cards_business_gold_rewards_card_008.md** - Business Gold Rewards Card: Travel Perks and Concierge Services
- **doc_business_credit_cards_business_gold_rewards_card_009.md** - Business Gold Rewards Card: Maximizing Your Operations Rewards
- **doc_business_credit_cards_business_gold_rewards_card_010.md** - Business Gold Rewards Card: Referral Program - Earn Rewards for Referring Businesses
- **doc_business_credit_cards_business_platinum_rewards_card_001.md** - Platinum Rewards Card: Elite Business Credit Card Application
- **doc_business_credit_cards_business_platinum_rewards_card_002.md** - Business Platinum Rewards Card: Earning 4% Cash Back
- **doc_business_credit_cards_business_platinum_rewards_card_003.md** - Business Platinum Rewards Card: Which Media and Advertising Expenses Qualify?
- **doc_business_credit_cards_business_platinum_rewards_card_004.md** - Business Platinum Rewards Card: How to Use the Concierge Travel Desk
- **doc_business_credit_cards_business_platinum_rewards_card_005.md** - Business Platinum Rewards Card: Travel Planning and Booking Assistance Services
- **doc_business_credit_cards_business_platinum_rewards_card_006.md** - Business Platinum Rewards Card: Premium Travel Benefits and Insurance Coverage
- **doc_business_credit_cards_business_platinum_rewards_card_007.md** - Business Platinum Rewards Card: Software and Cloud Services - Qualifying Purchases
- **doc_business_credit_cards_business_platinum_rewards_card_008.md** - Business Platinum Rewards Card: Understanding Your Annual Fee and Premium Perks
- **doc_business_credit_cards_business_platinum_rewards_card_009.md** - Business Platinum Rewards Card: Dedicated Account Support and Priority Service
- **doc_business_credit_cards_business_platinum_rewards_card_010.md** - Business Platinum Rewards Card: Business Travel Itinerary Management
- **doc_business_credit_cards_business_platinum_rewards_card_011.md** - Business Platinum Rewards Card: New Customer First Year Fee Waiver
- **doc_business_credit_cards_business_platinum_rewards_card_012.md** - Business Platinum Rewards Card: Referral Program - Earn Rewards for Referring Businesses
- **doc_business_credit_cards_business_silver_rewards_card_001.md** - Business Silver Rewards Card: Getting Started and Eligibility Requirements
- **doc_business_credit_cards_business_silver_rewards_card_002.md** - Business Silver Rewards Card: How to Earn 10% Back on Travel and Software Purchases
- **doc_business_credit_cards_business_silver_rewards_card_003.md** - Business Silver Rewards Card: Which Travel Purchases Qualify for 10% Cash Back?
- **doc_business_credit_cards_business_silver_rewards_card_004.md** - Business Silver Rewards Card: Software and SaaS Spending - Qualifying Categories
- **doc_business_credit_cards_business_silver_rewards_card_005.md** - Business Silver Rewards Card: Exceptions and Exclusions
- **doc_business_credit_cards_business_silver_rewards_card_006.md** - Business Silver Rewards Card: Understanding Your Included Travel Insurance Coverage
- **doc_business_credit_cards_business_silver_rewards_card_007.md** - Business Silver Rewards Card: How to File a Travel Insurance Claim
- **doc_business_credit_cards_business_silver_rewards_card_008.md** - Business Silver Rewards Card: Purchase Protection Benefits Explained
- **doc_business_credit_cards_business_silver_rewards_card_009.md** - Business Silver Rewards Card: Annual Fee and Fee Waiver Options
- **doc_business_credit_cards_business_silver_rewards_card_010.md** - Business Silver Rewards Card: Maximizing Rewards on Business Travel and Tech Expenses
- **doc_business_credit_cards_business_silver_rewards_card_011.md** - Business Silver Rewards Card: New Customer First Year Fee Waiver
- **doc_business_credit_cards_business_silver_rewards_card_012.md** - Business Silver Rewards Card: Double Cash Back Promo
- **doc_business_credit_cards_business_silver_rewards_card_013.md** - Business Silver Rewards Card: Referral Program - Earn Rewards for Referring Businesses
- **doc_business_credit_cards_green_rewards_card_001.md** - Green Rewards Card: Apply for a Sustainable Business Credit Card
- **doc_business_credit_cards_green_rewards_card_002.md** - Green Rewards Card: How the Sustainability Tracker Works
- **doc_business_credit_cards_green_rewards_card_003.md** - Green Rewards Card: Understanding Your Carbon Footprint Report
- **doc_business_credit_cards_green_rewards_card_004.md** - Green Rewards Card: Earning Rewards on Eco-Friendly Purchases
- **doc_business_credit_cards_green_rewards_card_005.md** - Green Rewards Card: Which Merchants Are Considered Sustainable?
- **doc_business_credit_cards_green_rewards_card_006.md** - Green Rewards Card: Carbon Offset Programs and Options
- **doc_business_credit_cards_green_rewards_card_007.md** - Green Rewards Card: Viewing Your Monthly Environmental Impact Dashboard
- **doc_business_credit_cards_green_rewards_card_008.md** - Green Rewards Card: Carbon Reporting for Business Accounting
- **doc_business_credit_cards_green_rewards_card_009.md** - Green Rewards Card: Setting Sustainability Goals for Your Business
- **doc_business_credit_cards_green_rewards_card_010.md** - Green Rewards Card: Referral Program - Earn Rewards for Referring Businesses
- **doc_business_credit_cards_silver_zoom_card_001.md** - Silver Zoom Card: Designed for Transportation and Logistics Businesses
- **doc_business_credit_cards_silver_zoom_card_002.md** - Silver Zoom Card: How to Get Fuel Discounts at Participating Stations
- **doc_business_credit_cards_silver_zoom_card_003.md** - Silver Zoom Card: Which Gas Stations and Fuel Networks Are Included?
- **doc_business_credit_cards_silver_zoom_card_004.md** - Silver Zoom Card: Tracking and Maximizing Your Fuel Savings
- **doc_business_credit_cards_silver_zoom_card_005.md** - Silver Zoom Card: Adding Fleet Drivers to Your Account
- **doc_business_credit_cards_silver_zoom_card_006.md** - Silver Zoom Card: Setting Spending Limits for Driver Cards
- **doc_business_credit_cards_silver_zoom_card_007.md** - Silver Zoom Card: Fuel Purchase Reporting and Analytics
- **doc_business_credit_cards_silver_zoom_card_008.md** - Silver Zoom Card: Eligibility Requirements for Logistics Businesses
- **doc_business_credit_cards_silver_zoom_card_009.md** - Silver Zoom Card: Managing Multiple Vehicle Expenses
- **doc_business_credit_cards_silver_zoom_card_010.md** - Silver Zoom Card: New Customer Promo - 0% APR on Fuel Purchases
- **doc_business_credit_cards_silver_zoom_card_011.md** - Silver Zoom Card: Referral Program - Earn Rewards for Referring Logistics Businesses
- **doc_business_credit_cards_silver_zoom_card_012.md** - Silver Zoom Card: Triple Cash Back Promo - Terms and Conditions
- **doc_business_credit_cards_silver_zoom_card_013.md** - Silver Zoom Card: How to Claim Your Triple Cash Back Promo
- **doc_business_credit_cards_virtual_card_management_001.md** - Virtual Card Management: Creating Instant Virtual Cards for Your Team
- **doc_business_credit_cards_virtual_card_management_002.md** - Virtual Card Management: How to Issue a Virtual Card in Seconds
- **doc_business_credit_cards_virtual_card_management_003.md** - Virtual Card Management: Revoking and Canceling Virtual Cards
- **doc_business_credit_cards_virtual_card_management_004.md** - Virtual Card Management: Setting Spending Limits Per Card
- **doc_business_credit_cards_virtual_card_management_005.md** - Virtual Card Management: Team-Based Spending Controls and Limits
- **doc_business_credit_cards_virtual_card_management_006.md** - Virtual Card Management: Tracking Virtual Card Purchases by Department
- **doc_business_credit_cards_virtual_card_management_007.md** - Virtual Card Management: Security Benefits of Virtual Cards
- **doc_business_credit_cards_virtual_card_management_008.md** - Virtual Card Management: Single-Use vs Multi-Use Virtual Cards
- **doc_business_credit_cards_virtual_card_management_009.md** - Virtual Card Management: Best Practices for Subscription and Vendor Payments
- **doc_business_credit_cards_virtual_card_management_010.md** - Virtual Card Management: Integrating Virtual Cards with Expense Management Tools
- **doc_business_savings_accounts_automatic_sweep_program_001.md** - Automatic Sweep: How Nightly Fund Transfers Work
- **doc_business_savings_accounts_automatic_sweep_program_002.md** - Automatic Sweep: Setting Up Your Sweep Program
- **doc_business_savings_accounts_automatic_sweep_program_003.md** - Automatic Sweep: Defining Excess Fund Thresholds
- **doc_business_savings_accounts_automatic_sweep_program_004.md** - Automatic Sweep: Which Accounts Can Use Automatic Sweeps?
- **doc_business_savings_accounts_automatic_sweep_program_005.md** - Automatic Sweep: Timing and Processing of Nightly Transfers
- **doc_business_savings_accounts_automatic_sweep_program_006.md** - Automatic Sweep: Pausing or Disabling Your Sweep Program
- **doc_business_savings_accounts_automatic_sweep_program_007.md** - Automatic Sweep: Optimizing Cash Flow with Smart Sweeps
- **doc_business_savings_accounts_automatic_sweep_program_008.md** - Automatic Sweep: Minimum Balance Requirements and Limits
- **doc_business_savings_accounts_bronze_saver_account_001.md** - Business Bronze Saver: Complete Guide
- **doc_business_savings_accounts_bronze_saver_account_002.md** - Bronze Saver Account specifications and requirements
- **doc_business_savings_accounts_bronze_saver_account_003.md** - Why am I being charged a fee on my Bronze Saver Account?
- **doc_business_savings_accounts_bronze_saver_account_004.md** - Bronze Saver Account: Interest payments and compounding
- **doc_business_savings_accounts_bronze_saver_account_005.md** - Bronze Saver Account: Mobile banking and deposit limits
- **doc_business_savings_accounts_bronze_saver_account_006.md** - Transferring funds to external banks: Bronze Saver
- **doc_business_savings_accounts_bronze_saver_account_007.md** - Bronze Saver Account: Adding authorized users to your account
- **doc_business_savings_accounts_bronze_saver_account_008.md** - Bronze Saver Account: Withdrawal limits explained
- **doc_business_savings_accounts_diamond_vault_001.md** - Diamond Vault Account: Complete Guide
- **doc_business_savings_accounts_diamond_vault_002.md** - Diamond Vault Account: Private banking services
- **doc_business_savings_accounts_diamond_vault_003.md** - Diamond Vault Account: Wire transfers and fees
- **doc_business_savings_accounts_diamond_vault_004.md** - Diamond Vault Account: Exclusive events and benefits
- **doc_business_savings_accounts_diamond_vault_005.md** - Diamond Vault Account: Mobile banking and deposit limits
- **doc_business_savings_accounts_diamond_vault_006.md** - Diamond Vault Account: Same-day transfers
- **doc_business_savings_accounts_diamond_vault_007.md** - Diamond Vault Account: Adding authorized users
- **doc_business_savings_accounts_diamond_vault_008.md** - Diamond Vault Account: Business Credit Card APY Bonuses
- **doc_business_savings_accounts_emerald_saver_001.md** - Emerald Saver Account: Complete Guide
- **doc_business_savings_accounts_emerald_saver_002.md** - Emerald Saver Account: How your savings help the environment
- **doc_business_savings_accounts_emerald_saver_003.md** - Emerald Saver vs Green Account (savings): What is the difference?
- **doc_business_savings_accounts_emerald_saver_004.md** - Emerald Saver Account: Boosting your APY with eco-partner products
- **doc_business_savings_accounts_emerald_saver_005.md** - Emerald Saver Account: Mobile banking and deposit limits
- **doc_business_savings_accounts_emerald_saver_006.md** - Emerald Saver Account: Transferring funds to external banks
- **doc_business_savings_accounts_emerald_saver_007.md** - Emerald Saver Account: Adding authorized users
- **doc_business_savings_accounts_emerald_saver_008.md** - Emerald Saver Account: Business Credit Card APY Bonuses
- **doc_business_savings_accounts_gold_plus_saver_001.md** - Gold Plus Saver Account: Complete Guide
- **doc_business_savings_accounts_gold_plus_saver_002.md** - Gold Plus Saver Account: Investment sweep integration
- **doc_business_savings_accounts_gold_plus_saver_003.md** - Gold Plus Saver Account: Priority support and quarterly reviews
- **doc_business_savings_accounts_gold_plus_saver_004.md** - Gold Plus Saver Account: Withdrawal limits
- **doc_business_savings_accounts_gold_plus_saver_005.md** - Gold Plus Saver Account: Mobile banking and deposit limits
- **doc_business_savings_accounts_gold_plus_saver_006.md** - Gold Plus Saver Account: Transferring funds to external banks
- **doc_business_savings_accounts_gold_plus_saver_007.md** - Gold Plus Saver Account: Same-day ACH transfers
- **doc_business_savings_accounts_gold_saver_account_001.md** - Gold Saver Account: Understanding Tiered Interest Rates up to 3.25%
- **doc_business_savings_accounts_gold_saver_account_002.md** - Gold Saver Account: How Automated Sweeps Work with Business Checking
- **doc_business_savings_accounts_gold_saver_account_003.md** - Gold Saver Account: Optimizing Your Balance for Maximum Interest
- **doc_business_savings_accounts_gold_saver_account_004.md** - Gold Saver Account: Integration with Business Checking
- **doc_business_savings_accounts_gold_saver_account_005.md** - Gold Saver Account: Mobile banking and deposit limits
- **doc_business_savings_accounts_gold_saver_account_006.md** - Gold Saver Account: Same-day ACH and wire transfers
- **doc_business_savings_accounts_gold_saver_account_007.md** - Gold Saver Account: Transferring funds to external banks
- **doc_business_savings_accounts_gold_saver_account_008.md** - Gold Saver Account: Adding authorized users to your account
- **doc_business_savings_accounts_gold_saver_account_009.md** - Gold Saver Account: Business Credit Card APY Bonuses
- **doc_business_savings_accounts_joint_business_holders___user_roles_001.md** - Joint Business Holders: Adding Multiple Account Owners
- **doc_business_savings_accounts_joint_business_holders___user_roles_002.md** - User Roles: Understanding Admin, Finance, and Auditor Access Levels
- **doc_business_savings_accounts_joint_business_holders___user_roles_003.md** - User Roles: Setting Up and Managing User Permissions
- **doc_business_savings_accounts_joint_business_holders___user_roles_004.md** - User Roles: What Can Each Role Access and Modify?
- **doc_business_savings_accounts_joint_business_holders___user_roles_005.md** - Joint Business Holders: Legal Requirements and Documentation
- **doc_business_savings_accounts_joint_business_holders___user_roles_006.md** - User Roles: Removing or Changing User Access
- **doc_business_savings_accounts_joint_business_holders___user_roles_007.md** - Joint Business Holders: Rights and Responsibilities of Each Holder
- **doc_business_savings_accounts_joint_business_holders___user_roles_008.md** - User Roles: Best Practices for Access Management
- **doc_business_savings_accounts_joint_business_holders___user_roles_009.md** - User Roles: Audit Trail and Activity Monitoring by Role
- **doc_business_savings_accounts_platinum_reserve_account_001.md** - Platinum Reserve Account: Enterprise Business Savings Overview
- **doc_business_savings_accounts_platinum_reserve_account_002.md** - Platinum Reserve Account: Daily Interest Compounding Explained
- **doc_business_savings_accounts_platinum_reserve_account_003.md** - Platinum Reserve Account: Using the Liquidity Reporting Dashboard
- **doc_business_savings_accounts_platinum_reserve_account_004.md** - Platinum Reserve Account: Eligibility Requirements for Enterprises
- **doc_business_savings_accounts_platinum_reserve_account_005.md** - Platinum Reserve Account: Advanced Cash Flow Analytics and Reporting
- **doc_business_savings_accounts_platinum_reserve_account_006.md** - Platinum Reserve Account: Integration with Enterprise Treasury Tools
- **doc_business_savings_accounts_platinum_reserve_account_007.md** - Platinum Reserve Account: Same-day transfers and wire capabilities
- **doc_business_savings_accounts_platinum_reserve_account_008.md** - Platinum Reserve Account: Mobile banking and deposit limits
- **doc_business_savings_accounts_platinum_reserve_account_009.md** - Platinum Reserve Account: Managing authorized users
- **doc_business_savings_accounts_platinum_reserve_account_010.md** - Platinum Reserve Account: Business Credit Card APY Bonuses
- **doc_business_savings_accounts_silver_plus_saver_001.md** - Silver Plus Saver Account: Complete Guide
- **doc_business_savings_accounts_silver_plus_saver_002.md** - Silver Plus Saver vs Silver Saver Account: What is the difference?
- **doc_business_savings_accounts_silver_plus_saver_003.md** - Silver Plus Saver Account: Relationship bonuses explained
- **doc_business_savings_accounts_silver_plus_saver_004.md** - Silver Plus Saver Account: Mobile banking and deposit limits
- **doc_business_savings_accounts_silver_plus_saver_005.md** - Silver Plus Saver Account: Transferring funds to external banks
- **doc_business_savings_accounts_silver_plus_saver_006.md** - Silver Plus Saver Account: Same-day ACH and interest compounding
- **doc_business_savings_accounts_silver_plus_saver_007.md** - Silver Plus Saver Account: Adding authorized users
- **doc_business_savings_accounts_silver_saver_account_001.md** - Silver Saver Account: Higher Interest Rates for Larger Balances
- **doc_business_savings_accounts_silver_saver_account_002.md** - Silver Saver Account: Understanding the $25K Balance Threshold
- **doc_business_savings_accounts_silver_saver_account_003.md** - Silver Saver Account: What Happens If I Exceed Withdrawal Limits?
- **doc_business_savings_accounts_silver_saver_account_004.md** - Silver Saver Account: Maximizing Your Interest Earnings
- **doc_business_savings_accounts_silver_saver_account_005.md** - Silver Saver Account: Upgrading from Bronze Saver
- **doc_business_savings_accounts_silver_saver_account_006.md** - Silver Saver Account: Mobile banking and deposit limits
- **doc_business_savings_accounts_silver_saver_account_007.md** - Silver Saver Account: Transferring funds to external banks
- **doc_business_savings_accounts_silver_saver_account_008.md** - Silver Saver Account: Adding authorized users
- **doc_buy_now_pay_later_bnpl_bronze_001.md** - BNPL Bronze: How to Activate and Use Your Installment Plan
- **doc_buy_now_pay_later_bnpl_bronze_002.md** - BNPL Bronze: Understanding Your $500 Credit Limit and Minimum Purchase Amount
- **doc_buy_now_pay_later_bnpl_bronze_003.md** - BNPL Bronze: 4 Installments Over 6 Weeks Explained
- **doc_buy_now_pay_later_bnpl_bronze_004.md** - BNPL Bronze: Interest-Free Payments and Upfront Charge Percentage
- **doc_buy_now_pay_later_bnpl_bronze_005.md** - BNPL Bronze: Late Fees, Maximum Late Charges, and How to Avoid Them
- **doc_buy_now_pay_later_bnpl_bronze_006.md** - BNPL Bronze: Eligible Purchases and Allowed Shopping Channels
- **doc_buy_now_pay_later_bnpl_bronze_007.md** - BNPL Bronze: Managing Multiple Active Plans on Your Account
- **doc_buy_now_pay_later_bnpl_bronze_008.md** - BNPL Bronze: Product Details, Terms, and Key Limits in USD
- **doc_buy_now_pay_later_bnpl_diamond_001.md** - BNPL Diamond: Eligibility Segments and Minimum Spend Requirements Explained
- **doc_buy_now_pay_later_bnpl_diamond_002.md** - BNPL Diamond: Enterprise Revenue Criteria and High-Limit Qualification
- **doc_buy_now_pay_later_bnpl_diamond_003.md** - BNPL Diamond: Understanding Your Maximum BNPL Limit and Concurrent Plans
- **doc_buy_now_pay_later_bnpl_diamond_004.md** - BNPL Diamond: 12-Month Standard Term and Flexible Repayment Options
- **doc_buy_now_pay_later_bnpl_diamond_005.md** - BNPL Diamond: APR, Introductory Interest-Free Period, and Late Payment Fees
- **doc_buy_now_pay_later_bnpl_diamond_006.md** - BNPL Diamond: Integrating with Accounting Tools and Supported Platforms
- **doc_buy_now_pay_later_bnpl_diamond_007.md** - BNPL Diamond: Concierge Dispute Support, Response Times, and What to Expect
- **doc_buy_now_pay_later_bnpl_diamond_008.md** - BNPL Diamond: How to Apply, Get Approved, and Start Using Your Plan
- **doc_buy_now_pay_later_bnpl_gold_001.md** - BNPL Gold: How to Apply and Get Started
- **doc_buy_now_pay_later_bnpl_gold_002.md** - BNPL Gold: Credit Limits and Eligibility Explained
- **doc_buy_now_pay_later_bnpl_gold_003.md** - BNPL Gold: Understanding Your 0% Intro APR and Standard APR
- **doc_buy_now_pay_later_bnpl_gold_004.md** - BNPL Gold: Using Auto-Debit and Setting Your Payment Date
- **doc_buy_now_pay_later_bnpl_gold_005.md** - BNPL Gold: Transaction Size Limits and Purchase Eligibility
- **doc_buy_now_pay_later_bnpl_gold_006.md** - BNPL Gold: Repayment Terms, Monthly Installments, and Plan Limits
- **doc_buy_now_pay_later_bnpl_gold_007.md** - BNPL Gold: Late Fees, Minimum Payments, and Avoiding Penalties
- **doc_buy_now_pay_later_bnpl_gold_008.md** - BNPL Gold: Business Eligibility and Using BNPL for Your Company
- **doc_buy_now_pay_later_bnpl_management_dashboard_001.md** - BNPL Management Dashboard: Viewing Your Total Active BNPL Plans and Balances
- **doc_buy_now_pay_later_bnpl_management_dashboard_002.md** - BNPL Management Dashboard: Tracking Overdue Plans and Due Dates
- **doc_buy_now_pay_later_bnpl_management_dashboard_003.md** - BNPL Management Dashboard: Using Date Range Filters to See Upcoming Payments
- **doc_buy_now_pay_later_bnpl_management_dashboard_004.md** - BNPL Management Dashboard: Including or Excluding Closed BNPL Plans
- **doc_buy_now_pay_later_bnpl_management_dashboard_005.md** - BNPL Management Dashboard: How Early-Payment Rewards Work and When They Apply
- **doc_buy_now_pay_later_bnpl_management_dashboard_006.md** - BNPL Management Dashboard: Understanding Reward Types, Rates, and Maximum Amounts
- **doc_buy_now_pay_later_bnpl_management_dashboard_007.md** - BNPL Management Dashboard: Viewing Projected Early-Payment Rewards
- **doc_buy_now_pay_later_bnpl_management_dashboard_008.md** - BNPL Management Dashboard: Sorting and Highlighting Plans for Better Tracking
- **doc_buy_now_pay_later_bnpl_platinum_001.md** - BNPL Platinum: How to Activate 6-Month Financing on Your Account
- **doc_buy_now_pay_later_bnpl_platinum_002.md** - BNPL Platinum: Understanding Your 6-Month Financing Term
- **doc_buy_now_pay_later_bnpl_platinum_003.md** - BNPL Platinum: How Your Variable APR Is Calculated (Indexes and Margins Explained)
- **doc_buy_now_pay_later_bnpl_platinum_004.md** - BNPL Platinum: Minimum and Maximum Purchase Amounts for Financing
- **doc_buy_now_pay_later_bnpl_platinum_005.md** - BNPL Platinum: Purchase Protection Coverage, Limits, and Eligibility Window
- **doc_buy_now_pay_later_bnpl_platinum_006.md** - BNPL Platinum: Late Fees, Due Dates, and How to Avoid Charges
- **doc_buy_now_pay_later_bnpl_platinum_007.md** - BNPL Platinum: Prepayment Rules, Penalties, and Paying Off Early
- **doc_buy_now_pay_later_bnpl_platinum_008.md** - BNPL Platinum: Determining Your APR Range Based on Credit Tier
- **doc_buy_now_pay_later_bnpl_silver_001.md** - BNPL Silver: How to Enable Buy Now, Pay Later on Your Account
- **doc_buy_now_pay_later_bnpl_silver_002.md** - BNPL Silver: Eligible Purchase Amounts and Transaction Limits
- **doc_buy_now_pay_later_bnpl_silver_003.md** - BNPL Silver: Choosing Your Installment Term and Payment Frequency
- **doc_buy_now_pay_later_bnpl_silver_004.md** - BNPL Silver: Interest, Processing Fees, and Total Cost Explained
- **doc_buy_now_pay_later_bnpl_silver_005.md** - BNPL Silver: Soft Credit Checks, Providers, and Inquiry Limits
- **doc_buy_now_pay_later_bnpl_silver_006.md** - BNPL Silver: Dynamic Repayment Reminders, Timing, and Channels
- **doc_buy_now_pay_later_bnpl_silver_007.md** - BNPL Silver: Late Payment Fees and What Happens If You Miss a Payment
- **doc_buy_now_pay_later_bnpl_silver_008.md** - BNPL Silver: Getting Started, Eligibility, and Common Questions
- **doc_checking_accounts_blue_account_001.md** - Blue Account at a glance
- **doc_checking_accounts_blue_account_002.md** - Blue Account specifications and requirements
- **doc_checking_accounts_blue_account_003.md** - Blue Account features and fees
- **doc_checking_accounts_blue_account_004.md** - Setting up overdraft protection for your Blue Account
- **doc_checking_accounts_blue_account_005.md** - Understanding Blue Account overdraft fees
- **doc_checking_accounts_blue_account_006.md** - Blue Account: How overdraft transfer fees work
- **doc_checking_accounts_blue_account_007.md** - Blue Account: Managing your balance with low-balance alerts
- **doc_checking_accounts_blue_account_008.md** - Why am I being charged an overdraft fee on my Blue Account?
- **doc_checking_accounts_blue_account_009.md** - How do I avoid overdraft fees on my Blue Account?
- **doc_checking_accounts_blue_account_010.md** - FAQ: Blue Account
- **doc_checking_accounts_blue_account_011.md** - Transferring funds to another bank
- **doc_checking_accounts_blue_account_012.md** - International ATM withdrawals for Blue Account
- **doc_checking_accounts_bluest_account_001.md** - Bluest Account at a glance
- **doc_checking_accounts_bluest_account_002.md** - Bluest Account specifications and requirements
- **doc_checking_accounts_bluest_account_003.md** - Bluest Account exclusive benefits
- **doc_checking_accounts_bluest_account_004.md** - Your dedicated personal banker
- **doc_checking_accounts_bluest_account_005.md** - Wire transfers for Bluest members
- **doc_checking_accounts_bluest_account_006.md** - What happens if my balance falls below the minimum?
- **doc_checking_accounts_bluest_account_007.md** - ATM fee rebates for Bluest
- **doc_checking_accounts_bluest_account_008.md** - Bluest vs. other checking accounts
- **doc_checking_accounts_bluest_account_009.md** - FAQ: Bluest Account
- **doc_checking_accounts_bluest_account_010.md** - Bluest Account: Out-of-network ATM fees
- **doc_checking_accounts_checking_accounts__general__001.md** - Welcome to Rho-Bank Checking
- **doc_checking_accounts_checking_accounts__general__002.md** - FAQ: Troubleshooting
- **doc_checking_accounts_checking_accounts__general__003.md** - FAQ: Common Error Codes
- **doc_checking_accounts_checking_accounts__general__004.md** - What is an authorization hold?
- **doc_checking_accounts_checking_accounts__general__005.md** - Direct deposits — setup and timing
- **doc_checking_accounts_checking_accounts__general__006.md** - Why does my account have a lower balance than I expected or have a negative balance?
- **doc_checking_accounts_checking_accounts__general__007.md** - I didn't receive my direct deposit
- **doc_checking_accounts_checking_accounts__general__008.md** - Why hasn't my direct deposit arrived yet?
- **doc_checking_accounts_checking_accounts__general__009.md** - Internal: Direct Deposit Delay Inquiries Protocol
- **doc_checking_accounts_dark_green_account_001.md** - Dark Green Account at a glance
- **doc_checking_accounts_dark_green_account_002.md** - Dark Green Account specifications and requirements
- **doc_checking_accounts_dark_green_account_003.md** - Verifying your student enrollment
- **doc_checking_accounts_dark_green_account_004.md** - Earning rewards for good grades
- **doc_checking_accounts_dark_green_account_005.md** - Student loan payment benefits
- **doc_checking_accounts_dark_green_account_006.md** - Partner discounts for students
- **doc_checking_accounts_dark_green_account_007.md** - Graduating from your Dark Green Account
- **doc_checking_accounts_dark_green_account_008.md** - Why was my Dark Green Account enrollment verification rejected?
- **doc_checking_accounts_dark_green_account_009.md** - FAQ: Dark Green Account
- **doc_checking_accounts_dark_green_account_010.md** - Dark Green Account Referral Program
- **doc_checking_accounts_evergreen_account_001.md** - Evergreen Account at a glance
- **doc_checking_accounts_evergreen_account_002.md** - Evergreen Account specifications and requirements
- **doc_checking_accounts_evergreen_account_003.md** - How the tree-planting program works
- **doc_checking_accounts_evergreen_account_004.md** - Tracking your environmental impact
- **doc_checking_accounts_evergreen_account_005.md** - How Rho calculates carbon offsets
- **doc_checking_accounts_evergreen_account_006.md** - Eligibility and fees
- **doc_checking_accounts_evergreen_account_007.md** - FAQ: Evergreen Account
- **doc_checking_accounts_evergreen_account_008.md** - Out-of-network ATM fees for Evergreen Account
- **doc_checking_accounts_gold_years_account_001.md** - Gold Years Account at a glance
- **doc_checking_accounts_gold_years_account_002.md** - Gold Years Account specifications and requirements
- **doc_checking_accounts_gold_years_account_003.md** - Free checks and paper statements for seniors
- **doc_checking_accounts_gold_years_account_004.md** - Prescription discount program
- **doc_checking_accounts_gold_years_account_005.md** - Dedicated senior support services
- **doc_checking_accounts_gold_years_account_006.md** - Social Security direct deposit setup
- **doc_checking_accounts_gold_years_account_007.md** - Estate planning partner services
- **doc_checking_accounts_gold_years_account_008.md** - Why can't I open a Gold Years Account?
- **doc_checking_accounts_gold_years_account_009.md** - FAQ: Gold Years Account
- **doc_checking_accounts_gold_years_account_010.md** - Gold Years Account Referral Program
- **doc_checking_accounts_green_account__checking__001.md** - Green Account (checking) at a glance
- **doc_checking_accounts_green_account__checking__002.md** - Green Account (checking) specifications and requirements
- **doc_checking_accounts_green_account__checking__003.md** - No-overdraft checking explained
- **doc_checking_accounts_green_account__checking__004.md** - How Green Account (checking) prevents overdrafts
- **doc_checking_accounts_green_account__checking__005.md** - Managing your spending limit in a Green Account (checking)
- **doc_checking_accounts_green_account__checking__006.md** - Green Account (checking): What happens if a payment exceeds your balance?
- **doc_checking_accounts_green_account__checking__007.md** - Green Account (checking): Troubleshooting: declined transactions
- **doc_checking_accounts_green_account__checking__008.md** - Why was my Green Account (checking) transaction declined?
- **doc_checking_accounts_green_account__checking__009.md** - Why can't I overdraft my Green Account (checking)?
- **doc_checking_accounts_green_account__checking__010.md** - FAQ: Green Account (checking)
- **doc_checking_accounts_green_account__checking__011.md** - Transferring funds to another bank
- **doc_checking_accounts_green_account__checking__012.md** - Foreign ATM fees for Green Account (checking)
- **doc_checking_accounts_green_fee-free_account_001.md** - Green Fee-Free Account: Complete Guide and Table of Contents
- **doc_checking_accounts_green_fee-free_account_002.md** - Green Fee-Free Account — summary
- **doc_checking_accounts_green_fee-free_account_003.md** - Green Fee-Free Account at a glance
- **doc_checking_accounts_green_fee-free_account_004.md** - Green Fee-Free Account specifications and requirements
- **doc_checking_accounts_green_fee-free_account_005.md** - Green Fee-Free Account features
- **doc_checking_accounts_green_fee-free_account_006.md** - FAQ: Green Fee-Free Account
- **doc_checking_accounts_green_fee-free_account_007.md** - Green Fee-Free Account Referral Program
- **doc_checking_accounts_light_blue_account_001.md** - Light Blue Account — summary
- **doc_checking_accounts_light_blue_account_002.md** - Light Blue Account at a glance
- **doc_checking_accounts_light_blue_account_003.md** - Light Blue Account specifications and requirements
- **doc_checking_accounts_light_blue_account_004.md** - Light Blue Account features and fees
- **doc_checking_accounts_light_blue_account_005.md** - FAQ: Light Blue Account
- **doc_checking_accounts_light_blue_account_006.md** - Foreign ATM withdrawal fees for Light Blue Account
- **doc_checking_accounts_light_blue_account_007.md** - Light Blue Account Referral Program
- **doc_checking_accounts_light_green_account_001.md** - Light Green Account at a glance
- **doc_checking_accounts_light_green_account_002.md** - Light Green Account specifications and requirements
- **doc_checking_accounts_light_green_account_003.md** - Who can open a Light Green Account?
- **doc_checking_accounts_light_green_account_004.md** - Parent or guardian requirements
- **doc_checking_accounts_light_green_account_005.md** - Spending limits and safety features for minors
- **doc_checking_accounts_light_green_account_006.md** - Teaching teens about saving and budgeting
- **doc_checking_accounts_light_green_account_007.md** - Adding funds to your teen's account
- **doc_checking_accounts_light_green_account_008.md** - What happens when the primary account holder turns 25
- **doc_checking_accounts_light_green_account_009.md** - Monitoring transactions in a Light Green Account
- **doc_checking_accounts_light_green_account_010.md** - How to transition a Light Green Account to Blue or Green Account (checking)
- **doc_checking_accounts_light_green_account_011.md** - How do I remove spending limits on my teen's account?
- **doc_checking_accounts_light_green_account_012.md** - FAQ: Teen Checking Accounts
- **doc_checking_accounts_light_green_account_013.md** - Understanding foreign ATM fees for teens
- **doc_checking_accounts_purple_account_001.md** - Purple Account at a glance
- **doc_checking_accounts_purple_account_002.md** - Purple Account specifications and requirements
- **doc_checking_accounts_purple_account_003.md** - No foreign transaction fees explained
- **doc_checking_accounts_purple_account_004.md** - Using ATMs worldwide
- **doc_checking_accounts_purple_account_005.md** - Multi-currency wallet features
- **doc_checking_accounts_purple_account_006.md** - Travel insurance benefits
- **doc_checking_accounts_purple_account_007.md** - Emergency card replacement while traveling
- **doc_checking_accounts_purple_account_008.md** - Airport lounge access
- **doc_checking_accounts_purple_account_009.md** - Currency conversion rates and timing
- **doc_checking_accounts_purple_account_010.md** - Why was my ATM fee not rebated?
- **doc_checking_accounts_purple_account_011.md** - FAQ: Purple Account
- **doc_checking_accounts_purple_account_012.md** - Out-of-network ATM fees for Purple Account
- **doc_credit_cards_bronze_rewards_card_001.md** - Bronze Rewards Card: How to Apply and Get Started
- **doc_credit_cards_bronze_rewards_card_002.md** - Bronze Rewards Card: Understanding Your 1% Cash Back Rewards
- **doc_credit_cards_bronze_rewards_card_003.md** - Bronze Rewards Card: When and How Are Cash Back Rewards Paid?
- **doc_credit_cards_bronze_rewards_card_004.md** - Bronze Rewards Card: Monthly Reporting Tools and Transaction Tracking
- **doc_credit_cards_bronze_rewards_card_005.md** - Bronze Rewards Card: Making Payments and Managing Your Balance
- **doc_credit_cards_bronze_rewards_card_006.md** - Bronze Rewards Card: Credit Limit and Spending Guidelines
- **doc_credit_cards_bronze_rewards_card_007.md** - Bronze Rewards Card: Redeeming Your Rewards and Cash Back Options
- **doc_credit_cards_bronze_rewards_card_008.md** - Bronze Rewards Card: New Customer Promo - 0% APR for the First Year
- **doc_credit_cards_credit_card_account_logistics_001.md** - How can I close a credit card account?
- **doc_credit_cards_credit_card_account_logistics_002.md** - Internal: Processing Credit Card Account Closures
- **doc_credit_cards_credit_card_account_logistics_003.md** - Internal: Credit Card Retention Protocol
- **doc_credit_cards_credit_card_account_logistics_004.md** - How can I request a CLI?
- **doc_credit_cards_credit_card_account_logistics_005.md** - CLI Eligibility Requirements by Card Tier
- **doc_credit_cards_credit_card_account_logistics_006.md** - Internal: CLI Payment History and Approval Criteria
- **doc_credit_cards_credit_card_account_logistics_007.md** - Internal: Processing CLI Approvals and Denials
- **doc_credit_cards_credit_card_account_logistics_008.md** - Internal: Downgrading a Credit Card to a No-Annual-Fee Card
- **doc_credit_cards_credit_card_account_logistics_009.md** - Paying Off Credit Card Balance from Checking Account (Internal)
- **doc_credit_cards_credit_card_replacements_001.md** - How to Order a Replacement Credit Card (Internal)
- **doc_credit_cards_credit_card_replacements_002.md** - Credit Card Replacement Shipping Options and Fees
- **doc_credit_cards_credit_card_replacements_003.md** - What Happens When You Order a Replacement Card
- **doc_credit_cards_credit_card_replacements_004.md** - Why was your credit card replacement request rejected
- **doc_credit_cards_credit_card_replacements_005.md** - Checking Pending Replacement Card Orders (Internal)
- **doc_credit_cards_credit_cards__general__001.md** - Internal: Understanding Credit Card Referral Statuses.
- **doc_credit_cards_credit_cards__general__002.md** - Credit Card Referral Offers and Restrictions
- **doc_credit_cards_credit_cards__general__003.md** - Submitting a Cash Back Dispute (Internal)
- **doc_credit_cards_credit_cards__general__004.md** - Applying Resolved Cash Back Dispute Corrections (Internal)
- **doc_credit_cards_credit_cards__general__005.md** - Credit Card Rewards Categories
- **doc_credit_cards_credit_cards__general__006.md** - Internal: Credit Card Rewards - Points vs Cash Back Representation
- **doc_credit_cards_credit_cards__general__007.md** - Credit Card Rewards Points Rounding Policy
- **doc_credit_cards_credit_cards__general__008.md** - Credit Card Rewards Redemption Options
- **doc_credit_cards_credit_cards__general__009.md** - Generating a Credit Card Referral Link (Internal)
- **doc_credit_cards_credit_cards__general__010.md** - Internal: Credit Card Purchase Decline Diagnostic Procedure
- **doc_credit_cards_credit_cards__general__011.md** - Internal: Backend Incident 11/13
- **doc_credit_cards_credit_cards__general__012.md** - Internal: Backend Incident 11/14 - Erroneous Credit Bureau Reporting
- **doc_credit_cards_credit_cards__general__013.md** - How to Find Your Rho-Bank Credit Card Numbers Online
- **doc_credit_cards_credit_cards__general__014.md** - Filing a Credit Card Transaction Dispute (Internal)
- **doc_credit_cards_credit_cards__general__015.md** - Provisional Credit Eligibility Guidelines (Internal)
- **doc_credit_cards_credit_cards__general__016.md** - Checking User Dispute History (Internal)
- **doc_credit_cards_credit_cards__general__017.md** - (Internal) Applying a Credit Card Statement Credit
- **doc_credit_cards_credit_cards__general__018.md** - How to Dispute a Credit Card Transaction
- **doc_credit_cards_credit_cards__general__019.md** - Understanding Your Credit Card Rewards Categories
- **doc_credit_cards_credit_cards__general__020.md** - Refer a Friend to Rho-Bank and Earn Rewards
- **doc_credit_cards_credit_cards__general__021.md** - Think You're Missing Cash Back? Here's How to Get Help
- **doc_credit_cards_credit_cards__general__022.md** - What is Provisional Credit?
- **doc_credit_cards_credit_cards__general__023.md** - Having Trouble Making Purchases with Your Card?
- **doc_credit_cards_credit_cards__general__024.md** - Getting the Most from Your Credit Card Rewards
- **doc_credit_cards_credit_cards__general__025.md** - Request a Statement Credit or Fee Adjustment
- **doc_credit_cards_crypto-cash_back_001.md** - Crypto-Cash Back: Getting Started with Crypto Rewards
- **doc_credit_cards_crypto-cash_back_002.md** - Crypto-Cash Back: How to Redeem Rewards in Your Crypto Wallet
- **doc_credit_cards_crypto-cash_back_003.md** - Crypto-Cash Back: Supported Cryptocurrencies and Wallets
- **doc_credit_cards_crypto-cash_back_004.md** - Crypto-Cash Back: Linking Your Crypto Wallet to Your Account
- **doc_credit_cards_crypto-cash_back_005.md** - Crypto-Cash Back: Understanding Crypto Reward Conversion Rates
- **doc_credit_cards_crypto-cash_back_006.md** - Crypto-Cash Back: Security and Safety of Crypto Redemptions
- **doc_credit_cards_crypto-cash_back_007.md** - Crypto-Cash Back: Tax Reporting for Cryptocurrency Rewards
- **doc_credit_cards_crypto-cash_back_008.md** - Crypto-Cash Back: Frequently Asked Questions About Crypto Rewards
- **doc_credit_cards_crypto-cash_back_009.md** - Crypto-Cash Back: FAQ-2
- **doc_credit_cards_diamond_elite_card_001.md** - Diamond Elite Card: Invitation-Only Membership - How to Qualify
- **doc_credit_cards_diamond_elite_card_002.md** - Diamond Elite Card: Getting Started with Your Elite Account
- **doc_credit_cards_diamond_elite_card_003.md** - Diamond Elite Card: Earning 5% Cash Back on All Purchases
- **doc_credit_cards_diamond_elite_card_004.md** - Diamond Elite Card: Maximizing Your 5% Rewards on Every Transaction
- **doc_credit_cards_diamond_elite_card_005.md** - Diamond Elite Card: Premium Credit Limits and Account Management
- **doc_credit_cards_diamond_elite_card_006.md** - Diamond Elite Card: Understanding Your Annual Fee and Elite Benefits
- **doc_credit_cards_diamond_elite_card_007.md** - Diamond Elite Card: Exclusive Benefits and 24/7 Concierge Services
- **doc_credit_cards_diamond_elite_card_008.md** - Diamond Elite Card: No Foreign Transaction Fees - Global Purchasing Power
- **doc_credit_cards_diamond_elite_card_009.md** - Diamond Elite Card: Invitation Promo Offer - Spend and Earn Rewards
- **doc_credit_cards_ecocard_001.md** - EcoCard: Apply for a Sustainable Rewards Credit Card
- **doc_credit_cards_ecocard_002.md** - EcoCard: How to Earn Sustainability Points on Green Purchases
- **doc_credit_cards_ecocard_003.md** - EcoCard: Which Merchants and Purchases Qualify as Green?
- **doc_credit_cards_ecocard_004.md** - EcoCard: Exceptions and Exclusions
- **doc_credit_cards_ecocard_005.md** - EcoCard: Redeeming Sustainability Points for Rewards
- **doc_credit_cards_ecocard_006.md** - EcoCard: Tracking Your Environmental Impact
- **doc_credit_cards_ecocard_007.md** - EcoCard: Partner Merchants and Eco-Friendly Shopping Guide
- **doc_credit_cards_ecocard_008.md** - EcoCard: Making Payments and Managing Your Account
- **doc_credit_cards_ecocard_009.md** - EcoCard: New Customer Promo - Earn Bonus Sustainability Points
- **doc_credit_cards_ecocard_010.md** - EcoCard: New Customer Promo - 0% APR for the First Year
- **doc_credit_cards_ecocard_011.md** - EcoCard: Referral Program - Earn Rewards for Referring Friends
- **doc_credit_cards_gold_rewards_card_001.md** - Gold Rewards Card: Apply for a Premium Credit Card
- **doc_credit_cards_gold_rewards_card_002.md** - Gold Rewards Card: Concierge Support Services - What's Included
- **doc_credit_cards_gold_rewards_card_003.md** - Gold Rewards Card: How to Contact Your Dedicated Concierge
- **doc_credit_cards_gold_rewards_card_004.md** - Gold Rewards Card: No Foreign Transaction Fees - International Purchases Made Easy
- **doc_credit_cards_gold_rewards_card_005.md** - Gold Rewards Card: Managing Your Account and Payment Options
- **doc_credit_cards_platinum_rewards_card_001.md** - Platinum Rewards Card: Getting Started with Your Platinum Account
- **doc_credit_cards_platinum_rewards_card_002.md** - Platinum Rewards Card: Earning 10% Cash Back
- **doc_credit_cards_platinum_rewards_card_003.md** - Platinum Rewards Card: Enhanced Fraud Protection Features Explained
- **doc_credit_cards_platinum_rewards_card_004.md** - Platinum Rewards Card: What to Do If You Suspect Fraudulent Activity
- **doc_credit_cards_platinum_rewards_card_005.md** - Platinum Rewards Card: Free Virtual Cards - How to Create and Manage
- **doc_credit_cards_platinum_rewards_card_006.md** - Platinum Rewards Card: Using Virtual Cards for Online Shopping and Subscriptions
- **doc_credit_cards_platinum_rewards_card_007.md** - Platinum Rewards Card: Maximizing Your Premium Benefits and Rewards
- **doc_credit_cards_platinum_rewards_card_008.md** - Platinum Rewards Card: Referral Program - Earn Rewards for Referring Friends
- **doc_credit_cards_platinum_rewards_card_009.md** - Platinum Rewards Card: New Customer Promo - First Year Annual Fee Waived
- **doc_credit_cards_platinum_rewards_card_010.md** - Platinum Rewards Card: Rebate Eligibility and Calcluation
- **doc_credit_cards_silver_rewards_card_001.md** - Silver Rewards Card: Getting Started and Eligibility Requirements
- **doc_credit_cards_silver_rewards_card_002.md** - Silver Rewards Card: How to Earn 4% Cash Back on Travel and Software Purchases
- **doc_credit_cards_silver_rewards_card_003.md** - Silver Rewards Card: Which Travel Purchases Qualify for 4% Cash Back?
- **doc_credit_cards_silver_rewards_card_004.md** - Silver Rewards Card: Software and SaaS Spending - Qualifying Categories
- **doc_credit_cards_silver_rewards_card_005.md** - Silver Rewards Card: Understanding Your Included Travel Insurance Coverage
- **doc_credit_cards_silver_rewards_card_006.md** - Silver Rewards Card: How to File a Travel Insurance Claim
- **doc_credit_cards_silver_rewards_card_007.md** - Silver Rewards Card: Making Payments and Managing Your Account
- **doc_credit_cards_silver_rewards_card_008.md** - Silver Rewards Card: Maximizing Rewards on Business Travel and Tech Expenses
- **doc_credit_cards_silver_rewards_card_009.md** - Silver Rewards Card: New Customer Promo - 0% APR for the First Year
- **doc_credit_cards_silver_rewards_card_010.md** - Silver Rewards Card: New Customer Promo - $500 Statement Credit Signup Bonus
- **doc_credit_cards_silver_rewards_card_011.md** - Silver Rewards Card: Referral Program - Earn Rewards for Referring Friends
- **doc_credit_cards_virtual_card_management_001.md** - Virtual Card Management: Getting Started with Virtual Cards
- **doc_credit_cards_virtual_card_management_002.md** - Virtual Card Management: How to Instantly Issue a Virtual Card
- **doc_credit_cards_virtual_card_management_003.md** - Virtual Card Management: Revoking and Canceling Virtual Cards
- **doc_credit_cards_virtual_card_management_004.md** - Virtual Card Management: Setting Up Team Spending Caps and Limits
- **doc_credit_cards_virtual_card_management_005.md** - Virtual Card Management: Creating One-Time Use Cards for Security
- **doc_credit_cards_virtual_card_management_006.md** - Virtual Card Management: Managing Multiple Virtual Cards for Your Team
- **doc_credit_cards_virtual_card_management_007.md** - Virtual Card Management: Best Practices for Virtual Card Security
- **doc_credit_cards_virtual_card_management_008.md** - Virtual Card Management: Tracking Virtual Card Transactions and Spending
- **doc_customer_support_special_support_codes_001.md** - Account Recovery Bypass Code (Internal)
- **doc_everyone_pay_everyone_pay_001.md** - Everyone Pay: Getting Started Guide
- **doc_everyone_pay_everyone_pay_002.md** - What is Everyone Pay?
- **doc_everyone_pay_everyone_pay_003.md** - How to send and receive money
- **doc_everyone_pay_everyone_pay_004.md** - Adding and verifying contacts
- **doc_everyone_pay_everyone_pay_005.md** - Sending limits — per day and month
- **doc_everyone_pay_everyone_pay_006.md** - Payment processing times
- **doc_everyone_pay_everyone_pay_007.md** - Canceling a payment
- **doc_everyone_pay_everyone_pay_008.md** - Disputing a payment
- **doc_everyone_pay_everyone_pay_009.md** - Blocking and unblocking users
- **doc_everyone_pay_everyone_pay_010.md** - Common security questions for Everyone Pay
- **doc_everyone_pay_everyone_pay_011.md** - Why can't I send money to this contact?
- **doc_everyone_pay_everyone_pay_012.md** - Why was my Everyone Pay transaction declined?
- **doc_everyone_pay_everyone_pay_013.md** - Why is my payment still pending?
- **doc_everyone_pay_everyone_pay_014.md** - Why can't I cancel my payment?
- **doc_everyone_pay_everyone_pay_015.md** - FAQ: Everyone Pay
- **doc_everyone_pay_qr_transfers_001.md** - QR Transfers: Complete Guide
- **doc_everyone_pay_qr_transfers_002.md** - How QR transfers work
- **doc_everyone_pay_qr_transfers_003.md** - Scanning and sending payments
- **doc_everyone_pay_qr_transfers_004.md** - Troubleshooting: QR not scanning
- **doc_everyone_pay_qr_transfers_005.md** - Why isn't my QR code scanning?
- **doc_everyone_pay_qr_transfers_006.md** - Why did my QR transfer fail?
- **doc_everyone_pay_qr_transfers_007.md** - Security tips for QR transfers
- **doc_everyone_pay_scheduled_payments_001.md** - Scheduled Payments: Complete Guide
- **doc_everyone_pay_scheduled_payments_002.md** - Setting up recurring payments
- **doc_everyone_pay_scheduled_payments_003.md** - Editing or canceling a scheduled payment
- **doc_everyone_pay_scheduled_payments_004.md** - Using Everyone Pay for rent or allowance
- **doc_everyone_pay_scheduled_payments_005.md** - Troubleshooting: payment didn't go through
- **doc_everyone_pay_scheduled_payments_006.md** - Why didn't my scheduled payment go through?
- **doc_everyone_pay_scheduled_payments_007.md** - Why can't I edit my scheduled payment?
- **doc_everyone_pay_scheduled_payments_008.md** - How do I stop a recurring payment?
- **doc_everyone_pay_sending_limits_001.md** - Sending Limits: Complete Guide
- **doc_everyone_pay_sending_limits_002.md** - Sending limits — per day and month
- **doc_everyone_pay_sending_limits_003.md** - What is Everyone Pay?
- **doc_everyone_pay_sending_limits_004.md** - Why did I reach my sending limit?
- **doc_everyone_pay_sending_limits_005.md** - How do I increase my Everyone Pay sending limit?
- **doc_everyone_pay_sending_limits_006.md** - Why can't I send more than $500 per day?
- **doc_everyone_pay_sending_limits_007.md** - What's the difference between daily and monthly limits?
- **doc_everyone_pay_split-the-bill_001.md** - Split-the-Bill: Complete Guide
- **doc_everyone_pay_split-the-bill_002.md** - Setting up a group payment
- **doc_everyone_pay_split-the-bill_003.md** - Managing group balances
- **doc_everyone_pay_split-the-bill_004.md** - Splitting restaurant or rent bills
- **doc_everyone_pay_split-the-bill_005.md** - Troubleshooting: uncollected payments
- **doc_everyone_pay_split-the-bill_006.md** - Why hasn't my group member paid yet?
- **doc_everyone_pay_split-the-bill_007.md** - Why can't I create a split payment?
- **doc_everyone_pay_split-the-bill_008.md** - How do I remove someone from a split payment?
- **doc_everyone_pay_user_blocking_001.md** - User Blocking: Complete Guide
- **doc_everyone_pay_user_blocking_002.md** - Blocking and unblocking users
- **doc_everyone_pay_user_blocking_003.md** - Common security questions for Everyone Pay
- **doc_everyone_pay_user_blocking_004.md** - Why can't I block this user?
- **doc_everyone_pay_user_blocking_005.md** - How do I unblock someone I previously blocked?
- **doc_everyone_pay_user_blocking_006.md** - Can blocked users still send me payment requests?
- **doc_personal_subscriptions_rho_bank_plus_001.md** - Rho Bank Plus: Premium Membership Overview and Benefits
- **doc_personal_subscriptions_rho_bank_plus_002.md** - Rho Bank Plus: Margin Lending Guide and Interest Rates
- **doc_savings_accounts_bronze_account_001.md** - Bronze Savings Account: Complete Guide
- **doc_savings_accounts_bronze_account_002.md** - Bronze Account — summary
- **doc_savings_accounts_bronze_account_003.md** - Bronze Account specifications and requirements
- **doc_savings_accounts_bronze_account_004.md** - How to avoid monthly fees
- **doc_savings_accounts_bronze_account_005.md** - Why am I being charged a fee on my Bronze Account?
- **doc_savings_accounts_bronze_account_006.md** - Bronze Account: Why did my interest rate change?
- **doc_savings_accounts_bronze_account_007.md** - Bronze Account: Getting interest payments
- **doc_savings_accounts_bronze_account_008.md** - Bronze Account: Mobile banking and deposit limits
- **doc_savings_accounts_bronze_account_009.md** - Transferring funds to external banks: Bronze Accounts
- **doc_savings_accounts_diamond_elite_account_001.md** - Diamond Elite Savings Account: Complete Guide
- **doc_savings_accounts_diamond_elite_account_002.md** - Diamond Elite Account — summary
- **doc_savings_accounts_diamond_elite_account_003.md** - Diamond Elite Account specifications and requirements
- **doc_savings_accounts_diamond_elite_account_004.md** - How to receive a Diamond Elite invitation
- **doc_savings_accounts_diamond_elite_account_005.md** - Private banking services for Diamond Elite members
- **doc_savings_accounts_diamond_elite_account_006.md** - Diamond Elite: Investment advisory and wealth management
- **doc_savings_accounts_diamond_elite_account_007.md** - Diamond Elite: Exclusive events and experiences
- **doc_savings_accounts_diamond_elite_account_008.md** - Why was my Diamond Elite application declined?
- **doc_savings_accounts_diamond_elite_account_009.md** - FAQ: Diamond Elite Account
- **doc_savings_accounts_diamond_elite_account_010.md** - Diamond Elite Account: How is interest calculated and when is it paid?
- **doc_savings_accounts_diamond_elite_account_011.md** - Diamond Elite Account: Credit Card APY Bonuses
- **doc_savings_accounts_gold_account_001.md** - Gold Savings Account: Complete Guide
- **doc_savings_accounts_gold_account_002.md** - Gold Account at a glance
- **doc_savings_accounts_gold_account_003.md** - Gold Account specifications and requirements
- **doc_savings_accounts_gold_account_004.md** - Gold Account: High-yield savings explained
- **doc_savings_accounts_gold_account_005.md** - Gold Account: No-transfer-fee advantage
- **doc_savings_accounts_gold_account_006.md** - Gold Account: Transferring funds between accounts instantly
- **doc_savings_accounts_gold_account_007.md** - Troubleshooting: transfer delays
- **doc_savings_accounts_gold_account_008.md** - Why was I charged a transfer fee on my Gold Account?
- **doc_savings_accounts_gold_account_009.md** - Why don't I qualify for the highest interest rate?
- **doc_savings_accounts_gold_account_010.md** - Gold Account: How is interest calculated and when is it paid?
- **doc_savings_accounts_gold_account_011.md** - Gold Account: Mobile banking and deposit limits
- **doc_savings_accounts_gold_account_012.md** - Gold Account: Transferring funds to external banks
- **doc_savings_accounts_gold_account_013.md** - Gold Account: Gold Rewards Card holder benefits
- **doc_savings_accounts_gold_account_014.md** - Gold Account: Credit Card APY Bonuses
- **doc_savings_accounts_gold_plus_account_001.md** - Gold Plus Account specifications and requirements
- **doc_savings_accounts_gold_plus_account_002.md** - Gold Plus: Differences from Gold Account
- **doc_savings_accounts_gold_plus_account_003.md** - Gold Plus: Investment sweep feature explained
- **doc_savings_accounts_gold_plus_account_004.md** - Gold Plus: Exclusive perks and features
- **doc_savings_accounts_gold_plus_account_005.md** - FAQ: Gold Plus Account
- **doc_savings_accounts_gold_plus_account_006.md** - Gold Plus Account: How is interest calculated and when is it paid?
- **doc_savings_accounts_gold_plus_account_007.md** - Gold Plus Account: Mobile banking and deposit limits
- **doc_savings_accounts_gold_plus_account_008.md** - Gold Plus Account: Transferring funds to external banks
- **doc_savings_accounts_gold_plus_account_009.md** - Gold Plus Account: Credit Card APY Bonuses
- **doc_savings_accounts_green_account__savings__001.md** - Green Account (savings) specifications and requirements
- **doc_savings_accounts_green_account__savings__002.md** - Green Account (savings): How your savings help the environment
- **doc_savings_accounts_green_account__savings__003.md** - Green Account (savings): Pairing with EcoCard for bonus APY
- **doc_savings_accounts_green_account__savings__004.md** - Green Account (savings): Why is paperless required?
- **doc_savings_accounts_green_account__savings__005.md** - FAQ: Green Account (savings)
- **doc_savings_accounts_green_account__savings__006.md** - Green Account (savings): How is interest calculated and when is it paid?
- **doc_savings_accounts_green_account__savings__007.md** - Green Account (savings): Mobile banking and deposit limits
- **doc_savings_accounts_green_account__savings__008.md** - Green Account (savings): Transferring funds to external banks
- **doc_savings_accounts_green_account__savings__009.md** - Green Account (savings): EcoCard holder exclusive benefits
- **doc_savings_accounts_platinum_account_001.md** - Platinum Savings Account: Complete Guide
- **doc_savings_accounts_platinum_account_002.md** - Platinum Account at a glance
- **doc_savings_accounts_platinum_account_003.md** - Platinum Account specifications and requirements
- **doc_savings_accounts_platinum_account_004.md** - Concierge banking services explained
- **doc_savings_accounts_platinum_account_005.md** - Extended FDIC insurance for high-balance accounts
- **doc_savings_accounts_platinum_account_006.md** - Why don't I qualify for Platinum Account benefits?
- **doc_savings_accounts_platinum_account_007.md** - Platinum Account: How is interest calculated and when is it paid?
- **doc_savings_accounts_platinum_account_008.md** - Platinum Account: Mobile banking and deposit limits
- **doc_savings_accounts_platinum_account_009.md** - Platinum Account: Same-day external transfers
- **doc_savings_accounts_platinum_account_010.md** - Platinum Account: Credit Card APY Bonuses
- **doc_savings_accounts_platinum_plus_account_001.md** - Platinum Plus Account specifications and requirements
- **doc_savings_accounts_platinum_plus_account_002.md** - Platinum Plus: Differences from Platinum Account
- **doc_savings_accounts_platinum_plus_account_003.md** - Platinum Plus: Wealth management services
- **doc_savings_accounts_platinum_plus_account_004.md** - Platinum Plus: Exclusive perks and features
- **doc_savings_accounts_platinum_plus_account_005.md** - FAQ: Platinum Plus Account
- **doc_savings_accounts_platinum_plus_account_006.md** - Platinum Plus Account: How is interest calculated and when is it paid?
- **doc_savings_accounts_platinum_plus_account_007.md** - Platinum Plus Account: Mobile banking and deposit limits
- **doc_savings_accounts_platinum_plus_account_008.md** - Platinum Plus Account: Same-day external transfers
- **doc_savings_accounts_platinum_plus_account_009.md** - Platinum Plus Account: Credit Card APY Bonuses
- **doc_savings_accounts_silver_account_001.md** - Silver Account specifications and requirements
- **doc_savings_accounts_silver_account_002.md** - Linking Silver Account with a Rho checking account
- **doc_savings_accounts_silver_account_003.md** - Interest-rate tiers explained
- **doc_savings_accounts_silver_account_004.md** - Why don't I qualify for Silver Account benefits?
- **doc_savings_accounts_silver_account_005.md** - FAQ: Silver Account
- **doc_savings_accounts_silver_account_006.md** - Silver Account: How is interest calculated and when is it paid?
- **doc_savings_accounts_silver_account_007.md** - Silver Account: Mobile banking and deposit limits
- **doc_savings_accounts_silver_account_008.md** - Silver Account: Transferring funds to external banks
- **doc_savings_accounts_silver_account_009.md** - Silver Account: Platinum Rewards Card holder benefits
- **doc_savings_accounts_silver_plus_account_001.md** - Silver Plus Account specifications and requirements
- **doc_savings_accounts_silver_plus_account_002.md** - Silver Plus: Differences from Silver Account
- **doc_savings_accounts_silver_plus_account_003.md** - Silver Plus: ATM fee rebate program
- **doc_savings_accounts_silver_plus_account_004.md** - Silver Plus: Exclusive perks and features
- **doc_savings_accounts_silver_plus_account_005.md** - FAQ: Silver Plus Account
- **doc_savings_accounts_silver_plus_account_006.md** - Silver Plus Account: How is interest calculated and when is it paid?
- **doc_savings_accounts_silver_plus_account_007.md** - Silver Plus Account: Mobile banking and deposit limits
- **doc_savings_accounts_silver_plus_account_008.md** - Silver Plus Account: Transferring funds to external banks
- **doc_savings_accounts_silver_plus_account_009.md** - Silver Plus Account: Credit Card APY Bonuses
./doc_savings_accounts_gold_account_014.md
./doc_credit_cards_credit_cards__general__006.md
./doc_business_credit_cards_business_silver_rewards_card_012.md
./doc_savings_accounts_bronze_account_007.md
./doc_business_credit_cards_business_silver_rewards_card_006.md
./doc_business_credit_cards_business_silver_rewards_card_007.md
./doc_credit_cards_silver_rewards_card_005.md
./doc_savings_accounts_silver_plus_account_009.md
./doc_credit_cards_silver_rewards_card_011.md
./doc_business_savings_accounts_diamond_vault_008.md
./doc_savings_accounts_platinum_account_010.md
./doc_credit_cards_credit_card_replacements_002.md
./doc_business_credit_cards_business_silver_rewards_card_010.md
./doc_credit_cards_silver_rewards_card_010.md
./doc_credit_cards_silver_rewards_card_008.md
./doc_business_savings_accounts_gold_plus_saver_001.md
./doc_credit_cards_credit_card_replacements_001.md
./doc_business_savings_accounts_silver_saver_account_001.md
./doc_business_credit_cards_business_silver_rewards_card_008.md
./doc_business_credit_cards_business_silver_rewards_card_002.md
./doc_credit_cards_silver_rewards_card_001.md
./doc_business_credit_cards_business_silver_rewards_card_004.md
./doc_credit_cards_silver_rewards_card_007.md
./doc_savings_accounts_gold_plus_account_009.md
./doc_savings_accounts_silver_account_006.md
./doc_business_credit_cards_business_silver_rewards_card_011.md
./doc_business_savings_accounts_bronze_saver_account_001.md
./doc_savings_accounts_green_account__savings__003.md
./doc_credit_cards_silver_rewards_card_002.md
./doc_business_savings_accounts_gold_saver_account_009.md
./doc_savings_accounts_diamond_elite_account_011.md
./doc_business_credit_cards_business_silver_rewards_card_003.md
./INDEX.md
./doc_credit_cards_silver_rewards_card_003.md
./doc_credit_cards_silver_rewards_card_006.md
./doc_credit_cards_silver_rewards_card_004.md
./doc_business_savings_accounts_emerald_saver_008.md
./doc_business_credit_cards_business_silver_rewards_card_005.md
./doc_savings_accounts_platinum_plus_account_009.md
./doc_credit_cards_credit_cards__general__015.md
./doc_business_credit_cards_business_silver_rewards_card_001.md
./doc_business_savings_accounts_platinum_reserve_account_010.md
./doc_business_credit_cards_business_silver_rewards_card_009.md
./doc_credit_cards_silver_rewards_card_009.md
./doc_business_credit_cards_business_silver_rewards_card_013.md
./doc_business_savings_accounts_silver_plus_saver_001.md

**assistant → shell:** {"command": "cat doc_credit_cards_credit_cards__general__003.md doc_credit_cards_credit_cards__general__006.md doc_credit_cards_credit_cards__general__007.md doc_credit_cards_credit_cards__general__005.md"}

**tool:** # Submitting a Cash Back Dispute (Internal)

## When to use this
Use this process when a customer believes there is a discrepancy between the cash back they received for a particular transaction and the cash back they should have received. This process applies to all credit card transactions.

## How to submit the dispute
Instead of a step-by-step workflow, instruct the customer to use the tool directly:

- Tool to provide to the user: `submit_cash_back_dispute_0589(user_id: str, transaction_id: str)`
- Tell the customer to run the tool with their own user_id and the specific transaction_id for the purchase in question.

## Agent notes
- Confirm the customer has the correct transaction_id before they submit.
- Do not collect sensitive card details; the tool uses the identifiers provided by the user.
- Advise the customer that supporting context (e.g., category or promotion expectations) may be requested later during review, but the submission itself is initiated with the tool above.# Internal: Credit Card Rewards - Points vs Cash Back Representation

## Rewards representation
Due to a legacy backend migration that has not yet been completed, all credit card rewards in the transaction database are stored as 'points' regardless of whether the card is a cash back card or a points-based card. For cash back cards (Bronze Rewards Card, Silver Rewards Card, Gold Rewards Card, Platinum Rewards Card, Diamond Elite Card, Crypto-Cash Back, Business Bronze Rewards Card, Business Silver Rewards Card, Green Rewards Card, Silver Zoom Card, Business Gold Rewards Card, and Business Platinum Rewards Card), these 'points' should be interpreted as cash back with a conversion rate of 1 point = $0.01 when redeemed as a statement credit or as a credit to the user's Rho-Bank checking account. For example, 250 points equals $2.50 in cash back, which matches a 2.5% cash back rate on a $100 purchase. The only true points-based card is the EcoCard, which earns 'sustainability points' (5 points per dollar on green purchases, 1 point per dollar on other purchases) that also redeem at $0.01 per point when redeemed as a statement credit or checking account credit.# Credit Card Rewards Points Rounding Policy

## Rounding policy
All credit card rewards points are rounded down to the nearest whole number. When calculating points earned on a purchase, the system always truncates (floors) fractional points rather than rounding to the nearest integer. For example, a $126.77 purchase on a 1% cash back card earns 126 points (not 127), because 126.77 × 0.01 = 1.2677 points, which rounds down to 1 point per dollar, giving 126 points. Similarly, a $99.99 purchase at 2.5% cash back would calculate to 249.975 points but would award 249 points. This rounding policy applies consistently across all credit card types and reward categories.# Credit Card Rewards Categories

## Eligible rewards categories
The following are the valid transaction categories that can earn rewards on Rho-Bank credit cards:

- Travel (flights, hotels, car rentals, etc.)
- Software (subscriptions, licenses, cloud services)
- Media (advertising spend, streaming services)
- Green (eco-friendly and sustainable merchants)
- Sustainable (same as Green, used interchangeably)
- Operations (business operational expenses)
- Transportation (logistics, shipping, rideshare)
- Groceries (supermarkets, grocery stores)
- Dining (restaurants, cafes, food delivery)
- Entertainment (movies, concerts, events)
- Utilities (electric, gas, water, internet bills)
- Shopping (retail stores, online marketplaces)
- Other (transactions that fall under card-specific exclusions or exceptions, do not list any examples here)

## Notes
- Each credit card has its own bonus categories with higher reward rates.
- Purchases in non-bonus categories earn the card’s default rate.
- This document lists categories only and does not specify reward rates for any card.

**assistant → shell:** {"command": "cat doc_credit_cards_silver_rewards_card_002.md doc_credit_cards_silver_rewards_card_003.md doc_credit_cards_silver_rewards_card_004.md"}

**tool:** # Silver Rewards Card: How to Earn 4% Cash Back on Travel and Software Purchases

## Earning at the Enhanced Rate
- Pay with your card for eligible travel and software transactions to earn 4.0% back

## Steps to Ensure You Earn Properly
1. Use your card directly with the merchant; third-party wallets and processors may alter merchant classification
2. Check that the merchant is categorized in a travel or software/SaaS merchant category; rewards are based on the merchant’s submitted category
3. Allow transactions to post; rewards are calculated after posting, not at authorization
4. Keep receipts and invoices; if a purchase is misclassified, you may be asked to provide documentation during a rewards review

## Transactions That Typically Do Not Qualify
- Gift cards, person-to-person payments, fees, interest, and insurance premiums charged by the bank
- Returned or refunded purchases (rewards are reversed when credits post)

## Visibility of Rewards
- You will see pending earnings soon after posting; final amounts may adjust if the merchant updates the transaction or issues a credit# Silver Rewards Card: Which Travel Purchases Qualify for 4% Cash Back?

## Travel Purchases That Typically Qualify
You earn 4.0% back when the transaction is categorized as travel. Examples include:
- Airlines and airfare booked directly with carriers
- Hotels, resorts, and motels
- Car rentals and car-sharing agencies
- Passenger rail (intercity trains), ferries, and long-distance bus operators
- Online travel agencies and tour operators when they are the merchant of record
- Rideshare and taxis when processed under a travel category

## Situations That May Not Qualify
- In-flight or onboard purchases coded as retail or food rather than travel
- Vacation rentals or home-sharing platforms coded under real estate or “miscellaneous” categories
- Parking garages, tolls, or local transit if the merchant is not classified as travel
- Purchases made through third-party payment processors that mask the travel category

## Tips to Ensure Eligibility
- Book directly with travel providers or well-known travel agencies that act as merchant of record
- Review your statement after posting; if an eligible trip did not earn at the travel rate, contact support with receipts for a category review# Silver Rewards Card: Software and SaaS Spending - Qualifying Categories

## What Counts as Software/SaaS
Transactions must be categorized by the merchant as software or SaaS to earn 4.0% back. Examples include:
- Subscription-based productivity suites, collaboration tools, and project management platforms
- Cloud services, developer tools, version control, CI/CD platforms, and code hosting
- Cybersecurity subscriptions, endpoint protection, and identity management
- Design, analytics, CRM, ERP, and accounting software subscriptions

## Purchases That May Not Qualify
- Hardware devices, peripherals, or equipment sold by tech retailers
- Telecommunications, internet service providers, or hosting billed under telecom categories
- App store gift cards and prepaid stored-value products

## Tips to Ensure Proper Category
- Purchase directly from the software provider’s billing portal or official sales channel
- Keep invoices showing the subscription description; if the transaction is miscategorized, you may request a category review
- Avoid paying through third-party processors that may categorize the charge as generic services

**assistant → shell:** {"command": "cat doc_credit_cards_silver_rewards_card_001.md doc_credit_cards_silver_rewards_card_008.md; grep -ri \"default rate\\|all other purchases\\|1% on\\|1.0%\" doc_credit_cards_silver_rewards_card_*.md"}

**tool:** # Silver Rewards Card: Getting Started and Eligibility Requirements

## Eligibility Requirements
- Minimum credit score: $680
- Annual fee: $0.00
- Standard APR for carried balances after any promotional period: 18.99%
- Introductory APR for new customers (first year): 0%

### International Usage
- Foreign transaction fee without a premium subscription: 2.75%
- Foreign transaction fee with a premium subscription: 0%

## Credit Limits
- Initial credit limits typically range from $10,000 to $100,000, subject to approval

## Included Protections (At a Glance)
- Purchase protection: eligible items are protected for 90 days after purchase, up to $3,000 per claim

## Rewards Basics When You Get Started
- Purchases outside top categories earn 1.0% back

## Virtual Card Availability
- Virtual card management features are No for this card

## How to Apply
1. Complete the online application with your personal information and income details
2. Consent to a credit check and submit any requested documentation
3. Review your preliminary credit limit offer and terms
4. Accept the terms to finalize your account setup

## After Approval
- Activate your physical card upon arrival and add it to your preferred digital wallets
- Set up account alerts and autopay to avoid missed payments
- Review your benefits summary, including purchase protection and international fee terms noted above# Silver Rewards Card: Maximizing Rewards on Business Travel and Tech Expenses

## Booking and Subscription Tactics
- Route air, lodging, and transportation charges through your card to accelerate earnings in top travel categories
- Put recurring software and SaaS subscriptions on autopay with your card to ensure consistent category rewards

## International Strategy
- If you maintain the premium subscription, international purchases incur 0% in foreign transaction fees, which helps preserve your rewards when traveling abroad

## Don’t Overlook Everyday Spend
- Use your card for non-category purchases to earn at least 1.0% back, then consolidate redemptions once you reach 17

## Redemption Timing
- Plan redemptions around billing cycles so new earnings post and you meet the 17 in a single statement period

## Administrative Tips
- Keep vendor lists current; when switching software providers, update your card on file so you continue to capture rewards without interruption
- Review posted transactions to confirm category alignment; request a review if a known travel or software purchase appears misclassifieddoc_credit_cards_silver_rewards_card_001.md:- Purchases outside top categories earn 1.0% back
doc_credit_cards_silver_rewards_card_008.md:- Use your card for non-category purchases to earn at least 1.0% back, then consolidate redemptions once you reach 17

**assistant → shell:** {"command": "cat doc_business_credit_cards_business_silver_rewards_card_002.md doc_business_credit_cards_business_silver_rewards_card_005.md doc_business_credit_cards_business_silver_rewards_card_012.md doc_business_credit_cards_business_silver_rewards_card_001.md"}

**tool:** # Business Silver Rewards Card: How to Earn 10% Back on Travel and Software Purchases

## Earning rates
- Earn 10.0% cash back on eligible travel and software purchases
- Earn 1.0% cash back on all other purchases

## Making sure your purchase qualifies
- The merchant must process the transaction under a travel or software-related merchant category
- Online purchases should be billed by the service provider itself or an authorized platform categorized as travel or software
- If a merchant sells multiple categories of goods or services, your charge must be coded under a qualifying category to receive 10.0%

## Best practices
- Use this card for airfare, lodging, car rentals, transit, and other travel booked directly with providers or recognized travel agencies
- Use this card to pay recurring SaaS subscriptions and cloud-based software billed by software providers
- Keep receipts and confirmations in case a merchant’s category coding needs review# Business Silver Rewards Card: Exceptions and Exclusions

## Important exclusions
The following specific merchants do NOT earn the 10.0% bonus rate and instead earn the standard 1.0% rate:

**Corporate Expense Platform Merchants (Travel Exclusion):**
- Concur
- SAP Concur
- Expensify
- Navan

**Hardware/Electronics Merchants (Software Exclusion):**
- Apple
- Microsoft
- Dell

**Gaming Subscription Merchants (Software Exclusion):**
- Xbox Game Pass
- PlayStation Plus
- Nintendo Switch Online

**Online Learning Platform Merchants (Software Exclusion):**
- Coursera
- Udemy
- LinkedIn Learning
- Skillshare
- Pluralsight# Business Silver Rewards Card: Double Cash Back Promo

## Limited-time offer: double your cash back
New Business Silver Rewards Card customers can earn 2x cash back on all purchases for the first 6 months after opening their account.

## How it works
- Travel and software purchases that normally earn 10.0% are multiplied by 2 during the promotional period
- Other purchases that normally earn 1.0% are multiplied by 2 during the promotional period
- The increased rewards rate is automatically applied to qualifying transactions; no separate enrollment is required

## Promo period
- Start date: 2024-11-14
- End date: 2025-11-14

## Important details
- The 6-month window begins on your account opening date, not the promotional start date
- You must open your account during the promotional period to qualify
- Merchant category exclusions continue to apply (see Exceptions and Exclusions)
- Cash back is credited to your account at the end of each billing cycle consistent with program rules# Business Silver Rewards Card: Getting Started and Eligibility Requirements

## Eligibility requirements
- Minimum personal credit score (FICO): $700
- Established businesses may qualify based on a Dun & Bradstreet PAYDEX of at least 47; newer businesses may be considered with strong personal credit

## Credit limits
- Approved credit limits generally range from $17,500 to $112,500, subject to credit review and business financials

## Pricing you should expect at approval
- Annual fee: $122.50
- First-year promotional annual fee for new customers: 0
- APR on carried balances: 18.99%

## How to get started
- Prepare your business information: legal name, address, formation details, employer identifiers, ownership structure, and estimated annual revenue
- Be ready to authorize a personal credit pull and, if applicable, provide business credit information
- Submit your application with accurate contact details so we can request any supporting documentation promptly
- After approval, set up account access, add authorized users, and configure payment preferences

## Tips to strengthen your application
- Maintain a personal credit profile at or above $700 and keep utilization low prior to applying
- If your business has a PAYDEX score, ensure vendor payment histories are up to date to support at least 47
- Have recent financial documentation available if requested, such as bank statements or accountant-prepared records

**assistant → shell:** {"command": "cat doc_credit_cards_silver_rewards_card_009.md doc_credit_cards_silver_rewards_card_010.md; grep -ril \"exclusion\" . | grep -i silver"}

**tool:** # Silver Rewards Card: New Customer Promo - 0% APR for the First Year

## Promotional Rate Details
- Introductory APR on carried balances for new customers: 0%
- Standard APR after the promotional period ends: 18.99%

## Offer Window
- Applications and account openings must occur within: 2025-10-15 to 2025-12-31

## How It Works
1. Open a new account within the offer window
2. For the first year from account opening, balances you carry are charged 0%
3. After the first year, your APR reverts to 18.99%

## Important Considerations
- Pay at least the statement minimum each month to keep your account in good standing throughout the promo period
- Plan larger purchases early in your first year to maximize the time at 0%
- When the promo ends, any remaining balance begins to accrue interest at 18.99%# Silver Rewards Card: New Customer Promo - $500 Statement Credit Signup Bonus

## Bonus Amount and Requirement
- Earn a statement credit of $500 when you spend $2,000 in the first 1 month(s) after account opening

## Offer Window
- Accounts must be opened within: 2025-01-01 to 2025-06-30

## How to Qualify
1. Open a new account during the offer window
2. Make qualifying purchases that post to your account totaling $2,000 within 1 month(s) from account opening
3. Keep your account open and in good standing through the bonus fulfillment date

## What Counts Toward the Threshold
- Net purchases (purchases minus returns/credits) that post within the qualifying period

## When You’ll Receive the Credit
- The $500 typically posts within two billing cycles after you meet the spending requirement

## Tips
- Set recurring bills and subscriptions on your card immediately after approval to reach $2,000 within 1 month(s)
- Track posted transactions (not just authorizations) to confirm progress toward the threshold./doc_business_credit_cards_business_silver_rewards_card_012.md
./doc_business_credit_cards_business_silver_rewards_card_006.md
./doc_business_credit_cards_business_silver_rewards_card_007.md
./doc_credit_cards_silver_rewards_card_005.md
./doc_business_credit_cards_business_silver_rewards_card_008.md
./doc_credit_cards_silver_rewards_card_006.md
./doc_business_credit_cards_business_silver_rewards_card_005.md

**assistant → shell:** {"command": "cat doc_credit_cards_credit_cards__general__021.md doc_credit_cards_credit_cards__general__004.md"}

**tool:** # Think You're Missing Cash Back? Here's How to Get Help

At Rho-Bank, we want to make sure you receive every dollar of cash back that you've earned. We understand that occasionally there may be discrepancies between the cash back you expected to receive on a purchase and what actually appeared on your account. If you've reviewed your rewards and something doesn't seem right, we're here to help investigate and make it right.

## When to Reach Out

You might want to contact us about your cash back if:

- A purchase that should have earned bonus cash back only received the standard rate
- You made a purchase in a bonus category but didn't receive the higher reward rate
- The cash back amount seems lower than what you calculated based on your card's rates
- A qualifying purchase doesn't show any rewards at all
- You believe a merchant was miscategorized and you should have earned a different rate

## How We Can Help

Our customer service team has the tools to investigate cash back discrepancies and review your transaction history. When you contact us, we'll look into the specific transaction, verify the merchant category, check the applicable reward rates for your card, and determine if an adjustment is needed. If we find that you were indeed shortchanged on your rewards, we'll make sure the correct amount is applied to your account.

## What You'll Need

To help us investigate quickly, it's helpful to have the following information ready when you contact us:

- The date of the transaction in question
- The merchant name and purchase amount
- The cash back amount you received versus what you expected
- Any receipts or documentation you have from the purchase

## Contact Us

If you think you're missing cash back or have any questions about how your rewards are calculated, please reach out to our customer service team. You can call us at 1-800-RHO-BANK, chat with us through the Rho-Bank mobile app, or visit rhobank.com/help. We're committed to ensuring you get the full value of your Rho-Bank credit card rewards.# Applying Resolved Cash Back Dispute Corrections (Internal)

## Overview
After a cash back dispute is resolved and approved, you must update the affected transaction(s) with the correct rewards value.

## Required steps
1) Identify transactions requiring correction
- Look up the user’s resolved disputes in the cash_back_disputes database to find the transaction_id values that need rewards adjustments.

2) Unlock the internal update tool
- Use `unlock_discoverable_agent_tool` to unlock the agent tool `update_transaction_rewards_3847`.

3) Recalculate the correct rewards
- Recalculate the correct rewards for each transaction based on the card type, transaction category, and any applicable promotions.
- Important: Do not rely on any expected_rewards field in the dispute record. Independently verify rates and eligibility.

4) Apply the corrected rewards
- Use `call_discoverable_agent_tool` to invoke `update_transaction_rewards_3847` with:
  - transaction_id: the exact transaction identifier
  - new_rewards_earned: a string formatted as 'X points' where X is the correct whole-number point value

## Compliance and accuracy
- Ensure your calculation reflects the proper base rate and any active bonus category or promotion at the time of purchase.
- Confirm the update in credit_card_transaction_history after the tool call completes.
- Retain your calculation notes in the internal case record for auditability.

**assistant:** Thank you, Amara — you're verified. I've reviewed both of your cards' transactions against the rewards rules for each card, and here's what I found.

**How your cards earn**
- **Silver Rewards Card:** 4% on Travel and Software, 1% on everything else.
- **Business Silver Rewards Card:** 10% on Travel and Software, 1% on everything else. Because you opened it on 02/13/2025 during the Double Cash Back promo, all purchases through the first 6 months (until ~08/13/2025) earned 2x (20% / 2%). Certain merchants (e.g., Microsoft, Coursera) are excluded from the bonus rate and earn the base rate.
- Rewards display as "points," where 1 point = $0.01.

**Silver Rewards Card** – All of your travel/software and everyday purchases were credited at the correct rate or higher. (One item, GitHub Enterprise on 05/18, actually posted at 10% rather than 4% — in your favor — and a few everyday purchases were rounded up by 1 point.)

**Business Silver Rewards Card** – Three transactions appear to have been **under-credited**:

| Date | Merchant | Amount | Received | Should Have Received | Shortfall |
|---|---|---|---|---|---|
| 03/22/2025 | JetBlue Airways (Travel, during 2x promo) | $315.00 | 3,150 pts ($31.50) | 6,300 pts ($63.00) | 3,150 pts |
| 08/25/2025 | Southwest Airlines (Travel) | $380.00 | 1,520 pts ($15.20) | 3,800 pts ($38.00) | 2,280 pts |
| 09/15/2025 | Zoom Video (Software) | $149.99 | 600 pts ($6.00) | 1,499 pts ($14.99) | 899 pts |

That's a total shortfall of about **6,329 points ($63.29)**. Everything else on the business card matched expectations (Microsoft 365 and Coursera correctly earned the base rate due to merchant exclusions).

**Next step:** You can submit a cash back dispute for each of these three transactions yourself through a self-service tool. Would you like me to provide it so you can file disputes for the JetBlue, Southwest, and Zoom transactions?

**user:** Yeah, please send the self-service tool link or whatever you use for that. I don’t have my phone on me, so I can’t pull up statements—if it just needs those basics (JetBlue, Southwest, Zoom) that’d be perfect.

**assistant → give_discoverable_user_tool:** {"discoverable_tool_name": "submit_cash_back_dispute_0589", "arguments": "{\"user_id\": \"755bcb4d5d\", \"transaction_id\": \"txn_a8f1c2d3e403\"}"}

**tool:** Tool given to user: submit_cash_back_dispute_0589
Description: Submit a cash back dispute for a specific transaction.
Arguments: {
  "user_id": "755bcb4d5d",
  "transaction_id": "txn_a8f1c2d3e403"
}

The user can now execute this action by calling `call_discoverable_user_tool` with discoverable_tool_name='submit_cash_back_dispute_0589' and the same arguments.

**assistant:** No problem — you don't need your statements, I've pulled the transaction IDs for you. I've provided the self-service tool **`submit_cash_back_dispute_0589`** to you. It takes two arguments: your `user_id` and the `transaction_id` for the purchase in question.

Please run it once for each of the three transactions, using these exact values:

1. **JetBlue Airways (03/22/2025, $315.00)**
   `submit_cash_back_dispute_0589(user_id="755bcb4d5d", transaction_id="txn_a8f1c2d3e403")`

2. **Southwest Airlines (08/25/2025, $380.00)**
   `submit_cash_back_dispute_0589(user_id="755bcb4d5d", transaction_id="txn_a8f1c2d3e410")`

3. **Zoom Video (09/15/2025, $149.99)**
   `submit_cash_back_dispute_0589(user_id="755bcb4d5d", transaction_id="txn_a8f1c2d3e411")`

I've pre-loaded the first one (JetBlue) for you; just swap in the other two transaction IDs for the Southwest and Zoom submissions. Once submitted, each dispute goes into review — you may be asked for supporting context (such as the expected category or promo) later, but no documentation is needed to submit now.

Let me know once you've submitted them, or if you'd like any help along the way!

**user:** 

**tool:** Cash back dispute submitted successfully. Your case has been queued for review.

Executed: submit_cash_back_dispute_0589
Arguments: {
  "user_id": "755bcb4d5d",
  "transaction_id": "txn_a8f1c2d3e403"
}
Dispute ID: dsp_c16631293b89
Status: SUBMITTED - Your dispute has been queued for review.

**user:** 

**tool:** Cash back dispute submitted successfully. Your case has been queued for review.

Executed: submit_cash_back_dispute_0589
Arguments: {
  "user_id": "755bcb4d5d",
  "transaction_id": "txn_a8f1c2d3e410"
}
Dispute ID: dsp_ebffb9d8483f
Status: SUBMITTED - Your dispute has been queued for review.

**user:** 

**tool:** Cash back dispute submitted successfully. Your case has been queued for review.

Executed: submit_cash_back_dispute_0589
Arguments: {
  "user_id": "755bcb4d5d",
  "transaction_id": "txn_a8f1c2d3e411"
}
Dispute ID: dsp_4a1414aa6359
Status: SUBMITTED - Your dispute has been queued for review.

**user:** All set — I submitted all three cash back disputes.

- JetBlue (03/22/2025) — Dispute ID **dsp_c16631293b89** (Status: Submitted)
- Southwest (08/25/2025) — Dispute ID **dsp_ebffb9d8483f** (Status: Submitted)
- Zoom (09/15/2025) — Dispute ID **dsp_4a1414aa6359** (Status: Submitted)

Let me know if there’s anything else you want me to double-check while these are in review. ###STOP###

