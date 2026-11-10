# task_025 · trial 0 · resolved · ended: user_stop

**assistant:** Hi! How can I help you today?

**user:** Hi, I’m looking to purchase a company-wide Apple Music subscription for my business — it’s going to cost $100,000 as a single payment. I want to open a new business credit card that will give me the best return on this purchase. What are my options?

→ assistant calls `KB_search_bm25` with `{"query": "business credit card rewards streaming subscription"}`

**tool:** 1. Understanding Your Credit Card Rewards Categories
   ID: doc_credit_cards_credit_cards_(general)_019
   Score: 12.6912
   Content: One of the great benefits of your Rho-Bank credit card is the ability to earn rewards on your everyday purchases. To help you maximize your earnings, it's important to understand how reward categories work and which types of purchases qualify for bonus rewards on your specific card.

## How Reward Categories Work

Every Rho-Bank credit card earns rewards on all purchases, but certain categories of spending may earn you bonus rewards at a higher rate. Your card has a base reward rate that applies to all purchases, plus bonus categories where you can earn even more. The specific bonus categories and rates vary depending on which Rho-Bank credit card you have.

## Available Reward Categories

Rho-Bank recognizes the following spending categories for rewards purposes:

- **Travel**: Flights, hotels, car rentals, cruises, and other travel-related purchases
- **Dining**: Restaurants, cafes, bars, and food delivery services
- **Groceries**: Supermarkets, grocery stores, and specialty food shops
- **Entertainment**: Movies, concerts, sporting events, and streaming services
- **Shopping**: Retail stores, department stores, and online marketplaces
- **Utilities**: Electric, gas, water, internet, and phone bills
- **Transportation**: Rideshare services, public transit, parking, and tolls
- **Software**: Software subscriptions, licenses, and cloud services
- **Media**: Advertising spend and digital media purchases
- **Green/Sustainable**: Eco-friendly merchants and sustainable products
- **Operations**: Business operational expenses

## Maximizing Your Rewards

To get the most out of your rewards, we recommend using your Rho-Bank credit card for purchases in your card's bonus categories whenever possible. You can find your specific card's bonus categories and reward rates by logging into your account on the Rho-Bank mobile app or website, or by reviewing your card's terms and conditions.

If you have any questions about which category a particular merchant falls into, or how to maximize your rewards earning potential, our customer service team is always happy to help. Give us a call at 1-800-RHO-BANK or chat with us through the mobile app.

2. Credit Card Rewards Categories
   ID: doc_credit_cards_credit_cards_(general)_005
   Score: 12.2487
   Content: ## Eligible rewards categories
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

3. Internal: Credit Card Rewards - Points vs Cash Back Representation
   ID: doc_credit_cards_credit_cards_(general)_006
   Score: 11.3123
   Content: ## Rewards representation
Due to a legacy backend migration that has not yet been completed, all credit card rewards in the transaction database are stored as 'points' regardless of whether the card is a cash back card or a points-based card. For cash back cards (Bronze Rewards Card, Silver Rewards Card, Gold Rewards Card, Platinum Rewards Card, Diamond Elite Card, Crypto-Cash Back, Business Bronze Rewards Card, Business Silver Rewards Card, Green Rewards Card, Silver Zoom Card, Business Gold Rewards Card, and Business Platinum Rewards Card), these 'points' should be interpreted as cash back with a conversion rate of 1 point = $0.01 when redeemed as a statement credit or as a credit to the user's Rho-Bank checking account. For example, 250 points equals $2.50 in cash back, which matches a 2.5% cash back rate on a $100 purchase. The only true points-based card is the EcoCard, which earns 'sustainability points' (5 points per dollar on green purchases, 1 point per dollar on other purchases) that also redeem at $0.01 per point when redeemed as a statement credit or checking account credit.

4. Gold Rewards Card: Apply for a Premium Credit Card
   ID: doc_credit_cards_gold_rewards_card_001
   Score: 10.9421
   Content: ## Eligibility and Key Terms
- Minimum credit score required: $720
- Rho‑Bank+ premium subscription required: Yes
- Annual fee: $0.00
- Cash back on all purchases: 2.5%
- APR on carried balances: 17.99%
- Foreign transaction fee: 0%
- Late payment fee: $32.50
- Minimum rewards needed to redeem: $15
- Purchase protection: 105 days from purchase, up to $7,500 per covered claim
- Virtual card management available: Yes

## Credit Limits
Approved credit limits typically range from $17,500 to $112,500. Your initial limit is determined during underwriting based on your credit profile and supplied financial information.

## How to Apply
1. Confirm you have an active Rho‑Bank+ subscription (required: Yes).
2. Prepare documentation typically requested during credit evaluation (e.g., government ID, income details, and address verification).
3. Complete the application in your Rho‑Bank dashboard and consent to a credit pull (minimum score: $720).
4. Review key terms before submission:
   - APR range on carried balances: 17.99%
   - Annual fee: $0.00
   - Late payment fee: $32.50
   - Rewards redemption minimum: $15
   - Purchase protection period and cap: 105 days; $7,500 per claim
   - Foreign transaction fee: 0%
5. Submit and monitor your application status. If additional information is needed, you will receive a prompt to upload or confirm details.
6. Upon approval, your initial limit will be set within $17,500–$112,500.

## After Approval
- If enabled on your account, you can create and manage virtual cards immediately: Yes.
- Purchases made after activation begin earning cash back at 2.5% and are covered by purchase protection for 105 days up to $7,500 per claim.

5. Business Bronze Rewards Card: Exceptions and Exclusions
   ID: doc_business_credit_cards_business_bronze_rewards_card_003
   Score: 10.6900
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

6. Provisional Credit Eligibility Guidelines (Internal)
   ID: doc_credit_cards_credit_cards_(general)_015
   Score: 10.4742
   Content: ## Overview
When processing a credit card transaction dispute, you must determine whether the customer is eligible for provisional credit. Provisional credit means the disputed amount is temporarily credited back to the customer's account while the investigation is ongoing.

## Eligibility Criteria — Customer is ELIGIBLE for provisional credit if ALL of the following are true:
1. Account Standing: The customer's credit card account has been open for at least 60 days
2. Dispute Reason Category: The dispute reason is one of the following:
   - 'unauthorized_fraudulent_charge'
   - 'duplicate_charge'
   - 'goods_services_not_received' (only if purchase was made more than 30 days ago)
3. Dispute Amount: The transaction amount must be at least $25.00 and must not exceed the card's maximum provisional credit limit based on card tier (see below)
4. Previous Disputes: The customer has not filed more than 2 disputes in the past 12 months
5. Contacted Merchant: For non-fraud disputes (anything other than 'unauthorized_fraudulent_charge'), the customer must have attempted to resolve with the merchant first

## Maximum Provisional Credit Limits by Card Tier
- Entry Tier (Bronze Rewards Card, EcoCard, Business Bronze Rewards Card, Crypto-Cash Back Card): Maximum $2500.00
- Mid Tier (Silver Rewards Card, Business Silver Rewards Card, Green Rewards Card, Silver Zoom Card): Maximum $5000.00
- Premium Tier (Gold Rewards Card, Business Gold Rewards Card): Maximum $10000.00
- Elite Tier (Platinum Rewards Card, Business Platinum Rewards Card): Maximum $15000.00
- Invitation Tier (Diamond Elite Card): Maximum $25000.00

## NOT Eligible Scenarios
- Account is less than 60 days old
- Dispute reason is 'incorrect_amount', 'goods_services_not_as_described', 'canceled_subscription_still_charging', or 'refund_never_processed'
- Transaction amount is under $25.00 or exceeds the card's maximum limit based on tier
- For non-fraud disputes: customer did not contact merchant first
- 'goods_services_not_received' where purchase was made within the last 30 days (merchant may still be processing delivery)

7. Internal: Downgrading a Credit Card to a No-Annual-Fee Card
   ID: doc_credit_cards_credit_card_account_logistics_008
   Score: 10.1419
   Content: ## When to Offer a Downgrade

Offer a downgrade in these scenarios:

1. Retention protocol: The customer wants to close due to annual fee concerns and has been a customer for less than 2 years (not eligible for annual fee waiver).
2. Customer request: The customer explicitly asks to downgrade to avoid annual fees.

## Available No-Annual-Fee Card Options

- Personal cards: Bronze Rewards Card (no annual fee)
- Business cards: Business Bronze Rewards Card (no annual fee)

Customers may only downgrade within the same category (personal to personal, business to business).

## Downgrade Process

1. Verify the customer's identity using standard procedures.
2. Confirm the customer wants to proceed and understands benefit changes.
3. Inform the customer that account history and credit line will be preserved, and rewards rates/benefits will change to match the new tier.
4. Use downgrade_credit_card_3847 to process the downgrade.

Tool arguments for downgrade_credit_card_3847:
- credit_card_account_id (string, required): The credit card account ID to downgrade.
- user_id (string, required): The customer's unique identifier.
- target_card_type (string, required): One of 'Bronze Rewards Card' (personal) or 'Business Bronze Rewards Card' (business).

## Important Notes

- Credit limit, account number, and account history are preserved during a downgrade.
- Unredeemed rewards points transfer at the same value.
- The downgrade takes effect immediately; the customer continues using the existing physical card until the new card arrives, typically within several business days.
- If the annual fee was paid recently, the customer may be eligible for a prorated refund.

8. Silver Zoom Card: Designed for Transportation and Logistics Businesses
   ID: doc_business_credit_cards_silver_zoom_card_001
   Score: 9.1900
   Content: ## Who this card is built for
- Established transportation and logistics operators that can meet both personal and business credit criteria
- Review focuses on a combination of your personal credit profile at or above $680 and a business credit profile at or above 57

## Fuel and rewards structure tailored to fleets
- Receive a recurring fuel price reduction of 10.0% at participating stations
- Earn elevated rewards on sector-specific spend at 3.0%
- Earn a flat rate on all other eligible purchases at 1.0%

## Pricing and promotional highlights
- Annual membership is $74.50
- New customer fuel financing promotion: an APR of 0% on fuel purchases for the first year, subject to eligibility and timing rules

## Why this mix matters for logistics
- The combination of 10.0% off at the pump and ongoing rewards helps reduce total cost per mile
- Dual-credit evaluation using $680 and 57 supports approval decisions designed for established operators

9. Gold Account: Credit Card APY Bonuses
   ID: doc_savings_accounts_gold_account_014
   Score: 9.1362
   Content: ## Credit card APY bonuses for Gold Account holders

Holding certain Rho-Bank credit cards provides bonus APY on your Gold Account balance. These bonuses are automatically applied when you hold both the credit card and savings account under the same customer profile.

### Available credit card APY bonuses

| Credit Card | APY Bonus |
|---|---|
| Bronze Rewards Card | +0.15% |
| Silver Rewards Card | +0.2% |
| Gold Rewards Card | +0.025% |
| Platinum Rewards Card | +0.15% |
| Diamond Elite Card | +0.3% |
| EcoCard | +0.6% |
| Green Rewards Card | +0.35% |
| Crypto-Cash Back Card | +0% |

### Additional Gold Rewards Card benefits

Gold Rewards Card holders receive an additional benefit: the minimum balance requirement is reduced from $10,000 to $5,000.

### How to qualify

1. Open or maintain a Gold Account.
2. Hold an eligible Rho-Bank credit card under the same customer profile.
3. The APY bonus is automatically applied to your Gold Account earnings.

10. Platinum Account: Credit Card APY Bonuses
   ID: doc_savings_accounts_platinum_account_010
   Score: 9.0417
   Content: ## Credit card APY bonuses for Platinum Account holders

Holding certain Rho-Bank credit cards provides bonus APY on your Platinum Account balance. These bonuses are automatically applied when you hold both the credit card and savings account under the same customer profile.

### Available credit card APY bonuses

| Credit Card | APY Bonus |
|---|---|
| Bronze Rewards Card | +0% |
| Silver Rewards Card | +0% |
| Gold Rewards Card | +0.15% |
| Platinum Rewards Card | +0.25% |
| Diamond Elite Card | +0.35% |
| EcoCard | +0% |
| Green Rewards Card | +0% |
| Crypto-Cash Back Card | +0% |

### How to qualify

1. Open or maintain a Platinum Account.
2. Hold an eligible Rho-Bank credit card under the same customer profile.
3. The APY bonus is automatically applied to your Platinum Account earnings.


[Timing: retrieval=3ms, reranking=0ms, total=3ms]

**assistant:** Let me look at all business cards in detail.
→ assistant calls `shell` with `{"command": "ls | grep -i business; ls | wc -l"}`

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
doc_business_credit_cards_business_bronze_rewards_card_001.md
doc_business_credit_cards_business_bronze_rewards_card_002.md
doc_business_credit_cards_business_bronze_rewards_card_003.md
doc_business_credit_cards_business_bronze_rewards_card_004.md
doc_business_credit_cards_business_bronze_rewards_card_005.md
doc_business_credit_cards_business_bronze_rewards_card_006.md
doc_business_credit_cards_business_bronze_rewards_card_007.md
doc_business_credit_cards_business_bronze_rewards_card_008.md
doc_business_credit_cards_business_bronze_rewards_card_009.md
doc_business_credit_cards_business_bronze_rewards_card_010.md
doc_business_credit_cards_business_bronze_rewards_card_011.md
doc_business_credit_cards_business_gold_rewards_card_001.md
doc_business_credit_cards_business_gold_rewards_card_002.md
doc_business_credit_cards_business_gold_rewards_card_003.md
doc_business_credit_cards_business_gold_rewards_card_004.md
doc_business_credit_cards_business_gold_rewards_card_005.md
doc_business_credit_cards_business_gold_rewards_card_006.md
doc_business_credit_cards_business_gold_rewards_card_007.md
doc_business_credit_cards_business_gold_rewards_card_008.md
doc_business_credit_cards_business_gold_rewards_card_009.md
doc_business_credit_cards_business_gold_rewards_card_010.md
doc_business_credit_cards_business_platinum_rewards_card_001.md
doc_business_credit_cards_business_platinum_rewards_card_002.md
doc_business_credit_cards_business_platinum_rewards_card_003.md
doc_business_credit_cards_business_platinum_rewards_card_004.md
doc_business_credit_cards_business_platinum_rewards_card_005.md
doc_business_credit_cards_business_platinum_rewards_card_006.md
doc_business_credit_cards_business_platinum_rewards_card_007.md
doc_business_credit_cards_business_platinum_rewards_card_008.md
doc_business_credit_cards_business_platinum_rewards_card_009.md
doc_business_credit_cards_business_platinum_rewards_card_010.md
doc_business_credit_cards_business_platinum_rewards_card_011.md
doc_business_credit_cards_business_platinum_rewards_card_012.md
doc_business_credit_cards_business_silver_rewards_card_001.md
doc_business_credit_cards_business_silver_rewards_card_002.md
doc_business_credit_cards_business_silver_rewards_card_003.md
doc_business_credit_cards_business_silver_rewards_card_004.md
doc_business_credit_cards_business_silver_rewards_card_005.md
doc_business_credit_cards_business_silver_rewards_card_006.md
doc_business_credit_cards_business_silver_rewards_card_007.md
doc_business_credit_cards_business_silver_rewards_card_008.md
doc_business_credit_cards_business_silver_rewards_card_009.md
doc_business_credit_cards_business_silver_rewards_card_010.md
doc_business_credit_cards_business_silver_rewards_card_011.md
doc_business_credit_cards_business_silver_rewards_card_012.md
doc_business_credit_cards_business_silver_rewards_card_013.md
doc_business_credit_cards_green_rewards_card_001.md
doc_business_credit_cards_green_rewards_card_002.md
doc_business_credit_cards_green_rewards_card_003.md
doc_business_credit_cards_green_rewards_card_004.md
doc_business_credit_cards_green_rewards_card_005.md
doc_business_credit_cards_green_rewards_card_006.md
doc_business_credit_cards_green_rewards_card_007.md
doc_business_credit_cards_green_rewards_card_008.md
doc_business_credit_cards_green_rewards_card_009.md
doc_business_credit_cards_green_rewards_card_010.md
doc_business_credit_cards_silver_zoom_card_001.md
doc_business_credit_cards_silver_zoom_card_002.md
doc_business_credit_cards_silver_zoom_card_003.md
doc_business_credit_cards_silver_zoom_card_004.md
doc_business_credit_cards_silver_zoom_card_005.md
doc_business_credit_cards_silver_zoom_card_006.md
doc_business_credit_cards_silver_zoom_card_007.md
doc_business_credit_cards_silver_zoom_card_008.md
doc_business_credit_cards_silver_zoom_card_009.md
doc_business_credit_cards_silver_zoom_card_010.md
doc_business_credit_cards_silver_zoom_card_011.md
doc_business_credit_cards_silver_zoom_card_012.md
doc_business_credit_cards_silver_zoom_card_013.md
doc_business_credit_cards_virtual_card_management_001.md
doc_business_credit_cards_virtual_card_management_002.md
doc_business_credit_cards_virtual_card_management_003.md
doc_business_credit_cards_virtual_card_management_004.md
doc_business_credit_cards_virtual_card_management_005.md
doc_business_credit_cards_virtual_card_management_006.md
doc_business_credit_cards_virtual_card_management_007.md
doc_business_credit_cards_virtual_card_management_008.md
doc_business_credit_cards_virtual_card_management_009.md
doc_business_credit_cards_virtual_card_management_010.md
doc_business_savings_accounts_automatic_sweep_program_001.md
doc_business_savings_accounts_automatic_sweep_program_002.md
doc_business_savings_accounts_automatic_sweep_program_003.md
doc_business_savings_accounts_automatic_sweep_program_004.md
doc_business_savings_accounts_automatic_sweep_program_005.md
doc_business_savings_accounts_automatic_sweep_program_006.md
doc_business_savings_accounts_automatic_sweep_program_007.md
doc_business_savings_accounts_automatic_sweep_program_008.md
doc_business_savings_accounts_bronze_saver_account_001.md
doc_business_savings_accounts_bronze_saver_account_002.md
doc_business_savings_accounts_bronze_saver_account_003.md
doc_business_savings_accounts_bronze_saver_account_004.md
doc_business_savings_accounts_bronze_saver_account_005.md
doc_business_savings_accounts_bronze_saver_account_006.md
doc_business_savings_accounts_bronze_saver_account_007.md
doc_business_savings_accounts_bronze_saver_account_008.md
doc_business_savings_accounts_diamond_vault_001.md
doc_business_savings_accounts_diamond_vault_002.md
doc_business_savings_accounts_diamond_vault_003.md
doc_business_savings_accounts_diamond_vault_004.md
doc_business_savings_accounts_diamond_vault_005.md
doc_business_savings_accounts_diamond_vault_006.md
doc_business_savings_accounts_diamond_vault_007.md
doc_business_savings_accounts_diamond_vault_008.md
doc_business_savings_accounts_emerald_saver_001.md
doc_business_savings_accounts_emerald_saver_002.md
doc_business_savings_accounts_emerald_saver_003.md
doc_business_savings_accounts_emerald_saver_004.md
doc_business_savings_accounts_emerald_saver_005.md
doc_business_savings_accounts_emerald_saver_006.md
doc_business_savings_accounts_emerald_saver_007.md
doc_business_savings_accounts_emerald_saver_008.md
doc_business_savings_accounts_gold_plus_saver_001.md
doc_business_savings_accounts_gold_plus_saver_002.md
doc_business_savings_accounts_gold_plus_saver_003.md
doc_business_savings_accounts_gold_plus_saver_004.md
doc_business_savings_accounts_gold_plus_saver_005.md
doc_business_savings_accounts_gold_plus_saver_006.md
doc_business_savings_accounts_gold_plus_saver_007.md
doc_business_savings_accounts_gold_saver_account_001.md
doc_business_savings_accounts_gold_saver_account_002.md
doc_business_savings_accounts_gold_saver_account_003.md
doc_business_savings_accounts_gold_saver_account_004.md
doc_business_savings_accounts_gold_saver_account_005.md
doc_business_savings_accounts_gold_saver_account_006.md
doc_business_savings_accounts_gold_saver_account_007.md
doc_business_savings_accounts_gold_saver_account_008.md
doc_business_savings_accounts_gold_saver_account_009.md
doc_business_savings_accounts_joint_business_holders___user_roles_001.md
doc_business_savings_accounts_joint_business_holders___user_roles_002.md
doc_business_savings_accounts_joint_business_holders___user_roles_003.md
doc_business_savings_accounts_joint_business_holders___user_roles_004.md
doc_business_savings_accounts_joint_business_holders___user_roles_005.md
doc_business_savings_accounts_joint_business_holders___user_roles_006.md
doc_business_savings_accounts_joint_business_holders___user_roles_007.md
doc_business_savings_accounts_joint_business_holders___user_roles_008.md
doc_business_savings_accounts_joint_business_holders___user_roles_009.md
doc_business_savings_accounts_platinum_reserve_account_001.md
doc_business_savings_accounts_platinum_reserve_account_002.md
doc_business_savings_accounts_platinum_reserve_account_003.md
doc_business_savings_accounts_platinum_reserve_account_004.md
doc_business_savings_accounts_platinum_reserve_account_005.md
doc_business_savings_accounts_platinum_reserve_account_006.md
doc_business_savings_accounts_platinum_reserve_account_007.md
doc_business_savings_accounts_platinum_reserve_account_008.md
doc_business_savings_accounts_platinum_reserve_account_009.md
doc_business_savings_accounts_platinum_reserve_account_010.md
doc_business_savings_accounts_silver_plus_saver_001.md
doc_business_savings_accounts_silver_plus_saver_002.md
doc_business_savings_accounts_silver_plus_saver_003.md
doc_business_savings_accounts_silver_plus_saver_004.md
doc_business_savings_accounts_silver_plus_saver_005.md
doc_business_savings_accounts_silver_plus_saver_006.md
doc_business_savings_accounts_silver_plus_saver_007.md
doc_business_savings_accounts_silver_saver_account_001.md
doc_business_savings_accounts_silver_saver_account_002.md
doc_business_savings_accounts_silver_saver_account_003.md
doc_business_savings_accounts_silver_saver_account_004.md
doc_business_savings_accounts_silver_saver_account_005.md
doc_business_savings_accounts_silver_saver_account_006.md
doc_business_savings_accounts_silver_saver_account_007.md
doc_business_savings_accounts_silver_saver_account_008.md
699


→ assistant calls `shell` with `{"command": "for c in business_bronze_rewards_card business_silver_rewards_card business_gold_rewards_card business_platinum_rewards_card green_rewards_card silver_zoom_card; do echo \"=========== $c\"; cat doc_business_credit_cards_${c}_*.md | grep -iE \"cash back|reward|%|media|stream|entertain|apple|music|limit|cap|exclu|fee|score|require\" | head -80; done"}`

**tool:** =========== business_bronze_rewards_card
# Business Bronze Rewards Card: How to Apply and Get Started
- Meet a minimum personal FICO score of $660.
- If your business has established credit, a Dun & Bradstreet PAYDEX score of 20 or higher is helpful; it is not required for new businesses.
- Earn 1.0% cash back on eligible purchases.
- Annual fee: $0.00.
- Variable APR on carried balances: 20.49%.
4. If your business has a PAYDEX score, include it to help underwriting; a minimum of 20 is used when applicable.
- Plan larger purchases early so they count toward the $10,000 requirement within 2 months.# Business Bronze Rewards Card: Understanding Your 1% Cash Back Rewards
## Earning Cash Back
- You earn 1.0% cash back on eligible purchases charged to your account.
- Rewards are calculated on the net transaction amount after any credits or returns.
## Tracking Rewards
- Your rewards balance updates as purchases post to your account.
- You can monitor earned and available rewards in your account dashboard.
## Redeeming Rewards
- You can redeem once your available rewards balance is at least $37.
## Tips to Maximize Rewards
- Consolidate everyday business spending on your card to grow rewards faster.
- Avoid returns or cancellations that reduce your net eligible spend and earned rewards.# Business Bronze Rewards Card: Exceptions and Exclusions
## Merchants Not Eligible for Cash Back
The following specific merchants earn **0% cash back** on your Business Bronze Rewards Card:
The following SaaS merchants earn the 1.0% rate for the first 12 months of your subscription only. After the initial 12-month period, auto-renewals and continued subscription payments to these merchants earn 0% cash back:
- Salesforce# Business Bronze Rewards Card: When and How Are Cash Back Rewards Paid?
## When Rewards Are Earned
- You earn cash back at the rate of 1.0% on eligible purchases when transactions post to your account.
- If a purchase is returned or credited, the associated rewards are reversed.
## How Rewards Are Paid
- Once your available rewards balance reaches $37, you can request redemption.
- Keep receipts and invoices for any disputed charges; adjustments can affect your rewards balance.# Business Bronze Rewards Card: Credit Limit and Spending Guidelines
## Your Credit Limit
- Underwriting considers your personal credit standing; applications should meet at least $660. If available, a business PAYDEX score of 20 or higher may be considered.
- Keep spending within your assigned credit limit and monitor available credit before initiating large transactions.
- For purchases made outside the United States or with non‑USD merchants, a foreign transaction fee of 2.75% applies to the converted amount.
- Provide up-to-date financial information to support your request and allow time for underwriting review.# Business Bronze Rewards Card: No Annual Fee - What Does This Mean?
## What "No Annual Fee" Means for You
- You are not charged an annual membership fee; the annual fee is $0.00.
- You can keep the account open for ongoing business spending without planning for an annual renewal bill.# Business Bronze Rewards Card: Making Payments and Managing Your Balance
- If you do not pay your statement balance in full by the due date, interest accrues at a variable rate of 20.49%.
## Avoiding Fees and Managing Cash Flow
- Pay at least your minimum due by the due date on your statement to avoid a late fee of $32.50.
- Consider paying more frequently to reduce average daily balance and interest charges when carrying a balance at 20.49%.
- A late payment fee of $32.50 may be assessed, and interest continues to accrue at 20.49% on unpaid balances.
- Make the past‑due payment as soon as possible to minimize additional interest.# Business Bronze Rewards Card: Travel Benefits and Purchase Protection
- Purchases made outside the United States or with non‑USD merchants incur a foreign transaction fee of 2.75% on the converted amount.
- Some merchants abroad may process transactions in U.S. dollars; confirm the currency at checkout to understand whether the 2.75% will apply.
- If a transaction is refunded, any associated rewards will be reversed accordingly.# Business Bronze Rewards Card: Adding Employee Cards to Your Account
- If an employee leaves your company, lock or cancel their card immediately and reassign remaining budget as needed.
- Review transaction activity regularly to ensure spending aligns with your internal policies.# Business Bronze Rewards Card: New Customer Promo Offer Details
- Track your progress toward the $10,000 requirement in your account dashboard.
- Keep documentation of purchases in case a transaction posts later than expected.# Business Bronze Rewards Card: Referral Program - Earn Rewards for Referring Businesses
3. You will be notified when the referral is approved and has met the $1,750 requirement within 90 days.
- Encourage referrals to complete their initial purchases early to meet the $1,750 requirement within 90 days.
=========== business_silver_rewards_card
# Business Silver Rewards Card: Getting Started and Eligibility Requirements
## Eligibility requirements
- Minimum personal credit score (FICO): $700
## Credit limits
- Approved credit limits generally range from $17,500 to $112,500, subject to credit review and business financials
- Annual fee: $122.50
- First-year promotional annual fee for new customers: 0
- APR on carried balances: 18.99%
- If your business has a PAYDEX score, ensure vendor payment histories are up to date to support at least 47
- Have recent financial documentation available if requested, such as bank statements or accountant-prepared records# Business Silver Rewards Card: How to Earn 10% Back on Travel and Software Purchases
- Earn 10.0% cash back on eligible travel and software purchases
- Earn 1.0% cash back on all other purchases
- If a merchant sells multiple categories of goods or services, your charge must be coded under a qualifying category to receive 10.0%
- Keep receipts and confirmations in case a merchant’s category coding needs review# Business Silver Rewards Card: Which Travel Purchases Qualify for 10% Cash Back?
## Eligible travel categories for 10.0%
- Split purchases may post in separate categories; only the portion coded as travel will earn 10.0%# Business Silver Rewards Card: Software and SaaS Spending - Qualifying Categories
## Software categories that typically qualify for 10.0%
- If a vendor sells both services and physical goods, only the portion coded as software will earn the 10.0% rate# Business Silver Rewards Card: Exceptions and Exclusions
## Important exclusions
The following specific merchants do NOT earn the 10.0% bonus rate and instead earn the standard 1.0% rate:
**Corporate Expense Platform Merchants (Travel Exclusion):**
**Hardware/Electronics Merchants (Software Exclusion):**
- Apple
**Gaming Subscription Merchants (Software Exclusion):**
**Online Learning Platform Merchants (Software Exclusion):**
- Pluralsight# Business Silver Rewards Card: Understanding Your Included Travel Insurance Coverage
## Coverage limits and documentation
- The maximum payable amount per trip is $15,000, subject to policy terms, conditions, and exclusions
- Claims require proof of payment with your card and supporting documentation such as itineraries, receipts, confirmations, and official reports when applicable
- Report incidents as soon as possible and follow the claim instructions provided by the administrator# Business Silver Rewards Card: How to File a Travel Insurance Claim
- Approved claims are paid up to $15,000 per trip, consistent with policy limits and exclusions
- Once processed, you will receive confirmation of the outcome and any approved reimbursement method# Business Silver Rewards Card: Purchase Protection Benefits Explained
- The maximum reimbursement per covered claim is $7,500, subject to terms, conditions, and exclusions
- For high-value items up to $7,500, store serial numbers and photos to streamline verification
- If a merchant offers a replacement or repair, follow those steps and include any paperwork in your claim submission# Business Silver Rewards Card: Annual Fee and Fee Waiver Options
## Standard annual fee
- The standard annual fee for card membership is $122.50
## First-year fee waiver for new customers
- New customers receive a first-year annual fee of 0
- After the first year, the annual fee returns to $122.50 for subsequent years
## How to confirm your fee status
- Review your first statement to verify that the fee reflects 0 if you qualified as a new customer
- If you have questions about eligibility or billing, contact support with your account details# Business Silver Rewards Card: Maximizing Rewards on Business Travel and Tech Expenses
- Route eligible travel and software transactions to this card to earn 10.0%
- Use this card for other purchases when convenient to earn 1.0%
## Leverage credits and minimize fees
- For international trips, consider the foreign transaction fee of 1.5% when deciding which card to use abroad
- Assign each card to the employee or department most likely to incur travel or software charges to capture 10.0%
- If a charge appears misclassified, request a review so the correct earning rate can be applied# Business Silver Rewards Card: New Customer First Year Fee Waiver
- New customers who open an account during the promotional window receive a first-year annual fee of 0
- After the first year, the standard annual fee of $122.50 applies
- The fee waiver is applied to the first year of card membership at 0
- Review your cardholder agreement and first statement for confirmation of the waived fee
- If the statement does not reflect 0, contact support so your account can be reviewed for promotional eligibility# Business Silver Rewards Card: Double Cash Back Promo
## Limited-time offer: double your cash back
New Business Silver Rewards Card customers can earn 2x cash back on all purchases for the first 6 months after opening their account.
- Travel and software purchases that normally earn 10.0% are multiplied by 2 during the promotional period
- Other purchases that normally earn 1.0% are multiplied by 2 during the promotional period
- The increased rewards rate is automatically applied to qualifying transactions; no separate enrollment is required
- Merchant category exclusions continue to apply (see Exceptions and Exclusions)
- Cash back is credited to your account at the end of each billing cycle consistent with program rules# Business Silver Rewards Card: Referral Program - Earn Rewards for Referring Businesses
- Earn $100 for each successful referral that is approved and meets the qualifying spend requirement
## Qualifying spend requirement
=========== business_gold_rewards_card
# Business Gold Rewards Card: Premium Business Credit Card Overview
- Minimum personal credit score required: $735
- Minimum business credit score required on the PAYDEX scale: 67
## Rewards Structure
- Operations spending earns cash back at 2.5%.
- All other purchases earn cash back at 1.0%.
- Annual fee: $200.00
- APR for carried balances: 17.99%
## Credit Limits
- Initial credit limits start at $37,500 and may reach up to $225,000, subject to credit approval and ongoing account review.
- Ensure your business credit profile is up to date to meet the 67 requirement.
- Prepare recent financial statements to support a credit limit request within the $37,500 to $225,000 range.# Business Gold Rewards Card: Earning 2.5% Cash Back on Operations Spending
- Eligible operations purchases earn cash back at 2.5%.
- Purchases that do not qualify as operations spend earn cash back at 1.0%.
## Posting of Rewards
- Cash back is calculated based on the merchant category code assigned by the merchant’s payment processor.
- Rewards typically appear after the transaction posts and clears. Returns or credits reduce earned cash back in the same statement period where possible.
- For recurring bills, confirm the merchant category remains aligned with operations functions to retain the 2.5% rate.
- If a purchase you believe is operations-related does not earn at 2.5%, contact support with the transaction details and merchant information for review.
- Centralize routine operational expenses on the card to maximize the 2.5% rate.
- Keep receipts and invoices that clearly reflect the operational nature of the purchase to support any category review.# Business Gold Rewards Card: What Qualifies as Operations Spend?
Transactions commonly recognized as operations spend and eligible for 2.5% may include:
- Transactions categorized as entertainment, personal, or non-business use
- If a transaction is misclassified, request a review and provide documentation that demonstrates its operational purpose.# Business Gold Rewards Card: How to Access Airport Lounges Worldwide
- Review participating lounges and entry requirements before you travel.
- Present your Business Gold Rewards Card and a valid boarding pass matching the name on the card account.
- If you exhaust the 9 allocation, standard lounge entry fees or restrictions may apply at the lounge’s discretion.
- Carry a digital and physical card where possible to ensure successful scanning at different lounges.# Business Gold Rewards Card: Lounge Access Network and Participating Airports
- Operating hours, capacity controls, and guest policies vary by lounge and are subject to change by the lounge operator.
- Check lounge capacity status before arriving at the airport when available.
- If a lounge is unavailable, search nearby terminals or partner lounges within the same network.# Business Gold Rewards Card: Adding Authorized Users with Lounge Access
- Authorized users may access participating lounges subject to the same entry requirements as the primary cardholder.
- Revoke or replace cards promptly when roles change to maintain security and benefit control.# Business Gold Rewards Card: Annual Fee and Premium Benefits Guide
## Annual Fee
- The annual fee for the Business Gold Rewards Card is $200.00. It bills to your primary account each anniversary.
- Foreign transaction fee: 0%. Use your card abroad without added currency conversion fees from this charge.
- Set alerts for large travel purchases to verify that credits post as expected.# Business Gold Rewards Card: Travel Perks and Concierge Services
- Foreign transaction fee: 0%. This enables international purchases without added foreign transaction charges from this fee.
- Keep confirmation emails accessible in case the concierge or credits team requests verification.# Business Gold Rewards Card: Maximizing Your Operations Rewards
- Route recurring operational expenses to your card to earn 2.5%.
- Issue up to 37 employee cards to capture decentralized operational purchases under the same rewards structure.
- When operations coding is not applicable, you still earn 1.0%.
- Centralize billing for software and utilities under a single account to standardize coding and reduce missed rewards.
- Review statements for returns or credits that may adjust previously earned cash back, and verify corrections.
- Keep documentation for high-value operational purchases in case you need to substantiate category eligibility.# Business Gold Rewards Card: Referral Program - Earn Rewards for Referring Businesses
## Qualifying Spend Requirement
=========== business_platinum_rewards_card
# Platinum Rewards Card: Elite Business Credit Card Application
- Minimum personal credit score required: $765
- Minimum business credit score (PAYDEX) for established businesses: 77
- Carried balance APR: 16.99%
## Credit Limits
- Minimum approved credit limit: $75,000
- Maximum available credit limit: $400,000
- Provide recent business financials and bank references to support the requested credit limit within the available range.
- Requested limits are aligned with financial capacity and may be adjusted within $75,000 to $400,000.
- If approved, your account will reflect the applicable APR for carried balances of 16.99%.
- Match your requested limit to demonstrated cash flow and liquidity.
- Respond promptly to any verification requests to avoid processing delays.# Business Platinum Rewards Card: Earning 4% Cash Back
- Travel, software, and media advertising purchases earn 4.0%.
- Purchases outside those categories earn 1.5%.
## How Rewards Accrue
- Cash back is calculated on net purchases after returns or credits.
- Transactions must be submitted by merchants under eligible category codes to receive the enhanced rate of 4.0%.
- Cash equivalents, balance transfers, and fees do not earn cash back.
- Keep invoices and receipts for category review if you believe a purchase should qualify for 4.0%.# Business Platinum Rewards Card: Which Media and Advertising Expenses Qualify?
## Qualifying Media and Advertising Purchases
- Digital advertising on major online platforms when processed under advertising or media merchant categories.
- Search engine marketing, social media ads, programmatic display, and retargeting buys.
- Streaming audio and video media placements when billed by media networks or ad platforms.
- Direct buys from publishers and networks where the merchant category code reflects advertising or media services.
- Public relations services, talent fees, or event sponsorships coded outside advertising or media categories.
- Invoices routed through non-media intermediaries that use general business services merchant codes.
- Qualifying media and advertising transactions earn 4.0% when the merchant submits the charge under an eligible category.
- Request direct billing from the advertising or media platform rather than via agencies.
- Verify how the merchant classifies transactions before completing large media buys.
- Keep documentation for any review in case a charge posts under a non-qualifying category.# Business Platinum Rewards Card: How to Use the Concierge Travel Desk
- Provide traveler profiles and company travel guidelines to streamline booking.
- Fare or rate preferences, cabin or room types, loyalty program IDs, and any approval requirements.
- Save confirmation emails or documents for reconciliation and expense reporting.# Business Platinum Rewards Card: Travel Planning and Booking Assistance Services
- Concierge travel desk availability: Yes# Business Platinum Rewards Card: Premium Travel Benefits and Insurance Coverage
- Unlimited lounge visits included: Yes
- Foreign transaction fee on international purchases: 0%
- Pay in local currency where appropriate to maximize acceptance and clarity of charges.# Business Platinum Rewards Card: Software and Cloud Services - Qualifying Purchases
- When merchants classify transactions as software or cloud services, you earn 4.0%.
- If a purchase does not fall under those categories, it earns 1.5%.
## Common Exclusions
- Confirm merchant category coding for large renewals to ensure the enhanced rate of 4.0%.
- Keep invoices for category review if a charge appears misclassified.# Business Platinum Rewards Card: Understanding Your Annual Fee and Premium Perks
## Annual Fee Structure
- Standard annual fee: $450.00
- First-year promotional annual fee: 0
- Airport lounge visits unlimited: Yes
- Foreign transaction fee on international purchases: 0%
## How Fees Are Assessed
- The annual fee is billed to your account and renews automatically unless the account is closed before renewal.
- If the promotional first-year fee applies, your first statement will reflect 0 for that period.
- Add traveler and loyalty details to streamline lounge access and insurance eligibility.
- Use your card for covered purchases to activate benefits tied to $17,500 and $62,500.# Business Platinum Rewards Card: Dedicated Account Support and Priority Service
- Receive proactive reviews of account activity to align limits and controls with your business needs.# Business Platinum Rewards Card: Business Travel Itinerary Management
- Concierge travel desk availability: Yes# Business Platinum Rewards Card: New Customer First Year Fee Waiver
## First-Year Annual Fee
- First-year annual fee for new customers: 0
- After the first year, the standard annual fee applies: $450.00
- The fee waiver is reflected on your account during the first year of membership, provided your account is opened within the promotion dates.
- The account must remain open and in good standing to retain the waived first-year fee.# Business Platinum Rewards Card: Referral Program - Earn Rewards for Referring Businesses
## Annual Limits
- Bonuses are credited to your account after verification that approval and qualifying spend requirements have been met.
=========== green_rewards_card
# Green Rewards Card: Apply for a Sustainable Business Credit Card
- Newer businesses: If you have limited business credit history, your personal profile can support your application.
| Annual fee | $100.00 |
| Purchase APR | 19.99% |
| Foreign transaction fee | 1.0% |
## Credit Limits
- Your limit is determined by factors such as revenue consistency, existing obligations, and overall credit profile.
- Review your assigned limit and applicable pricing, including $100.00, 19.99%, and 1.0%.
- Add the card to your digital wallet and configure expense categories for smoother reconciliation.# Green Rewards Card: How the Sustainability Tracker Works
- Encourage consistent receipt tagging to improve activity-level reporting.# Green Rewards Card: Understanding Your Carbon Footprint Report
- Keep the exported file with your accounting records so that your emissions narrative and financials remain aligned.# Green Rewards Card: Earning Rewards on Eco-Friendly Purchases
## Cash Back Earning Rates
| Purchase Type | Cash Back Rate |
| Sustainable/eco-friendly merchants | 3.0% |
| Other purchases | 1.0% |
## How Rewards Accrue
- Rewards are tracked per transaction and consolidate at the account level.
- Purchases at qualifying sustainable merchants earn 3.0%, while all other eligible purchases earn 1.0%.
- Cash equivalents, balance transfers, and certain fees do not earn rewards.
- Reversed or refunded transactions forfeit associated rewards in the period the adjustment posts.
## Managing and Viewing Rewards
- Track rewards by merchant category and by cardholder in your rewards dashboard.
- Use category filters to confirm which transactions qualified for 3.0%.
- Encourage team members to use the card at qualifying sustainable merchants whenever applicable.# Green Rewards Card: Which Merchants Are Considered Sustainable?
- Check for sustainability badges in your transaction feed where available.
## Rewards Reminder
- Purchases at qualifying sustainable merchants earn 3.0% cash back when processed under eligible categories.# Green Rewards Card: Carbon Offset Programs and Options
- Use the per-dollar rate of $0.06 and the tree threshold of $3,000 to plan targeted initiatives.# Green Rewards Card: Viewing Your Monthly Environmental Impact Dashboard
- Revisit settings to ensure your view matches your monthly preference.# Green Rewards Card: Carbon Reporting for Business Accounting
- Establish a workflow to review and approve the monthly file before posting entries.# Green Rewards Card: Setting Sustainability Goals for Your Business
- Translate the target into required card spend using $0.06 as your conversion factor.
- Share progress with stakeholders to maintain accountability.# Green Rewards Card: Referral Program - Earn Rewards for Referring Businesses
## Earning Limits and Timing
- Qualifying purchases exclude cash equivalents, certain fees, and returned transactions.
- View referral status in your rewards section, including approval state and qualifying spend progress.
- Encourage referrals to complete required steps promptly so their spend can count toward the qualification window.
=========== silver_zoom_card
## Fuel and rewards structure tailored to fleets
- Receive a recurring fuel price reduction of 10.0% at participating stations
- Earn elevated rewards on sector-specific spend at 3.0%
- Earn a flat rate on all other eligible purchases at 1.0%
- New customer fuel financing promotion: an APR of 0% on fuel purchases for the first year, subject to eligibility and timing rules
- The combination of 10.0% off at the pump and ongoing rewards helps reduce total cost per mile
- Eligible transactions receive a price reduction or statement adjustment equal to 10.0% of the fuel subtotal when processed by a participating location
- If the reduction does not appear immediately, it may post as an adjustment after the transaction settles
- If a location appears out of network, choose a nearby participating station to receive the full 10.0% benefit# Silver Zoom Card: Which Gas Stations and Fuel Networks Are Included?
- At participating sites, eligible fuel purchases qualify for a reduction equal to 10.0%
- Review transaction details to see discounts taken at 10.0% and rewards accruals on eligible transportation categories at 3.0%
- Discounted fuel savings can be approximated as: fuel subtotal multiplied by 10.0% at participating locations
- Rewards on eligible transportation and logistics expenses accrue at 3.0%
- Prioritize in-network stations to capture the 10.0% reduction consistently
- Concentrate eligible spend under transportation and logistics categories to earn 3.0%
- Match receipts to statement entries where the 10.0% discount and rewards appear as separate line items
- Confirm how many employee cards you need and ensure the total requested is within the account cap of 60
- Review card issuance regularly to remain within the limit of 60# Silver Zoom Card: Setting Spending Limits for Driver Cards
- Set a daily cap per card up to $1,250 to manage fueling and incidental expenses
- Apply limits consistently across as many as 60 issued under your account
## How to set limits
- Choose whether the limit applies to all purchases or only to specific categories such as fuel
- Adjust limits based on route distance, vehicle type, and current fuel pricing trends
- Temporarily raise or lower the threshold for unusual routes, then restore the standard limit afterwards
- Review declines associated with limit controls and refine thresholds as needed# Silver Zoom Card: Fuel Purchase Reporting and Analytics
- A fuel summary view highlights total fuel spend and reductions applied at 10.0%
- Category earnings reports display rewards accrued on eligible transportation and logistics expenses at 3.0%
- Discount rate field: confirms application of 10.0% to eligible fuel transactions
- Category rate field: confirms earning at 3.0% for qualifying spend
- Filter by merchant category and participating network to isolate transactions with the 10.0% reduction
- Export to your accounting system to reconcile reductions and rewards against route plans
- Periodically audit a sample of receipts against reported discounts and rewards# Silver Zoom Card: Eligibility Requirements for Logistics Businesses
## Credit profile requirements
## Credit limits
- Carried balances accrue interest at 20.99%
- Annual membership fee is $74.50
- Set a per‑driver daily spending cap at or below $1,250 to match expected fueling needs
- Use category controls to ensure the cap applies primarily to fuel and approved logistics expenses
- For fueling outside your home country, transactions are subject to a foreign transaction fee of 2.5%
- Plan routes and budgets with this fee in mind when operating internationally
- Tag purchases by vehicle or route to streamline cost-per-mile analysis
- Review exceptions where spend nears or exceeds the $1,250 threshold and adjust controls if necessary# Silver Zoom Card: New Customer Promo - 0% APR on Fuel Purchases
- An introductory APR of 0% on fuel purchases for the first year for new customers who qualify under the promotional window
- Once the promotional fuel period ends, new fuel purchases and any remaining promotional balances accrue interest at your standard rate of 20.99% when carried
- Continue to make at least the minimum required payments by each statement due date to keep the promotional terms in effect# Silver Zoom Card: Referral Program - Earn Rewards for Referring Logistics Businesses
- Earn a referral bonus of $150 for each referred logistics business that is approved and meets the qualifying spend requirement
- Bonuses are capped at 10 per calendar year
- Bonuses are typically issued to the referrer after the referred business satisfies the $7,500 requirement within 90 days
- If the referred business returns purchases or does not meet the requirement, no bonus is issued# Silver Zoom Card: Triple Cash Back Promo - Terms and Conditions
- The triple cash back earning period lasts 9 months from account opening once properly claimed
- Transportation and logistics category purchases earn 3 × 3.0% during the promo period
- All other eligible purchases earn 3 × 1.0% during the promo period
## Important limitations
- After 9 months, earnings revert to the standard rates of 3.0% and 1.0%# Silver Zoom Card: How to Claim Your Triple Cash Back Promo
- Select the triple cash back offer and confirm your acceptance within the claim window


→ assistant calls `shell` with `{"command": "grep -il \"apple\\|streaming\\|music\" doc_business_credit_cards_* doc_credit_cards_* | xargs -I{} sh -c 'echo \"=== {}\"; cat {}'"}`

**tool:** === doc_business_credit_cards_business_platinum_rewards_card_003.md
# Business Platinum Rewards Card: Which Media and Advertising Expenses Qualify?

## Qualifying Media and Advertising Purchases
- Digital advertising on major online platforms when processed under advertising or media merchant categories.
- Search engine marketing, social media ads, programmatic display, and retargeting buys.
- Streaming audio and video media placements when billed by media networks or ad platforms.
- Direct buys from publishers and networks where the merchant category code reflects advertising or media services.

## What Typically Does Not Qualify
- Creative agency retainers, consulting, or production costs if billed under professional services categories.
- Public relations services, talent fees, or event sponsorships coded outside advertising or media categories.
- Invoices routed through non-media intermediaries that use general business services merchant codes.

## Earning Rate
- Qualifying media and advertising transactions earn 4.0% when the merchant submits the charge under an eligible category.

## Tips to Ensure Proper Categorization
- Request direct billing from the advertising or media platform rather than via agencies.
- Verify how the merchant classifies transactions before completing large media buys.
- Keep documentation for any review in case a charge posts under a non-qualifying category.=== doc_business_credit_cards_business_silver_rewards_card_005.md
# Business Silver Rewards Card: Exceptions and Exclusions

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
- Pluralsight=== doc_credit_cards_credit_cards__general__005.md
# Credit Card Rewards Categories

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
- This document lists categories only and does not specify reward rates for any card.=== doc_credit_cards_credit_cards__general__019.md
# Understanding Your Credit Card Rewards Categories

One of the great benefits of your Rho-Bank credit card is the ability to earn rewards on your everyday purchases. To help you maximize your earnings, it's important to understand how reward categories work and which types of purchases qualify for bonus rewards on your specific card.

## How Reward Categories Work

Every Rho-Bank credit card earns rewards on all purchases, but certain categories of spending may earn you bonus rewards at a higher rate. Your card has a base reward rate that applies to all purchases, plus bonus categories where you can earn even more. The specific bonus categories and rates vary depending on which Rho-Bank credit card you have.

## Available Reward Categories

Rho-Bank recognizes the following spending categories for rewards purposes:

- **Travel**: Flights, hotels, car rentals, cruises, and other travel-related purchases
- **Dining**: Restaurants, cafes, bars, and food delivery services
- **Groceries**: Supermarkets, grocery stores, and specialty food shops
- **Entertainment**: Movies, concerts, sporting events, and streaming services
- **Shopping**: Retail stores, department stores, and online marketplaces
- **Utilities**: Electric, gas, water, internet, and phone bills
- **Transportation**: Rideshare services, public transit, parking, and tolls
- **Software**: Software subscriptions, licenses, and cloud services
- **Media**: Advertising spend and digital media purchases
- **Green/Sustainable**: Eco-friendly merchants and sustainable products
- **Operations**: Business operational expenses

## Maximizing Your Rewards

To get the most out of your rewards, we recommend using your Rho-Bank credit card for purchases in your card's bonus categories whenever possible. You can find your specific card's bonus categories and reward rates by logging into your account on the Rho-Bank mobile app or website, or by reviewing your card's terms and conditions.

If you have any questions about which category a particular merchant falls into, or how to maximize your rewards earning potential, our customer service team is always happy to help. Give us a call at 1-800-RHO-BANK or chat with us through the mobile app.

→ assistant calls `shell` with `{"command": "cat doc_business_credit_cards_business_platinum_rewards_card_009.md doc_business_credit_cards_silver_zoom_card_0{06,09,10,11,12,13}.md doc_business_credit_cards_business_bronze_rewards_card_010.md; grep -il \"promo\" doc_business_credit_cards_*"}`

**tool:** # Business Platinum Rewards Card: Dedicated Account Support and Priority Service

## Dedicated Account Management
- Dedicated account manager assigned: Yes
- Your manager assists with card setup, policy configuration, and ongoing optimization.

## Employee Card Management
- Maximum number of employee cards that can be issued: 75
- Request cards with tailored spending controls and category restrictions.
- Replace, suspend, or reissue cards quickly through your manager or the admin portal.

## Priority Support
- Escalate urgent items such as fraud concerns, credit line adjustments, and travel booking issues.
- Receive proactive reviews of account activity to align limits and controls with your business needs.# Silver Zoom Card: Setting Spending Limits for Driver Cards

## What you can control
- Set a daily cap per card up to $1,250 to manage fueling and incidental expenses
- Apply limits consistently across as many as 60 issued under your account

## How to set limits
- Go to the card controls section for the selected driver card
- Enable daily spending controls and set the threshold at or below $1,250
- Choose whether the limit applies to all purchases or only to specific categories such as fuel
- Save and confirm the settings, then notify the driver of the available daily budget

## Ongoing management
- Adjust limits based on route distance, vehicle type, and current fuel pricing trends
- Temporarily raise or lower the threshold for unusual routes, then restore the standard limit afterwards
- Review declines associated with limit controls and refine thresholds as needed# Silver Zoom Card: Managing Multiple Vehicle Expenses

## Structure your fleet spending
- Issue driver cards to team members who need purchasing authority, up to a total of 60 under your account
- Assign cards to specific vehicles or routes for clearer allocation and reconciliation

## Control daily exposure
- Set a per‑driver daily spending cap at or below $1,250 to match expected fueling needs
- Use category controls to ensure the cap applies primarily to fuel and approved logistics expenses

## International purchases
- For fueling outside your home country, transactions are subject to a foreign transaction fee of 2.5%
- Plan routes and budgets with this fee in mind when operating internationally

## Reconciliation workflow
- Tag purchases by vehicle or route to streamline cost-per-mile analysis
- Review exceptions where spend nears or exceeds the $1,250 threshold and adjust controls if necessary# Silver Zoom Card: New Customer Promo - 0% APR on Fuel Purchases

## What the promotion offers
- An introductory APR of 0% on fuel purchases for the first year for new customers who qualify under the promotional window

## Promotional window
- Accounts opened within 2025-09-15 through 2025-12-31 are eligible, subject to approval and program rules

## After the promotional period
- Once the promotional fuel period ends, new fuel purchases and any remaining promotional balances accrue interest at your standard rate of 20.99% when carried

## How to make the most of it
- Channel fuel spend to participating stations during the promotional period to pair the financing benefit with eligible fuel savings
- Continue to make at least the minimum required payments by each statement due date to keep the promotional terms in effect# Silver Zoom Card: Referral Program - Earn Rewards for Referring Logistics Businesses

## How the referral program works
- Earn a referral bonus of $150 for each referred logistics business that is approved and meets the qualifying spend requirement
- Bonuses are capped at 10 per calendar year

## Qualifying activity for your referral
- The referred business must complete purchases totaling $7,500 within 90 days of account opening
- Only one referrer is credited per approved account

## Program timing
- No new referrals are accepted after 2025-10-01

## How to refer
- Share your referral link or submit the referral through your account portal before the referred business applies
- Ensure the referred business uses your link or is otherwise properly attributed during application
- Track referral status within your account until the qualifying spend threshold of $7,500 is met

## Payout details
- Bonuses are typically issued to the referrer after the referred business satisfies the $7,500 requirement within 90 days
- If the referred business returns purchases or does not meet the requirement, no bonus is issued# Silver Zoom Card: Triple Cash Back Promo - Terms and Conditions

## Eligibility and claim window
- You must claim the promotion within the window starting 2024-01-01 and ending 2025-12-31
- Claims outside this period are not eligible

## Promo duration and scope
- The triple cash back earning period lasts 9 months from account opening once properly claimed
- The multiplier applies only during the defined promo duration

## How the multiplier applies
- Transportation and logistics category purchases earn 3 × 3.0% during the promo period
- All other eligible purchases earn 3 × 1.0% during the promo period

## Important limitations
- The multiplier applies to standard rates only and does not stack with other overlapping category accelerators beyond the defined 3
- Returns or credits reduce promotional earnings accordingly
- If the promo is not claimed within 2024-01-01 to 2025-12-31, no retroactive adjustments are made

## End of promotion
- After 9 months, earnings revert to the standard rates of 3.0% and 1.0%# Silver Zoom Card: How to Claim Your Triple Cash Back Promo

## Before you start
- Confirm that you are within the eligible claim dates of 2024-01-01 through 2025-12-31
- Ensure your account is open and in good standing

## Claim and activation
- Open your account portal and navigate to promotional offers
- Select the triple cash back offer and confirm your acceptance within the claim window
- Watch for an on‑screen or email confirmation that the offer is active

## After you claim
- Your promotional earning period runs for 9 months from account opening
- During the promo period, your earnings reflect a multiplier of 3
- Monitor your statements to verify promotional earnings are posting correctly

## If you miss the window
- Claims submitted after 2025-12-31 are not eligible for activation
- You can continue earning at your standard rates if the promo is not activated in time# Business Bronze Rewards Card: New Customer Promo Offer Details

## Promotion Window
- This offer is available from 2025-10-01 through 2026-03-31.

## How to Qualify
- Make net new purchases totaling $10,000 within the first 2 months of account opening.
- After qualification, a $500 statement credit will be applied to your account.

## What Counts Toward the Threshold
- Only net purchases count; returns or credits reduce your qualifying total.
- Transactions must post to your account within 2 months to be included.

## Tracking and Fulfillment
- Track your progress toward the $10,000 requirement in your account dashboard.
- The $500 will appear on a future statement after qualification is confirmed.

## Tips for Success
- Plan larger purchases early in the period so they post within 2 months.
- Keep documentation of purchases in case a transaction posts later than expected.doc_business_credit_cards_business_bronze_rewards_card_001.md
doc_business_credit_cards_business_bronze_rewards_card_010.md
doc_business_credit_cards_business_gold_rewards_card_010.md
doc_business_credit_cards_business_platinum_rewards_card_008.md
doc_business_credit_cards_business_platinum_rewards_card_011.md
doc_business_credit_cards_business_silver_rewards_card_001.md
doc_business_credit_cards_business_silver_rewards_card_011.md
doc_business_credit_cards_business_silver_rewards_card_012.md
doc_business_credit_cards_silver_zoom_card_001.md
doc_business_credit_cards_silver_zoom_card_010.md
doc_business_credit_cards_silver_zoom_card_012.md
doc_business_credit_cards_silver_zoom_card_013.md


→ assistant calls `shell` with `{"command": "cat doc_business_credit_cards_business_silver_rewards_card_01{1,2}.md doc_business_credit_cards_business_platinum_rewards_card_0{02,07,08,11}.md doc_business_credit_cards_silver_zoom_card_0{02,03,04,05,07,08}.md doc_business_credit_cards_green_rewards_card_001.md doc_business_credit_cards_business_bronze_rewards_card_005.md | grep -v \"^$\""}`

**tool:** # Business Silver Rewards Card: New Customer First Year Fee Waiver
## Offer terms
- New customers who open an account during the promotional window receive a first-year annual fee of 0
- After the first year, the standard annual fee of $122.50 applies
## Promotional window
- Start date: 2025-11-15
- End date: 2026-01-15
## How to qualify
- Open your account within the promotional window and meet new customer eligibility criteria
- The fee waiver is applied to the first year of card membership at 0
## Verifying the waiver
- Review your cardholder agreement and first statement for confirmation of the waived fee
- If the statement does not reflect 0, contact support so your account can be reviewed for promotional eligibility# Business Silver Rewards Card: Double Cash Back Promo
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
- Cash back is credited to your account at the end of each billing cycle consistent with program rules# Business Platinum Rewards Card: Earning 4% Cash Back
## Eligible Categories for Enhanced Earnings
- Travel, software, and media advertising purchases earn 4.0%.
- Typical qualifying transactions include airfare, hotels, car rentals, rideshare, SaaS subscriptions, cloud hosting, and digital ad placements when coded to those categories.
## Other Purchases
- Purchases outside those categories earn 1.5%.
## How Rewards Accrue
- Cash back is calculated on net purchases after returns or credits.
- Transactions must be submitted by merchants under eligible category codes to receive the enhanced rate of 4.0%.
- Cash equivalents, balance transfers, and fees do not earn cash back.
## Best Practices
- Confirm that your vendors process payments under the appropriate merchant category.
- Keep invoices and receipts for category review if you believe a purchase should qualify for 4.0%.# Business Platinum Rewards Card: Software and Cloud Services - Qualifying Purchases
## Qualifying Software and Cloud Purchases
- SaaS subscriptions and enterprise software licenses billed by software providers.
- Cloud infrastructure, platform, and storage services billed by cloud vendors.
- Developer tools, analytics platforms, and productivity suites billed under software or cloud services categories.
## Earning Rates
- When merchants classify transactions as software or cloud services, you earn 4.0%.
- If a purchase does not fall under those categories, it earns 1.5%.
## Common Exclusions
- Hardware, peripherals, or on-premise equipment billed as physical goods.
- Professional services, implementation, or consulting billed under services categories.
- Telecom or internet access charges not categorized as software or cloud.
## Tips for Proper Categorization
- Purchase directly from software or cloud vendors rather than general resellers when possible.
- Confirm merchant category coding for large renewals to ensure the enhanced rate of 4.0%.
- Keep invoices for category review if a charge appears misclassified.# Business Platinum Rewards Card: Understanding Your Annual Fee and Premium Perks
## Annual Fee Structure
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
- Use your card for covered purchases to activate benefits tied to $17,500 and $62,500.# Business Platinum Rewards Card: New Customer First Year Fee Waiver
## Promotion Window
- Offer start date: 2025-11-01
- Offer end date: 2026-02-28
## First-Year Annual Fee
- First-year annual fee for new customers: 0
- After the first year, the standard annual fee applies: $450.00
## How the Waiver Is Applied
- The fee waiver is reflected on your account during the first year of membership, provided your account is opened within the promotion dates.
- If you upgrade or change products during the promotional period, the waiver continues only if the account remains eligible.
## Eligibility Guidelines
- Offer is for new customers only and applies to new accounts opened within the promotion window.
- The account must remain open and in good standing to retain the waived first-year fee.# Silver Zoom Card: How to Get Fuel Discounts at Participating Stations
## Confirm participation before you fuel
- Check that the station belongs to one of the 10 participating fuel networks
- Look for participation indicators at the pump or register, or verify within your account tools
## Using your card at the pump or inside
- Present your card at payment and ensure the merchant is processing the transaction under a participating fuel category
- Keep the receipt in case you need to reconcile discounts later
## How the discount is applied
- Eligible transactions receive a price reduction or statement adjustment equal to 10.0% of the fuel subtotal when processed by a participating location
- If the reduction does not appear immediately, it may post as an adjustment after the transaction settles
## Tips to ensure consistent savings
- Favor stations within the 10 supported networks on regular routes
- If a location appears out of network, choose a nearby participating station to receive the full 10.0% benefit# Silver Zoom Card: Which Gas Stations and Fuel Networks Are Included?
## Program coverage
- The fuel discount program spans 10 national and regional fuel networks
- Each participating station within these networks is eligible for the discount when transactions are processed under supported fuel categories
## How to verify a station
- Look for in-network signage or prompts indicating program acceptance
- Confirm participation in your account tools before fueling at unfamiliar locations
## Discount availability at included stations
- At participating sites, eligible fuel purchases qualify for a reduction equal to 10.0%
- Non-participating locations do not receive the network-based reduction
## Keeping your list current
- Network participation can change. Recheck a station's status periodically, especially along new routes# Silver Zoom Card: Tracking and Maximizing Your Fuel Savings
## Track your savings
- Review transaction details to see discounts taken at 10.0% and rewards accruals on eligible transportation categories at 3.0%
- Filter statements by merchant category and fuel tags to isolate qualifying activity
- Export transaction data for route-level or vehicle-level analysis
## Simple ways to quantify impact
- Discounted fuel savings can be approximated as: fuel subtotal multiplied by 10.0% at participating locations
- Rewards on eligible transportation and logistics expenses accrue at 3.0%
## Maximize the benefit
- Prioritize in-network stations to capture the 10.0% reduction consistently
- Concentrate eligible spend under transportation and logistics categories to earn 3.0%
- Schedule fueling at participating stations along planned routes to avoid out-of-network purchases
## Reconcile and validate
- Match receipts to statement entries where the 10.0% discount and rewards appear as separate line items
- Flag any transactions that did not receive the expected reduction for follow‑up# Silver Zoom Card: Adding Fleet Drivers to Your Account
## Before you add drivers
- Confirm how many employee cards you need and ensure the total requested is within the account cap of 60
## How to add driver cards
- Open your account management portal and navigate to the user or card management section
- Add a driver by entering their name and contact details, then submit the request for an additional card
- Assign role permissions appropriate to the driver’s responsibilities
- Monitor the card’s activation status and confirm receipt with the driver
## Best practices
- Maintain an active roster and promptly deactivate cards for drivers who no longer need access
- Separate dispatch and fueling roles if different employees handle purchases
- Review card issuance regularly to remain within the limit of 60# Silver Zoom Card: Fuel Purchase Reporting and Analytics
## Views that matter
- A fuel summary view highlights total fuel spend and reductions applied at 10.0%
- Category earnings reports display rewards accrued on eligible transportation and logistics expenses at 3.0%
## Key fields to review
- Discount rate field: confirms application of 10.0% to eligible fuel transactions
- Category rate field: confirms earning at 3.0% for qualifying spend
- Merchant and station identifiers: help you confirm in-network activity and compare station-level pricing
## How to analyze
- Filter by merchant category and participating network to isolate transactions with the 10.0% reduction
- Export to your accounting system to reconcile reductions and rewards against route plans
- Use variance checks to spot out-of-network fueling where the expected discount did not apply
## Reporting hygiene
- Tag transactions with vehicle or route identifiers for clearer allocation
- Periodically audit a sample of receipts against reported discounts and rewards# Silver Zoom Card: Eligibility Requirements for Logistics Businesses
## Credit profile requirements
- Personal credit is evaluated and must be at or above $680 as a secondary approval factor
- Business credit is evaluated and must be at or above 57 on the PAYDEX scale for logistics business approval
## Credit limits
- Approved accounts receive a credit line within a range that starts at $22,500 and can extend up to $175,000 based on underwriting
## Cost of credit and membership
- Carried balances accrue interest at 20.99%
- Annual membership fee is $74.50
## Additional considerations
- Meeting the minimum criteria of $680 and 57 does not guarantee approval
- Underwriting may consider business tenure, industry stability, and current financial performance
- Ensure all application details are accurate and complete to avoid delays# Green Rewards Card: Apply for a Sustainable Business Credit Card
## Eligibility
- Personal credit: You should meet a minimum FICO threshold of $690.
- Business credit: Established businesses are evaluated using a PAYDEX threshold of 42 in addition to personal credit history.
- Newer businesses: If you have limited business credit history, your personal profile can support your application.
## Costs and Rates
| Cost or Rate | Amount |
| --- | --- |
| Annual fee | $100.00 |
| Purchase APR | 19.99% |
| Foreign transaction fee | 1.0% |
## Credit Limits
- Approved credit lines are assigned within a range from $14,000 to $75,000.
- Your limit is determined by factors such as revenue consistency, existing obligations, and overall credit profile.
## Employee Cards
- You can issue up to 15 employee cards to team members.
- Set category-level controls, customize spending rules, and monitor transactions in real time.
## What You Need to Apply
- Business legal name and structure, tax identification, and operating address
- Beneficial owner and authorized officer details
- Recent business financial information and estimated monthly card spend
- Agreement to a personal guaranty where applicable
## How to Apply
- Complete the application in your online banking portal.
- Provide business and owner details, upload requested documents, and review disclosures.
- Submit your application for credit review. You will be notified when a decision is reached.
## After You Submit
- If approved, activate your card and set controls for any employee cards.
- Review your assigned limit and applicable pricing, including $100.00, 19.99%, and 1.0%.
- Add the card to your digital wallet and configure expense categories for smoother reconciliation.# Business Bronze Rewards Card: Credit Limit and Spending Guidelines
## Your Credit Limit
- Your initial credit line will be at least $10,000 and may be as high as $50,000 based on underwriting.
- Underwriting considers your personal credit standing; applications should meet at least $660. If available, a business PAYDEX score of 20 or higher may be considered.
## Responsible Spending
- Keep spending within your assigned credit limit and monitor available credit before initiating large transactions.
- Pay down balances proactively to free up available credit for upcoming expenses.
- Avoid declines by confirming vendor billing dates and authorization holds.
## International Use
- For purchases made outside the United States or with non‑USD merchants, a foreign transaction fee of 2.75% applies to the converted amount.
## Requesting a Credit Line Review
- You can request a credit line increase after demonstrating consistent payment behavior and responsible utilization.
- Provide up-to-date financial information to support your request and allow time for underwriting review.


→ assistant calls `get_current_time` with `{}`

**tool:** The current time is 2025-11-14 03:40:00 EST.

**assistant:** I compared all six Rho-Bank business cards for a single $100,000 Apple Music payment. Three can't take a charge that size, and the best of the rest is the Silver Zoom Card at about 3% back.

**Cards that can't take a $100,000 charge**
- **Business Bronze Rewards:** the highest credit limit is $50,000.
- **Green Rewards:** the highest credit limit is $75,000.
- **Business Silver Rewards:** the highest limit is $112,500, but Apple is excluded from its 10% bonus, so this purchase earns 1%. Its double cash back promo for new accounts ends today (11/14/2025). Even if you opened today and qualified, you'd get 2%, which is still lower than the options below. There's also a $122.50 annual fee.

**Your strongest options**

1. **Silver Zoom Card: about 3% (~$2,925 after the fee)**
   - Its triple cash back promo pays 3 × 1% = 3% on purchases like this for the first 9 months after you open the account. You have to claim it in your account portal by 12/31/2025.
   - On $100,000, that's about $3,000. Subtract the $74.50 annual fee and you net about $2,925.
   - Credit limits go up to $175,000.
   - The catch: this card is built for transportation and logistics businesses. You need a personal credit score of 680+ and a business credit score (Dun & Bradstreet PAYDEX) of 57+. If your business isn't in that industry, it may not be an option.

2. **Business Platinum Rewards Card: 1.5% to 4% ($1,500 to $4,000)**
   - It pays 4% on travel, software and media advertising, and 1.5% on everything else.
   - What you'd earn depends on how Apple codes the charge. Its media bonus is aimed at advertising, so a music subscription may not qualify. If the charge counts as software or media, you'd get $4,000. If not, $1,500.
   - Credit limits run from $75,000 to $400,000.
   - The $450 annual fee is waived in the first year for new customers who open by 2/28/2026.
   - You need a personal credit score of 765+ and, for established businesses, a PAYDEX score of 77+.

3. **Business Gold Rewards Card: 1% (~$800 after the fee)**
   - Its 2.5% bonus is for operations spending, which leaves out entertainment, so this purchase would earn 1%. That's about $1,000, minus the $200 annual fee.

**My recommendation:** If your business qualifies for the Silver Zoom Card, it's the most reliable choice at about 3%. The Business Platinum Card could pay more (4%), but only if Apple codes the charge as software or media, so check with Apple before you buy. If it doesn't qualify, you'd get 1.5%.

For either card, the purchase has to fit within the credit limit you're approved for. Underwriting sets that limit, and the top of each range isn't guaranteed.

Would you like more detail on either card or how to apply?

**user:** Okay, helpful. But I run a software development company — not transportation/logistics. So I’m guessing the Silver Zoom Card might not even be available to me.

Given that, can you make *one* recommendation for what I should apply for to maximize the return on this **single $100,000 Apple Music** charge?

And can you confirm the math in plain terms — like, worst case vs best case — for the card you recommend (cash back %, expected $ back, and any annual fee I’d actually pay this first year)?

**assistant:** **My recommendation is the Business Platinum Rewards Card.** You're probably right that the Silver Zoom Card wouldn't fit your business. Without it, the Platinum card beats every other Rho-Bank business card on this purchase, even if the charge earns only its base rate.

**The math for one $100,000 Apple Music charge:**

| | Worst case | Best case |
|---|---|---|
| When this happens | The charge is coded outside the bonus categories | Apple codes the charge as software or media |
| Cash back rate | 1.5% | 4.0% |
| Cash back | **$1,500** | **$4,000** |
| First-year annual fee | $0* | $0* |
| **Net return in year one** | **$1,500** | **$4,000** |

*The first-year fee is waived for new customers who open an account between 11/1/2025 and 2/28/2026, so you'd qualify if you apply now. The fee becomes $450 a year starting in year two.

**Why it wins:**
- **Its worst case beats the other cards.** 1.5% is higher than what the others would pay on this charge. Business Silver excludes Apple from its bonus, so it pays 1%, and the double cash back promo for new accounts ends today. Business Gold pays 1% and charges a $200 annual fee. Bronze and Green have credit limits below $100,000.
- **Its credit limits fit the purchase.** Approved limits run from $75,000 to $400,000.

**Before you apply:**
1. **You need strong credit scores:** a personal credit score of at least 765 and, for established businesses, a Dun & Bradstreet PAYDEX score of at least 77.
2. **Ask for a limit above $100,000.** If you're approved below that, the charge won't fit. Recent business financials and bank references help support a higher limit.
3. **Ask Apple how they'll code the charge.** The 4% rate depends on Apple billing it as software or media. Their system sets this, and Rho-Bank can't guarantee it. If you buy directly from Apple instead of through a reseller, it's more likely to qualify. Keep the invoice in case you need to ask for a category review.
4. **If you have the cash, pay the balance in full.** Carrying it costs 16.99% APR, and the interest would quickly outweigh the cash back.

Would you like help with anything else, like how to apply or how to redeem the cash back?

**user:** So if I understand correctly: with the **Business Platinum Rewards Card**, on my one-time **$100,000 Apple Music** purchase I’ll get **$1,500 back worst case (1.5%)** and up to **$4,000 back best case (4%)**, and **the first-year annual fee is $0** since I’d be opening within the promo window. Correct?

If yes, let’s apply for the **Business Platinum Rewards Card**.

→ assistant calls `shell` with `{"command": "grep -ril \"apply\\|application\" . | xargs grep -il \"tool\\|call\\|_[0-9]\\{4\\}\" | head -30; grep -rn \"apply_for\\|submit_.*application\\|application_[0-9]\\|open_.*card\" . | head"}`

**tool:** ./doc_credit_cards_gold_rewards_card_001.md
./doc_business_credit_cards_business_silver_rewards_card_012.md
./doc_bank_accounts_bank_accounts__general__043.md
./doc_credit_cards_ecocard_011.md
./doc_bank_accounts_bank_accounts__general__048.md
./doc_credit_cards_silver_rewards_card_005.md
./doc_business_checking_accounts_beige_010.md
./doc_bank_accounts_bank_accounts__general__047.md
./doc_credit_cards_credit_card_account_logistics_003.md
./doc_credit_cards_bronze_rewards_card_004.md
./doc_credit_cards_credit_cards__general__004.md
./doc_credit_cards_silver_rewards_card_011.md
./doc_business_credit_cards_silver_zoom_card_011.md
./doc_bank_accounts_bank_accounts__general__042.md
./doc_credit_cards_credit_cards__general__009.md
./doc_business_credit_cards_business_silver_rewards_card_010.md
./doc_bank_accounts_bank_accounts__general__044.md
./doc_credit_cards_virtual_card_management_006.md
./doc_business_checking_accounts_navy_blue_002.md
./doc_business_checking_accounts_beige_005.md
./doc_business_credit_cards_silver_zoom_card_007.md
./doc_credit_cards_credit_cards__general__010.md
./doc_credit_cards_platinum_rewards_card_008.md
./doc_checking_accounts_purple_account_006.md
./doc_business_credit_cards_virtual_card_management_007.md
./doc_business_checking_accounts_true_blue_005.md
./doc_savings_accounts_platinum_account_001.md
./doc_bank_accounts_bank_accounts__general__041.md
./doc_buy_now_pay_later_bnpl_platinum_008.md
./doc_business_credit_cards_business_gold_rewards_card_008.md


→ assistant calls `KB_search_bm25` with `{"query": "apply for credit card application tool submit"}`

**tool:** 1. Internal: Processing CLI Approvals and Denials
   ID: doc_credit_cards_credit_card_account_logistics_007
   Score: 15.3567
   Content: ## Purpose

This document outlines the step-by-step workflow for agents to process CLI requests, including verification and final decision. These steps MUST be followed in the exact order listed.

## Step 0: Confirm Requested Amount is Within Limits

BEFORE submitting any CLI request, verify that the customer's requested increase amount is within the maximum allowed for their card tier. If the requested amount exceeds the limit, inform the customer of the maximum allowed and ask them to adjust their request. Do NOT submit a request that exceeds the tier limit.

## Step 1: Submit the CLI Request

Once the customer has confirmed a valid increase amount within their tier's limits, submit the request on their behalf using the submit_credit_limit_increase_request_7392 tool. This creates a formal record of the request before eligibility checks are performed. Eligibility checks are internal and not exposed to customers, so the submission must happen first.

### Tool Arguments: submit_credit_limit_increase_request_7392

- credit_card_account_id (string, required): The credit card account ID.
- user_id (string, required): The customer's unique identifier.
- requested_increase_amount (integer, required): The dollar amount by which to increase the credit limit (e.g., 1000 for a $1,000 increase).

## Step 2: Verify Basic Eligibility

After submitting the request, verify the customer meets all eligibility requirements. You MUST check ALL of the following eligibility criteria before making an approval or denial decision as this ensures complete audit records.

1. Account Age: Verify the account has been open for the minimum required days for their card tier.
2. Cooldown Period: Use the get_credit_limit_increase_history_4829 tool to check if the customer has submitted a request within the cooldown period for their card tier. If a request exists within this period, deny the new request and inform the customer when they will be eligible to submit again.
3. No Pending Disputes: Verify the account has no active disputes.
4. No Pending Replacement Cards: Verify there are no outstanding replacement card orders for this account. If a replacement is pending, the CLI cannot be processed until the replacement is delivered or cancelled.
5. Account Good Standing: The account must be current with no past-due balance.
6. Credit Utilization: Verify current utilization is below the maximum threshold for their card tier.

### Tool Arguments: get_credit_limit_increase_history_4829

- credit_card_account_id (string, required): The credit card account ID to check.

## Step 3: Verify Payment History and Requested Amount

.

## Step 4: Process the Decision

If all requirements are met and the requested amount is within limits:

1. Use the approve_credit_limit_increase_5847 tool to approve and apply the increase.

### Tool Arguments: approve_credit_limit_increase_5847

- credit_card_account_id (string, required): The credit card account ID.
- user_id (string, required): The customer's unique identifier.
- new_credit_limit (float, required): The new total credit limit to set.

If any requirements are not met:

1. Use the deny_credit_limit_increase_5848 tool to record the denial.

### Tool Arguments: deny_credit_limit_increase_5848

- credit_card_account_id (string, required): The credit card account ID.
- user_id (string, required): The customer's unique identifier.
- denial_reason (string, required): Must be one of: 'insufficient_account_age', 'cooldown_period_active', 'pending_disputes', 'pending_replacement_card', 'past_due_balance', 'high_utilization', 'insufficient_payment_history', 'requested_amount_exceeds_limit', 'other'.

## Step 5: Communicate the Decision

Inform the customer of the decision and provide next steps. For approvals, confirm the new credit limit. For denials, explain the reason and when they may be eligible to reapply.

2. (Internal) Applying a Credit Card Statement Credit
   ID: doc_credit_cards_credit_cards_(general)_017
   Score: 15.1426
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

3. Green Rewards Card: Apply for a Sustainable Business Credit Card
   ID: doc_business_credit_cards_green_rewards_card_001
   Score: 15.0211
   Content: ## Eligibility
- Personal credit: You should meet a minimum FICO threshold of $690.
- Business credit: Established businesses are evaluated using a PAYDEX threshold of 42 in addition to personal credit history.
- Newer businesses: If you have limited business credit history, your personal profile can support your application.

## Costs and Rates
| Cost or Rate | Amount |
| --- | --- |
| Annual fee | $100.00 |
| Purchase APR | 19.99% |
| Foreign transaction fee | 1.0% |

## Credit Limits
- Approved credit lines are assigned within a range from $14,000 to $75,000.
- Your limit is determined by factors such as revenue consistency, existing obligations, and overall credit profile.

## Employee Cards
- You can issue up to 15 employee cards to team members.
- Set category-level controls, customize spending rules, and monitor transactions in real time.

## What You Need to Apply
- Business legal name and structure, tax identification, and operating address
- Beneficial owner and authorized officer details
- Recent business financial information and estimated monthly card spend
- Agreement to a personal guaranty where applicable

## How to Apply
- Complete the application in your online banking portal.
- Provide business and owner details, upload requested documents, and review disclosures.
- Submit your application for credit review. You will be notified when a decision is reached.

## After You Submit
- If approved, activate your card and set controls for any employee cards.
- Review your assigned limit and applicable pricing, including $100.00, 19.99%, and 1.0%.
- Add the card to your digital wallet and configure expense categories for smoother reconciliation.

4. BNPL Gold: How to Apply and Get Started
   ID: doc_buy_now_pay_later_bnpl_gold_001
   Score: 14.1631
   Content: ## What you need before you apply
- Business details and ownership information
- Verification documents as requested during the application flow
- A primary email and mobile number for account communications

## How to apply
1. Start a BNPL Gold application from your account.
2. Enter required business and financial information.
3. Review and confirm disclosures.
4. Submit your application for review.

## Approval and your starting limit
- If approved, your initial BNPL Gold credit limit will fall between [value not available] and [value not available].
- You can begin using BNPL Gold once you accept your agreement.

## Get started with your first plan
- The standard repayment term for plans created under BNPL Gold is [value not available] months.
- You receive an introductory promotional APR of [value not available] for the first [value not available] days from plan start.
- After the introductory period ends, the standard APR [value not available] applies to any remaining balance on the plan.

## Tips for a smooth first purchase
- Confirm your available credit before checkout to ensure your purchase fits within your approved limit.
- Review your plan schedule at creation so you know your monthly payment amount and timing.
- Keep contact and billing information current so you receive payment reminders and statements.

5. EcoCard: Apply for a Sustainable Rewards Credit Card
   ID: doc_credit_cards_ecocard_001
   Score: 14.0503
   Content: ## Eligibility and Requirements
- Minimum credit score required to apply: 0 (0 indicates no credit score requirement).
- You must provide standard identity and income information during the application.

## How to Apply
1. Complete the online application with your personal and financial details.
2. Review the pricing and terms, including fees and interest rates listed below.
3. Submit the application and, if approved, activate your card upon receipt to begin using it.

## Credit Limits Upon Approval
- Initial credit lines are determined at approval and typically range from $5,000 to $50,000.
- Credit line increases may be reviewed over time based on account history and usage.

## Rates and Fees You Agree To
- Purchase APR on carried balances: 19.99%.
- Annual fee: $50.00.
- Foreign transaction fee (international purchases): 1.0%.
- Late payment fee (if you miss the due date): $32.50.

## New-Customer Bonus at Sign-Up
- Spend $5,000 in eligible purchases within your first month to earn $2,000 sustainability points.
- Bonus points are awarded once the threshold is met and the qualifying period closes.

## After You Apply
- You will receive a decision or a request for additional documentation.
- If approved, set up your online account and add your card to a digital wallet to start using it immediately after activation.

6. Platinum Rewards Card: Elite Business Credit Card Application
   ID: doc_business_credit_cards_business_platinum_rewards_card_001
   Score: 13.9377
   Content: ## Eligibility and Credit Standards
- Minimum personal credit score required: $765
- Minimum business credit score (PAYDEX) for established businesses: 77
- Carried balance APR: 16.99%

## Credit Limits
- Minimum approved credit limit: $75,000
- Maximum available credit limit: $400,000

## How to Apply
- Prepare business legal details, ownership structure, and verification documents that substantiate established operations and strong credit history.
- Ensure the primary owner or officer meets the minimum personal credit threshold and is prepared to authorize a credit review.
- Provide recent business financials and bank references to support the requested credit limit within the available range.
- Submit the application through your online dashboard or with assistance from a commercial lending specialist; include accurate contact information for follow-up.

## Underwriting Review
- Your application is evaluated against both personal and business credit standards, including $765 and 77.
- Requested limits are aligned with financial capacity and may be adjusted within $75,000 to $400,000.
- If approved, your account will reflect the applicable APR for carried balances of 16.99%.

## Tips for a Smooth Review
- Match your requested limit to demonstrated cash flow and liquidity.
- Provide consistent legal names across all documents and tax records.
- Respond promptly to any verification requests to avoid processing delays.

7. Silver Rewards Card: Getting Started and Eligibility Requirements
   ID: doc_credit_cards_silver_rewards_card_001
   Score: 13.6310
   Content: ## Eligibility Requirements
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
- Review your benefits summary, including purchase protection and international fee terms noted above

8. Internal: Submitting Interest Discrepancy Reports
   ID: doc_bank_accounts_bank_accounts_(general)_044
   Score: 13.4878
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

9. Submitting a Cash Back Dispute (Internal)
   ID: doc_credit_cards_credit_cards_(general)_003
   Score: 13.2675
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

10. Internal: Processing Credit Card Account Closures
   ID: doc_credit_cards_credit_card_account_logistics_002
   Score: 12.8153
   Content: ## Closure Process

1. Verify the customer's identity using standard verification procedures.
2. Confirm all eligibility requirements are met per 'How can I close a credit card account?':
   - Outstanding balance must be $0.00 dollars
   - No pending disputes (pending disputes allowed: No)
   - Account age at least 60 days
   - No pending replacement cards
3. Use the close_credit_card_account_7834 tool to process the closure. Refer to 'How can I close a credit card account?' for eligibility requirements.

### Tool Arguments

- credit_card_account_id (string): The unique identifier for the credit card account to be closed. This can be found in the customer's account profile or by looking up their credit card accounts.
- user_id (string): The unique identifier for the customer requesting the closure. This should match the authenticated user.

### Post-Closure Communication

- The customer will receive a confirmation email and a final statement within several business days after the account is closed.
- If asked, remind the customer:
  - Unredeemed rewards can be redeemed for 45 days after the closure request; thereafter, they are forfeited.
  - If an annual fee posted recently, a full refund may apply if the closure occurs within 37 days of the fee charge.


[Timing: retrieval=3ms, reranking=0ms, total=3ms]

→ assistant calls `shell` with `{"command": "grep -rn \"credit_card_application\\|apply_for\\|open_credit_card\\|application_[0-9]\" . | head; grep -rln \"Internal.*[Aa]pplication\\|card application\" . | head"}`

**tool:** (no output)

→ assistant calls `KB_search_dense` with `{"query": "agent submits new credit card application on behalf of customer", "k": 5}`

**tool:** 1. Generating a Credit Card Referral Link (Internal)
   ID: doc_credit_cards_credit_cards_(general)_009
   Score: 0.6838
   Content: ## Pre-check before providing the referral tool
- Search the knowledge base to confirm the specific card has a documented referral program.
- Verify the customer’s understanding of referral terms. If the customer cites terms that do not match any documented program, clarify the discrepancy.
- If no referral program is documented for the requested card, or if the user’s claimed terms are incorrect, or if there is reason to believe the referral will be automatically rejected, explain why and do not provide a referral link tool. Do not transfer to a human in these cases.

## How the user generates their referral link
- Provide the customer with this tool and instruct them to run it themselves:
  - `get_referral_link(user_id: str, card_name: str)`
- Tell the customer to pass their own user_id and the exact card name (for example, 'Gold Rewards Card').
- When the tool is called successfully, a referral record is created with status 'NO_PROGRESS'. The referred person can then use the generated link to apply.

## Important reminders
- Weekly limit: Customers can receive at most 2 referral bonuses in any rolling 7-day window; the third and subsequent referrals in that window are automatically denied.
- Agents must not generate the link on the customer’s behalf.
- Reiterate the correct referral terms as documented to prevent confusion and complaints.

2. (Internal) Applying a Credit Card Statement Credit
   ID: doc_credit_cards_credit_cards_(general)_017
   Score: 0.6651
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

3. Filing a Credit Card Transaction Dispute (Internal)
   ID: doc_credit_cards_credit_cards_(general)_014
   Score: 0.6498
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

4. Internal: Filing a Debit Card Transaction Dispute
   ID: doc_bank_accounts_bank_accounts_(general)_031
   Score: 0.6475
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

5. Paying Off Credit Card Balance from Checking Account (Internal)
   ID: doc_credit_cards_credit_card_account_logistics_009
   Score: 0.6264
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


[Timing: retrieval=8ms, reranking=0ms, total=24ms]

**assistant:** Yes, that's right, with two conditions:

- **$1,500 to $4,000 back:** You get the 1.5% base rate no matter how the charge is coded. You only get 4% if Apple bills it under a software or media category. Rho-Bank can't control or guarantee that.
- **$0 annual fee in year one:** This applies to new customers who open the account between 11/1/2025 and 2/28/2026, so you qualify if you open it now. From year two on, the fee is $450 a year unless you close the card before it renews.

**How to apply:** I can't submit a credit card application from this chat. Rho-Bank takes Business Platinum applications in two ways:
1. **Your Rho-Bank online dashboard.** You can apply there yourself.
2. **With a commercial lending specialist.** They can walk you through the application.

**What to have ready:**
- Your business's legal details, ownership structure, and documents showing it's established
- Recent business financials and bank references to support the credit limit you ask for
- Consent from the main owner or officer to a credit review
- Your contact details, entered the same way everywhere, for follow-up

**To get the most from this purchase:**
- **Ask for a limit above $100,000,** for example $110,000 to $125,000. Your financials should be able to back up that amount. Limits range from $75,000 to $400,000.
- **Check your eligibility:** you need a personal credit score of 765 or higher and a PAYDEX score of 77 or higher.
- **Wait to make the Apple purchase until after approval.** Only purchases made on the open card earn cash back.
- **Buy directly from Apple,** keep the invoice, and pay the balance in full to avoid the 16.99% APR.

Would you like me to connect you with a specialist for help with the application, or will you apply through your dashboard?

→ user calls `apply_for_credit_card` with `{"card_type": "Business Platinum Rewards Card", "customer_name": "Sarah Mitchell", "annual_income": 250000, "rho_bank_subscription": false}`

**tool:** Credit card application submitted:
Your application has been successfully submitted. You will receive a decision within 5-7 business days via email.

**user:** I’ll apply through the dashboard.

I just submitted an application for the **Business Platinum Rewards Card** under **Sarah Mitchell**, **$250,000** annual income, and **no Rho-Bank+ subscription**.

It says the application was successfully submitted and I’ll get a decision in **5–7 business days via email**.###STOP###
