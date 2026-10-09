# Lumora Pay — Product FAQ

*Fictional sample document created for the docsearch-ai-rag demo. Lumora Pay is not a real product.*

Lumora Pay is a payments API that lets online businesses accept cards, bank transfers and digital wallets through a single integration.

## Getting Started

### How do I create an account?

Sign up at the Lumora Pay dashboard with your business email. You will get immediate access to sandbox mode. To process live payments you must complete business verification (KYB), which usually takes 2 to 3 business days.

### What do I need for business verification?

You need your company registration document, the tax identification number, a government-issued ID for every beneficial owner holding 25% or more, and a bank statement from the last 3 months for the payout account.

### Is there a sandbox?

Yes. Every account includes a sandbox with test API keys prefixed with `sk_test_`. Live keys are prefixed with `sk_live_`. Sandbox transactions never move real money.

## Pricing

### How much does Lumora Pay cost?

Domestic card payments cost 2.9% + USD 0.30 per successful transaction. International cards add 1.5%. Bank transfers cost a flat USD 0.80 per transaction. There are no setup fees and no monthly fees.

### Do you offer volume discounts?

Businesses processing more than USD 250,000 per month can request custom pricing from the sales team.

### Are refunds charged?

Refunds are free, but the original processing fee is not returned.

## Payouts

### When do I receive my money?

Standard payouts arrive in your bank account 2 business days after the payment is captured. Instant payouts are available for eligible accounts for a fee of 1% of the payout amount, with a minimum of USD 0.50.

### Can I change my payout schedule?

Yes. You can choose daily, weekly or monthly payouts in the dashboard under Settings > Payouts.

## Disputes and Chargebacks

### What happens when a customer disputes a payment?

The disputed amount plus a USD 15 dispute fee is withheld from your balance. You have 7 calendar days to submit evidence through the dashboard. If you win the dispute, both the amount and the fee are returned.

## Technical Integration

### What are the API rate limits?

The live API allows 100 requests per second per account. The sandbox allows 25 requests per second. Requests over the limit receive HTTP status 429 and should be retried with exponential backoff.

### How do webhooks work?

Lumora Pay sends webhook events such as `payment.succeeded` and `payout.paid` to your HTTPS endpoint. Each request is signed with an HMAC-SHA256 signature in the `Lumora-Signature` header. Failed deliveries are retried for up to 72 hours.

### How do I avoid duplicate charges?

Send an `Idempotency-Key` header with every POST request. Keys are stored for 24 hours, and repeating a request with the same key returns the original response instead of creating a new charge.

### Which currencies are supported?

Lumora Pay supports 35 currencies for card payments. Payouts are available in USD, EUR, MXN, COP and BRL.

## Security and Compliance

### Is Lumora Pay PCI compliant?

Yes. Lumora Pay is certified as a PCI DSS Level 1 service provider. Card data is tokenized and never touches your servers when you use our hosted fields.

### Do you support 3D Secure?

Yes. 3D Secure 2 is applied automatically when required by regulation or when our risk engine flags a payment.
