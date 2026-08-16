"""Generate instruction-tuning dataset for customer support domain."""
import json
from pathlib import Path

EXAMPLES = [
    {"instruction": "How do I reset my password?", "input": "", "output": "Go to the login page and click 'Forgot Password'. Enter your email address, and we'll send a reset link within 2 minutes. The link expires after 24 hours."},
    {"instruction": "What is your return policy?", "input": "", "output": "We accept returns within 30 days of purchase. Items must be unused and in original packaging. Refunds are processed within 5-7 business days to your original payment method."},
    {"instruction": "How do I track my order?", "input": "", "output": "Log into your account and visit 'My Orders'. Click on your order number to see real-time tracking information. You'll also receive automated email updates at each shipping milestone."},
    {"instruction": "My order arrived damaged. What should I do?", "input": "", "output": "We're sorry to hear that. Please take photos of the damaged item and packaging, then contact our support team within 48 hours. We'll arrange a free replacement or full refund — your choice."},
    {"instruction": "Can I change my shipping address after placing an order?", "input": "", "output": "Address changes are possible if the order hasn't shipped yet. Contact support immediately with your order number. Once dispatched, we're unable to redirect the package."},
    {"instruction": "Do you offer international shipping?", "input": "", "output": "Yes, we ship to over 50 countries. International shipping takes 7-14 business days. Customs duties and taxes may apply and are the buyer's responsibility."},
    {"instruction": "How do I cancel my subscription?", "input": "", "output": "You can cancel anytime in Account Settings → Subscriptions → Cancel. Your access continues until the end of the current billing period. No cancellation fees apply."},
    {"instruction": "What payment methods do you accept?", "input": "", "output": "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay. All transactions are secured with 256-bit SSL encryption."},
    {"instruction": "Is my personal data safe?", "input": "", "output": "Yes. We are GDPR and POPIA compliant. We never sell your data to third parties. You can request a full data export or deletion at any time from your account settings."},
    {"instruction": "How do I apply a discount code?", "input": "", "output": "Enter your discount code in the 'Promo Code' field at checkout, then click 'Apply'. The discount will reflect in your order total immediately."},
    {"instruction": "Why was my payment declined?", "input": "", "output": "Common reasons include insufficient funds, incorrect card details, or your bank blocking the transaction. Try a different card or contact your bank. You can also try PayPal as an alternative."},
    {"instruction": "Do you have a loyalty programme?", "input": "", "output": "Yes! Our Rewards Club gives you 1 point per R10 spent. 100 points = R10 discount. Points expire after 12 months of inactivity. Enrol free in your account dashboard."},
    {"instruction": "How long does standard shipping take?", "input": "", "output": "Standard shipping within South Africa takes 3-5 business days. Gauteng orders often arrive in 2 business days. Express shipping (1-2 days) is available at checkout."},
    {"instruction": "Can I exchange an item for a different size?", "input": "", "output": "Yes. Initiate an exchange within 30 days via 'My Orders'. Select the new size and we'll ship it once we receive the return. Exchanges are free for South African customers."},
    {"instruction": "What is your warranty policy?", "input": "", "output": "All products carry a 12-month warranty against manufacturing defects. Extended warranties are available for purchase. The warranty does not cover accidental damage or normal wear."},
    {"instruction": "How do I contact customer support?", "input": "", "output": "Reach us via live chat (available 8am-8pm SAST), email support@example.com (response within 4 hours), or call 0800-123-456 (Monday-Friday, 8am-5pm)."},
    {"instruction": "Can I buy now and pay later?", "input": "", "output": "Yes, we offer PayFlex and Payflex instalment options at checkout. You can split your purchase into 4 equal payments over 6 weeks, interest-free."},
    {"instruction": "What happens if I miss a payment?", "input": "", "output": "A missed payment triggers a 3-day grace period with email reminders. After that, a R50 late fee applies. After 30 days, the account is referred to collections."},
    {"instruction": "How do I update my billing details?", "input": "", "output": "Go to Account → Payment Methods → Edit. Update your card details and click Save. Changes apply to all future transactions."},
    {"instruction": "Do you price match?", "input": "", "output": "Yes! If you find the same product cheaper at an authorised retailer within 7 days of purchase, we'll refund the difference. Submit a price match request with a screenshot of the competitor's price."},
]

# Pad to 50 examples by slight variation
EXTRA = [
    {"instruction": f"Question about order {1000 + i}", "input": "", "output": "Please contact our support team with your order number and we'll assist you promptly."}
    for i in range(30)
]
ALL = EXAMPLES + EXTRA

split = int(len(ALL) * 0.8)
train, val = ALL[:split], ALL[split:]

Path("data").mkdir(exist_ok=True)
Path("data/train.jsonl").write_text("\n".join(json.dumps(e) for e in train))
Path("data/eval.jsonl").write_text("\n".join(json.dumps(e) for e in val))
print(f"Created {len(train)} train / {len(val)} eval examples.")
