# Fixing Telegram Webhook Issues

If the bot fails to receive messages, the Telegram webhook might be configured incorrectly. This guide explains how to verify and fix the webhook manually.

## 1. Verify the current webhook

1. Ensure `TELEGRAM_BOT_TOKEN` is set in your `.env` file and is not a placeholder.
2. Retrieve webhook information:

   ```bash
   curl -s https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getWebhookInfo
   ```

   The `url` field should point to your domain, e.g. `https://your-app.com/webhook`. If it is empty or contains `<your-telegram-bot-token>`, you need to reset the webhook.

## 2. Delete the existing webhook

```bash
curl -X POST https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/deleteWebhook
```

## 3. Set the correct webhook

Replace `YOUR_URL` with the domain where the bot is deployed (without a trailing slash):

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"url": "https://YOUR_URL/webhook", "drop_pending_updates": true}' \
  https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook
```

Check again with `getWebhookInfo` to confirm that the new URL is in place.

## Automated option

You can run the included script to perform these checks automatically:

```bash
python fix_webhook.py
```

The script will display environment information, attempt to fix the webhook, and verify the result.
