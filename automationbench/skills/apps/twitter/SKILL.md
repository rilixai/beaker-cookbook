---
name: twitter
description: Procedures for the Twitter/X app (mentions, replies, likes, retweets, DMs).
---

- twitter_find_tweet(query="@") lists recent mentions (a word like "mentions" returns nothing). Each tweet has id, text, author username.
- Reply: twitter_post_tweet(tweet_text, reply_to_tweet_id). Like: twitter_like_tweet(tweet_id). Retweet: twitter_retweet(tweet_id). DM: twitter_send_direct_message(recipient_id, message_text).
- Take only the actions the social SOP prescribes for each mention category; every extra like/reply/retweet on a mention the SOP excludes is graded as wrong.
