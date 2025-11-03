"""
Usage:
------
This script collects Reddit comments from a specific subreddit
within a given UTC time range.

You can adjust the following parameters:
-----------------------------------------
1. Reddit credentials:!!!!!!!!You can use mine directly, but please do not upload it to GitHub
   - client_id, client_secret, username, password, user_agent

2. Time range (UTC):
   - start_time = datetime(YYYY, MM, DD, HH, MM, SS, tzinfo=timezone.utc)
   - end_time   = datetime(YYYY, MM, DD, HH, MM, SS, tzinfo=timezone.utc)

3. Scraping parameters:
   - subreddit_name: name of the subreddit to scrape (e.g., "Trump")
   - limit_posts: how many posts to check at most
   - limit_comments_per_post: max number of comments per post
   - max_total_comments: global upper limit on total comments collected

Output:
-------
1. Full CSV file:
   {subreddit_name}_{start_date}_{end_date}.csv
   → Includes post_id, comment_id, title, time, score, and text.

2. Text-only cleaned CSV file:
   {subreddit_name}_{start_date}_{end_date}_text.csv
   → Contains only the text column, !!!!!!!!!But you need to do some simple preprocessing just like you did with the dataset before.
"""

import praw
import pandas as pd
import re
from datetime import datetime, timezone

# ==== Reddit credentials ====
reddit = praw.Reddit(
    client_id="nyMZ3cPTsyKYt6HtAT23Dg",
    client_secret="g26ULvbjkdJT-uaTEfax8VHEolYBiQ",
    user_agent="bert-sentiment-analysis",
    username="jiamao0772621n",
    password="aa112233."
)

# ==== Time range (UTC) ====
# Note: Reddit timestamps are in UTC, not local time.
start_time = datetime(2025, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
end_time   = datetime(2025, 10, 2, 0, 0, 0, tzinfo=timezone.utc)

start_timestamp = start_time.timestamp()
end_timestamp = end_time.timestamp()

# ==== Scraping parameters ====
subreddit_name = "Trump"
limit_posts = 500             # Maximum number of posts to check
limit_comments_per_post = 40  # Maximum comments per post
max_total_comments = 100      # Global maximum number of comments to collect

# ==== Begin scraping ====
subreddit = reddit.subreddit(subreddit_name)
data = []
count = 0

print(f"📅 Collecting comments from r/{subreddit_name} between {start_time} and {end_time} (max {max_total_comments} comments)...")

for submission in subreddit.new(limit=limit_posts):
    if count >= max_total_comments:
        break  # Stop when total comment limit is reached

    # Check if post is within the specified time window
    if start_timestamp <= submission.created_utc <= end_timestamp:
        submission.comments.replace_more(limit=0)
        for comment in submission.comments[:limit_comments_per_post]:
            if count >= max_total_comments:
                break

            text = comment.body.strip()
            if text and text.lower() not in ["[deleted]", "[removed]"]:
                data.append({
                    "post_id": submission.id,
                    "comment_id": comment.id,
                    "post_title": submission.title,
                    "created_utc": datetime.fromtimestamp(comment.created_utc, tz=timezone.utc),
                    "score": comment.score,
                    "text": text
                })
                count += 1

print(f"✅ Collected {count} comments in total.")

# ==== Save full CSV ====
df = pd.DataFrame(data)
filename_full = f"{subreddit_name}_{start_time.date()}_{end_time.date()}.csv"
df.to_csv(filename_full, index=False, encoding="utf-8-sig")
print(f"💾 Full dataset saved: {filename_full}")

# ==== Clean text and export text-only CSV ====
def clean_text(text):
    # Remove URLs, markdown links, mentions, and excessive whitespace
    text = re.sub(r"http\S+|www\.\S+|discord\.gg/\S+", "", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # remove markdown links
    text = re.sub(r"\*{1,2}|\~|\_|\>", "", text)  # remove markdown symbols
    text = re.sub(r"\s+", " ", text).strip()
    # Remove bot-related text
    if "I am a bot" in text or "contact the moderators" in text:
        return ""
    return text

if not df.empty:
    text_only = df["text"].dropna().apply(clean_text)
    text_only = text_only[text_only != ""].reset_index(drop=True)

    filename_text = f"{subreddit_name}_{start_time.date()}_{end_time.date()}_text.csv"
    text_only.to_csv(filename_text, index=False, header=["text"], encoding="utf-8-sig")
    print(f"💾 Clean text-only file saved: {filename_text}")
else:
    print("⚠️ No comments found — text-only file not generated.")

print("✅ All tasks completed successfully.")
