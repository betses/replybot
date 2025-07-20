import os
import re
import discord
import asyncio
from dotenv import load_dotenv
from discord.ext import commands
from openai import OpenAI
from typing import Optional
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
import logging

# Required packages: discord.py, python-dotenv, openai

# ─── 1) Load environment variables ─────────────────────────────────────────────
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENAI_KEY    = os.getenv("OPENAI_API_KEY")

if not DISCORD_TOKEN or not OPENAI_KEY:
    print("❌ Missing DISCORD_BOT_TOKEN or OPENAI_API_KEY in .env")
    exit(1)

# ─── 2) Instantiate OpenAI client ───────────────────────────────────────────────
client = OpenAI(api_key=OPENAI_KEY)

# ─── 3) Configure Discord bot ───────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ─── 4) Safety caps ─────────────────────────────────────────────────────────────
MAX_COMMENTS        = 20    # never generate more than 20 replies
MAX_TOKENS_PER_CALL = 1024  # cap tokens per request

# ─── 5) Parser: extract tweet text and dynamic comment count ─────────────────────
def parse_tweetshift_message(content: str) -> tuple[str, int]:
    # Find all C tags, including ranges (e.g., 1-3C, 10-20C, 3C)
    matches = re.findall(r"(\d+)(?:-(\d+))?C\b", content, flags=re.IGNORECASE)
    counts = []
    for m in matches:
        if m[1]:
            counts.append(int(m[1]))  # Use upper bound of range
        else:
            counts.append(int(m[0]))  # Use single value
    count = max(counts) if counts else 0
    count = min(count, MAX_COMMENTS)
    # Split into lines and skip the first line if it looks like metadata
    lines = content.splitlines()
    if lines:
        first_line = lines[0].strip()
        # If the first line contains only tags, mentions, or brackets, skip it
        if re.match(r"^(@\w+\s*)*([\d\-]+[lLrRcC]\s*[/+]*\s*)+.*(\[.*\])?\s*$", first_line):
            lines = lines[1:]
    text = "\n".join(lines).strip()
    # Remove any remaining metric tags and mentions from the text
    text = re.sub(r"\s*\d+(?:-\d+)?[lLrRcC]\b", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[\/|\\]", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text, count

# ─── 6) Core: generate GPT-4.1 replies dynamically with exact batch template ─────
async def generate_replies(tweet: str, n: int) -> list[str]:
    n = max(1, min(n, MAX_COMMENTS))
    max_tokens = min(150 * n, MAX_TOKENS_PER_CALL)

    # Distribute n replies as evenly as possible among 5 batches
    batch_sizes = [n // 5] * 5
    for i in range(n % 5):
        batch_sizes[i] += 1
    batch_ranges = []
    start = 1
    for size in batch_sizes:
        end = start + size - 1
        batch_ranges.append((start, end))
        start = end + 1

    prompt = (
        f"""
# TASK

Generate {n} Twitter replies to the post provided below.

# FORMAT

Deliver the replies in 5 **distinct batches**, distributed as evenly as possible ({' / '.join([f'{r[0]}–{r[1]}' for r in batch_ranges])}). Each batch should reflect a different **set of personas and tones**, as described below.

# GENERAL OUTPUT RULES

* Replies must feel human, authentic, and platform-native
* Avoid robotic or repetitive phrasing
* Reference specific details or phrases from the original tweet
* Do **not** generate usernames
* Do **not** include hashtags or emojis or em dashes
* Do **not** write anything before or after the replies — only the {n} raw replies in order
* Maintain conversational relevance to the input post
* Drop the results without any titles for example "BATCH 1: " 

# BATCHES

### BATCH 1: *Tech-savvy developers and chain analysts*

**Tone:** Analytical, skeptical, precise
**Focus:** architecture, scalability, implementation
Replies {batch_ranges[0][0]}–{batch_ranges[0][1]} should reflect developers or researchers dissecting the post.
**Length of sentence**: Long. Two sentences. 10-40 words

### BATCH 2: *Crypto bros and hype-driven traders*

**Tone:** Hypey, excited, informal
**Focus:** Comeback narratives, price potential, mainstream attention
Replies {batch_ranges[1][0]}–{batch_ranges[1][1]} should sound like excited traders or degens reacting to the visibility and timing.
Use emojis on these replies where fitting. 
**Length of sentence**: Very short. 2-8 words
---

### BATCH 3: *Founders, VCs, and long-term observers*

**Tone:** Strategic, thoughtful, veteran-like
**Focus:** Team background, timing
Replies {batch_ranges[2][0]}–{batch_ranges[2][1]} should reflect seasoned builders, funders, or advisors
**Length of sentence**: Medium. 1 full sentence. 6-15 words

### BATCH 4: *Meme lords and sarcastic skeptics*

**Tone:** Ironic, dismissive, sharp, or confused
**Focus:** Complexity, jargon, marketing moves, questionable decentralization
Replies {batch_ranges[3][0]}–{batch_ranges[3][1]} should be meme-y, chaotic, or questioning the relevance
**Length of sentence**: Very short. 2-8 words

### BATCH 5: *Casual observers and curious newcomers*

**Tone:** Curious, tentative, learning mode
**Focus:** Trying to understand, wondering if it’s worth following or using
Replies {batch_ranges[4][0]}–{batch_ranges[4][1]} should sound like newcomers trying to make sense of what’s going on.
**Length of sentence**: Medium. 1 full sentence. 6-15 words

# INPUT POST:
{tweet}
"""
    )

    system_msg: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": "You are simulating a real Twitter conversation. Follow the user's style instructions exactly."
    }
    user_msg: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": prompt
    }

    try:
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4.1",
            messages=[system_msg, user_msg],
            temperature=0.8,
            max_tokens=max_tokens,
        )
    except Exception as e:
        print("⚠️ OpenAI API error:", e)
        return []

    body = resp.choices[0].message.content if resp.choices and resp.choices[0].message.content else None
    if not body:
        return []
    replies = [line.strip() for line in body.splitlines() if line.strip()]
    # Number the replies as an ordered list
    numbered_replies = [f"{i+1}. {reply}" for i, reply in enumerate(replies)]
    return numbered_replies[:n]

# ─── 7) Threading helper ───────────────────────────────────────────────────────
async def reply_in_thread(origin_msg: discord.Message, tweet: str, count: int):
    # Determine or create thread
    if isinstance(origin_msg.channel, discord.Thread):
        thread = origin_msg.channel
    else:
        try:
            thread = await origin_msg.create_thread(
                name=f"Replies-{count}-{tweet[:30]}…", auto_archive_duration=60
            )
        except Exception as e:
            return await origin_msg.channel.send(f"⚠️ Thread error: {e}")

    # Send header and replies in thread
    await thread.send(f"💬 Replies to: \"{tweet[:60]}…\"")
    replies = await generate_replies(tweet, count)
    if not replies:
        return await thread.send("⚠️ Couldn't generate replies.")

    for r in replies:
        await thread.send(r)

# ─── 8) Bot events ───────────────────────────────────────────────────────────
@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

@bot.event
async def on_message(message: discord.Message):
    if bot.user and message.author.id == bot.user.id:
        return

    if not isinstance(message.channel, discord.Thread):
        tweet, count = parse_tweetshift_message(message.content)
        if tweet and count > 0:
            await reply_in_thread(message, tweet, count)

    await bot.process_commands(message)

# ─── 9) Command: !replybot ────────────────────────────────────────────────────
@bot.command(name="replybot")
async def replybot(ctx: commands.Context, count: Optional[int] = None):
    try:
        channel = ctx.channel

        if isinstance(channel, discord.Thread):
            parts = channel.name.split('-')
            if len(parts) < 3:
                return await ctx.send("⚠️ Can't determine original tweet or count from thread name. Use a valid thread.")
            tweet_count = count if count is not None else int(parts[1])
            tweet = ' '.join(parts[2:])
            if tweet.endswith('…'):
                tweet = tweet[:-1]
            await reply_in_thread(ctx.message, tweet, tweet_count)
            return

        if not ctx.message.reference or not ctx.message.reference.message_id:
            return await ctx.send("⚠️ Please reply to a message with `!replybot` to specify the tweet.")

        ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        tweet, ts_count = parse_tweetshift_message(ref_msg.content)
        final_count = count if count is not None else ts_count
        if final_count == 0:
            return await ctx.send("⚠️ Couldn't determine count. Use `!replybot <count>`.")

        await reply_in_thread(ref_msg, tweet or ref_msg.content, final_count)
    except Exception as e:
        logging.exception("Unexpected error in !replybot command")
        await ctx.send("⚠️ An unexpected error occurred while processing your request.")

# ─── 10) Run the bot ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)