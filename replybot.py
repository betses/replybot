# replybot.py

import os
import re
import discord
import asyncio
from dotenv import load_dotenv
from discord.ext import commands
from openai import OpenAI

# ─── 1) Load environment variables ─────────────────────────────────────────────
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENAI_KEY    = os.getenv("OPENAI_API_KEY")

if not DISCORD_TOKEN or not OPENAI_KEY:
    print("❌ Missing DISCORD_BOT_TOKEN or OPENAI_API_KEY in .env")
    exit(1)

# ─── 2) Instantiate OpenAI client (synchronous) ────────────────────────────────
client = OpenAI(api_key=OPENAI_KEY)

# ─── 3) Configure Discord bot ───────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ─── 4) Safety caps ─────────────────────────────────────────────────────────────
MAX_COMMENTS        = 20    # never generate more than 20 replies
MAX_TOKENS_PER_CALL = 1024  # cap tokens per request

# ─── 5) Parser: extract tweet text and count (highest in a range) ──────────────
def parse_tweetshift_message(content: str) -> tuple[str, int]:
    """
    Scans all lines for 'XXC' or 'X-YC' and returns the tweet text (everything else)
    plus the count (highest of the range, or the single value).
    """
    lines = content.splitlines()
    count = 0
    tweet_lines = []

    for line in lines:
        m = re.search(r"(\d+)(?:-(\d+))?C", line)
        if m:
            # use upper bound if it's a range, else the single number
            count = int(m.group(2)) if m.group(2) else int(m.group(1))
        else:
            tweet_lines.append(line)

    tweet_text = "\n".join(tweet_lines).strip()
    return tweet_text, count

# ─── 6) Core: generate GPT replies with custom instructions ────────────────────
async def generate_replies(tweet: str, n: int) -> list[str]:
    # clamp count & tokens
    n = max(1, min(n, MAX_COMMENTS))
    max_tokens = min(150 * n, MAX_TOKENS_PER_CALL)

    system_msg = {
        "role": "system",
        "content": (
            "You are simulating a real Twitter conversation. "
            "Follow the user’s style instructions exactly."
        )
    }

    user_instructions = (
        f"Generate {n} Twitter replies to the post provided below. "
        "Each reply should feel like it was written by a real user (do not invent usernames): authentic, human, platform-native. "
        "Vary tone, cadence, and personality across replies to reflect diverse Twitter archetypes (developers, founders, hype people, "
        "tech analysts, crypto bros, skeptics, casual users). Use emojis, slang, sarcasm, enthusiasm, or technical jargon as appropriate. "
        "Keep them contextually relevant and varied in length—include 1-sentence quips, emoji-only reactions, short skeptical questions, "
        "and some longer thoughtful replies. Ensure 25–40% are under 10 words.\n\n"
        f"Original tweet:\n\"{tweet}\""
    )

    try:
        # run the blocking call in a thread so we don't block the Discord event loop
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4.1",
            messages=[system_msg, {"role": "user", "content": user_instructions}],
            temperature=0.8,
            max_tokens=max_tokens,
        )
    except Exception as e:
        print("⚠️ OpenAI API error:", e)
        return []

    body = resp.choices[0].message.content
    return [line.strip() for line in body.splitlines() if line.strip()]

# ─── 7) Threading helper ───────────────────────────────────────────────────────
async def reply_in_thread(original_msg: discord.Message, tweet: str, count: int):
    """Create a thread (or use channel), post header & generated replies."""
    # clamp again
    count = max(1, min(count, MAX_COMMENTS))

    try:
        thread = await original_msg.create_thread(
            name=f"Replies: {tweet[:40]}…",
            auto_archive_duration=60
        )
    except discord.HTTPException:
        thread = original_msg.channel

    header = f"💬 Replies to: “{tweet[:60]}…”"
    await thread.send(header)

    replies = await generate_replies(tweet, count)
    if not replies:
        return await thread.send("⚠️ Couldn’t generate replies.")

    for r in replies:
        await thread.send(r)

# ─── 8) Bot events & command ──────────────────────────────────────────────────
@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

@bot.event
async def on_message(message: discord.Message):
    # ignore bot's own messages
    if message.author == bot.user:
        return

    # Auto-reply to TweetShift posts
    tweet, count = parse_tweetshift_message(message.content)
    if tweet and count > 0:
        await reply_in_thread(message, tweet, count)

    # allow commands to run
    await bot.process_commands(message)

@bot.command(name="replybot")
async def replybot(ctx: commands.Context, count: int = None):
    """
    Usage:
      • Reply to a TweetShift post with !replybot         → uses its XXC
      • Reply to any message with !replybot <count>       → uses your specified count
    """
    ref = ctx.message.reference
    if not ref or not ref.message_id:
        return await ctx.send(
            "⚠️ Please reply to a message with `!replybot` or `!replybot <count>`."
        )

    try:
        ref_msg = await ctx.channel.fetch_message(ref.message_id)
    except discord.HTTPException:
        return await ctx.send("⚠️ Could not fetch the referenced message.")

    tweet_text, ts_count = parse_tweetshift_message(ref_msg.content)

    # decide final count
    if tweet_text and ts_count > 0 and count is None:
        final_count = ts_count
    elif count:
        final_count = count
    else:
        return await ctx.send(
            "⚠️ Couldn’t determine count. For non-TweetShift posts use `!replybot <count>`."
        )

    await reply_in_thread(ref_msg, tweet_text or ref_msg.content, final_count)

# ─── 9) Run the bot ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
