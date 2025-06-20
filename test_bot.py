# replybot.py

import os
import discord
import asyncio
from dotenv import load_dotenv
from discord.ext import commands
from openai import OpenAI

# 1) Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENAI_KEY    = os.getenv("OPENAI_API_KEY")

if not DISCORD_TOKEN or not OPENAI_KEY:
    print("❌ Missing DISCORD_BOT_TOKEN or OPENAI_API_KEY in .env")
    exit(1)

# 2) Instantiate the OpenAI client (synchronous)
client = OpenAI(api_key=OPENAI_KEY)

# 3) Configure Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# 4) Constants to prevent runaway values
MAX_COMMENTS = 20          # at most 20 replies
MAX_TOKENS_PER_CALL = 1024 # cap tokens per request

# 5) Helper to parse TweetShift‐style posts
def parse_tweetshift_message(content: str):
    lines = content.splitlines()
    if not lines:
        return "", 0
    last = lines[-1].strip()  # e.g. "80L / 10R / 20C"
    try:
        count = int(last.split("/")[-1].replace("C","").strip())
    except:
        count = 0
    tweet_text = "\n".join(lines[:-1]).strip()
    return tweet_text, count

# 6) Core: generate GPT replies with custom instructions
async def generate_replies(tweet: str, n: int) -> list[str]:
    # clamp n and compute token cap
    n = max(1, min(n, MAX_COMMENTS))
    desired = 150 * n
    max_tokens = min(desired, MAX_TOKENS_PER_CALL)

    system_msg = {
        "role": "system",
        "content": (
            "You are simulating a real Twitter conversation. "
            "When asked to generate replies, follow the user’s style instructions exactly."
        )
    }

    user_instructions = (
        f"Generate {n} Twitter replies to the post provided below. "
        "Each reply should feel like it was written by a real user (do not invent usernames): authentic, human, and platform-native. "
        "Vary tone, cadence, and personality across replies to reflect diverse Twitter archetypes such as developers, founders, hype people, tech analysts, crypto bros, skeptics, and casual users. "
        "Infuse replies with distinct stylistic traits like emojis, slang, sarcasm, enthusiasm, or technical jargon where appropriate to the persona. "
        "Maintain contextual relevance to the original tweet’s content, ensuring each reply feels like a legitimate and timely response that could be found in an actual Twitter thread. "
        "Simulate a believable, varied reply chain: not robotic, not repetitive, and true to Twitter’s conversational dynamics. "
        "Vary length like real Twitter users do: include 1-sentence quips, emoji-only reactions, short skeptical questions, and some longer thoughtful replies. "
        "Ensure 25–40% of replies are very short (under 10 words).\n\n"
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

    text = resp.choices[0].message.content
    # split on newlines, drop empty lines
    return [line.strip() for line in text.splitlines() if line.strip()]

# 7) Bot is ready
@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

# 8) Command: replybot
@bot.command(name="replybot")
async def replybot(ctx: commands.Context):
    # Must be a reply to another message
    ref = ctx.message.reference
    if not ref or not ref.message_id:
        return await ctx.send(
            "⚠️ Please reply to a TweetShift post with `!replybot` to generate replies."
        )

    # Fetch the referenced message
    try:
        ref_msg = await ctx.channel.fetch_message(ref.message_id)
    except Exception:
        return await ctx.send("⚠️ Could not fetch the referenced message.")

    tweet, count = parse_tweetshift_message(ref_msg.content)
    if not tweet or count <= 0:
        return await ctx.send(
            "⚠️ The referenced message doesn't look like a valid TweetShift post. "
            "Make sure it ends with something like `80L / 10R / 20C`."
        )

    count = max(1, min(count, MAX_COMMENTS))
    header = f"💬 Replies to: “{tweet[:60]}…”"

    # Attempt to create a thread on the original message
    try:
        thread = await ref_msg.create_thread(
            name=f"ReplyBot: {tweet[:50]}",
            auto_archive_duration=60  # in minutes
        )
    except Exception:
        thread = ctx.channel  # fallback to same channel

    # Send header & generate replies
    await thread.send(header)
    replies = await generate_replies(tweet, count)
    if not replies:
        return await thread.send("⚠️ Couldn’t generate replies for this tweet.")

    for r in replies:
        await thread.send(r)

# 9) Run the bot
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
