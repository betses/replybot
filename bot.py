import os
import asyncio
import discord
import openai
from dotenv import load_dotenv

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENAI_KEY    = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_KEY

intents = discord.Intents.default()
intents.message_content = True  # to read messages
bot = discord.Bot(intents=intents)  # or commands.Bot

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

@bot.event
async def on_message(message):
    # 1) Ignore your own bot
    if message.author == bot.user:
        return

    # 2) Detect TweetShift posts (you’ll refine this)
    if message.author.bot and "20C" in message.content:
        # Extract tweet text & count; here’s where you’ll parse it
        tweet_text, num_comments = parse_tweetshift_message(message.content)

        # 3) Call OpenAI to generate
        replies = await generate_replies(tweet_text, num_comments)

        # 4) Post back
        header = f"💬 Replies to: “{tweet_text[:80]}…”"
        await message.channel.send(header)
        for idx, r in enumerate(replies, 1):
            await message.channel.send(f"{idx}. {r}")

async def generate_replies(tweet: str, n: int) -> list[str]:
    system = {"role": "system", "content": "You’re a friendly community assistant."}
    user   = {"role": "user",   "content": f"Generate {n} concise replies to this tweet:\n\n“{tweet}”"}
    resp = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[system, user],
        temperature=0.7,
        max_tokens=150 * n  # adjust as needed
    )
    text = resp.choices[0].message.content
    # simple split by lines; refine as needed
    return [line.strip() for line in text.splitlines() if line.strip()]

def parse_tweetshift_message(content: str):
    lines = content.splitlines()
    # last line “80L / 10R / 20C”
    last = lines[-1]
    n = int(last.split("/")[-1].strip().replace("C",""))
    tweet = "\n".join(lines[:-1]).strip()
    return tweet, n

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
