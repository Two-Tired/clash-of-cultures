import os
import discord
from dotenv import load_dotenv
import logging

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

level = logging.INFO
fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=level, format=fmt)

class TwoTiredClient(discord.Client):
    async def on_ready(self):
        logging.info(f'{client.user} has connected to Discord!')

client = TwoTiredClient()
client.run(TOKEN)
