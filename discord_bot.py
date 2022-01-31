from discord.ext import commands
import discord
import logging
from dotenv import load_dotenv
import os
from combat import Army, Battle
from analysis import analyze_battle


load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
PREFIX = os.getenv('PREFIX')

level = logging.INFO
fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=level, format=fmt)


bot = commands.Bot(command_prefix='!')


@bot.command(
    name="hi",
    help="Simply greets you back."
)
async def say_hi(ctx):
    response = '!hi'
    await ctx.send(response)


@bot.command(
    name="coc",
    help="Simulates a simple battle in Clash of Cultures."
)
async def simple_battle(ctx,
                        a_I: int, a_C: int, a_E: int, a_L: int,
                        d_I: int, d_C: int, d_E: int, d_L: int):
    attacker = Army(a_I, a_C, a_E, a_L)
    defender = Army(d_I, d_C, d_E, d_L)
    battle = Battle(attacker, defender)
    logging.info(f'Analyzing battle: {attacker} attacking {defender}')

    s = analyze_battle(battle, True)
    if not s:
        raise Exception('None returned by analyze_battle!')
    logging.debug('  ...analysis done')

    s.seek(0)
    chart = discord.File(s, filename="battle_chart.png")
    logging.debug('  ...file created')
    embed = discord.Embed(title=f"{attacker} attacking {defender}")
    embed.set_image(url="attachment://battle_chart.png")
    logging.debug('  ...embed created')

    await ctx.send(embed=embed, file=chart)
    logging.debug('Command finished.')


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.errors.CheckFailure):
        await ctx.send('You do not have the correct role for this command.')
    elif isinstance(error, commands.errors.CommandInvokeError):
        await ctx.send(f'You provided invalid command arguments: {error}')
    else:
        raise error


bot.run(TOKEN)
