import combat
from combat import Battle
import matplotlib.pyplot as plt
import numpy as np
import io
from typing import Optional, BinaryIO


def analyze_battle(battle: Battle, to_data_stream=False) -> Optional[BinaryIO]:

    labels = ['4A', '3A', '2A', '1A', '0', '1D', '2D', '3D', '4D']

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.5))

    battle_result = battle.simulate()
    result = battle_result.aggregate()
    combat_bar = combat.to_combat_bar(result)

    rating = combat.rate_combat_bar(combat_bar)

    ax.matshow(combat_bar, vmin=0, vmax=.4)
    ax.plot(rating+4, 0.5, 'r+')
    ax.text(
        rating+4, 1, f'{rating:.2f}', color='red', ha='center', va='center')
    for (i, j), z in np.ndenumerate(combat_bar):
        ax.text(j, i, '{:.0f}'.format(
            z*100), ha='center', va='center')

    ax.set_xticks(range(9))
    ax.set_xticklabels(labels)
    ax.set_yticks([])

    if to_data_stream:
        data_stream = io.BytesIO()
        plt.savefig(data_stream, format='png', bbox_inches='tight')
        plt.close()
        return data_stream
    else:
        plt.show()
        return None
