from combat import Army, Battle, BattleType, SiegecraftType, battle_round
import matplotlib.pyplot as plt
import numpy as np


def analyze_single_unit_combat():
    single_unit_armies = [
        Army(infantry=1),
        Army(cavalry=1),
        Army(elephants=1),
        Army(leader=1)
    ]
    labels = ['Infantry', 'Cavalry', 'Elephant', 'Leader']

    fig, axs = plt.subplots(4, 4)
    for a, attacker in enumerate(single_unit_armies):
        for d, defender in enumerate(single_unit_armies):
            losses = battle_round(attacker, defender)
            win_matrix = losses['probability'].to_numpy().reshape(2, 2)
            axs[a, d].matshow(win_matrix, vmin=0, vmax=1)
            for (i, j), z in np.ndenumerate(win_matrix):
                axs[a, d].text(j, i, '{:0.1f}%'.format(
                    z*100), ha='center', va='center')

            axs[a, d].set_xticks([0, 1])
            axs[a, d].set_xticklabels(['d keeps', 'd looses'])
            axs[a, d].set_yticks([0, 1])
            axs[a, d].set_yticklabels(['a keeps', 'a looses'])
            if (a == 0):
                axs[a, d].set_title(labels[d], fontsize='large')
            if (d == 0):
                axs[a, d].set_ylabel(labels[a], fontsize='large')

    plt.suptitle('Win matrices for single unit armies, attacker: left, defender: top')
    plt.show()


if __name__ == '__main__':
    attacker = Army(infantry=1,
                    cavalry=0,
                    elephants=0,
                    leader=0,
                    siegecraft_type=SiegecraftType.NONE)

    defender = Army(infantry=0,
                    cavalry=0,
                    elephants=0,
                    leader=1,
                    fortress=False)

    battle = Battle(attacker=attacker, defender=defender)

    # a = attacker.combat_value_probabilities(first_round=False, opponent=defender, attacking=True)
    # d = defender.combat_value_probabilities(first_round=False, opponent=attacker, attacking=False)

    losses = battle_round(attacker, defender)
    print(losses)

    analyze_single_unit_combat()
