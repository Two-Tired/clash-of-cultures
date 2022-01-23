from combat import Army, Battle, BattleType, SiegecraftType, to_combat_bar
import matplotlib.pyplot as plt
import numpy as np


def analyze_single_unit_combat(fortress: bool = False, siegecraft_type: SiegecraftType = SiegecraftType.NONE):
    single_unit_armies = [
        Army(infantry=1, siegecraft_type=siegecraft_type, fortress=fortress),
        Army(cavalry=1, siegecraft_type=siegecraft_type, fortress=fortress),
        Army(elephants=1, siegecraft_type=siegecraft_type, fortress=fortress),
        Army(leader=1, siegecraft_type=siegecraft_type, fortress=fortress)
    ]
    labels = ['Infantry', 'Cavalry', 'Elephant', 'Leader']

    fig, axs = plt.subplots(4, 4)
    for a, attacker in enumerate(single_unit_armies):
        for d, defender in enumerate(single_unit_armies):
            losses = battle_round(attacker, defender, first_round=True)
            win_matrix = losses['probability'].to_numpy().reshape(2, 2)
            axs[a, d].matshow(win_matrix, vmin=0, vmax=.611)
            for (i, j), z in np.ndenumerate(win_matrix):
                axs[a, d].text(j, i, '{:0.1f}%'.format(
                    z*100), ha='center', va='center')

            axs[a, d].set_xticks([0, 1])
            axs[a, d].set_xticklabels(['d keeps', 'd looses'])
            axs[a, d].set_yticks([0, 1])
            axs[a, d].set_yticklabels(
                ['a keeps', 'a looses'], rotation=90, va='center')
            if (a == 0):
                axs[a, d].set_title(labels[d], fontsize='large')
            if (d == 0):
                axs[a, d].set_ylabel(labels[a], fontsize='large')

    plt.suptitle(
        'Win matrices for single unit armies, attacker: left, defender: top')
    plt.show()


def analyze_battle(battle: Battle):
    labels = ['4A', '3A', '2A', '1A', 'None', '1D', '2D', '3D', '4D']
    plot_labels_a = ['None', 'Steal Weapons', 'S-Cancle Attack', 'S-Cancle Ignore',
                     'S-Cancle Both', 'SW +\nS-Cancle Attack', 'SW +\nS-Cancle Ignore', 'SW +\nS-Cancle Both']
    plot_labels_d = ['None', 'Steal Weapons', 'Fortress', 'SW + Fortress']

    a_I = battle.attacker.infantry
    a_C = battle.attacker.cavalry
    a_E = battle.attacker.elephants
    a_L = battle.attacker.leader

    d_I = battle.defender.infantry
    d_C = battle.defender.cavalry
    d_E = battle.defender.elephants
    d_L = battle.defender.leader

    fig, axs = plt.subplots(8, 4)
    for a, attacker in enumerate([
        Army(a_I, a_C, a_E, a_L),
        Army(a_I, a_C, a_E, a_L, steel_weapons=True),
        Army(a_I, a_C, a_E, a_L, siegecraft_type=SiegecraftType.CANCEL_ATTACK),
        Army(a_I, a_C, a_E, a_L, siegecraft_type=SiegecraftType.CANCEL_IGNORE),
        Army(a_I, a_C, a_E, a_L, siegecraft_type=SiegecraftType.CANCEL_BOTH),
        Army(a_I, a_C, a_E, a_L, steel_weapons=True,
             siegecraft_type=SiegecraftType.CANCEL_ATTACK),
        Army(a_I, a_C, a_E, a_L, steel_weapons=True,
             siegecraft_type=SiegecraftType.CANCEL_IGNORE),
        Army(a_I, a_C, a_E, a_L, steel_weapons=True,
             siegecraft_type=SiegecraftType.CANCEL_BOTH),
    ]):
        for d, defender in enumerate([
            Army(d_I, d_C, d_E, d_L),
            Army(d_I, d_C, d_E, d_L, steel_weapons=True),
            Army(d_I, d_C, d_E, d_L, fortress=True),
            Army(d_I, d_C, d_E, d_L, steel_weapons=True, fortress=True),
        ]):
            curr_battle = Battle(attacker, defender)
            battle_result = curr_battle.simulate()
            result = battle_result.aggregate()
            matrix = to_combat_bar(result)

            axs[a, d].matshow(matrix, vmin=0, vmax=.4)
            for (i, j), z in np.ndenumerate(matrix):
                axs[a, d].text(j, i, '{:0.2f}'.format(
                    z), ha='center', va='center')

            axs[a, d].set_xticks(range(9))
            axs[a, d].set_xticklabels(labels)
            axs[a, d].set_yticks([])
            if (a == 0):
                axs[a, d].set_title(plot_labels_d[d], fontsize='large')
            if (d == 0):
                axs[a, d].set_ylabel(plot_labels_a[a], fontsize='large', rotation=0)
                axs[a, d].yaxis.set_label_coords(-.22, .5)

    plt.suptitle(
        f'Remaining units: {battle.attacker} (left) attacking {battle.defender} (top)')
    plt.show()


if __name__ == '__main__':
    attacker = Army(infantry=4,
                    cavalry=0,
                    elephants=0,
                    leader=0,
                    siegecraft_type=SiegecraftType.NONE)

    defender = Army(infantry=4,
                    cavalry=0,
                    elephants=0,
                    leader=0,
                    fortress=False)

    # a = attacker.combat_value_probabilities(first_round=False, opponent=defender, attacking=True)
    # d = defender.combat_value_probabilities(first_round=False, opponent=attacker, attacking=False)

    # losses = battle_round(attacker, defender)

    # analyze_single_unit_combat(fortress=False)

    battle = Battle(attacker, defender)
    # battle_result = battle.simulate()
    # result = battle_result.aggregate()
    # print(battle)
    # print(result)

    analyze_battle(battle)

    # import timeit, functools
    # t_simple = timeit.Timer(functools.partial(battle.simulate, True))
    # print(t_simple.timeit(3))
    # t_exact = timeit.Timer(functools.partial(battle.simulate, False))
    # print(t_exact.timeit(3))
