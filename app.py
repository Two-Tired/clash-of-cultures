
import combat
from combat import Army, Battle, SiegecraftType
import analysis
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True


def analyze_single_unit_combat(fortress: bool = False,
                               siegecraft_type: SiegecraftType
                               = SiegecraftType.NONE):
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
            losses = combat.battle_round(attacker, defender, first_round=True)
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


def analyze_battle_full(battle: Battle):
    labels = ['4A', '3A', '2A', '1A', '0', '1D', '2D', '3D', '4D']
    plot_labels_a = ['None', 'Steal Weapons', 'Cancel Attack', 'Cancel Ignore']
    plot_labels_d = ['None', 'Steal Weapons', 'Fortress', 'SW + Fortress']

    a_I = battle.attacker.infantry
    a_C = battle.attacker.cavalry
    a_E = battle.attacker.elephants
    a_L = battle.attacker.leader

    d_I = battle.defender.infantry
    d_C = battle.defender.cavalry
    d_E = battle.defender.elephants
    d_L = battle.defender.leader

    fig, axs = plt.subplots(4, 4, figsize=(15, 5))
    for a, attacker in enumerate([
        Army(a_I, a_C, a_E, a_L),
        Army(a_I, a_C, a_E, a_L, steel_weapons=True),
        Army(a_I, a_C, a_E, a_L, siegecraft_type=SiegecraftType.CANCEL_ATTACK),
        Army(a_I, a_C, a_E, a_L, siegecraft_type=SiegecraftType.CANCEL_IGNORE),
        Army(a_I, a_C, a_E, a_L, steel_weapons=True,
             siegecraft_type=SiegecraftType.CANCEL_ATTACK),
        Army(a_I, a_C, a_E, a_L, steel_weapons=True,
             siegecraft_type=SiegecraftType.CANCEL_IGNORE)
    ]):
        for d, defender in enumerate([
            Army(d_I, d_C, d_E, d_L),
            Army(d_I, d_C, d_E, d_L, steel_weapons=True),
            Army(d_I, d_C, d_E, d_L, fortress=True),
            Army(d_I, d_C, d_E, d_L, steel_weapons=True, fortress=True),
        ]):
            if (not defender.fortress) and \
                    (attacker.siegecraft_type is not SiegecraftType.NONE):
                continue

            r = a
            c = d
            if attacker.siegecraft_type is not SiegecraftType.NONE:
                if not attacker.steel_weapons:
                    c = d - 2
                else:
                    r = a - 2

            curr_battle = Battle(attacker, defender)
            battle_result = curr_battle.simulate()
            result = battle_result.aggregate()
            combat_bar = combat.to_combat_bar(result)

            rating = combat.rate_combat_bar(combat_bar)
            # print(r, c, rating)

            axs[r, c].matshow(combat_bar, vmin=0, vmax=.4)
            axs[r, c].plot(rating+4, 0.5, 'r+')
            axs[r, c].text(rating+4, 1, f'{rating:.2f}',
                           color='red', ha='center', va='center')
            for (i, j), z in np.ndenumerate(combat_bar):
                axs[r, c].text(j, i, '{:.0f}'.format(
                    z*100), ha='center', va='center')

            axs[r, c].set_xticks(range(9))
            axs[r, c].set_xticklabels(labels)
            axs[r, c].set_yticks([])
            if r == 0:
                axs[r, c].set_title(plot_labels_d[d], fontsize='large')
            # if (d == 0 or (d == 2 and a > 1)):
            if c == 0:
                axs[r, c].set_ylabel(
                    plot_labels_a[a], fontsize='large', rotation=0)
                axs[r, c].yaxis.set_label_coords(-.22, .25)
            if (c == 0 or c == 1) and r == 2:
                axs[r, c].set_title(plot_labels_d[d], fontsize='large')
            if (c == 2 and r > 1):
                axs[r, c].set_ylabel('+SW', fontsize='large', rotation=0)
                axs[r, c].yaxis.set_label_coords(-.1, .25)

    plt.suptitle(f'Remaining units: $\\mathcal{{{battle.attacker}}}$ (left) '
                 'attacking $\\mathcal{{{battle.defender}}}$ (top)')
    plt.show()


if __name__ == '__main__':
    attacker = Army(infantry=1,
                    cavalry=0,
                    elephants=0,
                    leader=0)

    defender = Army(infantry=1,
                    cavalry=0,
                    elephants=0,
                    leader=0)

    battle = Battle(attacker, defender)

    # import cProfile
    # import pstats

    # with cProfile.Profile() as pr:
    analysis.analyze_battle(battle)

    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats(100)
    # stats.dump_stats(filename='with_cache.prof')

    # import timeit, functools
    # t_simple = timeit.Timer(functools.partial(battle.simulate, True))
    # print(t_simple.timeit(3))
    # t_exact = timeit.Timer(functools.partial(battle.simulate, False))
    # print(t_exact.timeit(3))
