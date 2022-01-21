from enum import Enum
from collections import Counter
import attrs
import math
import numpy as np
import pandas as pd
from typing import Callable, Optional


class UnitType(Enum):
    '''
    Enum for the different unit types.
    A special case is the elephant, which is split up into three
    sub-units to allow for computation of their ability (it is
    relevant which of the dice's sides showed the elephant symbol)

    ...

    Values are selected such that bitwise comparisons are possible
    to determine whether some elephant is selected. For example:
        (is_elephant.value & UnitType.ELEPHANTS.value) > 0
    returns true when "is_elephant" is one of the elephant types
    '''
    ELEPHANTS_2 = 1
    ELEPHANTS_3 = 2
    ELEPHANTS_4 = 4
    ELEPHANTS = 7
    INFANTRY = 8
    CAVALRY = 16
    LEADER = 32
    SHIP = 64


class BattleType(Enum):
    '''Enum for different battle grounds.'''
    NAVAL = 1
    LANDING = 2
    LAND = 4


class SiegecraftType(Enum):
    '''Enum for selecting the type of siege the army payed for.'''
    NONE = 0
    CANCEL_ATTACK = 1
    CANCEL_IGNORE = 2
    CANCEL_BOTH = 3


def validate_max_army_count(instance, attribute, value):
    '''
    attrs-validator function for the maximum capacity of a single hex.
    '''
    if (instance.army_size > 4):
        raise ValueError(
            f'Maximum army size is 4!')


@attrs.define(frozen=True)
class Army:
    '''
    A class to represent a single army (i.e. group of military units).
    '''
    infantry: int = attrs.field(
        default=0,
        validator=[
            attrs.validators.instance_of(int),
            attrs.validators.ge(0),
            attrs.validators.le(4),
            validate_max_army_count
        ]
    )
    cavalry: int = attrs.field(
        default=0,
        validator=[
            attrs.validators.instance_of(int),
            attrs.validators.ge(0),
            attrs.validators.le(4),
            validate_max_army_count
        ]
    )
    elephants: int = attrs.field(
        default=0,
        validator=[
            attrs.validators.instance_of(int),
            attrs.validators.ge(0),
            attrs.validators.le(4),
            validate_max_army_count
        ]
    )
    leader: int = attrs.field(
        default=0,
        validator=[
            attrs.validators.instance_of(int),
            attrs.validators.ge(0),
            attrs.validators.le(1),
            validate_max_army_count
        ]
    )
    ships: int = attrs.field(
        default=0,
        validator=[
            attrs.validators.instance_of(int),
            attrs.validators.ge(0),
            attrs.validators.le(4)
        ]
    )
    fortress: bool = attrs.field(
        default=False,
        validator=attrs.validators.instance_of(bool)
    )
    steel_weapons: bool = attrs.field(
        default=False,
        validator=attrs.validators.instance_of(bool)
    )
    warships: bool = attrs.field(
        default=False,
        validator=attrs.validators.instance_of(bool)
    )
    siegecraft_type: SiegecraftType = attrs.field(
        default=SiegecraftType.NONE,
        validator=attrs.validators.instance_of(SiegecraftType)
    )

    @property
    def army_size(self):
        return self.infantry + self.cavalry + self.elephants + self.leader

    def combat_value_probabilities(self, first_round: bool = False, opponent: 'Army' = None, attacking: bool = False, battle_type: 'BattleType' = BattleType.LAND):
        '''
        Returns a list of probabilities with corresponding army values and ignored hits for this army when
        fighting the specified opponent in the specified environment.

        :param first_round: is this the first battle round?
        :param opponent:    the opposing army
        :param attacking:   are you attacking?
        :param type:        the type of battle you are fighting

        :returns: DataFrame with columns 'value', 'ignored_hits', 'probability'
        '''
        army_size_modifier = 0

        # fortresses
        if (
            not attacking and  # you are defending ... AND
            first_round and  # it is the first combat round ... AND
            self.fortress and  # you have a fortress ... AND
            (opponent == None or  # oppenent is not specified ... OR
             (opponent.siegecraft_type.value & SiegecraftType.CANCEL_ATTACK.value) == 0)  # opponent does NOT cancel attack
        ):
            army_size_modifier += 1

        # roll the dice!
        rolls = roll_dice(self.army_size + army_size_modifier)

        # combat abilities
        if (battle_type == BattleType.NAVAL):
            rolls = roll_dice(self.ships + army_size_modifier)

            applied_abilities = [
                (value, 0, prob)
                for value, prob
                in zip(
                    rolls['value'],
                    rolls['probability']
                )
            ]
        else:
            # roll the dice!
            rolls = roll_dice(self.army_size + army_size_modifier)

            applied_abilities = [
                self.apply_combat_abilities(value, i, c, e2, e3, e4, l, prob)
                for value, i, c, e2, e3, e4, l, prob
                in zip(
                    rolls['value'],
                    rolls[UnitType.INFANTRY],
                    rolls[UnitType.CAVALRY],
                    rolls[UnitType.ELEPHANTS_2],
                    rolls[UnitType.ELEPHANTS_3],
                    rolls[UnitType.ELEPHANTS_4],
                    rolls[UnitType.LEADER],
                    rolls['probability']
                )
            ]

        ignored_hits_modifier = 0
        # warships
        if (
            first_round and
            (battle_type.value & 3) > 0 and  # either naval battle or landing ... AND
            self.warships  # you have warships researched
        ):
            ignored_hits_modifier += 1

        # TODO: I am not sure if you can extend the list in the list comprehension above, then this step can be skipped
        applied_abilities_as_list = list()
        for r in applied_abilities:
            applied_abilities_as_list.extend(r)

        # group all entries with same value and ignored_hits and sum their probabilities
        combat_value_probs = pd.DataFrame(
            applied_abilities_as_list, columns=['value', 'ignored_hits', 'probability'])
        combat_value_probs['ignored_hits'] += ignored_hits_modifier
        combat_value_probs = combat_value_probs \
            .groupby(['value', 'ignored_hits']) \
            .sum() \
            .reset_index()

        # steel weapons
        combat_value_modifier = 0
        if (self.steel_weapons):
            combat_value_modifier += 1
            if (opponent == None or not opponent.steel_weapons):
                combat_value_modifier += 1
        combat_value_probs['value'] = combat_value_probs['value'] + \
            combat_value_modifier

        return combat_value_probs

    def apply_combat_abilities(self, value, i, c, e2, e3, e4, l, prob):
        '''
        Returns a list of rolls modified by the combat abilities of your units.

        :param value: original summed dice values
        :param i: rolled number of infantry abilities
        :param c: rolled number of cavalry abilities
        :param e2: rolled number of elephant abilities (on dice side 2)
        :param e3: rolled number of elephant abilities (on dice side 3)
        :param e4: rolled number of elephant abilities (on dice side 4)
        :param l: rolled number of leader abilities
        :param prob: probability of this dice roll
        :returns: list of tuples, each being one possible dice roll in the form: ('value', 'ignored_hits', 'probability')
        '''
        infantry_ability = self.infantry_ability(i)
        cavalry_ability = self.cavalry_ability(c)
        ignore_count, elephant_malus = self.elephant_ability(e2, e3, e4)
        leader_ability = self.is_leader_ability_active(l)

        combat_value = value + infantry_ability + cavalry_ability - elephant_malus
        if (not leader_ability):
            combat_value_prob = (
                combat_value, ignore_count, prob)
            return [combat_value_prob]
        else:
            return self.leader_ability(value, i, c, e2, e3, e4, prob)

    def infantry_ability(self, i):
        '''
        Returns the combat bonus through infantry ability activations.

        :param i: rolled number of infantry abilities
        :returns: combat bonus
        '''
        return min(i, self.infantry)

    def cavalry_ability(self, c):
        '''
        Returns the combat bonus through cavalry ability activations.

        :param c: rolled number of cavalry abilities
        :returns: combat bonus
        '''
        return min(c, self.cavalry) * 2

    def elephant_ability(self, e2, e3, e4):
        '''
        Returns the combat malus through elephant ability activations and the
        number of ignored hits.

        :param e2: rolled number of elephant abilities (on dice side 2)
        :param e3: rolled number of elephant abilities (on dice side 3)
        :param e4: rolled number of elephant abilities (on dice side 4)
        :returns: tuple (combat malus, ignored hits)
        '''
        e = sum([e2, e3, e4])
        ignore_count = min(e, self.elephants)
        malus = 0
        for i in range(self.elephants):
            if (e2 > 0):
                malus += 2
                e2 -= 1
                continue
            elif (e3 > 0):
                malus += 3
                e3 -= 1
                continue
            elif (e4 > 0):
                malus += 4
                e4 -= 1
                continue
            else:
                break

        return ignore_count, malus

    def is_leader_ability_active(self, l):
        '''
        Returns whether your leader ability gets activated by your roll

        :param l: rolled number of leader abilities
        :returns: whether leader ability is active
        '''
        return l > 0 and self.leader > 0

    def leader_ability(self, value, i, c, e2, e3, e4, prob):
        '''
        Returns a list of all possible combination when re-rolling the leader dice.
        Make sure that `is_leader_ability_active` is true!

        :param value: original summed dice values
        :param i: rolled number of infantry abilities
        :param c: rolled number of cavalry abilities
        :param e2: rolled number of elephant abilities (on dice side 2)
        :param e3: rolled number of elephant abilities (on dice side 3)
        :param e4: rolled number of elephant abilities (on dice side 4)
        :returns: list of tuples, each being one possible dice roll in the form: ('value', 'ignored_hits', 'probability')
        '''
        value = value - 1  # remove dice showing leader symbol

        infantry_ability = self.infantry_ability(i)
        cavalry_ability = self.cavalry_ability(c)
        ignore_count, elephant_malus = self.elephant_ability(e2, e3, e4)

        combat_value_probs = list()
        for new_roll in dice[1:]:
            additional_value = new_roll[0]
            # times 1.2 to account for missing 1's
            new_probability = prob * new_roll[2] * 1.2
            new_ignore_count = ignore_count

            if (new_roll[1] == UnitType.INFANTRY):
                additional_value += self.infantry_ability(
                    i+1) - infantry_ability
            elif (new_roll[1] == UnitType.CAVALRY):
                additional_value += self.cavalry_ability(i+1) - cavalry_ability
            elif (new_roll[1] == UnitType.ELEPHANTS_2):
                new_ignore_count, new_elephant_malus = self.elephant_ability(
                    e2+1, e3, e4)
                additional_value += elephant_malus - new_elephant_malus
            elif (new_roll[1] == UnitType.ELEPHANTS_3):
                new_ignore_count, new_elephant_malus = self.elephant_ability(
                    e2, e3+1, e4)
                additional_value += elephant_malus - new_elephant_malus
            elif (new_roll[1] == UnitType.ELEPHANTS_4):
                new_ignore_count, new_elephant_malus = self.elephant_ability(
                    e2, e3, e4+1)
                additional_value += elephant_malus - new_elephant_malus
            else:
                print(
                    "ERROR! All cases should be checked, this should never print.")

            new_value = value + additional_value

            combat_value_probs.append((
                new_value,
                new_ignore_count,
                new_probability
            ))
        return combat_value_probs


@attrs.define(frozen=True)
class Battle:
    '''
    Class representing a battle.
    '''
    attacker: Army = attrs.field(
        validator=attrs.validators.instance_of(Army)
    )
    defender: Army = attrs.field(
        validator=attrs.validators.instance_of(Army)
    )
    type: BattleType = attrs.field(
        default=BattleType.LAND,
        validator=attrs.validators.instance_of(BattleType)
    )


def values_to_hits(combat_value_probs):
    combat_value_probs = combat_value_probs.rename(columns={'value': 'hits'})
    combat_value_probs['hits'] = combat_value_probs['hits'].floordiv(5)

    combat_value_probs = combat_value_probs \
        .groupby(['hits', 'ignored_hits']) \
        .sum() \
        .reset_index()

    return combat_value_probs


def battle_round(attacker: 'Army', defender: 'Army', first_round: bool = False):
    '''
    Given the two fighting armies, this method returns the losses of a battle round.

    :param attacker: the attacking army
    :param defender: the defending army
    :param first_round: whether this is the first battle round
    :returns: DataFrame with columns 'losses_attacker', 'losses_defender', 'probability'
    '''
    a = attacker.combat_value_probabilities(first_round=first_round,
                                            opponent=defender,
                                            attacking=True)
    d = defender.combat_value_probabilities(first_round=first_round,
                                            opponent=attacker,
                                            attacking=False)

    a_hits = values_to_hits(a)
    d_hits = values_to_hits(d)

    # merge all rows of a with all rows of d (cross-product)
    cross = a_hits.merge(d_hits, 'cross', suffixes=('_attacker', '_defender'))

    # compute combined probability of each row
    cross['probability'] = cross['probability_attacker'] * \
        cross['probability_defender']

    # compute losses for each opponent
    cross['losses_attacker'] = cross['hits_defender'].subtract(
        cross['ignored_hits_attacker']).clip(0, attacker.army_size)
    cross['losses_defender'] = cross['hits_attacker'].subtract(
        cross['ignored_hits_defender']).clip(0, defender.army_size)

    # group losses
    losses = cross.groupby(['losses_attacker', 'losses_defender'])[
        'probability'].sum().reset_index()

    return losses


def max_count_reduce_strategy(hits_taken: int, army: Army) -> Optional[Army]:
    '''
    The basic strategy for removing units after taking hits. The strategy evolves around
    keeping a high unit diversity by reducing the unit type that has the most duplicates.
    In case of ties, units are removed in order 'infantry -> cavalry -> elephants -> leader'

    :param hits_taken: number of hits taken
    :param army: your army
    :returns: new army with reduced unit counts or None if
    '''
    # counts are given in order of increasing value such that removing the first
    # occurence of the maximum count always removes least important unit type
    counts = np.array([army.infantry, army.cavalry,
                      army.elephants, army.leader])

    for i in range(min(army.army_size, hits_taken)):
        largest_type = np.argmax(counts)
        counts[largest_type] -= 1

    if (sum(counts) == 0):
        return None

    counts = counts.tolist()
    new_army = attrs.evolve(
        army, infantry=counts[0], cavalry=counts[1], elephants=counts[2], leader=counts[3])
    return new_army


def reduce_armies(attacker: 'Army',
                  hits_attacker: int,
                  defender: 'Army',
                  hits_defender: int,
                  reduce_strategy: Callable[[int, Army], Army] = max_count_reduce_strategy) -> (Optional[Army], Optional[Army]):
    '''
    Returns the reduced armies when applying the specified reduce strategy.

    :param attacker: attacking army
    :param hits_atttacker: number of hits the attacker deals
    :param defender: defending army
    :param hits_defender: number of hits the defender deals
    :returns: tuple of the two Armies, which might be None due to loosing all units
    '''
    a = reduce_strategy(hits_defender, attacker)
    d = reduce_strategy(hits_attacker, defender)

    return a, d


# value, skill, probability
dice = [
    (1, UnitType.LEADER, 1/6),
    (2, UnitType.CAVALRY, 1/12),
    (2, UnitType.ELEPHANTS_2, 1/12),
    (3, UnitType.INFANTRY, 1/12),
    (3, UnitType.ELEPHANTS_3, 1/12),
    (4, UnitType.ELEPHANTS_4, 1/12),
    (4, UnitType.CAVALRY, 1/12),
    (5, UnitType.CAVALRY, 1/12),
    (5, UnitType.INFANTRY, 1/12),
    (6, UnitType.INFANTRY, 1/6)
]


def roll_dice(count):
    '''
    Rolls the specified amount of dice.

    :param count: number of dice to roll
    :returns: DataFrame with total roll value, the number of ability activations per unit and the probability of this roll
    '''
    sequences = dice_recursion(count)  # (value, type, prob)
    rolls = list()  # (value, prob, I, C, E, L)
    for sequence in sequences:
        total_value = sum(elem[0] for elem in sequence)
        c = Counter(elem[1] for elem in sequence)
        total_prob = math.prod(elem[2] for elem in sequence)
        roll = (total_value,
                total_prob,
                c[UnitType.INFANTRY],
                c[UnitType.CAVALRY],
                c[UnitType.ELEPHANTS_2],
                c[UnitType.ELEPHANTS_3],
                c[UnitType.ELEPHANTS_4],
                c[UnitType.LEADER])
        rolls.append(roll)

    rolls = pd.DataFrame(rolls, columns=[
        'value', 'probability', UnitType.INFANTRY, UnitType.CAVALRY, UnitType.ELEPHANTS_2, UnitType.ELEPHANTS_3, UnitType.ELEPHANTS_4, UnitType.LEADER])
    rolls = rolls.groupby(['value', UnitType.INFANTRY, UnitType.CAVALRY, UnitType.ELEPHANTS_2,
                          UnitType.ELEPHANTS_3, UnitType.ELEPHANTS_4, UnitType.LEADER]).sum().reset_index()
    return rolls


def dice_recursion(count, sequences=None):
    '''
    Used internally by `roll_dice` to get the correct rolls.
    '''
    if (count == 0):
        return sequences

    if sequences is None:
        new_sequences = [[d] for d in dice]
    else:
        new_sequences = list()
        for sequence in sequences:
            extended = [sequence + [d] for d in dice]
            new_sequences.extend(extended)

    return dice_recursion(count-1, new_sequences)
