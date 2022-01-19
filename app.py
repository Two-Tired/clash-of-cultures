from combat import Army, Battle, BattleType, SiegecraftType


if __name__ == '__main__':
    attacker = Army(infantry=0,
                    cavalry=0,
                    elephants=1,
                    leader=0,
                    siegecraft_type=SiegecraftType.NONE)

    defender = Army(infantry=1,
                    fortress=True)

    battle = Battle(attacker=attacker, defender=defender)

    a = attacker.combat_value_probabilities(first_round=False,
                                            opponent=defender,
                                            attacking=True)
    # d = defender.combat_value_probabilities(first_round=False, opponent=attacker, attacking=False)

    print(a)
