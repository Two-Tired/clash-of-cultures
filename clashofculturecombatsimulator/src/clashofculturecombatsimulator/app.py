"""
Simulates battles in the board game 'Clash of Cultures'
"""
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, CENTER, RIGHT


class ClashofCultureCombatSimulator(toga.App):

    def startup(self):
        main_box = toga.Box(style=Pack(direction=COLUMN, alignment=CENTER))

        battle_type_sel = toga.Selection('battle_type',
                                         items=['Land', 'Landing', 'Naval'],
                                         style=Pack(width=200, text_align=CENTER))

        att_box = toga.Box(style=Pack(direction=COLUMN, flex=1))
        att_lbl = toga.Label('Attacker', style=Pack(text_align=CENTER))

        att_civ_sel = toga.Selection('att_civ',
                                     items=['generic civ', 'Egypt', 'Vikings'])

        lbl_width = 10
        att_I_lbl = toga.Label('I', style=Pack(
            width=lbl_width, text_align=CENTER))
        att_I_sl = toga.Slider('att_I', range=(
            0, 4), default=0, tick_count=5, style=Pack(flex=1))
        att_I_box = toga.Box()
        att_I_box.add(att_I_lbl)
        att_I_box.add(att_I_sl)

        att_C_lbl = toga.Label('C', style=Pack(
            width=lbl_width, text_align=CENTER))
        att_C_sl = toga.Slider('att_C', range=(
            0, 4), default=0, tick_count=5, style=Pack(flex=1))
        att_C_box = toga.Box()
        att_C_box.add(att_C_lbl)
        att_C_box.add(att_C_sl)

        att_E_lbl = toga.Label('E', style=Pack(
            width=lbl_width, text_align=CENTER))
        att_E_sl = toga.Slider('att_E', range=(
            0, 4), default=0, tick_count=5, style=Pack(flex=1))
        att_E_box = toga.Box()
        att_E_box.add(att_E_lbl)
        att_E_box.add(att_E_sl)

        att_leader_sel = toga.Selection('att_leader',
                                        items=['no leader', 'leader 1', 'leader 2', 'leader 3'])

        att_sw_sel = toga.Selection('att_sw',
                                    items=['no steal weapons', 'steal weapons researched', 'steal weapons active'])
        att_siege_sel = toga.Selection('att_siege',
                                       items=['no siege', 'cancel extra hit', 'cancel ignore hit', 'cancel both'])

        att_box.add(att_lbl)
        att_box.add(att_civ_sel)
        att_box.add(att_I_box)
        att_box.add(att_C_box)
        att_box.add(att_E_box)
        att_box.add(att_leader_sel)
        att_box.add(att_sw_sel)
        att_box.add(att_siege_sel)

        def_box = toga.Box(style=Pack(direction=COLUMN, flex=1))
        def_lbl = toga.Label('Defender', style=Pack(text_align=CENTER))

        def_civ_sel = toga.Selection('att_civ',
                                     items=['generic civ', 'Egypt', 'Vikings'])

        def_I_lbl = toga.Label('I', style=Pack(
            width=lbl_width, text_align=CENTER))
        def_I_sl = toga.Slider('def_I', range=(
            0, 4), default=0, tick_count=5, style=Pack(flex=1))
        def_I_box = toga.Box()
        def_I_box.add(def_I_lbl)
        def_I_box.add(def_I_sl)

        def_C_lbl = toga.Label('C', style=Pack(
            width=lbl_width, text_align=CENTER))
        def_C_sl = toga.Slider('def_C', range=(
            0, 4), default=0, tick_count=5, style=Pack(flex=1))
        def_C_box = toga.Box()
        def_C_box.add(def_C_lbl)
        def_C_box.add(def_C_sl)

        def_E_lbl = toga.Label('E', style=Pack(
            width=lbl_width, text_align=CENTER))
        def_E_sl = toga.Slider('def_E', range=(
            0, 4), default=0, tick_count=5, style=Pack(flex=1))
        def_E_box = toga.Box()
        def_E_box.add(def_E_lbl)
        def_E_box.add(def_E_sl)

        def_leader_sel = toga.Selection('def_leader',
                                        items=['no leader', 'leader 1', 'leader 2', 'leader 3'])

        def_sw_sel = toga.Selection('def_sw',
                                    items=['no steal weapons', 'steal weapons researched', 'steal weapons active'])
        def_fortress_sel = toga.Selection('def_sw',
                                          items=['without fortress', 'fortress'])

        def_box.add(def_lbl)
        def_box.add(def_civ_sel)
        def_box.add(def_I_box)
        def_box.add(def_C_box)
        def_box.add(def_E_box)
        def_box.add(def_leader_sel)
        def_box.add(def_sw_sel)
        def_box.add(def_fortress_sel)

        opponents_box = toga.Box(style=Pack(direction=ROW))
        opponents_box.add(att_box)
        opponents_box.add(def_box)

        battle_sim_btn = toga.Button(
            'Simulate Battle', 'battle_sim', style=Pack(width=200, text_align=CENTER))
        result_txt = toga.MultilineTextInput(
            'result', style=Pack(flex=1), readonly=True, placeholder='Results will go here')

        main_box.add(battle_type_sel)
        main_box.add(opponents_box)
        main_box.add(battle_sim_btn)
        main_box.add(result_txt)

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()

    # def startup(self):
    #     main_box = toga.Box(style=Pack(direction=COLUMN))

    #     name_label = toga.Label(
    #         'Your name: ',
    #         style=Pack(padding=(0, 5))
    #     )
    #     self.name_input = toga.TextInput(style=Pack(flex=1))

    #     name_box = toga.Box(style=Pack(direction=ROW, padding=5))
    #     name_box.add(name_label)
    #     name_box.add(self.name_input)

    #     button = toga.Button(
    #         'Say Hello!',
    #         on_press=self.say_hello,
    #         style=Pack(padding=5)
    #     )

    #     main_box.add(name_box)
    #     main_box.add(button)

    #     self.main_window = toga.MainWindow(title=self.formal_name)
    #     self.main_window.content = main_box
    #     self.main_window.show()

    # def say_hello(self, widget):
    #     print("Hello", self.name_input.value)


def main():
    return ClashofCultureCombatSimulator()
