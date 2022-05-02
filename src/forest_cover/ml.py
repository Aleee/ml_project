import cmd2


class MLApp(cmd2.Cmd):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, auto_load_commands=False, **kwargs)

        self.hidden_commands.extend(['alias', 'edit', 'macro', 'run_pyscript', 'run_script', 'set', 'shell'])
        self.default_category = 'Встроенные команды'

        self.intro = style("Начните работу с выбора алгоритма (\'setmodel\'). "
                           "Справка: \'?\' или \'help\'. Выход: \'quit\'.",
                           bold=True)


def start():
    app = MLApp()
    app.cmdloop()


if __name__ == '__main__':
    start()
