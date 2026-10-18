import sys
from contextlib import contextmanager


COMMANDS = ["/chat", "/rag", "/docs", "/method", "/new", "/clear", "/sources",
            "/model", "/info", "/reindex", "/eval", "/help", "/exit"]


class TerminalUI:
    def __init__(self, plain=False):
        self.rich = not plain and sys.stdin.isatty() and sys.stdout.isatty()
        self.buffer = ""
        if self.rich:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import WordCompleter
            from prompt_toolkit.key_binding import KeyBindings
            from rich.console import Console
            self.console = Console(highlight=False)
            bindings = KeyBindings()

            @bindings.add("enter")
            def send(event):
                event.current_buffer.validate_and_handle()

            @bindings.add("escape", "enter")
            def newline(event):
                event.current_buffer.insert_text("\n")

            self.prompt = PromptSession(multiline=True, key_bindings=bindings,
                                        completer=WordCompleter(COMMANDS), complete_while_typing=False)

    def print(self, text="", dim=False):
        if self.rich:
            self.console.print(text, markup=False, style="dim" if dim else None)
        else:
            print(text, flush=True)

    def read(self, label):
        if self.rich:
            return self.prompt.prompt(f"你 · {label} › ").strip()
        return input(f"你 · {label} > ").strip()

    def push(self, text):
        self.buffer += text
        if not self.rich:
            print(text, end="", flush=True)

    @contextmanager
    def answer(self):
        self.buffer = ""
        self.print("\nQwen", dim=True)
        if self.rich:
            from rich.live import Live
            from rich.markdown import Markdown
            with Live(console=self.console, get_renderable=lambda: Markdown(self.buffer or "…"),
                      refresh_per_second=12, transient=True, vertical_overflow="crop"):
                yield self.push
            if self.buffer:
                self.console.print(Markdown(self.buffer))
        else:
            yield self.push
            print()
        self.print()
