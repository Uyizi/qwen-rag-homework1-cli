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

    def read(self):
        self.print()
        if self.rich:
            return self.prompt.prompt("› ").strip()
        return input("> ").strip()

    @contextmanager
    def busy(self, message, enabled=True):
        if self.rich and enabled:
            with self.console.status(message, spinner="dots") as status:
                yield status.update
        else:
            yield lambda message: None

    def push(self, text):
        self.buffer += text
        if not self.rich:
            print(text, end="", flush=True)

    @contextmanager
    def answer(self):
        self.buffer = ""
        self.print()
        if self.rich:
            from rich.live import Live
            from rich.markdown import Markdown
            try:
                with Live(console=self.console, get_renderable=lambda: Markdown(self.buffer or "…"),
                          refresh_per_second=12, transient=True, vertical_overflow="crop"):
                    yield self.push
            finally:
                if self.buffer:
                    self.console.print(Markdown(self.buffer))
        else:
            try:
                yield self.push
            finally:
                print()

    def markdown(self, text):
        if self.rich:
            from rich.markdown import Markdown
            self.console.print(Markdown(text))
        else:
            self.print(text)
