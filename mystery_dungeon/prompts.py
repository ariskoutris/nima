"""Model-facing wording: the turn instructions and the actions."""

INSTRUCTIONS = """\
You are playing Pokémon Mystery Dungeon: Red Rescue Team. Each turn you get a task, the actions \
you can take, your earlier turns and the current screenshot. Choose one of the listed actions \
that moves the task forward.

Earlier turns are listed oldest first; turn -1 is the most recent. Each shows the screenshot you \
saw then, your reasoning and the action you took. The screenshot of the following turn, or the \
current screenshot after turn -1, shows the result of that action. An operator turn is an action \
someone else took.

In your reasoning, state briefly:
1. What the current screenshot shows: where your character is and which way it faces, the \
characters and objects around it, and any dialogue box or menu with its text and cursor.
2. What changed since the previous screenshot, and whether your last action did what you \
expected. If not, say why.
3. Your plan for the task, the next action, and what you expect the next screenshot to show.

Choose DONE only when the current screenshot shows that every part of the task is complete; \
DONE ends the task. If you are not sure, choose an action that checks without undoing progress."""

ACTION_DESCRIPTIONS = {  # one per action in ACTIONS
    "UP": "move one tile up, or move the menu cursor up",
    "DOWN": "move one tile down, or move the menu cursor down",
    "LEFT": "move one tile left, or move the menu cursor left",
    "RIGHT": "move one tile right, or move the menu cursor right",
    "A": ("talk to or interact with what the character faces, confirm a menu choice, "
          "advance dialogue, or attack in a dungeon"),
    "B": "open the menu, cancel, or go back",
    "SELECT": "show the dungeon map",
    "WAIT": "press nothing and let the game run",
    "DONE": "report that the task is complete; this ends the task",
}
