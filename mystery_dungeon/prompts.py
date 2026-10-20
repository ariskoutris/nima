"""Model-facing wording: the turn instructions and the actions."""

INSTRUCTIONS = """\
You are playing Pokémon Mystery Dungeon: Red Rescue Team. Each turn you get a task, the actions \
you can take, your earlier turns, and the current screenshot with its game state. Choose one of \
the listed actions that moves the task forward.

Earlier turns are listed oldest first; turn -1 is the most recent. Each shows the screenshot and \
game state you saw then, your reasoning and the action you took; the next turn shows the result. \
An operator turn is an action someone else took. The game state is read from the game: your \
character's position as (x, y), with x growing rightward and y downward, the way it faces, and \
where the other characters on screen are relative to you. In towns one step is 16.

In your reasoning, state briefly:
1. What the current screenshot and game state show: where your character is and which way it \
faces, what is around it, and any dialogue box or menu with its text and cursor.
2. What your last action actually did, judged by what changed since the previous turn, including \
its game state. If nothing changed, say so and why.
3. Your plan for the task, the next action, and what you expect to see next.

Choose DONE only when the current state shows that every part of the task is complete; \
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
