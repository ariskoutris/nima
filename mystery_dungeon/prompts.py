"""Model-facing wording: the turn and report instructions and the actions."""

CONTEXT_LAYOUT = """\
Earlier turns are listed oldest first; turn -1 is the most recent. Each shows the screenshot and \
game state you saw then, your reasoning and the action you took; the next turn shows the result. \
An operator turn is an action someone else took. The game state is read from the game: your \
character's position as (x, y), with x growing rightward and y downward, the way it faces, and \
where the other characters on screen are relative to you. In towns one step is 16."""

INSTRUCTIONS = f"""\
You are playing Pokémon Mystery Dungeon: Red Rescue Team. Each turn you get a task, the actions \
you can take, your notes and earlier turns, and the current screenshot with its game state. \
Choose one of the listed actions that moves the task forward.

{CONTEXT_LAYOUT}

In your reasoning, state briefly:
1. What the current screenshot and game state show: where your character is and which way it \
faces, what is around it, and any dialogue box or menu with its text and cursor.
2. What your last action actually did, judged by what changed since the previous turn, including \
its game state. If nothing changed, say so and why.
3. Your plan for the task, the next action, and what you expect to see next.

Use the note to record a finding while it is on screen. This could include anything that could \
prove to be useful in the long term context of beating the game. For example, what an NPC said, \
what a shop sells or what a place contains. Notes stay with you for the whole task and are \
passed on when it ends, while earlier turns are forgotten. Leave the note empty on most turns.

Choose STOP to end the task in one of three cases, and say in the note which case applies and why:
- The current state shows that every part of the task is complete; say what shows it.
- You are stuck: say what you tried and what blocks you.
- You learned something that makes the task wrong or pointless; say what.
If you are not sure the task is complete, choose an action that checks without undoing progress."""

REPORT_INSTRUCTIONS = f"""\
You are playing Pokémon Mystery Dungeon: Red Rescue Team, and the action budget for your task \
has run out. You get the task, your notes and earlier turns, and the current screenshot with its \
game state; the actions are listed for reference, but you can take no more.

{CONTEXT_LAYOUT}

Say in a few sentences where you stand:
- If you were close to completing the task, say what remains to be done.
- If you were stuck, say what you tried and what blocked you.
- If the task looks wrong, pointless or impossible, say why."""

ACTION_DESCRIPTIONS = {  # one per action in ACTION_NAMES
    "UP": "move one tile up, or move the menu cursor up",
    "DOWN": "move one tile down, or move the menu cursor down",
    "LEFT": "move one tile left, or move the menu cursor left",
    "RIGHT": "move one tile right, or move the menu cursor right",
    "A": ("talk to or interact with what the character faces, confirm a menu choice, "
          "advance dialogue, or attack in a dungeon"),
    "B": "open the menu, cancel, or go back",
    "SELECT": "show the dungeon map",
    "WAIT": "press nothing and let the game run",
    "STOP": "end the task, giving the reason in the note",
}
