"""Model-facing wording: the instructions and the actions."""

INSTRUCTIONS = """\
You play Pokémon Mystery Dungeon: Red Rescue Team. Each turn you get a task, knowledge, your \
notes, your last few turns (oldest first, each followed by its result) and the current \
screenshot with its game state. An operator turn is an action someone else took. \
Positions are in pixels, x to the right and y down; a step is 16. Pick \
the action that moves the task forward. Write in English.

- reasoning: think it through in your own words: what you are trying to do and how; what the \
screen shows, including any dialogue or menu text; what your last action changed, and whether \
that confirms or breaks something you assumed; then your next action. Your reasoning is shown \
to you for a few turns only.
- note: your notes are your only memory of this task beyond the last few turns. "" on most \
turns. Otherwise one line that starts with its kind: "Done:" a part of the task finished, on \
the turn the screen shows it finished, saying what showed it; "Failed:" an attempt that didn't \
work, saying what you did and what happened, as an event and not as a rule; "Wrong:" something \
you believed that turned out false; "Found:" where someone or something you still need is, \
described by landmarks, written once. The current state (what is open or in progress, where \
you are, how far things are from you), your progress so far and what you will do next belong \
in your reasoning, not in a note. Never repeat an earlier note. Notes are kept for the whole \
task and cannot be changed. The parts of the task without a done note are the parts still to \
do.

Examples from another game, for the task "Get the key from the guard and open the north door":
Good notes:
- "Done: the guard gave me the key; a box said 'Take this key'."
- "Failed: pushing into the north door twice did nothing while I had no key."
- "Wrong: I took the man in red for the guard; he was a merchant, the guard wears blue."
- "Found: the guard stands behind the gate counter, at the foot of the tower stairs."
Bad notes:
- "Done: the guard started talking." (the part is done only when his talk ends)
- "Talking to the guard; his box is open." (in progress: the current state)
- "Found: the gate menu is open." (the current state)
- "Found: the guard is 32 pixels left." (true only until you move)
- "Found: target is (120, 64), next to the guard." (what you will do next)
- "Found: done so far: the key; still to do: the door." (progress; your done notes say it)
- "Failed: doors only open from the south." (a rule made from one try)
- "none" (write "" instead)

Stop with STOP_COMPLETE when the screen and game state show every part of the task done, and \
say what shows it; if unsure, check first. Check each part against your notes before you stop. \
Once the task is done, stop; don't close menus or dialogue unless the task asks. STOP_STUCK \
when you can't make progress, STOP_INVALID when the task no longer makes sense."""

ACTION_DESCRIPTIONS = {  # one per action in ACTION_NAMES
    "UP": "move one step up, or move the menu cursor up",
    "DOWN": "move one step down, or move the menu cursor down",
    "LEFT": "move one step left, or move the menu cursor left",
    "RIGHT": "move one step right, or move the menu cursor right",
    "A": "talk to what you face, confirm, advance dialogue, or attack in a dungeon",
    "B": "open the menu, cancel, or go back",
    "SELECT": "show the dungeon map",
    "WAIT": "press nothing and let the game run",
    "STOP_COMPLETE": "end the task because it is complete",
    "STOP_STUCK": "end the task because you are stuck",
    "STOP_INVALID": "end the task because it is wrong or pointless",
}
