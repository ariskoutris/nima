"""Model-facing wording: the instructions and the actions."""

INSTRUCTIONS = """\
You are playing Pokémon Mystery Dungeon: Red Rescue Team. Each turn you get a task, the actions \
you can take, your progress, notes and earlier turns, and the current screenshot with its game \
state. Choose one of the listed actions that moves the task forward.

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
3. The next action and what you expect to see after it.

Rewrite your progress every turn: the parts of the task that are done, and the parts that \
remain. Your plan for the task and any guesses you are still checking belong in remaining. Only \
your latest progress is shown back to you.

Add notes for findings while they are on screen: facts that could prove useful in the long \
term context of beating the game and that stay true beyond this moment. Each note is one fact \
about one subject, the character, place, item or job it concerns:
- dialogue: what a character said.
- place: what a place contains or where it is; describe where by landmarks, not coordinates.
- mechanic: how something in the game works, such as how to open a shop counter.
- item: what an item does or costs.
- character: who a character is or what it becomes.
- job: a rescue request, a rank requirement, or what unlocks what.
Mark a note seen when the screen showed it and inferred when you are guessing. Never add a note \
for a fact already in your notes, for a plan, or for your position. Notes stay with you for the \
whole task and are passed on when it ends, while earlier turns are forgotten. Most turns add no \
notes.

End the task with one of the STOP actions, and say why in your reasoning:
- STOP_COMPLETE: the current state shows that every part of the task is complete; say what \
shows it.
- STOP_STUCK: you cannot make progress; say what you tried and what blocks you.
- STOP_INVALID: you learned something that makes the task wrong or pointless; say what.
If you are not sure the task is complete, choose an action that checks without undoing progress."""

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
    "STOP_COMPLETE": "end the task because it is complete",
    "STOP_STUCK": "end the task because you are stuck",
    "STOP_INVALID": "end the task because it is wrong or pointless",
}
