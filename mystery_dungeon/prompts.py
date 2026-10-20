"""Model-facing wording: the turn instructions and the actions."""

INSTRUCTIONS = ("You are playing Pokémon Mystery Dungeon: Red Rescue Team. "
                "Choose the next action that moves the task forward, or DONE when the task is "
                "complete. Answer DONE only if every part of the task is done and the "
                "screenshots, earlier reasoning and earlier actions leave no doubt; DONE ends the "
                "task. If the task may be complete but you are not certain, choose an action that "
                "checks without undoing progress.\n\n"
                "Before choosing, reason through: "
                "1. What the current screenshot shows. "
                "2. Whether your last action did what you expected, and if not, why. "
                "3. Your plan, and what you expect the next screenshot to show. "
                "Continue with your existing action plan or adapt it.")
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
