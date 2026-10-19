"""Model-facing wording shared by every model."""

ACT = ("You are playing Pokémon Mystery Dungeon: Red Rescue Team. "
       "Which action should be taken next to make progress on the task?")
COMPLETE = ("You are playing Pokémon Mystery Dungeon: Red Rescue Team. "
            "Judging only from the visible screenshots, has the task clearly been completed?")
ACTION_DESCRIPTIONS = {
    "UP": "move one step up in the world, or move the menu cursor up",
    "DOWN": "move one step down in the world, or move the menu cursor down",
    "LEFT": "move one step left in the world, or move the menu cursor left",
    "RIGHT": "move one step right in the world, or move the menu cursor right",
    "A": ("press the A button to interact with or talk to what the character faces; "
          "confirm a menu choice; advance visible text or dialogue; attack in a dungeon"),
    "B": "press the B button to open the menu during play, cancel a command, or go back",
    "SELECT": "press SELECT to show the dungeon map",
    "WAIT": "press nothing and let the game advance",
}
