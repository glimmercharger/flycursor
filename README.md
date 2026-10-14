# flycursor
A real fruit fly's brain controls your mouse and keyboard, built in Python.

The scripts require the pip packages: `flybrain`, `pyautogui` and `pillow`. It also requires the fly's brain data which you can grab [here](https://github.com/alextitonis/fly.ai/releases/) or just run[...]

## For the scripts themselves:
Claude was nice enough to summerise both for me so here's the first one:

### flycursor.py
<details>
<summary>Show description of script</summary>

`fly_cursor.py` lets the fruit fly connectome drive your mouse, and remember.

The mouse cursor is the fly's body. It has a heading, like a walking fly.
Senses go in through the fly's own sensory neurons, motor commands come out of
its descending neurons, and the wiring in between is the MaleCNS connectome.

  IN   LC10a   (chase target)   <- a "food" spot on screen, on the side it is on
       LPLC2   (looming)        <- the window under the cursor growing
       LC4     (threat/escape)  <- the cursor getting near a screen edge
       tpGRN   (leg taste)      <- standing on food (flies taste with their feet)
       KC      (Kenyon cells)   <- WHICH window it is standing on, as a sparse code

  OUT  DNa02 L/R (steering)     -> turn heading left or right
       DNg100    (forward)      -> walk along heading
       MDN       (backward)     -> walk backwards
       DNp01     (giant fiber)  -> jump: a leap away
       DNg11     (punch)        -> whack a key: which key depends on which descending
                                   neurons fired that step (or the next letter of --text)
       feeding                  -> reaching food = a click (unless it dislikes the place)

MEMORY. The connectome itself never changes, so the fly is given the learning
rule real flies use in the mushroom body: when dopamine arrives while a set of
Kenyon cells is active, the synapses from those Kenyon cells onto the mushroom
body output neurons (MBONs) in that compartment weaken.

  * Each window the fly stands on activates its own sparse set of Kenyon cells
    (the fly's code for "this place"), chosen from the window's title.
  * Finding food fires the PAM reward dopamine neurons and weakens that place's
    synapses onto the avoidance MBONs: the fly comes to like the place.
  * Getting scared there (a loom, a screen edge, a giant-fiber jump) fires the
    PPL1 punishment dopamine neurons and weakens the synapses onto the approach
    MBONs: the fly comes to dislike it.
  * Liked places: it chases food harder and will feed (click) there.
    Disliked places: they look threatening, so it jumps and backs away, and it
    refuses to feed there.
  * The learned synapse strengths are saved to fly_memory.json next to this
    script and re-applied to the brain next time, so it remembers across runs.

EYES. Every 2 s a screenshot becomes the fly's world: a panorama is read off
the screen around the cursor and fed to its 6,006 photoreceptors. Every
--snapshot-every seconds a PNG goes into snapshots/ showing the screen, what
it saw, and which parts of the brain fired since the last snapshot.

Safety: `pyautogui`'s fail-safe is on, so slam the mouse into any screen corner
to abort. The run stops after --seconds. --no-click / --no-type switch those
off, and --dry-run moves nothing at all.

</details>
