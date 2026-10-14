# FlyCursor
A real fruit fly's brain controls your mouse and keyboard, built in Python.

The scripts require the pip packages: `flybrain`, `pyautogui` and `pillow`. It also requires the fly's brain data which you can grab [here](https://github.com/alextitonis/fly.ai/releases/) and then manually move `fly-data` folder next to `scripts`.

## For the scripts themselves:
Claude was nice enough to summarise both for me so here's the first one:

### fly_cursor.py
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

KEYBOARD. When at least two DNg11 “punch” neurons fire during the same 20 ms
simulation step, the fly presses a key through `pyautogui`. By default, the
exact descending neurons that fired determine which character is selected from
the letters, numbers, and spaces in the keyboard map. With `--text`, the fly
types the supplied text one character at a time instead. Use `--no-type` to
disable keyboard input, or `--dry-run` to print the keys without pressing them.

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
    MBONs: the fly comes to dislike the place.
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

### fly_vision.py
<details>
<summary>Show description of script</summary>

`fly_vision.py` is what the fly sees on the screen, and a picture of its brain at work.

Vision: a screenshot is the fly's world. From the cursor's position and heading
a 1-D panorama is read off the screen (luminance along rays fanning out over
the fly's field of view), then projected onto the 6,006 photoreceptors by their
azimuth, the same recipe as flybrain.eyes.Eyes uses for its synthetic blobs.
Brightness plus brightness *change* (motion) is what the photoreceptors get.

In this spiking model the photoreceptor signal fades in the lamina (the package
author says so too), so the panorama is ALSO run through a looming detector: a
dark patch that grows wider on one side drives that side's LPLC2 looming
neurons, the route that does reach the descending neurons.

Snapshots: every N seconds a PNG is written with the screenshot (fly, heading,
field of view, food), the panorama strip it saw, and a map of all 166,700
neurons with the ones firing above the noise floor lit up by region. Rendering
takes a few seconds, so it runs in a background thread.

</details>

## How to run
Once you've installed all of the dependences you need, you can just run `fly_cursor.py` and it will start working, it may take a few minutes for the fly brain to initiate. 
Everything else will be made for you such as the `.json` file for it's memory and the screenshots in a different folder.

## Credits and acknowledgements

- **FlyCursor:** Created and maintained by Will K. See the [MIT License](LICENSE) for the project copyright and terms.
- **Fly brain model and connectome data:** Built on [`flybrain`](https://github.com/alextitonis/fly.ai), by **Alex Titonis**. The brain data used by this project is distributed through the [`fly.ai` releases](https://github.com/alextitonis/fly.ai/releases/). Please review that project's license and attribution requirements when redistributing the data.
- **Biological connectome:** This project uses the **MaleCNS** fruit-fly connectome represented by `flybrain`; credit and citation should follow the `flybrain`/`fly.ai` project and the underlying connectome data source.
- **Python dependencies:** This project uses [NumPy](https://numpy.org/), [PyAutoGUI](https://github.com/asweigart/pyautogui), [PyGetWindow](https://github.com/asweigart/PyGetWindow), [Pillow](https://python-pillow.org/), and [Matplotlib](https://matplotlib.org/). These projects retain their own copyrights and licenses; consult their respective repositories or distributions for the applicable terms.
- **Documentation assistance:** Parts of the script descriptions in this README were summarized with assistance from **Claude**.

If you redistribute FlyCursor, keep the copyright and license notices in [LICENSE](LICENSE), preserve the upstream acknowledgements above, and verify the current license terms of each dependency and the `fly.ai` brain-data release you distribute.
