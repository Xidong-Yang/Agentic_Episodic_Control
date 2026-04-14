import numpy as np

MAX_MODEL_LEN = 14112
MAX_TOKENS = 8192
WORK_MEM_OBS_MAXLEN = 3
WORK_MEM_ACT_MAXLEN = 2

ENV_DESCRIPTIONS = {
    'BabyAI-GoToRedBallNoDists-v0': 'Go to the red ball. No distractors present.',
    'BabyAI-GoToLocal-v0': 'Go to an object, inside a single room with distractors.',
    'BabyAI-PickupLoc-v0': 'Pick up an object which may be described using its location. This is a single room environment.',
    'BabyAI-UnlockLocal-v0': 'Fetch a key and unlock a door (in the current room).',
    'BabyAI-PutNextLocal-v0': 'Put an object next to another object, inside a single room.',
    'BabyAI-FindObjS5-v0': 'Pick up an object (in a random room) Rooms have a size of 5 This level requires potentially exhaustive exploration.',
    'BabyAI-UnlockPickup-v0': 'Unlock a door, go to another room, then pick up a box in another room.',
    'BabyAI-KeyInBox-v0': 'Unlock a door. Key is in a box (in the current room).',
    'BabyAI-OpenTwoDoors-v0': 'Open door X, then open door Y The two doors are facing opposite directions.',
    'BabyAI-PickupDist-v0': 'Pick up an object The object to pick up is given by its type only, or by its color, or by its type and color. (in the current room, with distractors)',
    'BabyAI-MiniBossLevel-v0': 'Command can be any sentence drawn from the Baby Language grammar. Union of all competencies. This level is a superset of all other levels.',
    'BabyAI-ActionObjDoor-v0': '[pick up an object] or [go to an object or door] or [open a door] (in the current room)',
    'BabyAI-OpenDoorsOrderN4-v0': 'Open one or two doors in the order specified.',
    'BabyAI-UnlockToUnlock-v0': 'Unlock a door A that requires to unlock a door B before, then pick up the ball.',
    'BabyAI-OpenDoor-v0': 'Go to the door The door to open is given by its color or by its location. (always unlocked, in the current room)',
    'BabyAI-PutNextS7N4-v0': 'Task of the form: move the A next to the B and the C next to the D.',
    'BabyAI-GoToRedBlueBall-v0': 'Go to the red ball or to the blue ball. There is exactly one red or blue ball, and some distractors.',
    'BabyAI-MixedTrainLocal': (
        'Your task is sampled randomly from the following tasks: '
        'Go to <object>, a simple navigation task that requires reasoning abilities to choose the right plan given objects position; '
        'Pick up <object>, a reasoning task that combines navigation tasks; '
        'Put <object A> next to <object B>, which requires first reaching <object A>, picking it up, reaching <object B> and finally dropping <object A> next to <object B>; '
        'Pick up <object A> then go to <object B> and Go to <object B> after pick up <object A>, both serving to test reasoning abilities on temporal sequences; '
        'Unlock <door>, a task that includes inferring that a key is needed to unlock the door, finding the right key (i.e. the one colored as the door) and eventually using the toggle action with the key on the door.'
    ),
}


def argmax_with_random_tiebreak(values):
    """Return index with max value with random tie-breaking."""
    idxs = np.nonzero(values == np.max(values))[0]
    return np.random.choice(idxs)
