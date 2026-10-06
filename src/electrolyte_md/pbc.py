"""Periodic-boundary-condition utilities."""
import numpy as np

def minimum_image(displacement, box_lengths):
    displacement=np.asarray(displacement,dtype=float)
    box=np.asarray(box_lengths,dtype=float)
    return displacement-box*np.round(displacement/box)

def unwrap_step(previous_unwrapped, previous_wrapped, current_wrapped, box_lengths):
    return np.asarray(previous_unwrapped)+minimum_image(np.asarray(current_wrapped)-np.asarray(previous_wrapped),box_lengths)
