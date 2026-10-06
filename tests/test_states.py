from electrolyte_md.states import coordination_state


def test_coordination_states():
    assert coordination_state(3, 0) == "P"
    assert coordination_state(2, 1) == "PT"
    assert coordination_state(0, 2) == "T"
    assert coordination_state(0, 0) == "F"
