"""
trellis.py — Trellis FSM: encode, validate, load standard codes

Part of: EE597 Search-Designed Trellis Codes Project
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Constants (project-wide fixed values)
# ─────────────────────────────────────────────
N_CODED = 512       # coded block length
K_INFO  = 256       # information bits per block
N_STATES = 64       # trellis states
RATE = 0.5          # code rate
CONSTRAINT_LEN = 7  # K=7 for NASA code


# ─────────────────────────────────────────────
# Trellis dataclass
# ─────────────────────────────────────────────
@dataclass
class Trellis:
    n_states: int                   # S = 64
    next_state: np.ndarray          # shape (S, 2): next_state[s, input_bit]
    output_bits: np.ndarray         # shape (S, 2, 2): output_bits[s, input_bit, 0:2]
    name: str = "unnamed"

    def encode(self, info_bits: np.ndarray, init_state: int = 0) -> np.ndarray:
        """
        Encode info bits through the trellis with termination.

        Args:
            info_bits: shape (K,) binary array
            init_state: initial state (default 0)

        Returns:
            coded_bits: binary array, length = 2*(K + tail_bits)
        """
        state = init_state
        coded = []

        # Encode info bits
        for bit in info_bits:
            coded.extend(self.output_bits[state, bit])
            state = self.next_state[state, bit]

        # Termination: force back to state 0
        max_tail = self.n_states  # safety limit
        tail_count = 0
        while state != 0 and tail_count < max_tail:
            # For shift-register codes, input bit 0 shifts in a zero,
            # which always drives toward state 0
            term_bit = 0
            coded.extend(self.output_bits[state, term_bit])
            state = self.next_state[state, term_bit]
            tail_count += 1

        if state != 0:
            raise RuntimeError(f"Termination failed after {max_tail} tail bits")

        return np.array(coded, dtype=np.int8)

    def validate(self) -> dict[str, bool]:
        """
        Validate trellis properties.

        Returns:
            dict with keys: non_catastrophic, fully_connected, terminated
        """
        S = self.n_states
        results = {}

        # Check termination: from any state, can we reach state 0 using input 0?
        terminated = True
        for s in range(S):
            state = s
            for _ in range(S):
                state = self.next_state[state, 0]
                if state == 0:
                    break
            if state != 0:
                terminated = False
                break
        results['terminated'] = terminated

        # Check fully connected: all states reachable from state 0
        reachable = set()
        frontier = {0}
        while frontier:
            s = frontier.pop()
            if s in reachable:
                continue
            reachable.add(s)
            for bit in [0, 1]:
                ns = self.next_state[s, bit]
                if ns not in reachable:
                    frontier.add(ns)
        results['fully_connected'] = len(reachable) == S

        # Check non-catastrophic: different input sequences produce different
        # output sequences (simplified check via weight-1 input test)
        # A catastrophic code maps some weight-1 input to weight-0 output
        non_catastrophic = True
        for s in range(S):
            for bit in [0, 1]:
                out = self.output_bits[s, bit]
                ns = self.next_state[s, bit]
                # Check if there's a cycle with all-zero output
                if np.sum(out) == 0 and ns != 0 and bit == 1:
                    # Follow the cycle
                    state = ns
                    all_zero = True
                    for _ in range(S):
                        if np.sum(self.output_bits[state, 1]) != 0:
                            all_zero = False
                            break
                        state = self.next_state[state, 1]
                        if state == ns:
                            break
                    if all_zero and state == ns:
                        non_catastrophic = False
        results['non_catastrophic'] = non_catastrophic

        return results


# ─────────────────────────────────────────────
# NASA K=7 code loader
# ─────────────────────────────────────────────
def load_nasa_k7() -> Trellis:
    """
    Build 64-state trellis from NASA K=7 rate 1/2 convolutional code.
    Generator polynomials: G = [171_oct, 133_oct] = [0b1111001, 0b1011011]
    """
    g1 = 0b1111001  # 171 octal
    g2 = 0b1011011  # 133 octal
    K = CONSTRAINT_LEN
    S = 2 ** (K - 1)  # 64 states

    next_state = np.zeros((S, 2), dtype=np.int32)
    output_bits = np.zeros((S, 2, 2), dtype=np.int8)

    for state in range(S):
        for input_bit in [0, 1]:
            # Shift register: new bit enters from the left (MSB)
            # State represents the K-1 memory elements
            # Register = [input_bit, state_bits]
            register = (input_bit << (K - 1)) | state

            # Compute outputs by convolving with generator polynomials
            out1 = bin(register & g1).count('1') % 2
            out2 = bin(register & g2).count('1') % 2

            # Next state: shift right (drop LSB, new bit becomes MSB of state)
            ns = register >> 1

            next_state[state, input_bit] = ns
            output_bits[state, input_bit, 0] = out1
            output_bits[state, input_bit, 1] = out2

    return Trellis(
        n_states=S,
        next_state=next_state,
        output_bits=output_bits,
        name="NASA_K7_171_133",
    )


def create_random_trellis(
    n_states: int = N_STATES,
    rng: Optional[np.random.Generator] = None,
) -> Trellis:
    """Create a random trellis for baseline B3."""
    if rng is None:
        rng = np.random.default_rng()

    next_state = rng.integers(0, n_states, size=(n_states, 2), dtype=np.int32)
    output_bits = rng.integers(0, 2, size=(n_states, 2, 2), dtype=np.int8)

    # Ensure state 0 goes to state 0 on input 0 (for termination)
    next_state[0, 0] = 0

    return Trellis(
        n_states=n_states,
        next_state=next_state,
        output_bits=output_bits,
        name="random_trellis",
    )


# ─────────────────────────────────────────────
# Smoke test (run with: python trellis.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Running smoke test for trellis...")

    trellis = load_nasa_k7()
    assert trellis.n_states == 64, f"Expected 64 states, got {trellis.n_states}"
    assert trellis.next_state.shape == (64, 2), f"next_state shape: {trellis.next_state.shape}"
    assert trellis.output_bits.shape == (64, 2, 2), f"output_bits shape: {trellis.output_bits.shape}"

    # Validate properties
    props = trellis.validate()
    print(f"  Validation: {props}")
    assert props['terminated'], "NASA K=7 should terminate"
    assert props['fully_connected'], "NASA K=7 should be fully connected"
    assert props['non_catastrophic'], "NASA K=7 should be non-catastrophic"

    # Encode a test block
    rng = np.random.default_rng(42)
    info_bits = rng.integers(0, 2, K_INFO, dtype=np.int8)
    coded = trellis.encode(info_bits)
    print(f"  Encoded {K_INFO} info bits -> {len(coded)} coded bits")

    # For K=7, tail is K-1=6 bits, producing 12 coded bits
    expected_len = 2 * (K_INFO + CONSTRAINT_LEN - 1)
    assert len(coded) == expected_len, f"Expected {expected_len} coded bits, got {len(coded)}"

    # All coded bits should be 0 or 1
    assert np.all((coded == 0) | (coded == 1)), "Coded bits must be binary"

    # Encode all-zeros -> should get all-zero output (linear code property)
    zero_bits = np.zeros(K_INFO, dtype=np.int8)
    zero_coded = trellis.encode(zero_bits)
    assert np.all(zero_coded == 0), "All-zero input should produce all-zero output"

    print("PASS")
