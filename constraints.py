"""
constraints.py — Trellis constraint checkers (Python reference implementations)
"""

from __future__ import annotations

import heapq
from collections import deque

import numpy as np

from trellis import Trellis, load_nasa_k7


def hamming_weight(bits: np.ndarray) -> int:
    """Return the Hamming weight of a binary array."""
    return int(np.asarray(bits, dtype=np.int64).sum())


def is_fully_connected(trellis: Trellis) -> bool:
    """Check forward and reverse reachability of all states from state 0."""
    S = trellis.n_states
    next_state = np.asarray(trellis.next_state, dtype=np.int32)

    seen_fwd = np.zeros(S, dtype=bool)
    q = deque([0])
    while q:
        s = q.popleft()
        if seen_fwd[s]:
            continue
        seen_fwd[s] = True
        q.append(int(next_state[s, 0]))
        q.append(int(next_state[s, 1]))
    if not bool(seen_fwd.all()):
        return False

    rev_adj: list[list[int]] = [[] for _ in range(S)]
    for s in range(S):
        rev_adj[int(next_state[s, 0])].append(s)
        rev_adj[int(next_state[s, 1])].append(s)

    seen_rev = np.zeros(S, dtype=bool)
    q = deque([0])
    while q:
        s = q.popleft()
        if seen_rev[s]:
            continue
        seen_rev[s] = True
        q.extend(rev_adj[s])
    return bool(seen_rev.all())


def is_terminating(trellis: Trellis, max_tail: int = 6) -> bool:
    """Check that every state reaches state 0 within max_tail zero-input steps."""
    S = trellis.n_states
    next_state = np.asarray(trellis.next_state, dtype=np.int32)
    for s in range(S):
        state = s
        for _ in range(max_tail):
            if state == 0:
                break
            state = int(next_state[state, 0])
        if state != 0:
            return False
    return True


def _zero_weight_successors(trellis: Trellis, node: int) -> tuple[int, ...]:
    S = trellis.n_states
    s_a = node // S
    s_b = node % S
    out = trellis.output_bits
    nxt = trellis.next_state
    succ: list[int] = []
    for bit in (0, 1):
        if np.array_equal(out[s_a, bit], out[s_b, bit]):
            succ.append(int(nxt[s_a, bit]) * S + int(nxt[s_b, bit]))
    return tuple(succ)


def is_non_catastrophic(trellis: Trellis) -> bool:
    """
    Check the Lin-Costello state-pair difference graph criterion.

    A trellis is catastrophic iff some off-diagonal state-pair node lies on a
    zero-weight cycle under equal-input transitions.
    """
    S = trellis.n_states
    N = S * S
    zero_adj = [_zero_weight_successors(trellis, node) for node in range(N)]

    color = np.zeros(N, dtype=np.int8)
    in_cycle = np.zeros(N, dtype=bool)
    stack: list[int] = []
    stack_pos: dict[int, int] = {}

    def dfs(node: int) -> None:
        color[node] = 1
        stack_pos[node] = len(stack)
        stack.append(node)
        for nxt in zero_adj[node]:
            if color[nxt] == 0:
                dfs(nxt)
            elif color[nxt] == 1:
                start = stack_pos[nxt]
                for cyc_node in stack[start:]:
                    in_cycle[cyc_node] = True
        stack.pop()
        stack_pos.pop(node, None)
        color[node] = 2

    for node in range(N):
        if color[node] == 0:
            dfs(node)

    for node, on_cycle in enumerate(in_cycle):
        if on_cycle and (node // S) != (node % S):
            return False
    return True


def compute_dfree(trellis: Trellis) -> float:
    """
    Return the minimum Hamming weight non-trivial codeword.

    Catastrophic trellises are rejected here and reported as infinite free
    distance so the EA never treats them as acceptable candidates.
    """
    if not is_non_catastrophic(trellis):
        return float("inf")

    S = trellis.n_states
    END = S
    next_state = np.asarray(trellis.next_state, dtype=np.int32)
    output_bits = np.asarray(trellis.output_bits, dtype=np.int8)

    pq: list[tuple[int, int]] = []
    best = np.full(S + 1, np.iinfo(np.int32).max, dtype=np.int64)

    for bit in (0, 1):
        weight = hamming_weight(output_bits[0, bit])
        ns = int(next_state[0, bit])
        if ns == 0:
            if bit == 1:
                best[END] = min(best[END], weight)
                heapq.heappush(pq, (weight, END))
            continue
        if weight < best[ns]:
            best[ns] = weight
            heapq.heappush(pq, (weight, ns))

    while pq:
        dist, state = heapq.heappop(pq)
        if dist != int(best[state]):
            continue
        if state == END:
            return float(dist)
        for bit in (0, 1):
            ns = int(next_state[state, bit])
            new_dist = dist + hamming_weight(output_bits[state, bit])
            if ns == 0:
                if new_dist < best[END]:
                    best[END] = new_dist
                    heapq.heappush(pq, (new_dist, END))
            elif new_dist < best[ns]:
                best[ns] = new_dist
                heapq.heappush(pq, (new_dist, ns))

    return float("inf")


if __name__ == "__main__":
    nasa = load_nasa_k7()
    print(f"hamming_weight([1,0,1]) = {hamming_weight(np.array([1, 0, 1], dtype=np.int8))}")
    print(f"is_fully_connected(NASA K=7) = {is_fully_connected(nasa)}")
    print(f"is_terminating(NASA K=7) = {is_terminating(nasa)}")
    print(f"is_non_catastrophic(NASA K=7) = {is_non_catastrophic(nasa)}")
    print(f"compute_dfree(NASA K=7) = {compute_dfree(nasa)}")
