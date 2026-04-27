/*
 * phase3_core.c — C helpers for Phase 3 trellis search
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define INF_NEG (-1e18)
#define INF_I32 (2147483647)

static int32_t branch_weight(const int8_t *output_bits, int32_t state, int32_t bit) {
    int32_t idx = (state * 2 + bit) * 2;
    return (int32_t)output_bits[idx] + (int32_t)output_bits[idx + 1];
}

static void pair_to_bits(int32_t pair, int8_t *dst) {
    static const int8_t BITS_FROM_PAIR[4][2] = {
        {1, 1}, {1, 0}, {0, 1}, {0, 0}
    };
    dst[0] = BITS_FROM_PAIR[pair][0];
    dst[1] = BITS_FROM_PAIR[pair][1];
}

static uint64_t rng_next_u64(uint64_t *state) {
    *state = (*state * 6364136223846793005ULL) + 1442695040888963407ULL;
    return *state;
}

static int32_t rng_bounded(uint64_t *state, int32_t bound) {
    return (int32_t)(rng_next_u64(state) % (uint64_t)bound);
}

int32_t encode_c(
    const int32_t *next_state,
    const int8_t *output_bits,
    const int8_t *info_bits,
    int32_t S,
    int32_t K_INFO,
    int32_t memory,
    int8_t *coded_out
) {
    (void)S;
    int32_t state = 0;
    int32_t pos = 0;
    for (int32_t t = 0; t < K_INFO; t++) {
        int32_t bit = (int32_t)info_bits[t];
        int32_t idx = (state * 2 + bit) * 2;
        coded_out[pos++] = output_bits[idx];
        coded_out[pos++] = output_bits[idx + 1];
        state = next_state[state * 2 + bit];
    }
    for (int32_t t = 0; t < memory; t++) {
        int32_t idx = (state * 2) * 2;
        coded_out[pos++] = output_bits[idx];
        coded_out[pos++] = output_bits[idx + 1];
        state = next_state[state * 2];
    }
    return 0;
}

int32_t viterbi_neural_bm_c(
    const double *branch_metrics,
    const int32_t *next_state,
    const int32_t *index_table,
    int32_t S,
    int32_t T,
    int32_t K_INFO,
    int8_t *decoded
) {
    double *pm_cur = (double *)malloc((size_t)S * sizeof(double));
    double *pm_next = (double *)malloc((size_t)S * sizeof(double));
    int32_t *backtrack = (int32_t *)malloc((size_t)T * (size_t)S * sizeof(int32_t));
    int8_t *input_bits = (int8_t *)malloc((size_t)T * (size_t)S * sizeof(int8_t));
    if (pm_cur == NULL || pm_next == NULL || backtrack == NULL || input_bits == NULL) {
        free(pm_cur);
        free(pm_next);
        free(backtrack);
        free(input_bits);
        return -1;
    }

    for (int32_t s = 0; s < S; s++) {
        pm_cur[s] = INF_NEG;
    }
    pm_cur[0] = 0.0;

    for (int32_t t = 0; t < T; t++) {
        for (int32_t s = 0; s < S; s++) {
            pm_next[s] = INF_NEG;
            backtrack[t * S + s] = 0;
            input_bits[t * S + s] = 0;
        }
        for (int32_t s = 0; s < S; s++) {
            double src_metric = pm_cur[s];
            if (src_metric <= INF_NEG / 2.0) {
                continue;
            }
            for (int32_t bit = 0; bit < 2; bit++) {
                int32_t ns = next_state[s * 2 + bit];
                int32_t bm_idx = index_table[s * 2 + bit];
                double cand = src_metric + branch_metrics[t * 4 + bm_idx];
                if (cand > pm_next[ns]) {
                    pm_next[ns] = cand;
                    backtrack[t * S + ns] = s;
                    input_bits[t * S + ns] = (int8_t)bit;
                }
            }
        }
        {
            double *tmp = pm_cur;
            pm_cur = pm_next;
            pm_next = tmp;
        }
    }

    {
        int32_t state = 0;
        for (int32_t t = T - 1; t >= 0; t--) {
            if (t < K_INFO) {
                decoded[t] = input_bits[t * S + state];
            }
            state = backtrack[t * S + state];
        }
    }

    free(pm_cur);
    free(pm_next);
    free(backtrack);
    free(input_bits);
    return 0;
}

int32_t check_connectivity_c(const int32_t *next_state, int32_t S) {
    int8_t *seen = (int8_t *)calloc((size_t)S, sizeof(int8_t));
    int8_t *seen_rev = (int8_t *)calloc((size_t)S, sizeof(int8_t));
    if (seen == NULL || seen_rev == NULL) {
        free(seen);
        free(seen_rev);
        return 0;
    }

    seen[0] = 1;
    while (1) {
        int32_t changed = 0;
        for (int32_t s = 0; s < S; s++) {
            if (!seen[s]) {
                continue;
            }
            for (int32_t bit = 0; bit < 2; bit++) {
                int32_t ns = next_state[s * 2 + bit];
                if (!seen[ns]) {
                    seen[ns] = 1;
                    changed = 1;
                }
            }
        }
        if (!changed) {
            break;
        }
    }
    for (int32_t s = 0; s < S; s++) {
        if (!seen[s]) {
            free(seen);
            free(seen_rev);
            return 0;
        }
    }

    seen_rev[0] = 1;
    while (1) {
        int32_t changed = 0;
        for (int32_t src = 0; src < S; src++) {
            if (seen_rev[src]) {
                continue;
            }
            if (seen_rev[next_state[src * 2]] || seen_rev[next_state[src * 2 + 1]]) {
                seen_rev[src] = 1;
                changed = 1;
            }
        }
        if (!changed) {
            break;
        }
    }
    for (int32_t s = 0; s < S; s++) {
        if (!seen_rev[s]) {
            free(seen);
            free(seen_rev);
            return 0;
        }
    }

    free(seen);
    free(seen_rev);
    return 1;
}

int32_t check_termination_c(const int32_t *next_state, int32_t S, int32_t max_tail) {
    for (int32_t s = 0; s < S; s++) {
        int32_t state = s;
        for (int32_t t = 0; t < max_tail; t++) {
            if (state == 0) {
                break;
            }
            state = next_state[state * 2];
        }
        if (state != 0) {
            return 0;
        }
    }
    return 1;
}

static int32_t zero_weight_successor_count(
    const int32_t *next_state,
    const int8_t *output_bits,
    int32_t S,
    int32_t node,
    int32_t succ[2]
) {
    int32_t sa = node / S;
    int32_t sb = node % S;
    int32_t n = 0;
    for (int32_t bit = 0; bit < 2; bit++) {
        int32_t idx_a = (sa * 2 + bit) * 2;
        int32_t idx_b = (sb * 2 + bit) * 2;
        if (output_bits[idx_a] == output_bits[idx_b] && output_bits[idx_a + 1] == output_bits[idx_b + 1]) {
            succ[n++] = next_state[sa * 2 + bit] * S + next_state[sb * 2 + bit];
        }
    }
    return n;
}

int32_t check_noncatastrophic_c(const int32_t *next_state, const int8_t *output_bits, int32_t S) {
    int32_t N = S * S;
    int32_t *seen = (int32_t *)calloc((size_t)N, sizeof(int32_t));
    int32_t *stack = (int32_t *)malloc((size_t)N * sizeof(int32_t));
    if (seen == NULL || stack == NULL) {
        free(seen);
        free(stack);
        return 0;
    }

    int32_t stamp = 1;
    for (int32_t start = 0; start < N; start++) {
        if ((start / S) == (start % S)) {
            continue;
        }
        int32_t top = 0;
        stack[top++] = start;
        seen[start] = stamp;
        while (top > 0) {
            int32_t node = stack[--top];
            int32_t succ[2];
            int32_t n_succ = zero_weight_successor_count(next_state, output_bits, S, node, succ);
            for (int32_t i = 0; i < n_succ; i++) {
                if (succ[i] == start) {
                    free(seen);
                    free(stack);
                    return 0;
                }
                if (seen[succ[i]] != stamp) {
                    seen[succ[i]] = stamp;
                    stack[top++] = succ[i];
                }
            }
        }
        stamp++;
    }

    free(seen);
    free(stack);
    return 1;
}

int32_t compute_dfree_c(
    const int32_t *next_state,
    const int8_t *output_bits,
    int32_t S,
    int32_t *dfree_out
) {
    int32_t END = S;
    int32_t V = S + 1;
    int32_t *dist = (int32_t *)malloc((size_t)V * sizeof(int32_t));
    int8_t *visited = (int8_t *)calloc((size_t)V, sizeof(int8_t));
    if (dist == NULL || visited == NULL) {
        free(dist);
        free(visited);
        return -1;
    }

    if (!check_noncatastrophic_c(next_state, output_bits, S)) {
        free(dist);
        free(visited);
        return -1;
    }

    for (int32_t i = 0; i < V; i++) {
        dist[i] = INF_I32;
    }

    for (int32_t bit = 0; bit < 2; bit++) {
        int32_t w = branch_weight(output_bits, 0, bit);
        int32_t ns = next_state[bit];
        if (ns == 0) {
            if (bit == 1 && w < dist[END]) {
                dist[END] = w;
            }
        } else if (w < dist[ns]) {
            dist[ns] = w;
        }
    }

    while (1) {
        int32_t best_node = -1;
        int32_t best_dist = INF_I32;
        for (int32_t i = 0; i < V; i++) {
            if (!visited[i] && dist[i] < best_dist) {
                best_dist = dist[i];
                best_node = i;
            }
        }
        if (best_node < 0 || best_dist == INF_I32) {
            free(dist);
            free(visited);
            return -1;
        }
        if (best_node == END) {
            *dfree_out = best_dist;
            free(dist);
            free(visited);
            return 0;
        }
        visited[best_node] = 1;
        for (int32_t bit = 0; bit < 2; bit++) {
            int32_t ns = next_state[best_node * 2 + bit];
            int32_t nd = best_dist + branch_weight(output_bits, best_node, bit);
            if (ns == 0) {
                if (nd < dist[END]) {
                    dist[END] = nd;
                }
            } else if (nd < dist[ns]) {
                dist[ns] = nd;
            }
        }
    }
}

int32_t mutate_and_validate_c(
    const int32_t *in_next_state,
    const int32_t *in_output_pair,
    int32_t S,
    int32_t n_edges,
    int32_t max_attempts,
    int32_t dfree_target,
    uint64_t seed,
    int32_t *out_next_state,
    int32_t *out_output_pair,
    int8_t *out_output_bits,
    int32_t *out_dfree,
    int32_t *out_attempts
) {
    uint64_t rng_state = seed ? seed : 1ULL;
    size_t edge_count = (size_t)S * 2U;

    for (int32_t attempt = 1; attempt <= max_attempts; attempt++) {
        memcpy(out_next_state, in_next_state, edge_count * sizeof(int32_t));
        memcpy(out_output_pair, in_output_pair, edge_count * sizeof(int32_t));

        for (int32_t i = 0; i < n_edges; i++) {
            int32_t s = rng_bounded(&rng_state, S);
            int32_t bit = rng_bounded(&rng_state, 2);
            int32_t idx = s * 2 + bit;
            if (rng_bounded(&rng_state, 2) == 0) {
                out_next_state[idx] = rng_bounded(&rng_state, S);
            } else {
                out_output_pair[idx] = rng_bounded(&rng_state, 4);
            }
        }

        for (size_t i = 0; i < edge_count; i++) {
            pair_to_bits(out_output_pair[i], &out_output_bits[i * 2U]);
        }

        if (!check_connectivity_c(out_next_state, S)) {
            continue;
        }
        if (!check_termination_c(out_next_state, S, 6)) {
            continue;
        }
        if (!check_noncatastrophic_c(out_next_state, out_output_bits, S)) {
            continue;
        }
        if (compute_dfree_c(out_next_state, out_output_bits, S, out_dfree) != 0) {
            continue;
        }
        if (*out_dfree < dfree_target) {
            continue;
        }
        *out_attempts = attempt;
        return 0;
    }

    return -1;
}
