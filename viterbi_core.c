/*
 * viterbi_core.c — C implementation of Viterbi forward pass + traceback
 *
 * Part of: EE597 Search-Designed Trellis Codes Project
 *
 * Build: cc -O3 -shared -fPIC -o viterbi_core.so viterbi_core.c
 */

#include <stdlib.h>
#include <string.h>

#define INF_NEG (-1e18)

/*
 * viterbi_decode_c — Viterbi decoding (mismatched or oracle mode)
 *
 * Parameters:
 *   received   — (T * 2) doubles, received BPSK symbols
 *   interference — (T * 2) doubles or NULL; if non-NULL, subtract before metric
 *   rev_src    — (S * max_inc) int32, source states for each destination
 *   rev_bit    — (S * max_inc) int8, input bits for each incoming branch
 *   rev_exp    — (S * max_inc * 2) doubles, expected BPSK symbols per branch
 *   n_incoming — (S) int32, number of incoming branches per state
 *   decoded    — (K_INFO) int8, output decoded bits (caller-allocated)
 *   S          — number of trellis states
 *   T          — number of trellis time steps (N_CODED / 2)
 *   max_inc    — max incoming branches per state
 *   K_INFO     — number of info bits to output
 *   noise_var  — noise variance
 */
void viterbi_decode_c(
    const double *received,
    const double *interference,   /* NULL for mismatched mode */
    const int    *rev_src,
    const signed char *rev_bit,
    const double *rev_exp,
    const int    *n_incoming,
    signed char  *decoded,
    int S,
    int T,
    int max_inc,
    int K_INFO,
    double noise_var
) {
    double inv_2nv = -0.5 / noise_var;

    /* Allocate path metrics (current and next) */
    double *pm_cur  = (double *)malloc(S * sizeof(double));
    double *pm_next = (double *)malloc(S * sizeof(double));

    /* Backtrack and input_bits tables */
    int         *backtrack  = (int *)malloc(T * S * sizeof(int));
    signed char *input_bits = (signed char *)malloc(T * S * sizeof(signed char));

    /* Initialize: state 0 has metric 0, all others -inf */
    for (int s = 0; s < S; s++) {
        pm_cur[s] = INF_NEG;
    }
    pm_cur[0] = 0.0;

    /* Forward pass (ACS loop) */
    for (int t = 0; t < T; t++) {
        double rx0, rx1;

        if (interference != NULL) {
            rx0 = received[t * 2]     - interference[t * 2];
            rx1 = received[t * 2 + 1] - interference[t * 2 + 1];
        } else {
            rx0 = received[t * 2];
            rx1 = received[t * 2 + 1];
        }

        for (int ns = 0; ns < S; ns++) {
            int n_inc = n_incoming[ns];
            double best_metric = INF_NEG;
            int    best_src = 0;
            signed char best_bit = 0;

            for (int j = 0; j < n_inc; j++) {
                int src = rev_src[ns * max_inc + j];
                /* Expected symbols for this branch */
                double exp0 = rev_exp[(ns * max_inc + j) * 2];
                double exp1 = rev_exp[(ns * max_inc + j) * 2 + 1];

                double d0 = rx0 - exp0;
                double d1 = rx1 - exp1;
                double bm = inv_2nv * (d0 * d0 + d1 * d1);

                double total = pm_cur[src] + bm;
                if (total > best_metric) {
                    best_metric = total;
                    best_src = src;
                    best_bit = rev_bit[ns * max_inc + j];
                }
            }

            pm_next[ns] = best_metric;
            backtrack[t * S + ns]  = best_src;
            input_bits[t * S + ns] = best_bit;
        }

        /* Swap current and next */
        double *tmp = pm_cur;
        pm_cur = pm_next;
        pm_next = tmp;
    }

    /* Traceback from state 0 */
    int state = 0;
    for (int t = T - 1; t >= 0; t--) {
        if (t < K_INFO) {
            decoded[t] = input_bits[t * S + state];
        }
        state = backtrack[t * S + state];
    }

    free(pm_cur);
    free(pm_next);
    free(backtrack);
    free(input_bits);
}
