# SKILL.md — Weekly PhD Progress Report Generator (LaTeX → PDF)

## Purpose

This skill generates a comprehensive, supervisor-ready LaTeX progress report for a PhD student working on **Search-Designed Trellis Codes with Neural Decoding** (EE597, Abhishek Manjunath). The report is submitted to a supervisor weekly, **created from scratch every week** (not incrementally patched), and derived entirely from the current state of the codebase and results.

**Hard requirements from the supervisor:**
- IEEE-style formal prose, single-column, readable layout (NOT a two-column conference paper)
- Table of contents followed by numbered sections
- Mathematical formulations and derivations wherever a concept is introduced
- Figures embedded from `results/` wherever data exists
- A clear "This Week's Progress" section distinguishing new work from prior work
- No code snippets anywhere in the document
- No references section

The skill reads the codebase, determines what is actually implemented (not just what is planned), and writes prose that accurately reflects the current state — not aspirational descriptions of future work.

---

## Step 1 — Read Project Context

**Before writing a single line of LaTeX**, read these files:

```bash
cat CLAUDE.md
cat Phase0_Specification.txt
```

Extract and record the following — you will need all of these to write accurately:

| Item | Where to find it | What to record |
|------|-----------------|----------------|
| Current phase | CLAUDE.md phase headers | Which phase is active (0/1/2a/2b/2c/3/4) |
| Completed tasks | `[x]` checkboxes in CLAUDE.md | Exact task descriptions |
| Pending tasks | `[ ]` checkboxes in CLAUDE.md | Tasks not yet done |
| System parameters | CLAUDE.md table + Spec §1 | K, N, R, S, SNR range, INR range, op budget |
| Channel model | Spec §2 | The full formula for r[t], i[t], A |
| Decoder architecture | CLAUDE.md Phase 2 notes | GRU hidden size, op count derivation |
| Search algorithm | CLAUDE.md Phase 3 | EA parameters if decided |
| Success criteria | CLAUDE.md + Spec §6 | Numeric thresholds per phase |

---

## Step 2 — Inventory What Is Actually Implemented

This step is critical. **A file existing is not the same as it being implemented.** Read the actual code.

### 2a. Check which files exist
```bash
ls -la *.py 2>/dev/null
find . -name "*.py" | sort | xargs wc -l 2>/dev/null
```

### 2b. For each existing file, read its content to judge implementation depth
```bash
cat channel.py
cat trellis.py
cat decoders.py
cat neural_decoder.py
cat baselines.py
cat compute_cost.py
cat eval.py
cat search.py           2>/dev/null
cat fitness.py          2>/dev/null
cat neural_bm.py        2>/dev/null
cat interference_est.py 2>/dev/null
```

For each file, classify it as one of:
- **Fully implemented** — has real logic, not just `pass` or `raise NotImplementedError`
- **Partially implemented** — some functions done, some stubbed
- **Stub only** — file exists but contains only function signatures or `TODO` comments
- **Absent** — file does not exist yet

### 2c. Check test results
```bash
python -m pytest tests/ -v 2>/dev/null || python tests/test_encode_decode.py 2>/dev/null
python tests/test_awgn_theory.py 2>/dev/null
python tests/test_oracle_vs_mismatch.py 2>/dev/null
```

Record: which tests pass, which fail, which don't exist yet.

### 2d. Build an implementation summary table before writing

| Module | Status | Key functions present | Notes |
|--------|--------|----------------------|-------|
| channel.py | ? | ? | ? |
| trellis.py | ? | ? | ? |
| decoders.py | ? | ? | ? |
| neural_decoder.py | ? | ? | ? |
| baselines.py | ? | ? | ? |
| compute_cost.py | ? | ? | ? |
| eval.py | ? | ? | ? |

---

## Step 3 — Extract Numerical Results

```bash
ls results/ 2>/dev/null && find results/ -type f | sort
```

For each `.npz` file found:
```python
import numpy as np
data = np.load("results/FILENAME.npz")
print(data.files)
for k in data.files:
    print(k, data[k])
```

For each `.json` file found:
```python
import json
with open("results/FILENAME.json") as f:
    print(json.dumps(json.load(f), indent=2))
```

Record into a working table:

| Method | SNR (dB) | INR (dB) | BLER | Ops (M) | Latency (ms) |
|--------|----------|----------|------|---------|-------------|
| B1 | | | | | |
| B2 | | | | | |

If no results exist yet, note this — do not fabricate numbers.

---

## Step 4 — Identify and Verify Available Figures

```bash
find results/ -name "*.png" -o -name "*.pdf" 2>/dev/null | sort
# Check file sizes — zero-byte images will break LaTeX compilation
find results/ -name "*.png" | xargs ls -lh 2>/dev/null
```

**Image-to-section mapping:**

| Filename pattern | Section | Caption template |
|-----------------|---------|-----------------|
| `bler_vs_snr*.png` | Experimental Results — BLER vs. SNR | "BLER vs. SNR at INR = X dB. MC trials: $10^Y$." |
| `bler_vs_inr*.png` | Experimental Results — Robustness | "BLER vs. INR at SNR = X dB for all methods." |
| `training_loss*.png` | Neural Decoder — Training | "Training loss vs. epoch for N1 GRU decoder." |
| `compute_table*.png` | Compute Budget | "Measured operation counts per block." |
| `search_fitness*.png` | Trellis Search — Fitness | "Fitness vs. generation. Best and median shown." |
| `pareto*.png` | Trellis Search — Pareto | "Pareto frontier: BLER vs. compute." |
| `ablation*.png` | Ablation Study | "Ablation results at equal $2.1 \times 10^6$ op budget." |
| `trellis*.png` | System Model | "Trellis state transition diagram." |

**If an image file does not exist**, use this LaTeX placeholder — do NOT reference a missing file path:
```latex
\begin{figure}[h]
\centering
\framebox[0.82\textwidth]{%
  \parbox{0.80\textwidth}{\centering\vspace{1.8cm}
  \textit{[Figure pending: \texttt{eval.py} has not been run yet.]}
  \vspace{1.8cm}}}
\caption{BLER vs. SNR at INR = 5\,dB --- pending evaluation.}
\label{fig:bler_snr}
\end{figure}
```

---

## Step 5 — Determine Which Sections to Include

Include a section **only if there is real content to fill it**:

| Section | Include when | If absent |
|---------|-------------|-----------|
| Abstract | Always | — |
| 1. Introduction | Always | — |
| 2. System Model | Always | — |
| 3. Channel Model | Always | — |
| 4. Baseline Methods | Any baseline implemented | One sentence per unimplemented baseline |
| 5. GRU Neural Decoder | `neural_decoder.py` has real GRU code | One sentence: "scheduled for Phase 2a" |
| 6. Neural Branch Metric | `neural_bm.py` exists with real content | One sentence: "scheduled for Phase 2b" |
| 7. Interference Estimation | B5 in `baselines.py` implemented | One sentence |
| 8. Trellis Search | `search.py` has real content | One sentence |
| 9. Experimental Results | Any file in `results/` | Placeholder figures only |
| 10. Compute Budget | `compute_cost.py` exists | Analytical table with [TBD] measured values |
| 11. Ablation Study | Phase 4 data exists | Table with dashes |
| 12. Discussion | Results exist to discuss | One sentence |
| 13. This Week's Progress | Always | — |
| 14. Next Steps | Always | — |

---

## Step 6 — Write the LaTeX Document

**Key rules before writing:**
- Use `article` document class — NOT `IEEEtran` (breaks TOC, forces two columns)
- Describe only what is actually implemented; use "is planned for Phase X" for pending work
- Present tense for system description; past tense for completed experiments
- Never use Markdown table syntax inside `.tex` — always use `tabular` with `booktabs`
- For any section with pending content: one sentence stub only, do not pad with speculation

```latex
\documentclass[11pt,a4paper]{article}

% ── Packages ────────────────────────────────────────────────────────────────
\usepackage[margin=1.25in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{hyperref}
\usepackage[table]{xcolor}
\usepackage{subfig}
\usepackage{caption}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{titlesec}
\usepackage{parskip}
\usepackage{microtype}
\usepackage{fancyhdr}
\usepackage{mdframed}

% ── Page style ──────────────────────────────────────────────────────────────
\pagestyle{fancy}
\fancyhf{}
\rhead{EE597 Progress Report --- Week [N]}
\lhead{Abhishek Manjunath}
\cfoot{\thepage}

% ── Theorem environments ────────────────────────────────────────────────────
\newtheorem{definition}{Definition}[section]
\newtheorem{proposition}{Proposition}[section]
\newtheorem{remark}{Remark}[section]

% ── Convenience macros ──────────────────────────────────────────────────────
\newcommand{\bler}{\text{BLER}}
\newcommand{\snr}{\text{SNR}}
\newcommand{\inr}{\text{INR}}
\newcommand{\dfree}{d_{\text{free}}}

% ════════════════════════════════════════════════════════════════════════════
\begin{document}

\begin{titlepage}
\centering
\vspace*{2cm}
{\LARGE\bfseries Search-Designed Trellis Codes\\[0.4em] with Neural Decoding\par}
\vspace{1.5cm}
{\large Weekly Progress Report \quad\textbar\quad Week [N] \quad\textbar\quad [Month DD, YYYY]\par}
\vspace{1cm}
\rule{0.6\textwidth}{0.4pt}\\[0.8cm]
{\large Abhishek Manjunath\\[0.2em]
EE597 --- Advanced Topics in Communications\\[0.2em]
[University Name]\par}
\vfill
{\normalsize
\textbf{Current Phase:} [Phase X --- Name]\\[0.3em]
\textbf{Phase Status:} [In Progress / Complete]\\[0.3em]
\textbf{Codebase state as of:} [Date]}
\end{titlepage}

% ── Abstract ─────────────────────────────────────────────────────────────────
\begin{abstract}
% 3--4 sentences ONLY.
% State: (1) what the project investigates, (2) what has been completed this week,
% (3) the key numerical result if results exist, (4) the next milestone.
% Example structure (replace with real content):
%   This report documents progress on joint co-design of a 64-state convolutional trellis
%   and a neural decoder for AWGN channels impaired by periodic sinusoidal interference.
%   As of Week [N], [completed work]. [Key result if available.] [Next milestone.]
\end{abstract}

\newpage
\tableofcontents
\newpage

% ════════════════════════════════════════════════════════════════════════════
\section{Introduction and Motivation}

Modern communication systems deployed in industrial IoT environments encounter structured interference that standard AWGN channel models do not capture. Periodic sinusoidal interference from switching power supplies, variable-frequency motor drives, and other electromagnetic sources imposes a deterministic-yet-unknown additive component on the received signal. Classical minimum-distance decoders, designed for white Gaussian noise, are mismatched to this channel and suffer avoidable performance degradation.

This project investigates the \emph{joint co-design} of a convolutional trellis code and a learned neural decoder, both optimized for the sinusoidal interference regime. The central thesis is:

\begin{quote}
\itshape
``When channels are mismatched, structured, or computationally constrained,
the best end-to-end system may be obtained by co-designing the trellis and
a constrained learned decoder for the realistic impairment family.''
\end{quote}

The primary objective is not to surpass state-of-the-art codes on clean AWGN. The goals are:
\begin{enumerate}
  \item \textbf{Robustness:} Maintain reliable decoding across a wide range of interference-to-noise ratios.
  \item \textbf{Graceful degradation:} Performance degrades smoothly as interference parameters move outside the training distribution.
  \item \textbf{Compute efficiency:} All methods operate within $2.1 \times 10^6$ operations per block, enabling fair comparison.
\end{enumerate}

The system is evaluated using block error rate (BLER) at a target of $\bler = 10^{-3}$, across SNR $\in [0,10]$\,dB and INR $\in [-5, 15]$\,dB.

% ════════════════════════════════════════════════════════════════════════════
\section{System Model}

\subsection{Code Parameters}

The system uses BPSK modulation with constellation $x[t] \in \{-1,+1\}$. Each block encodes $K = 256$ information bits at rate $R = 1/2$, producing $N = 512$ coded symbols. Table~\ref{tab:params} lists all fixed system parameters.

\begin{table}[h]
\centering
\caption{Fixed System Parameters}
\label{tab:params}
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Modulation              & BPSK, $x[t] \in \{-1,+1\}$ \\
Information bits        & $K = 256$ \\
Coded symbols           & $N = 512$ \\
Code rate               & $R = 1/2$ \\
Trellis states          & $S = 64$ \\
Trellis structure       & Time-invariant, block-terminated \\
Computational budget    & $2.1 \times 10^6$ operations per block \\
SNR range               & $0$--$10$\,dB (focus: $5$\,dB) \\
INR range               & $-5$--$15$\,dB \\
BLER target             & $10^{-3}$ \\
MC trials (evaluation)  & $10^5$ \\
MC trials (search full) & $10^4$ \\
MC trials (search proxy)& $10^3$ \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Trellis Encoder}

\begin{definition}[Trellis Finite-State Machine]
A trellis encoder is the tuple $\mathcal{T} = (\mathcal{S},\,\mathcal{I},\,\mathcal{O},\,\delta,\,\lambda)$, where
$\mathcal{S} = \{0,\ldots,S-1\}$ is the state space,
$\mathcal{I} = \{0,1\}$ is the binary input alphabet,
$\mathcal{O} = \{-1,+1\}^2$ is the output symbol pair,
$\delta : \mathcal{S} \times \mathcal{I} \to \mathcal{S}$ is the next-state function, and
$\lambda : \mathcal{S} \times \mathcal{I} \to \mathcal{O}$ is the output labelling.
\end{definition}

Starting at initial state $s_0 = 0$, for each information bit $u_t \in \{0,1\}$:
\begin{align}
  s_{t+1} &= \delta(s_t,\, u_t), \\
  \mathbf{c}_t &= \lambda(s_t,\, u_t) \in \{-1,+1\}^2.
\end{align}
The block is \emph{terminated} by appending tail bits that return the encoder to state $0$, establishing $s_N = 0$ and enabling decoding with known boundary conditions.

\subsection{Structural Constraints on Valid Trellis Candidates}

All candidate trellises in Phase~3 must satisfy:
\begin{enumerate}
  \item \textbf{Non-catastrophic:} No cycle in the trellis produces zero output weight.
  \item \textbf{Full connectivity:} Every state is reachable from $s_0 = 0$, and $s_0$ is reachable from every state.
  \item \textbf{Minimum free distance:} $\dfree \geq 8$, providing correction of up to $\lfloor(8-1)/2\rfloor = 3$ bit errors.
\end{enumerate}

% ════════════════════════════════════════════════════════════════════════════
\section{Channel Model and Mathematical Formulation}

\subsection{Received Signal}

At discrete time $t \in \{0,\ldots,N-1\}$:
\begin{equation}
  r[t] = x[t] + n[t] + i[t],
  \label{eq:channel}
\end{equation}
where $n[t] \sim \mathcal{N}(0, \sigma^2)$ is i.i.d.\ AWGN and $i[t]$ is the sinusoidal interference.

\subsection{SNR and Noise Variance}

With unit symbol energy $E_s = 1$:
\begin{equation}
  \snr_{\text{lin}} = 10^{\snr_{\text{dB}}/10},
  \qquad
  \sigma^2 = \frac{1}{\snr_{\text{lin}}}.
\end{equation}

\subsection{Sinusoidal Interference}

\begin{equation}
  i[t] = A \cdot \sin\!\left(\frac{2\pi t}{P} + \varphi\right),
  \label{eq:interference}
\end{equation}
with parameters drawn independently per block:
\begin{align}
  P      &\sim \mathcal{U}_{\mathbb{Z}}\{8, \ldots, 32\}, \\
  \varphi &\sim \mathcal{U}[0, 2\pi).
\end{align}
Since $P \geq 8$ and $N = 512$, at least $\lfloor 512/8 \rfloor = 64$ complete cycles occur per block.

\subsection{Amplitude--INR Relationship}

A sinusoid $A\sin(\cdot)$ has average power $A^2/2$. The INR is therefore:
\begin{equation}
  \inr_{\text{lin}} = \frac{A^2/2}{\sigma^2}
  \implies
  A = \sqrt{2\,\inr_{\text{lin}}\,\sigma^2} = \sqrt{\frac{2\,\inr_{\text{lin}}}{\snr_{\text{lin}}}},
  \label{eq:amplitude}
\end{equation}
where $\inr_{\text{lin}} = 10^{\inr_{\text{dB}}/10}$.

\begin{remark}
At $\inr = \snr = 5$\,dB (both linear $\approx 3.16$), the amplitude $A = \sqrt{2} \approx 1.41$ exceeds the BPSK symbol magnitude of $1$. Interference therefore dominates the signal at this operating point, making the mismatched AWGN assumption highly inaccurate.
\end{remark}

\subsection{Training Distribution for Neural Methods}

All channel parameters are re-randomised per batch block:
\begin{equation}
  \snr_{\text{dB}} \sim \mathcal{U}[0, 10],\quad
  \inr_{\text{dB}} \sim \mathcal{U}[-5, 10],\quad
  P \sim \mathcal{U}_{\mathbb{Z}}\{8,\ldots,32\},\quad
  \varphi \sim \mathcal{U}[0, 2\pi).
\end{equation}
Evaluation uses both in-distribution and shifted test sets (e.g.\ $\inr > 10$\,dB, $P \notin [8,32]$) to assess generalisation.

% ════════════════════════════════════════════════════════════════════════════
\section{Baseline Methods}
% INSTRUCTION: One subsection per baseline that is at least partially implemented.
% For unimplemented baselines, write one sentence: "Baseline Bx is planned for Phase Y."
% Do not describe unimplemented code as if it works.

\subsection{B1: Mismatched Viterbi Decoder}

The primary baseline uses the NASA K=7 rate-1/2 convolutional code with generator polynomials $\mathbf{g}_1 = [1,0,1,1,0,1,1]$ and $\mathbf{g}_2 = [1,1,1,1,0,0,1]$ (octal: 133, 171), achieving $\dfree = 10$.

The Viterbi decoder applies squared Euclidean branch metrics under the assumption of pure AWGN, ignoring interference:
\begin{equation}
  \gamma_t^{\text{mm}}(s \to s')
  = -\sum_{j=1}^{2}\!\left(r_t^{(j)} - c_t^{(j)}\right)^2.
  \label{eq:mm_bm}
\end{equation}
The true metric under the actual channel (Eq.~\ref{eq:channel}) differs by a bias term $-2r_t^{(j)} i_t^{(j)} - (i_t^{(j)})^2$, which is structured and time-varying. B1 ignores this bias entirely.

\subsection{B2: Oracle Viterbi Decoder (Performance Ceiling)}

The oracle decoder knows $A$, $P$, $\varphi$ exactly per block, subtracts $i[t]$, and applies the matched metric to the cleaned signal $\tilde{r}[t] = r[t] - i[t] = x[t] + n[t]$:
\begin{equation}
  \gamma_t^{\text{oracle}}(s \to s')
  = -\sum_{j=1}^{2}\!\left(\tilde{r}_t^{(j)} - c_t^{(j)}\right)^2.
\end{equation}
B2 represents the performance ceiling. The gap B1 $-$ B2 (in dB at fixed BLER) quantifies the loss attributable solely to interference ignorance.

\subsection{B3: Random Trellis + Neural Decoder (Search Sanity Check)}

B3 uses a randomly generated 64-state trellis (subject to structural constraints) with the same GRU decoder as N1. Its sole purpose is to verify that Phase~3 search finds genuinely good structure: the searched trellis (S1) must beat B3 by $\geq 1$\,dB at $\bler = 10^{-3}$.

\subsection{B5: Interference Estimation and Cancellation + Viterbi}

B5 estimates $(A, P, \varphi)$ from the received signal via DSP, cancels the estimated interference, then applies mismatched Viterbi to the residual:

\begin{enumerate}
  \item \textbf{Frequency estimation.} Identify dominant frequency via periodogram:
  \begin{equation}
    \hat{P} = \arg\max_{P'} \left|\sum_{t=0}^{N-1} r[t]\, e^{-j2\pi t/P'}\right|^2.
  \end{equation}

  \item \textbf{Amplitude and phase estimation.} Given $\hat{P}$, solve least squares. The problem is linear in $[A'\cos\varphi',\; A'\sin\varphi']$ and admits a closed-form solution:
  \begin{equation}
    [\hat{A},\hat{\varphi}]
    = \arg\min_{A',\varphi'} \sum_{t=0}^{N-1}
      \!\left(r[t] - A'\sin\!\left(\tfrac{2\pi t}{\hat{P}} + \varphi'\right)\right)^2.
  \end{equation}

  \item \textbf{Cancellation and decoding.}
  $\tilde{r}[t] = r[t] - \hat{A}\sin(2\pi t/\hat{P} + \hat{\varphi})$, then B1 Viterbi on $\tilde{r}[t]$.
\end{enumerate}

B5 is critical for determining the regime where learning adds value: if DSP cancellation closes the gap between B1 and B2, the neural approach offers no benefit.

% ════════════════════════════════════════════════════════════════════════════
\section{GRU Neural Decoder (N1)}
% INSTRUCTION: Write at full depth only if neural_decoder.py has real GRU code.
% If stub only, replace section body with:
%   "The GRU end-to-end decoder (N1) is scheduled for Phase~2a and has not yet been implemented."

\subsection{Architecture}

The GRU decoder accepts the received sequence $\mathbf{r} = (r[0],\ldots,r[N-1])$ and produces estimated information bit probabilities $\hat{\mathbf{u}}$. At each step $t$:
\begin{align}
  z_t    &= \sigma\!\left(W_z r[t] + U_z h_{t-1} + b_z\right), \label{eq:gru_z}\\
  \rho_t  &= \sigma\!\left(W_\rho r[t] + U_\rho h_{t-1} + b_\rho\right), \label{eq:gru_r}\\
  \tilde{h}_t &= \tanh\!\left(W_h r[t] + U_h (\rho_t \odot h_{t-1}) + b_h\right),\label{eq:gru_cand}\\
  h_t    &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t, \label{eq:gru_up}\\
  \hat{u}_t &= \sigma\!\left(W_o h_t + b_o\right) \in (0,1), \label{eq:gru_out}
\end{align}
where $h_t \in \mathbb{R}^H$ with $H = 64$ units.

\subsection{Computational Budget}

Each GRU step requires three gate computations plus one output projection. The dominant cost is the recurrent matrix--vector multiply $U h_{t-1}$ at $H^2$ MACs each:
\begin{equation}
  \text{MACs}_{\text{step}}
  = 3(1 \cdot H + H^2) + (1 \cdot H + H^2) + H
  = 4H + 4H^2 + H.
\end{equation}
Over $N = 512$ steps with $H = 64$:
\begin{equation}
  \text{MACs}_{\text{total}}
  = N(4H + 4H^2 + H)
  = 512 \times 16{,}704
  \approx 8.55 \times 10^6.
\end{equation}

\begin{remark}
The analytical MAC count exceeds the $2.1$M budget by approximately $4\times$. The budget convention (MACs vs.\ FLOPs vs.\ multiply-adds) must be confirmed against the \texttt{compute\_cost.py} profiler output. Architecture adjustments --- smaller $H$, bidirectional layers, or reduced precision --- may be required if the profiler confirms the overrun.
\end{remark}

\subsection{Training Procedure}

The decoder minimises binary cross-entropy over information bit positions:
\begin{equation}
  \mathcal{L}(\theta)
  = -\frac{1}{K}\sum_{t \in \mathcal{I}}
    \bigl[u_t \log \hat{u}_t + (1-u_t)\log(1-\hat{u}_t)\bigr],
  \label{eq:bce}
\end{equation}
where $\mathcal{I}$ indexes the $K = 256$ information bit positions. Channel parameters are re-sampled per batch. Optimisation uses Adam ($\eta = 10^{-3}$, cosine annealing) with gradient clipping at norm $1.0$.

% ════════════════════════════════════════════════════════════════════════════
\section{Neural Branch Metric Estimator (N2)}
% INSTRUCTION: Write at full depth if neural_bm.py is implemented.
% Write as design notes (as below) if in ideation phase.
% If entirely pending, write one sentence only.

\subsection{Motivation}

N2 preserves the Viterbi dynamic-programming structure while replacing only the branch metric $\gamma_t(s \to s')$ with a small neural network. Unlike the end-to-end N1 decoder, this hybrid retains the combinatorial optimality guarantee of Viterbi given its metrics, while learning to estimate those metrics in a way that accounts for interference.

\subsection{Formulation}

The standard metric (Eq.~\ref{eq:mm_bm}) is replaced by:
\begin{equation}
  \hat{\gamma}_t(s \to s') = f_\theta\!\left(\mathbf{r}_{t-K_c:t+K_c},\; \mathbf{c}(s,s')\right),
\end{equation}
where $f_\theta$ is a small neural network, $\mathbf{r}_{t-K_c:t+K_c}$ is a context window of received samples, and $\mathbf{c}(s,s') \in \{-1,+1\}^2$ is the expected codeword for transition $(s \to s')$.

\subsection{Computational Constraint}

With $S \times 2 = 128$ transitions per step and $N = 512$ steps:
\begin{equation}
  128 \times N \times \text{ops}_{f_\theta} \leq 2.1 \times 10^6
  \implies
  \text{ops}_{f_\theta} \leq 32.
\end{equation}
This limits $f_\theta$ to a very small MLP (e.g.\ input $\to$ 16 units $\to$ scalar).

\subsection{Open Design Questions}

The following are empirical questions to be answered in Phase~2b experiments:
\begin{itemize}
  \item How large must context window $K_c$ be? Sweep: $K_c \in \{0, 2, 4, 8, 16\}$.
  \item Should $t \bmod \hat{P}$ be an explicit feature, or should periodicity be learned implicitly?
  \item Is a shared-weight architecture (one forward pass per time step, outputs all 128 metrics) more efficient than 128 separate evaluations?
\end{itemize}

% ════════════════════════════════════════════════════════════════════════════
\section{Trellis Search Algorithm}
% INSTRUCTION: Include only if search.py has real content. Otherwise one sentence.

\subsection{Search Problem Formulation}

\begin{equation}
  \mathcal{T}^{*}
  = \arg\min_{\mathcal{T} \,\in\, \mathcal{F}}
    \bler\!\left(\mathcal{T},\; f_{\theta^*(\mathcal{T})}\right)\big|_{\snr=5\,\text{dB},\; \inr=5\,\text{dB}},
\end{equation}
where $\mathcal{F}$ is the feasible set (non-catastrophic, connected, $\dfree \geq 8$) and $\theta^*(\mathcal{T})$ are decoder weights trained for trellis $\mathcal{T}$.

\subsection{Evolutionary Algorithm}

\begin{algorithm}[h]
\caption{Evolutionary Trellis Search}
\label{alg:ea}
\begin{algorithmic}[1]
\Require Population size $M$, generations $G$, full-eval interval $G_{\text{full}}$
\State \textbf{Initialise} $\mathcal{P}_0$: perturb K=7 code (rewire transitions, flip output labels; reject infeasible)
\For{$g = 1, 2, \ldots, G$}
  \ForAll{$\mathcal{T} \in \mathcal{P}_g$}
    \State Evaluate cheap fitness: $1{,}000$ MC trials, single $(\snr,\inr)$ point, no NN retrain
  \EndFor
  \If{$g \bmod G_{\text{full}} = 0$}
    \State Re-evaluate top-$k$: $10{,}000$ MC trials, multiple SNR points, retrain NN
  \EndIf
  \State Select parents via tournament selection (tournament size $\tau$)
  \State Apply mutation: rewire $m$ transitions; flip $f$ output labels
  \State Apply crossover: combine transition tables from two parents column-wise
  \State Discard offspring violating feasibility constraints
  \State $\mathcal{P}_{g+1} \leftarrow$ top-$M$ survivors by cheap fitness
\EndFor
\State \Return $\mathcal{T}^{*}$ with lowest full-eval fitness across all generations
\end{algorithmic}
\end{algorithm}

\subsection{Two-Tier Fitness Evaluation}

\begin{table}[h]
\centering
\caption{Two-Tier Fitness Evaluation Protocol}
\label{tab:fitness}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Tier} & \textbf{MC Trials} & \textbf{SNR/INR Points} & \textbf{NN Retrain} & \textbf{Frequency} \\
\midrule
Cheap proxy & $1{,}000$ & Single $(5\,\text{dB}, 5\,\text{dB})$ & No  & Every generation \\
Full eval   & $10{,}000$ & Multiple ($3, 5, 7$\,dB)            & Yes & Every $G_{\text{full}}$ gens \\
\bottomrule
\end{tabular}
\end{table}

% ════════════════════════════════════════════════════════════════════════════
\section{Compute Budget Analysis}

All methods must operate within $2.1 \times 10^6$ operations per block. Table~\ref{tab:compute} reports both analytical formulas and, where available, measured values from \texttt{compute\_cost.py}.

\begin{table}[h]
\centering
\caption{Computational Cost Per Block}
\label{tab:compute}
\begin{tabular}{@{}lllr@{}}
\toprule
\textbf{Method} & \textbf{Dominant operation} & \textbf{Analytical formula} & \textbf{Measured (M ops)} \\
\midrule
B1: Mismatched Viterbi & Add-compare-select (ACS)   & $S^2 \cdot N / 2$          & [TBD] \\
B2: Oracle Viterbi     & ACS                         & $S^2 \cdot N / 2$          & [TBD] \\
B5: IC + Viterbi       & FFT + ACS                   & $N\log_2\!N + S^2 N/2$     & [TBD] \\
N1: GRU ($H=64$)       & Recurrent GEMM              & $N(4H + 4H^2 + H)$         & [TBD] \\
N2: Neural BM          & Per-transition MLP          & $128 N \cdot \text{ops}_{f_\theta}$ & [TBD] \\
\bottomrule
\end{tabular}
\end{table}

\noindent\textit{[TBD] entries are populated with profiled values as each method is completed.}

% ════════════════════════════════════════════════════════════════════════════
\section{Experimental Results}
% INSTRUCTION: Include one subsection per result type with real data.
% For each figure: include \includegraphics if file exists; use \framebox placeholder otherwise.
% Never hard-code numbers --- always transcribe from results/*.npz or results/*.json.

\subsection{Validation Tests}

\begin{table}[h]
\centering
\caption{Validation Test Results}
\label{tab:tests}
\begin{tabular}{@{}llc@{}}
\toprule
\textbf{Test} & \textbf{Description} & \textbf{Status} \\
\midrule
Noiseless round-trip & Encode $\to$ decode in zero noise: $0$ bit errors expected & [PASS / FAIL / PENDING] \\
AWGN theory match    & B1 BLER within $\pm 0.2$\,dB of theoretical K=7 curve       & [PASS / FAIL / PENDING] \\
Oracle dominance     & B2 BLER $\leq$ B1 BLER at all tested (SNR, INR) pairs        & [PASS / FAIL / PENDING] \\
\bottomrule
\end{tabular}
\end{table}

\subsection{BLER vs.\ SNR Performance}

% Replace the \framebox block with the line below once results/bler_vs_snr.png exists:
% \includegraphics[width=0.82\textwidth]{../results/bler_vs_snr.png}

\begin{figure}[h]
\centering
\framebox[0.82\textwidth]{%
  \parbox{0.80\textwidth}{\centering\vspace{2cm}
  \textit{[Figure pending: run \texttt{eval.py} to generate BLER vs.\ SNR curves.]}
  \vspace{2cm}}}
\caption{BLER vs.\ SNR at INR = 5\,dB. Monte Carlo trials: $10^5$.}
\label{fig:bler_snr}
\end{figure}

\subsection{BLER vs.\ INR --- Robustness Analysis}

\begin{figure}[h]
\centering
\framebox[0.82\textwidth]{%
  \parbox{0.80\textwidth}{\centering\vspace{2cm}
  \textit{[Figure pending: INR sweep at SNR = 5\,dB.]}
  \vspace{2cm}}}
\caption{BLER vs.\ INR at SNR = 5\,dB for all implemented methods.}
\label{fig:bler_inr}
\end{figure}

\subsection{Neural Decoder Training Curves}
% INSTRUCTION: Include this subsection only after training has been run.

\begin{figure}[h]
\centering
\framebox[0.82\textwidth]{%
  \parbox{0.80\textwidth}{\centering\vspace{2cm}
  \textit{[Figure pending: N1 GRU training loss vs.\ epoch.]}
  \vspace{2cm}}}
\caption{Training loss (BCE, Eq.~\ref{eq:bce}) vs.\ epoch for the N1 GRU decoder.}
\label{fig:training}
\end{figure}

% ════════════════════════════════════════════════════════════════════════════
\section{Ablation Study}
% INSTRUCTION: Include in Phase 4. Until then, show the planned table structure
% with dashes to make the experimental design visible to the supervisor.

Table~\ref{tab:ablation} presents the planned ablation matrix. All methods operate at the equal $2.1 \times 10^6$ op budget. Results are populated during Phase~4.

\begin{table}[h]
\centering
\caption{Ablation Matrix --- BLER at SNR = 5\,dB, INR = 5\,dB, $10^5$ MC trials}
\label{tab:ablation}
\begin{tabular}{@{}llllc@{}}
\toprule
\textbf{ID} & \textbf{Trellis} & \textbf{Decoder} & \textbf{Role} & \textbf{BLER} \\
\midrule
B1 & K=7 classical & Mismatched Viterbi  & Primary baseline          & --- \\
B2 & K=7 classical & Oracle Viterbi      & Performance ceiling       & --- \\
B5 & K=7 classical & IC + Viterbi        & DSP cancellation baseline & --- \\
N1 & K=7 classical & GRU end-to-end      & Decoder contribution      & --- \\
N2 & K=7 classical & Neural BM + Viterbi & Structured hybrid         & --- \\
B3 & Random         & GRU end-to-end     & Search sanity check       & --- \\
S1 & Searched       & GRU end-to-end     & \textbf{Core result}      & --- \\
S2 & Searched       & Neural BM          & Optional extension        & --- \\
\bottomrule
\end{tabular}
\end{table}

% ════════════════════════════════════════════════════════════════════════════
\section{Discussion and Analysis}
% INSTRUCTION: Write this section when experimental results exist.
% If no results yet, write: "Discussion will be added once Phase 1 evaluation results are available."
% When results exist, address the questions below in prose paragraphs (not bullet points).

% Questions to address when results exist:
% - At what INR does B5 fail and why (estimation error regime)?
% - Does the GRU exploit periodicity or learn something else?
% - Does the searched trellis show implicit interleaving structure?
% - Which method dominates in which (SNR, INR) regime?
% - How does performance degrade under distribution shift (P outside [8,32], INR > 10 dB)?

% ════════════════════════════════════════════════════════════════════════════
\section{This Week's Progress}

The following tasks were completed during the reporting period (Week~[N]):

\begin{itemize}
  \item[$\bullet$] [Specific completed task 1 --- be precise: name the function, module, and what was verified.]
  \item[$\bullet$] [Specific completed task 2]
  \item[$\bullet$] [Specific completed task 3]
\end{itemize}

\noindent Issues encountered and resolved this week:
\begin{itemize}
  \item[$\circ$] [Describe issue and resolution concisely.]
\end{itemize}

\noindent Issues currently open:
\begin{itemize}
  \item[$\circ$] [Describe unresolved issue and planned resolution.]
\end{itemize}

% ════════════════════════════════════════════════════════════════════════════
\section{Next Steps}

The following tasks are scheduled for the next reporting period, drawn from the Phase~[X] checklist in \texttt{CLAUDE.md}:

\begin{enumerate}
  \item [Task from CLAUDE.md pending checklist]
  \item [Task from CLAUDE.md pending checklist]
  \item [Task from CLAUDE.md pending checklist]
\end{enumerate}

\noindent The success criterion for the next milestone is: [state the exact numeric criterion from CLAUDE.md, e.g.\ ``N1 beats B1 by $\geq 0.5$\,dB at $\bler = 10^{-3}$''].

% ════════════════════════════════════════════════════════════════════════════
\end{document}
```

---

## Step 7 — Compile to PDF

```bash
cd /path/to/report/

# First pass: builds aux, TOC entries, cross-references
pdflatex -interaction=nonstopmode -halt-on-error weekly_report_week_N.tex

# Second pass: resolves TOC page numbers and \ref labels
pdflatex -interaction=nonstopmode -halt-on-error weekly_report_week_N.tex

# Check for fatal errors
grep "^!" weekly_report_week_N.log | head -20

# Check for missing figures (these cause hard failures)
grep -i "cannot find\|No such file" weekly_report_week_N.log
```

**If packages are missing:**
```bash
sudo apt-get install -y texlive-full
# or individually:
sudo tlmgr install mdframed booktabs algorithms algorithmicx
```

**Critical:** Never add an `\includegraphics` reference to a file that does not exist. Use the `\framebox` placeholder shown in the template instead. Missing image files cause hard compilation failures with an uninformative error message.

---

## Step 8 — Quality Checklist

**Content accuracy:**
- [ ] Abstract contains only completed work — no aspirational claims stated as fact
- [ ] Every module described as "implemented" was confirmed in Step 2b by reading its code, not just checking file existence
- [ ] Every numerical value in the body comes from `results/` files or `CLAUDE.md` — not from memory
- [ ] "This Week's Progress" lists only work done this week, not cumulative progress
- [ ] "Next Steps" matches exactly the unchecked `[ ]` items in `CLAUDE.md`

**Mathematical consistency:**
- [ ] $K$, $N$, $R$, $S$, $H$ defined once in Table~1 and reused without re-definition
- [ ] SNR and INR always specified as dB or linear explicitly
- [ ] All `\ref{eq:...}` and `\ref{tab:...}` labels resolve (no "??" in output PDF)

**LaTeX hygiene:**
- [ ] No Markdown table syntax (`|---|`) anywhere inside the `.tex` file
- [ ] No `\includegraphics` pointing to a file that was not confirmed to exist in Step 4
- [ ] No `[TBD]` in sections where real measured data is available
- [ ] Two `pdflatex` passes completed (TOC page numbers correct)
- [ ] No `\cite` commands (no references section)
- [ ] Zero errors in `.log` file (overfull hbox warnings acceptable)

---

## Step 9 — Phase-Specific Depth Guide

| Phase | Sections at full depth | Sections as one-sentence stubs |
|-------|------------------------|-------------------------------|
| Phase 0 | 1 (Intro), 2 (System), 3 (Channel) | 4--14 |
| Phase 1 | 1--4, 10 (Compute), 9 (Validation only) | 5--8, 11--12 |
| Phase 2a | 1--5, 9 (BLER + training curves), 10 | 6--8, 11--12 |
| Phase 2b | 1--6, 9, 10 | 7--8, 11--12 |
| Phase 3 | 1--10 | 11 (Discussion) |
| Phase 4 | All sections | — |

**Rule:** Write less and be accurate rather than write more and speculate. One accurate sentence is more valuable to a supervisor than a paragraph of hedged guesswork about unimplemented components.
